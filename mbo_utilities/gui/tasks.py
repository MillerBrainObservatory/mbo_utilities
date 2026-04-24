"""
Task registry for background worker processes.

This module contains the actual logic for background tasks:
- TaskMonitor: Helper for reporting progress to a JSON sidecar file.
- task_save_as: Generic array conversion/saving.
- task_suite2p: Suite2p pipeline with safe serial extraction + parallel processing.
- TASKS: Registry mapping task names to functions.
"""

from __future__ import annotations

import json
import logging
import time
import os
import traceback
from pathlib import Path

from mbo_utilities import imread
from mbo_utilities.writer import imwrite
from mbo_utilities.arrays._registration import (
    compute_axial_shifts,
    validate_axial_shifts,
)
from mbo_utilities.metadata import get_param

logger = logging.getLogger("mbo.worker.tasks")


class _ChannelView:
    """4D TZYX view of a single channel from 5D TCZYX data.

    Wraps a lazy array and presents it as 4D by fixing the channel index.
    Used to feed single-channel data to pipelines that expect TZYX input.
    """

    def __init__(self, arr, channel_0idx: int):
        self._arr = arr
        self._ch = channel_0idx
        self._metadata_override = None

    @property
    def shape(self):
        s = self._arr.shape
        return (s[0], s[2], s[3], s[4])

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        s = self._arr.shape5d if hasattr(self._arr, "shape5d") else self._arr.shape
        return (s[0], 1, s[2], s[3], s[4])

    @property
    def shape5d(self) -> tuple[int, int, int, int, int]:
        return self._shape5d()

    @property
    def ndim(self):
        return 4

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def metadata(self):
        if self._metadata_override is not None:
            return self._metadata_override
        md = dict(getattr(self._arr, "metadata", {}))
        md["num_color_channels"] = 1
        return md

    @metadata.setter
    def metadata(self, value):
        self._metadata_override = value

    @property
    def filenames(self):
        return getattr(self._arr, "filenames", [])

    @property
    def num_planes(self):
        return self._arr.shape[2]

    @property
    def num_color_channels(self):
        return 1

    @property
    def num_channels(self):
        return self._arr.shape[2]

    @property
    def dims(self):
        return ("T", "Z", "Y", "X")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        return self._arr[(key[0], self._ch, key[1], key[2], key[3])]

    def _imwrite(self, outpath, planes=None, ext=".tiff", **kwargs):
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(
            self, outpath, planes=planes, ext=ext, **kwargs
        )


class TaskMonitor:
    """
    Helper to report task progress to a JSON sidecar file.
    Plugins into the ProcessManager on the GUI side.
    """

    def __init__(self, output_dir: Path | str, uuid: str | None = None):
        self.output_dir = Path(output_dir)
        self.pid = os.getpid()
        self.uuid = uuid
        # Sidecar file: progress_{pid}.json or progress_{uuid}.json in the log directory
        # We assume output_dir might be the data dir, so let's try to find a logs dir
        # or just put it in a standard location if possible.
        # Actually, ProcessManager expects to just read info.
        # Let's write to a standard location that ProcessManager knows about.
        # Standard: ~/.mbo/logs/progress_{pid}.json
        self.log_dir = Path.home() / ".mbo" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.uuid:
            self.progress_file = self.log_dir / f"progress_{self.uuid}.json"
        else:
            self.progress_file = self.log_dir / f"progress_{self.pid}.json"

    def update(self, progress: float, message: str, state: str = "running", details: dict | None = None):
        """
        Update progress file.
        progress: 0.0 to 1.0
        message: Short status description
        state: "running", "completed", "error".
        """
        data = {
            "pid": self.pid,
            "uuid": self.uuid,
            "timestamp": time.time(),
            "status": state,
            "progress": progress,
            "message": message,
            "details": details or {}
        }
        try:
            # Write atomically to avoid race conditions with readers.
            # On windows the replace can fail with WinError 5 when the gui
            # has the sidecar open for read (default share flags deny
            # rename); retry briefly so a reader blip doesn't drop an
            # update.
            tmp_file = self.progress_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            for _ in range(10):
                try:
                    tmp_file.replace(self.progress_file)
                    break
                except PermissionError:
                    time.sleep(0.05)
        except Exception:
            pass  # Non-blocking

    def finish(self, message: str = "Task completed"):
        self.update(1.0, message, state="completed")

    def fail(self, error: str, details: str | dict | None = None):
        self.update(0.0, f"Error: {error}", state="error", details=details)


def task_save_as(args: dict, logger: logging.Logger) -> None:
    """
    Generic save/convert task.
    Supports saving any readable array to .zarr, .h5, .tiff, .bin, etc.
    """
    monitor = TaskMonitor(args.get("output_dir", "."), uuid=args.get("_uuid"))
    monitor.update(0.0, "Initializing save task...")

    input_path = args["input_path"]
    output_path = Path(args["output_path"]) # Full path including extension
    output_dir = args.get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Optional params
    planes = args.get("planes")
    channels = args.get("channels")  # list of 1-based channel indices or None
    rois = args.get("rois")
    frames = args.get("frames")  # list of 1-based frame indices or None
    metadata = args.get("metadata", {})

    # Load array
    monitor.update(0.05, f"Loading {Path(input_path).name}...")
    logger.info(f"Loading {input_path}")
    arr = imread(input_path)

    # Apply on-the-fly settings if supported
    if hasattr(arr, "fix_phase"):
        arr.fix_phase = args.get("fix_phase", True)
    if hasattr(arr, "use_fft"):
        arr.use_fft = args.get("use_fft", True)

    # Handle Z-Registration
    register_z = args.get("register_z", False)
    if register_z:
        monitor.update(0.05, "Checking Z-registration status...")

        # merge existing array metadata with overrides
        combined_meta = getattr(arr, "metadata", {}).copy()
        combined_meta.update(metadata)

        # axial registration knobs (optional; gui can set via task args)
        axial_max_frames = int(args.get("max_frames", 200))
        axial_chunk_frames = int(args.get("chunk_frames", 10))
        axial_max_reg_xy = int(args.get("max_reg_xy", 150))

        def _reg_cb(progress, msg=""):
            monitor.update(0.05 + 0.05 * progress, f"Registration: {msg}")

        num_planes = get_param(combined_meta, "nplanes") or getattr(arr, "num_planes", 1)
        reg_error: Exception | None = None

        if validate_axial_shifts(combined_meta, num_planes):
            logger.info("using plane_shifts already present in metadata.")
            metadata["plane_shifts"] = list(combined_meta["plane_shifts"])
            metadata["apply_shift"] = True
            monitor.update(0.1, "Z-registration ready (cached).")
        else:
            logger.info("computing axial plane shifts...")
            try:
                compute_axial_shifts(
                    arr,
                    metadata=combined_meta,
                    max_frames=axial_max_frames,
                    chunk_frames=axial_chunk_frames,
                    max_reg_xy=axial_max_reg_xy,
                    progress_callback=_reg_cb,
                )
            except Exception as e:
                reg_error = e

            if reg_error is None and validate_axial_shifts(combined_meta, num_planes):
                metadata["plane_shifts"] = list(combined_meta["plane_shifts"])
                metadata["plane_shifts_params"] = combined_meta.get("plane_shifts_params")
                metadata["apply_shift"] = True
                monitor.update(0.1, "Z-registration ready.")
            else:
                if reg_error is not None:
                    logger.warning(
                        f"Z-registration failed: {reg_error}. proceeding without axial shift."
                    )
                else:
                    logger.warning(
                        "axial registration produced no valid plane_shifts. "
                        "proceeding without axial shift."
                    )
                metadata["apply_shift"] = False

    monitor.update(0.1, f"Saving to {output_path.name}...")

    # Define progress callback for imwrite. Multiple writers in mbo call
    # this with different conventions:
    #   - cb(fraction)                 — _writers.py per-chunk
    #   - cb(current, total)           — generic per-frame writers
    #   - cb(1.0, "Complete")          — _imwrite_base completion sentinel
    # so we have to peek at `total` to figure out which form we got. If
    # `total` is numeric, treat as (current, total) progress; otherwise
    # treat it as a message override.
    def _progress_cb(current, total=None, **kwargs):
        msg = "Writing..."
        if total is None:
            # fraction-only form
            try:
                p = 0.1 + 0.9 * float(current)
            except (TypeError, ValueError):
                p = 0.5
        elif isinstance(total, (int, float)) and not isinstance(total, bool):
            # (current, total) form — guard against div-by-zero by clamping
            # the divisor to >= 1 (max, not min — the previous code used
            # min(total,1) which always returns 1 for total>=1 and made the
            # division a no-op).
            denom = max(float(total), 1.0)
            p = 0.1 + 0.9 * (float(current) / denom)
            msg = f"Writing frame {current}/{total}"
        else:
            # second positional arg is a string (e.g. "Complete") — treat
            # `current` as a fraction and `total` as the status message.
            try:
                p = 0.1 + 0.9 * float(current)
            except (TypeError, ValueError):
                p = 1.0
            msg = str(total)

        # throttle updates to avoid IO thrashing
        monitor.update(p, msg)

    try:
        # Determine extension: explicit > from path > default
        ext = args.get("ext")
        if not ext:
            ext = output_path.suffix if output_path.suffix else ".zarr"

        # If output_path is a directory-like path (no extension) and we invoke imwrite,
        # it treats it as a directory.
        # If output_path has extension, it treats it as file.
        # However, for _imwrite (ScanImageArray), it generally expects 'outdir'.

        # Ensure output directory exists
        if not output_path.suffix:
             output_path.mkdir(parents=True, exist_ok=True)
        else:
             output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing to {output_path} with ext={ext}")

        # We use the internal _imwrite method of the array if available for optimization,
        # else generic imwrite.
        if hasattr(arr, "_imwrite"):
            # ScanImageArray._imwrite takes 'outdir'
            # If output_path is a file path, we should probably pass parent as outdir?
            # Existing usage suggests outdir is the target folder.
            # If saving to zarr, outdir is the .zarr folder.
            arr._imwrite(
                outpath=output_path,
                ext=ext,
                planes=planes,
                channels=channels,
                frames=frames,
                roi=rois,
                overwrite=True,
                metadata_overrides=metadata,
                progress_callback=_progress_cb,
                **args.get("kwargs", {})
            )
        else:
            # Fallback for generic arrays
            imwrite(
                arr,
                output_path,
                ext=ext,
                planes=planes,
                channels=channels,
                frames=frames,
                roi=rois,
                overwrite=True,
                metadata=metadata,
                progress_callback=_progress_cb,
                **args.get("kwargs", {})
            )

        monitor.finish(f"Saved to {output_path.name}")
        logger.info("Save completed")

    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        raise


def task_suite2p(args: dict, logger: logging.Logger) -> None:
    """
    Suite2p pipeline task.

    Delegates entirely to lbm_suite2p_python.pipeline which handles:
    - input loading (paths, lists of paths, arrays)
    - plane iteration for volumetric data
    - binary extraction and suite2p processing
    """
    from lbm_suite2p_python import pipeline

    monitor = TaskMonitor(args.get("output_dir", "."), uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing Suite2p pipeline...")

    input_path = args["input_path"]
    output_dir = Path(args["output_dir"])
    planes = args.get("planes")
    # the GUI's target_timepoints defaults to -1 (meaning "all frames").
    # normalize to None so downstream code (lsp's pipeline, writer_kwargs)
    # doesn't propagate -1 as a literal num_frames — that makes
    # generate_plane_dirname produce "zplane01" (no tp suffix) and can
    # confuse frame-count arithmetic in the or-chain.
    num_timepoints = args.get("num_timepoints")
    if num_timepoints is not None and num_timepoints <= 0:
        num_timepoints = None
    # Config can arrive as either a legacy flat `ops` dict OR as the new
    # upstream-shaped pair (`settings`, `db`). Normalize to flat ops here
    # so the rest of this function keeps working with its existing
    # mutation pipeline; the lbm run_plane patch re-splits at the bottom.
    ops = dict(args.get("ops") or {})
    incoming_settings = args.get("settings")
    incoming_db = args.get("db")
    if incoming_settings is not None or incoming_db is not None:
        try:
            from lbm_suite2p_python.db_settings import db_settings_to_ops
            flattened = db_settings_to_ops(incoming_db, incoming_settings)
        except ImportError:
            flattened = {**(incoming_db or {}), **(incoming_settings or {})}
        # ops overrides on top of the flattened base
        flattened.update(ops)
        ops = flattened
    s2p_settings = args.get("s2p_settings", {})

    # Merge GUI-set custom metadata (e.g. dz from the metadata editor)
    # into ops. The user explicitly set these in the editor, so they win
    # over both ops defaults and the source file's voxel size.
    custom_metadata = args.get("custom_metadata") or {}
    if custom_metadata:
        ops.update(custom_metadata)
        logger.info(f"Applied custom metadata to ops: {sorted(custom_metadata.keys())}")

    # ALWAYS load source metadata and propagate fs/dz to ops, even when
    # no stride selection is present. lbm's default_ops ships fs=10.0
    # hardcoded, and that leaks into ops.npy unless we explicitly
    # replace it with the source value here. This was the root cause of
    # repeated "ops.npy fs=10 even though my source is 14Hz" reports.
    from mbo_utilities.metadata import OutputMetadata, get_param

    _src_arr = None
    try:
        _src_arr = imread(input_path)
        src_meta = dict(getattr(_src_arr, "metadata", {}) or {})
        src_shape = (
            tuple(_src_arr.shape5d) if hasattr(_src_arr, "shape5d") else None
        )
    except Exception as e:
        logger.warning(
            f"task_suite2p: could not load source metadata for reactive "
            f"fs/dz scaling: {e}. Falling back to ops defaults."
        )
        src_meta = {}
        src_shape = None

    # Carry over the user's editor metadata so it wins over the source.
    if custom_metadata:
        src_meta.update(custom_metadata)

    raw_src_fs = get_param(src_meta, "fs") if src_meta else None
    raw_src_dz = get_param(src_meta, "dz") if src_meta else None
    logger.info(
        f"task_suite2p: source fs={raw_src_fs}, dz={raw_src_dz}, "
        f"input_path={input_path!r}"
    )

    if raw_src_fs is None:
        logger.warning(
            "task_suite2p: source metadata has NO fs field — "
            "ops.npy fs will fall through to lbm_suite2p_python's default "
            "(10 Hz). To fix: set fs via the metadata editor, or fix the "
            "source TIFF metadata."
        )
    else:
        # Pull the raw source fs into ops, replacing lbm's hardcoded 10.
        # The reactive scaling block below may overwrite this with a
        # stride-scaled value when tp_indices is non-None.
        if ops.get("fs") in (None, 10.0):
            ops["fs"] = float(raw_src_fs)
            logger.info(f"task_suite2p: replaced lbm default fs=10 with source fs={raw_src_fs}")

    # Same for dz — only override the lbm default, not a user-set value.
    if raw_src_dz is not None:
        if ops.get("dz") in (None, 1.0):
            ops["dz"] = float(raw_src_dz)
            logger.info(f"task_suite2p: pulled source dz={raw_src_dz} into ops")

    # Reactively scale fs/dz via OutputMetadata when the user has a
    # timepoint or z-plane stride selection. This block REPLACES the
    # source values pulled above with stride-scaled versions when
    # appropriate.
    tp_indices = args.get("tp_indices")
    selected_planes_0based = args.get("selected_planes_0based")
    if tp_indices is not None or selected_planes_0based is not None:
        if src_meta:
            selections = {}
            if tp_indices is not None:
                selections["T"] = list(tp_indices)
            if selected_planes_0based is not None:
                selections["Z"] = list(selected_planes_0based)

            out_meta = OutputMetadata(
                source=src_meta,
                source_shape=src_shape,
                source_dims=("T", "C", "Z", "Y", "X"),
                selections=selections,
            )

            scaled = out_meta.to_dict()
            # Merge reactive values into ops:
            # - When the scaled value is non-None, write it (the
            #   normal path — source had the field, we scaled it).
            # - When the scaled value IS None, EXPLICITLY remove the
            #   key from ops so downstream code can't pick up a stale
            #   default and pretend it's the answer. The user will see
            #   None in ops.npy and immediately know something's wrong,
            #   instead of silently getting fs=10.
            for key in ("fs", "dz", "dx", "dy", "z_step",
                        "umPerPixZ", "umPerPixX", "umPerPixY"):
                if key in scaled and scaled[key] is not None:
                    ops[key] = scaled[key]
                elif key in ("fs", "dz") and key in ops:
                    # only fs/dz get the strict "remove if missing"
                    # treatment — the others have sensible 1.0 defaults
                    # in get_voxel_size and aren't worth surfacing as
                    # missing-data signals.
                    ops.pop(key, None)
            logger.info(
                f"task_suite2p: applied reactive metadata -> "
                f"fs={ops.get('fs')}, dz={ops.get('dz')} "
                f"(t-stride from {len(tp_indices) if tp_indices else 0} indices, "
                f"z-stride from {len(selected_planes_0based) if selected_planes_0based else 0} planes)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    # display name for logging
    if isinstance(input_path, list):
        if len(input_path) == 1:
            input_path = input_path[0]  # unwrap single-item list
            display_name = Path(input_path).name
        else:
            display_name = f"{len(input_path)} files ({Path(input_path[0]).name}...)"
    else:
        display_name = Path(input_path).name

    monitor.update(0.05, f"Running pipeline: {display_name}...")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Planes: {planes}")

    # axial registration: produce per-plane offsets and plumb them into
    # ops so lsp's run_plane can pass them through to mbo's _write_plane
    # as shift_vector. the shift_vector path bypasses _write_plane's
    # data_raw.bin guard, so the suite2p binary gets written with each
    # plane translated to its globally-aligned y/x.
    #
    # sources for shifts (priority order):
    #   1. source metadata already carries plane_shifts (computed earlier)
    #   2. compute fresh via compute_axial_shifts on the source array
    #
    # Branch A: if the source file was saved with axial shifts already
    # baked into pixel data, it carries `_pad_yrange`/`_pad_xrange` in its
    # metadata. Propagate them into ops so lsp's cellpose clip uses the
    # valid region, and disable register_z so we don't re-shift already-
    # aligned pixels.
    register_z = args.get("register_z", False)
    if "_pad_yrange" in src_meta and "_pad_xrange" in src_meta:
        ops["_pad_yrange"] = list(src_meta["_pad_yrange"])
        ops["_pad_xrange"] = list(src_meta["_pad_xrange"])
        if register_z:
            logger.info(
                "task_suite2p: source has baked-in axial shifts; "
                "skipping register_z to avoid double-application."
            )
            register_z = False
        else:
            logger.info(
                "task_suite2p: detected baked-in axial shifts from source "
                f"metadata (_pad_yrange={ops['_pad_yrange']}, "
                f"_pad_xrange={ops['_pad_xrange']})."
            )

    if register_z:
        # reuse the array already loaded at the top of the task
        _src_arr_for_reg = _src_arr
        if _src_arr_for_reg is None:
            try:
                _src_arr_for_reg = imread(input_path)
            except Exception as e:
                logger.warning(
                    f"task_suite2p: cannot open source for axial registration: {e}. "
                    f"skipping axial registration."
                )
                _src_arr_for_reg = None

        num_planes_reg = None
        if _src_arr_for_reg is not None:
            try:
                num_planes_reg = int(_src_arr_for_reg.shape5d[2])
            except Exception as e:
                logger.warning(
                    f"task_suite2p: cannot probe num_planes: {e}. "
                    f"skipping axial registration."
                )

        if num_planes_reg is not None and num_planes_reg > 1:
            combined_meta = dict(getattr(_src_arr_for_reg, "metadata", {}) or {})
            combined_meta.update(custom_metadata)

            if not validate_axial_shifts(combined_meta, num_planes_reg):
                logger.info("task_suite2p: computing axial plane shifts...")
                try:
                    compute_axial_shifts(
                        _src_arr_for_reg,
                        metadata=combined_meta,
                        max_frames=int(args.get("max_frames", 200)),
                        chunk_frames=int(args.get("chunk_frames", 10)),
                        max_reg_xy=int(args.get("max_reg_xy", 150)),
                    )
                except Exception as e:
                    logger.warning(
                        f"task_suite2p: axial registration failed: {e}. "
                        f"binaries will be written without axial shift."
                    )

            if validate_axial_shifts(combined_meta, num_planes_reg):
                ops["apply_shift"] = True
                ops["plane_shifts"] = list(combined_meta["plane_shifts"])
                logger.info(
                    f"task_suite2p: axial shifts wired into ops "
                    f"(apply_shift=True, {num_planes_reg} planes)"
                )
            else:
                logger.warning(
                    "task_suite2p: no valid plane_shifts produced; "
                    "binaries will be written without axial shift."
                )

    # seed nframes into ops so lsp's generate_plane_dirname always has a
    # frame count for the directory name. explicit num_timepoints wins;
    # otherwise fall back to the source array's T dimension.
    if num_timepoints is not None and num_timepoints > 0:
        ops["nframes"] = num_timepoints
    elif src_shape is not None:
        ops["nframes"] = int(src_shape[0])

    # per-channel extraction: wrap data as 4D TZYX so pipeline sees single channel
    channel = args.get("channel")
    pipeline_input = input_path
    if channel is not None:
        ops["functional_chan"] = 1
        ops["align_by_chan"] = 1
        logger.info(f"Single-channel extraction: channel {channel}")

        # register _ChannelView as recognized lazy array type
        import lbm_suite2p_python.utils as _lsp_utils
        if "_ChannelView" not in _lsp_utils._LAZY_ARRAY_TYPES:
            _lsp_utils._LAZY_ARRAY_TYPES = _lsp_utils._LAZY_ARRAY_TYPES + ("_ChannelView",)

        # load data and wrap with channel view (presents 5D as 4D TZYX)
        arr = imread(input_path)
        pipeline_input = _ChannelView(arr, channel - 1)
        logger.info(f"Wrapped as 4D view: shape={pipeline_input.shape}")

    # build writer_kwargs for phase correction settings
    writer_kwargs = {
        "fix_phase": args.get("fix_phase", True),
        "use_fft": args.get("use_fft", True),
    }

    try:
        monitor.update(0.1, "Running Suite2p...")

        # progress callback maps plane/step to 0.1-0.95 range
        def _progress(plane=0, total_planes=1, step="", message="", **kw):
            base = 0.1 + 0.85 * (plane / max(total_planes, 1))
            offsets = {
                "plane_start": 0.0,
                "writing_binary": 0.01,
                "suite2p": 0.05,
                "postprocessing": 0.70,
                "plane_done": 0.85 / max(total_planes, 1),
                "done": 0.85 / max(total_planes, 1),
            }
            frac = base + offsets.get(step, 0)
            monitor.update(min(frac, 0.95), message)

        pipeline(
            pipeline_input,
            save_path=str(output_dir),
            ops=ops,
            planes=planes,
            num_timepoints=num_timepoints,
            keep_raw=s2p_settings.get("keep_raw", False),
            keep_reg=s2p_settings.get("keep_reg", True),
            force_reg=s2p_settings.get("force_reg", False),
            force_detect=s2p_settings.get("force_detect", False),
            dff_window_size=s2p_settings.get("dff_window_size", 300),
            dff_percentile=s2p_settings.get("dff_percentile", 20),
            dff_smooth_window=s2p_settings.get("dff_smooth_window"),
            writer_kwargs=writer_kwargs,
            progress_callback=_progress,
        )

        monitor.finish("Suite2p pipeline completed.")
        logger.info("Suite2p completed successfully")

    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"Suite2p failed: {e}")
        raise

# Registry
TASKS = {
    "save_as": task_save_as,
    "suite2p": task_suite2p
}
