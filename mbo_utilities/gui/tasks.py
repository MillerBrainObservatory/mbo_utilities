"""
Task registry for background worker processes.

This module contains the actual logic for background tasks:
- TaskMonitor: Helper for reporting progress to a JSON sidecar file.
- task_save_as: Generic array conversion/saving.
- task_suite2p: Suite2p pipeline.
- TASKS: Registry mapping task names to functions.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import os
import traceback
from pathlib import Path

from mbo_utilities import imread, log
from mbo_utilities.writer import imwrite
from mbo_utilities.arrays._registration import (
    compute_axial_shifts,
    validate_axial_shifts,
)
from mbo_utilities.metadata import get_param

logger = log.get("worker.tasks")


from mbo_utilities.arrays._channel_view import _ChannelView


def _auto_workers(num_planes: int, *, use_gpu: bool = False) -> int:
    """Pick a worker count from hardware capacity and dataset size.

    Bounded by ``num_planes`` (no point spawning more workers than tasks),
    leaves CPU headroom for the OS + GUI, and budgets ~8 GB of available
    RAM per worker (one suite2p plane fits comfortably in that).

    GPU consumers (cellpose on GPU, etc.) are capped at 2 workers since
    they contend for one device — set ``use_gpu=True`` to enable that.
    """
    cpu = os.cpu_count() or 4
    cpu_workers = max(1, cpu - 2)
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        mem_workers = max(1, int(avail_gb // 8))
    except Exception:
        mem_workers = cpu_workers
    cap = 2 if use_gpu else 12
    return max(1, min(num_planes, cpu_workers, mem_workers, cap))


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
        from mbo_utilities.preferences import get_mbo_dirs
        self.log_dir = get_mbo_dirs()["logs"]
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
    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
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
        axial_max_reg_xy = int(args.get("max_reg_xy", 30))

        def _reg_cb(progress, msg=""):
            monitor.update(0.05 + 0.05 * progress, f"Registration: {msg}")

        num_planes = get_param(combined_meta, "nplanes") or getattr(arr, "num_planes", 1)
        reg_error: Exception | None = None

        if validate_axial_shifts(combined_meta, num_planes):
            logger.info("using plane_shifts already present in metadata.")
            metadata["plane_shifts"] = list(combined_meta["plane_shifts"])
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
                monitor.update(0.1, "Z-registration ready.")
            else:
                if reg_error is not None:
                    logger.warning(
                        f"Z-registration failed: {reg_error}."
                    )
                else:
                    logger.warning(
                        "axial registration produced no valid plane_shifts."
                    )

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

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
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
    logger.info(
        f"task_suite2p: parallel knobs from GUI -> "
        f"workers={s2p_settings.get('workers')!r}, "
        f"threads_per_worker={s2p_settings.get('threads_per_worker')!r}, "
        f"skip_volumetric={s2p_settings.get('skip_volumetric')!r}; "
        f"s2p_settings has workers key: {'workers' in s2p_settings}"
    )

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

    # axial registration: compute per-plane offsets and store in
    # ops["plane_shifts"]. shifts are not applied to the bin pixels —
    # downstream viewers consume the metadata to align planes at render
    # time (see AxiallyAlignedView).
    register_z = args.get("register_z", False)

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
                        max_reg_xy=int(args.get("max_reg_xy", 30)),
                    )
                except Exception as e:
                    logger.warning(
                        f"task_suite2p: axial registration failed: {e}. "
                        f"binaries will be written without axial shift."
                    )

            if validate_axial_shifts(combined_meta, num_planes_reg):
                ops["plane_shifts"] = list(combined_meta["plane_shifts"])
                logger.info(
                    f"task_suite2p: axial shifts stored in ops "
                    f"({num_planes_reg} planes)"
                )
            else:
                logger.warning(
                    "task_suite2p: no valid plane_shifts produced."
                )

    # seed nframes into ops so lsp's generate_plane_dirname always has a
    # frame count for the directory name. explicit num_timepoints wins;
    # otherwise fall back to the source array's T dimension.
    if num_timepoints is not None and num_timepoints > 0:
        ops["nframes"] = num_timepoints
    elif src_shape is not None:
        ops["nframes"] = int(src_shape[0])

    # per-channel extraction: pass channel through reader_kwargs so each
    # parallel worker re-creates the same 4D TZYX wrap after its own
    # imread (the wrap can't survive pickling — workers re-open from
    # paths). Path-based pipeline_input survives unchanged.
    channel = args.get("channel")
    pipeline_input = input_path
    reader_kwargs: dict = {}
    if channel is not None:
        ops["functional_chan"] = 1
        ops["align_by_chan"] = 1
        reader_kwargs["channel"] = int(channel) - 1
        logger.info(f"Single-channel extraction: channel {channel} (zero-based: {channel - 1})")

        # register _ChannelView as a recognized lazy array type in lsp's
        # subprocess workers (they'll re-import on imread).
        import lbm_suite2p_python.utils as _lsp_utils
        if "_ChannelView" not in _lsp_utils._LAZY_ARRAY_TYPES:
            _lsp_utils._LAZY_ARRAY_TYPES = _lsp_utils._LAZY_ARRAY_TYPES + ("_ChannelView",)

    # Resolve workers: pass-through user choice, but honour 0/None as
    # "auto" using hardware capacity. lsp also has its own auto path,
    # but ours additionally bounds by available RAM and respects a GPU
    # cap. Passing the resolved number through means logs say a real
    # count instead of "auto/0".
    raw_workers = s2p_settings.get("workers")
    if raw_workers in (None, 0):
        n_planes = len(planes) if planes else 1
        # suite2p reg/detect run on the GPU unless torch_device is cpu or
        # the env policy forces CPU; GPU workers contend for one device so
        # _auto_workers caps them. Key off the real device, not rastermap.
        from mbo_utilities.gpu import gpu_compute_disabled
        device = str(
            ops.get("torch_device") or s2p_settings.get("torch_device") or "cuda"
        ).lower()
        use_gpu = (not gpu_compute_disabled()) and not device.startswith("cpu")
        resolved = _auto_workers(n_planes, use_gpu=use_gpu)
        if resolved != raw_workers:
            logger.info(
                f"task_suite2p: workers=auto resolved to {resolved} "
                f"(num_planes={n_planes}, cpu={os.cpu_count()})"
            )
            s2p_settings = dict(s2p_settings)
            s2p_settings["workers"] = resolved

    # build writer_kwargs for phase correction settings
    writer_kwargs = {
        "fix_phase": args.get("fix_phase", True),
        "use_fft": args.get("use_fft", True),
    }

    # Rastermap Force → drop cached model.npy in every plane subdir under
    # output_dir before pipeline runs. lsp's plot_zplane_figures already
    # deletes the rastermap PNG every run, but reuses model.npy when its
    # isort length matches n_accepted. Force == "recompute from scratch".
    if s2p_settings.get("force_rastermap") and s2p_settings.get("rastermap_kwargs") is not None:
        try:
            removed = 0
            for cached in Path(output_dir).glob("plane*/model.npy"):
                try:
                    cached.unlink()
                    removed += 1
                except OSError as _e:
                    logger.warning(f"force_rastermap: could not remove {cached}: {_e}")
            if removed:
                logger.info(f"force_rastermap: removed {removed} cached model.npy file(s)")
        except Exception as _e:
            logger.warning(f"force_rastermap pre-clean failed: {_e}")

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

        if tp_indices:
            logger.info(
                f"task_suite2p: frame_indices len={len(tp_indices)}, "
                f"first={tp_indices[0]}, last={tp_indices[-1]} "
                f"(expected dirname tp{tp_indices[0] + 1:05d}-{tp_indices[-1] + 1:05d})"
            )
        else:
            logger.info("task_suite2p: frame_indices=None (no slice — full range will be used)")

        pipeline(
            pipeline_input,
            save_path=str(output_dir),
            ops=ops,
            planes=planes,
            num_timepoints=num_timepoints,
            frame_indices=tp_indices,
            keep_raw=s2p_settings.get("keep_raw", False),
            keep_reg=s2p_settings.get("keep_reg", True),
            force_reg=s2p_settings.get("force_reg", False),
            force_detect=s2p_settings.get("force_detect", False),
            accept_all_cells=s2p_settings.get("accept_all_cells", False),
            norm_method=s2p_settings.get("norm_method", "dff"),
            dff_window_size=s2p_settings.get("dff_window_size", 300),
            dff_percentile=s2p_settings.get("dff_percentile", 20),
            dff_smooth_window=s2p_settings.get("dff_smooth_window"),
            save_json=s2p_settings.get("save_json", False),
            correct_neuropil=s2p_settings.get("correct_neuropil", True),
            cell_filters=s2p_settings.get("cell_filters") or None,
            # unified rastermap api: dict keyed by mode ("planar" /
            # "volumetric"). presence-of-key = mode on; empty sub-dict
            # = use lsp's built-in defaults; None overall = both off.
            # built gui-side by build_rastermap_kwargs() in settings.py.
            rastermap_kwargs=s2p_settings.get("rastermap_kwargs"),
            writer_kwargs=writer_kwargs,
            reader_kwargs=reader_kwargs or None,
            progress_callback=_progress,
            workers=s2p_settings.get("workers", 1),
            skip_volumetric=s2p_settings.get("skip_volumetric", False),
            threads_per_worker=s2p_settings.get("threads_per_worker"),
        )

        monitor.finish("Suite2p pipeline completed.")
        logger.info("Suite2p completed successfully")

    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"Suite2p failed: {e}")
        raise


def _bridge_consolidate_logging(worker_logger: logging.Logger) -> None:
    """Attach the worker's handlers to the consolidator's mbo logger.

    Every ``mbo.*`` logger has ``propagate=False`` (see ``mbo_utilities.log``),
    and the worker's per-task file handler is bound to ``mbo.worker``
    only. Without this bridge the consolidator's INFO messages
    (``consolidate corrected: src=... out=... ...``, ``consolidate: wrote ...``)
    go nowhere in the subprocess. Idempotent — same handler is never
    added twice.
    """
    cons_logger = logging.getLogger("mbo.arrays.isoview.consolidate")
    cons_logger.setLevel(logging.INFO)
    sources: list[logging.Handler] = list(worker_logger.handlers)
    for h in logging.getLogger("mbo").handlers:
        if h not in sources:
            sources.append(h)
    for h in sources:
        if h not in cons_logger.handlers:
            cons_logger.addHandler(h)


def task_isoview(args: dict, logger: logging.Logger) -> None:
    """Isoview consolidator pipeline task.

    Calls ``consolidate_isoview`` and forwards its per-stage progress
    to ``TaskMonitor`` via a stage-weighted mapping. The consolidator's
    progress callback fires with ``(stage, current, total)`` per
    ``(t, c)`` slab written; we bucket the stage into one of four
    weight slots so the bar moves smoothly across the whole run
    instead of resetting at every level/subgroup transition.
    """
    from mbo_utilities.arrays.isoview import consolidate_isoview

    _bridge_consolidate_logging(logger)

    monitor = TaskMonitor(
        args.get("output_path", "."), uuid=args.get("_uuid")
    )
    monitor.update(0.01, "Initializing isoview consolidator...")

    input_path = args["input_path"]
    output_path = args["output_path"]
    kind = args.get("kind")
    overwrite = bool(args.get("overwrite", False))
    pyramid = bool(args.get("pyramid", True))
    pyramid_max_layers = int(args.get("pyramid_max_layers", 4))
    compressor = args.get("compressor", "zstd")
    compression_level = int(args.get("compression_level", 3))

    logger.info(f"consolidate: input={input_path}")
    logger.info(f"consolidate: output={output_path}")
    logger.info(
        f"consolidate: kind={kind or 'auto'} overwrite={overwrite} "
        f"pyramid={pyramid} max_levels={pyramid_max_layers} "
        f"codec={compressor}@{compression_level}"
    )

    # Stage-weight budget. Numbers reflect empirical timing on the
    # 61-timepoint corrected demo (image: ~90s × 4 levels = 360s,
    # seg: ~60s × 4 levels = 240s, projections: ~30s × 3 axes × 4
    # levels = 360s, aux + misc: ~30s). Normalized to sum to 1.0.
    stage_weights = {
        "image": 0.40,
        "seg": 0.15,
        "proj": 0.30,
        "aux": 0.10,
    }
    stage_weights["misc"] = 1.0 - sum(stage_weights.values())

    def _classify(stage: str) -> str:
        stem = stage.split("_", 1)[0]
        if stem == "image":
            return "image"
        if stem == "seg":
            return "seg"
        if stem in ("max", "raw"):
            return "proj"
        if stem == "aux":
            return "aux"
        return "misc"

    # Track cumulative budget consumed by completed stages so the bar
    # never goes backward when a new stage starts.
    consumed: dict[str, float] = {k: 0.0 for k in stage_weights}
    bucket_done: dict[str, set[str]] = {k: set() for k in stage_weights}
    seen_stages: set[str] = set()

    def cb(stage: str, current: int, total: int) -> None:
        if total <= 0:
            return
        # First sighting of this sub-stage → one INFO line so the user
        # sees the pipeline moving. Per-(t,c) progress within a stage
        # would spam the log; the TaskMonitor still gets every update.
        if stage not in seen_stages:
            seen_stages.add(stage)
            logger.info(f"consolidate: stage '{stage}' starting (total={total})")
        bucket = _classify(stage)
        weight = stage_weights[bucket]
        # consolidator fires many sub-stages per bucket (e.g. image_l0,
        # image_l1, image_l2, image_l3 all map to "image"). Split the
        # bucket's weight evenly across the sub-stages we've seen so far.
        bucket_done[bucket].add(stage)
        n_substages = max(1, len(bucket_done[bucket]))
        substage_weight = weight / n_substages
        substage_frac = current / total
        # consumed for this bucket: every prior sub-stage in this bucket
        # is treated as fully done (1.0); current sub-stage at substage_frac.
        bucket_consumed = (
            (len(bucket_done[bucket]) - 1) * substage_weight
            + substage_frac * substage_weight
        )
        consumed[bucket] = max(consumed[bucket], bucket_consumed)
        total_frac = sum(consumed.values())
        # cap at 0.99 — actual finalize() pushes to 1.0 on success.
        monitor.update(min(total_frac, 0.99), f"{stage}: {current}/{total}")

    try:
        out = consolidate_isoview(
            input_path,
            output_path,
            kind=kind,
            overwrite=overwrite,
            pyramid=pyramid,
            pyramid_max_layers=pyramid_max_layers,
            compressor=compressor,
            compression_level=compression_level,
            progress_callback=cb,
        )
        monitor.finish(f"Wrote {out}")
        logger.info(f"isoview consolidator wrote {out}")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"isoview consolidator failed: {e}")
        raise


def _build_isoview_processing_config(args: dict):
    """Assemble an :class:`isoview.ProcessingConfig` from a task args dict.

    Only forwards keys the user explicitly set — everything else falls
    through to ProcessingConfig's own defaults (which themselves get
    further refined by ``update_from_metadata`` once the XML is read
    inside the pipeline). Path-typed fields are coerced to ``Path``.
    """
    from isoview import ProcessingConfig

    config_kwargs: dict = {}

    # required: input_dir always provided by the widget
    config_kwargs["input_dir"] = Path(args["input_path"])
    if args.get("output_dir"):
        config_kwargs["output_dir"] = Path(args["output_dir"])

    # plain pass-throughs — keep keys only when the widget supplied them
    for key in (
        "output_suffix",
        "stitcher_suffix",
        "output_format",
        "compression",
        "compression_level",
        "overwrite",
        "workers",
        "log",
        "pyramid",
        "pyramid_max_layers",
        "timepoints",
        "specimens",
        "segment_mode",
        "gauss_kernel",
        "gauss_sigma",
        "segment_threshold",
        "splitting",
        "apply_segmentation_mask",
        "background_percentile",
        "mask_percentile",
        "subsample_factor",
        "do_tenengrad",
        "blending_method",
        "blending_range",
        "transition_plane",
        "flip_z",
        "flip_horizontal",
        "flip_vertical",
        "rotation",
        "front_flag",
        # Per-view overrides — dict[int, T] keyed by view ID. Resolved
        # by isoview's ProcessingConfig.get_blend(view); views absent
        # from a dict fall back to the scalar above.
        "blending_method_by_view",
        "blending_range_by_view",
        "transition_plane_by_view",
        "front_flag_by_view",
        "flip_z_by_view",
        "flip_horizontal_by_view",
        "flip_vertical_by_view",
        "rotation_by_view",
        "search_offsets_x_by_view",
        "search_offsets_y_by_view",
        "pixel_spacing_z",
        "detection_objective_mag",
        "pixel_spacing_camera",
        "crop_left",
        "crop_top",
        "crop_front",
        "crop_width",
        "crop_height",
        "crop_depth",
    ):
        val = args.get(key)
        if val is not None:
            config_kwargs[key] = val

    # tuple-typed: median_kernel
    if args.get("median_kernel") is not None:
        config_kwargs["median_kernel"] = tuple(args["median_kernel"])
    if args.get("search_offsets_x") is not None:
        config_kwargs["search_offsets_x"] = tuple(args["search_offsets_x"])
    if args.get("search_offsets_y") is not None:
        config_kwargs["search_offsets_y"] = tuple(args["search_offsets_y"])
    for key in ("search_offsets_x_by_view", "search_offsets_y_by_view"):
        d = args.get(key)
        if d is not None:
            config_kwargs[key] = {int(v): tuple(off) for v, off in d.items()}

    # crop_* dicts are keyed by camera index; JSON round-trips int keys to
    # strings, so coerce back to int or get_crop(camera) misses every entry.
    for key in ("crop_left", "crop_top", "crop_front",
                "crop_width", "crop_height", "crop_depth"):
        d = config_kwargs.get(key)
        if isinstance(d, dict):
            config_kwargs[key] = {int(k): int(v) for k, v in d.items()}

    return ProcessingConfig(**config_kwargs)


class _ForwardingHandler(logging.Handler):
    """Mirror records to a fixed list of target handlers at our own level.

    Used to bridge the upstream ``isoview`` logger into the worker's
    log stream without inheriting the upstream's level config. isoview
    calls ``setup_logging()`` partway through ``multi_fuse`` which
    flips its logger to DEBUG — without this wrapper, our bridge would
    silently start emitting every DEBUG record. We forward by calling
    each target's ``handle`` (not ``emit``) so target-level filters,
    formatters, and filters apply.
    """

    def __init__(self, targets: list[logging.Handler], level=logging.INFO):
        super().__init__(level=level)
        self._targets = list(targets)

    def emit(self, record: logging.LogRecord) -> None:
        for h in self._targets:
            h.handle(record)


# Sentinel attribute marking the bridge we install. Per-process; reset
# implicitly because each worker is a fresh subprocess.
_ISOVIEW_BRIDGE_ATTR = "_mbo_isoview_bridge"


def _bridge_isoview_logging(worker_logger: logging.Logger) -> None:
    """Forward isoview pipeline logs to the worker's stream + file.

    isoview uses its own top-level ``isoview`` logger; by default only
    WARNING+ flows to stderr and INFO/DEBUG lands in
    ``<fused_dir>/<method>/fusion.log`` — which the GUI user can't see
    in real time. This helper attaches a forwarding
    handler that mirrors INFO+ records to the worker logger's handlers
    (the per-task file), sets ``isoview``'s
    propagation to ``False`` so nothing double-prints, and is idempotent
    — only one bridge is installed per process even if both isoview
    tasks fire (the second call refreshes its target list to pick up
    any handlers added since).
    """
    iso_logger = logging.getLogger("isoview")
    iso_logger.setLevel(logging.INFO)
    iso_logger.propagate = False

    # Mirror isoview records onto the worker logger's own handlers (the
    # per-task file, plus any GUI panel handler). Do NOT also add the root
    # `mbo` StreamHandler: the worker's stderr is already redirected to that
    # same per-task file (ProcessManager.spawn), so routing isoview through
    # the stream handler too writes every line to the file a second time.
    # mbo.* logs avoid this because propagate=False keeps them off the root
    # stream handler. Fall back to the root handlers only when the worker
    # logger has none (standalone invocation with no per-task file).
    targets: list[logging.Handler] = list(worker_logger.handlers)
    if not targets:
        targets = list(logging.getLogger("mbo").handlers)

    existing = getattr(iso_logger, _ISOVIEW_BRIDGE_ATTR, None)
    if isinstance(existing, _ForwardingHandler):
        existing._targets = targets  # refresh target list
        return

    bridge = _ForwardingHandler(targets, level=logging.INFO)
    iso_logger.addHandler(bridge)
    setattr(iso_logger, _ISOVIEW_BRIDGE_ATTR, bridge)


def task_correct_stack(args: dict, logger: logging.Logger) -> None:
    """IsoView ``correct_stack`` pipeline task.

    Runs the pixel-correction + segmentation + per-camera output stage
    against raw ``.stack`` data. Mirrors the ``~/repos/isoview/pipeline/
    correct_stack.py`` example script: assemble a ProcessingConfig from
    the args dict, call ``isoview.correct_stack(config)``.

    Progress: coarse — the upstream ``correct_stack`` has no progress
    callback hook, so the TaskMonitor only flips
    ``initializing`` → ``running`` → ``completed/failed``. The per-camera
    detail lands in the per-process log file (``~/.mbo/logs/...``).
    """
    from isoview import correct_stack

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing correct_stack...")

    try:
        config = _build_isoview_processing_config(args)
        logger.info(
            f"correct_stack: input={config.input_dir}, "
            f"output={config.output_dir}, format={config.output_format}, "
            f"workers={config.workers}, segment_mode={config.segment_mode}"
        )
        logger.info(f"isoview per-run log: {config.output_dir}/correct.log")
        monitor.update(0.05, "Running correct_stack (see log file for details)...")
        correct_stack(config)
        monitor.finish("correct_stack pipeline complete.")
        logger.info("correct_stack completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"correct_stack failed: {e}")
        raise


def task_isoview_raw_projections(args: dict, logger: logging.Logger) -> None:
    """Precompute raw XY max-projections for the segmentation / dead-pixel
    previews.

    Writes ``SPM##_TM######_CM##.xyProjection.tif`` into the flat sibling
    ``<raw_dir>.raw.projections/`` directory. Existing files are skipped
    so re-running (and the later ``correct_stack``) only fills gaps. No
    dependency on the upstream ``isoview`` package.

    Args: ``raw_dir`` (required), ``overwrite`` (optional, default False).
    """
    from mbo_utilities.arrays.isoview import make_raw_projections

    raw_dir = args.get("raw_dir")
    if not raw_dir:
        raise ValueError("raw_dir is required")
    overwrite = bool(args.get("overwrite", False))

    monitor = TaskMonitor(raw_dir, uuid=args.get("_uuid"))
    monitor.update(0.01, "Scanning raw stacks...")

    def cb(current: int, total: int, name: str) -> None:
        frac = current / total if total else 1.0
        monitor.update(min(0.99, frac), f"projection {current}/{total}: {name}")

    try:
        result = make_raw_projections(
            raw_dir, overwrite=overwrite, progress_callback=cb,
        )
        monitor.finish(
            f"raw projections: {result['written']} written, "
            f"{result['skipped']} existing -> {result['dir']}"
        )
        logger.info(
            f"raw projections: {result['written']} written, "
            f"{result['skipped']} skipped in {result['dir']}"
        )
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"raw projections failed: {e}")
        raise


def task_multi_fuse(args: dict, logger: logging.Logger) -> None:
    """IsoView ``multi_fuse`` pipeline task.

    Runs the camera-pair fusion stage on a corrected output tree.
    Assembles a ProcessingConfig (same kwargs format as
    :func:`task_correct_stack`) and calls ``isoview.multi_fuse(config)``.
    Progress is coarse — same caveat as ``task_correct_stack``: the
    per-pair detail lives in the log file.
    """
    from isoview import multi_fuse

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing multi_fuse...")

    try:
        config = _build_isoview_processing_config(args)
        fused_dir = Path(config.fused_dir)
        fusion_log_path = fused_dir / "fusion.log"
        logger.info(
            f"multi_fuse: input={config.input_dir}, "
            f"fused_dir={fused_dir}, blending={config.blending_method}, "
            f"workers={config.workers}"
        )
        logger.info(f"isoview per-run log: {fusion_log_path} (DEBUG records)")
        monitor.update(0.05, "Running multi_fuse (see log file for details)...")

        # multi_fuse runs the whole parallel loop with no progress callback,
        # so the worker watchdog (kills after MAX_STALL_MINUTES of unchanged
        # progress) would terminate a long but healthy run. Run it in a
        # thread and heartbeat from the count of fused timepoint dirs: the
        # count advances as each timepoint completes (keeping the watchdog
        # alive) and freezes if fusion genuinely hangs (so it still fires).
        total = max(len(config.timepoints), 1)
        result: dict = {}

        def _run():
            try:
                multi_fuse(config)
            except BaseException as exc:
                result["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=30)
            try:
                # new fused layout: <fused>/SPM##/TM######/ (timelapse) or
                # <fused>/SPM##/ (tiled) — count whichever advances.
                done = sum(
                    1 for p in fused_dir.glob("SPM*/TM*") if p.is_dir()
                ) or sum(1 for p in fused_dir.glob("SPM*") if p.is_dir())
            except OSError:
                done = 0
            frac = 0.05 + 0.9 * min(done / total, 1.0)
            monitor.update(frac, f"multi_fuse: {done}/{total} timepoints")
        if "error" in result:
            raise result["error"]

        # output_suffix is already encoded in fused_dir (isoview names the
        # tree <root>.fused_<suffix>), so the method subdir stays <method>.
        # No post-run rename — doing so would double-apply the suffix and
        # break resume (next run writes to <method>/, completed data sits
        # in <method>_<suffix>/).

        monitor.finish("multi_fuse pipeline complete.")
        logger.info("multi_fuse completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"multi_fuse failed: {e}")
        raise


def task_generate_bigstitcher(args: dict, logger: logging.Logger) -> None:
    """IsoView ``generate_bigstitcher_xml`` export task.

    Walks ``config.fused_dir`` for VW00/VW90 pairs and writes one BDV
    SpimData dataset per fusion method under ``config.stitcher_dir``.
    No resampling — the VW90 pre-rotation is baked into the calibration
    affine that BigStitcher applies at display time.

    Reuses the shared isoview ProcessingConfig builder + log bridge so
    the same path resolution and per-run log routing as correct_stack /
    multi_fuse apply.
    """
    from isoview import generate_bigstitcher_xml

    _bridge_isoview_logging(logger)

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing generate_bigstitcher_xml...")

    try:
        config = _build_isoview_processing_config(args)
        logger.info(
            f"generate_bigstitcher_xml: input={config.input_dir}, "
            f"fused_dir={config.fused_dir}, "
            f"stitcher_dir={config.stitcher_dir}"
        )
        monitor.update(0.05, "Writing BDV zarr + dataset.xml...")
        xml_path = generate_bigstitcher_xml(
            config,
            method=args.get("method"),
            coarse_align=args.get("coarse_align", False),
            bake_tile_positions=args.get("bake_tile_positions", False),
            orientation=args.get("orientation"),
        )
        logger.info(f"  wrote: {xml_path}")
        monitor.finish(f"generate_bigstitcher_xml complete: {xml_path}")
        logger.info("generate_bigstitcher_xml completed successfully")
    except Exception as e:
        monitor.fail(str(e), details={"traceback": traceback.format_exc()})
        logger.exception(f"generate_bigstitcher_xml failed: {e}")
        raise


def task_bigstitcher_register(args: dict, logger: logging.Logger) -> None:
    """Run bigstitcher-spark's coarse → fine registration pipeline.

    Mutates ``dataset.xml`` in place. Five spark stages run sequentially:

      1. ``detect-interestpoints`` (DoG)
      2. ``match-interestpoints`` (descriptor pass, e.g. PRECISE_TRANSLATION)
      3. ``solver`` (global optimization)
      4. ``match-interestpoints`` (ICP refinement, when ``do_icp_refine``)
      5. ``solver`` again

    Required args:
      - xml_path: path to dataset.xml
      - spark_root: path to bigstitcher-spark executables dir OR fat JAR

    Optional args: see :func:`isoview.bigstitcher.register`. Both the
    new keys (``coarse_method``, ``solver_method``, …) and the legacy
    ``match_algorithm`` alias are accepted.
    """
    from isoview import bigstitcher

    monitor = TaskMonitor(args.get("output_dir") or ".", uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing bigstitcher-spark...")

    xml_path = Path(args["xml_path"]).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"dataset.xml not found: {xml_path}")
    spark_root = Path(args["spark_root"]).resolve()
    if not spark_root.exists():
        raise FileNotFoundError(f"spark_root does not exist: {spark_root}")

    # Legacy alias: GUI used to send `match_algorithm`; rename to the
    # new `coarse_method` keyword if the caller hasn't already.
    if "coarse_method" not in args and "match_algorithm" in args:
        args = dict(args)
        args["coarse_method"] = args.pop("match_algorithm")

    runtime_keys = ("java_home", "spark_memory_gb", "spark_threads")

    def _pick(keys):
        return {
            k: args[k] for k in keys
            if k in args and args[k] is not None
        }

    detect_kwargs = _pick((
        "label", "sigma", "threshold",
        "downsample_xy", "downsample_z",
        "min_intensity", "max_intensity",
        "max_spots", "dog_type", "localization",
        "overlapping_only_detect",
        *runtime_keys,
    ))
    # bigstitcher.detect_interestpoints expects ``overlapping_only``, not the
    # prefixed name the register() facade uses.
    if "overlapping_only_detect" in detect_kwargs:
        detect_kwargs["overlapping_only"] = detect_kwargs.pop(
            "overlapping_only_detect"
        )

    common_match_keys = (
        "label",
        "transformation_model", "regularization_model", "lambda_reg",
        "view_registration_scope", "interestpoints_for_reg",
        *runtime_keys,
    )
    coarse_kwargs = _pick((
        *common_match_keys,
        "coarse_method",
        "num_neighbors", "redundancy", "significance", "search_radius",
        "coarse_ransac_max_error", "coarse_ransac_min_inliers",
    ))
    # rename register()-style names to match_interestpoints()-style.
    if "coarse_method" in coarse_kwargs:
        coarse_kwargs["method"] = coarse_kwargs.pop("coarse_method")
    if "coarse_ransac_max_error" in coarse_kwargs:
        coarse_kwargs["ransac_max_error"] = coarse_kwargs.pop(
            "coarse_ransac_max_error"
        )
    if "coarse_ransac_min_inliers" in coarse_kwargs:
        coarse_kwargs["ransac_min_inliers"] = coarse_kwargs.pop(
            "coarse_ransac_min_inliers"
        )

    icp_kwargs = _pick((
        *common_match_keys,
        "icp_iterations", "icp_max_error", "icp_use_ransac",
        "icp_ransac_max_error", "icp_ransac_min_inliers",
        "icp_ransac_iterations",
    ))
    icp_kwargs["method"] = "ICP"
    if "icp_ransac_max_error" in icp_kwargs:
        icp_kwargs["ransac_max_error"] = icp_kwargs.pop("icp_ransac_max_error")
    if "icp_ransac_min_inliers" in icp_kwargs:
        icp_kwargs["ransac_min_inliers"] = icp_kwargs.pop(
            "icp_ransac_min_inliers"
        )
    if "icp_ransac_iterations" in icp_kwargs:
        icp_kwargs["ransac_iterations"] = icp_kwargs.pop(
            "icp_ransac_iterations"
        )

    solver_kwargs = _pick((
        "label",
        "transformation_model", "regularization_model", "lambda_reg",
        "solver_method",
        "solver_relative_threshold", "solver_absolute_threshold",
        *runtime_keys,
    ))
    if "solver_method" in solver_kwargs:
        solver_kwargs["method"] = solver_kwargs.pop("solver_method")
    if "solver_relative_threshold" in solver_kwargs:
        solver_kwargs["relative_threshold"] = solver_kwargs.pop(
            "solver_relative_threshold"
        )
    if "solver_absolute_threshold" in solver_kwargs:
        solver_kwargs["absolute_threshold"] = solver_kwargs.pop(
            "solver_absolute_threshold"
        )

    do_icp = bool(args.get("do_icp_refine", True))

    monitor.update(0.05, "BigStitcher-Spark: detect-interestpoints")
    bigstitcher.detect_interestpoints(
        xml_path, spark_root, log=logger, **detect_kwargs
    )

    monitor.update(0.30, "BigStitcher-Spark: coarse match (descriptor)")
    bigstitcher.match_interestpoints(
        xml_path, spark_root, log=logger,
        clear_correspondences=True, **coarse_kwargs,
    )

    monitor.update(0.55, "BigStitcher-Spark: solver (post-coarse)")
    bigstitcher.solver(xml_path, spark_root, log=logger, **solver_kwargs)

    if do_icp:
        monitor.update(0.70, "BigStitcher-Spark: fine match (ICP)")
        bigstitcher.match_interestpoints(
            xml_path, spark_root, log=logger,
            clear_correspondences=True, **icp_kwargs,
        )

        monitor.update(0.90, "BigStitcher-Spark: solver (post-ICP)")
        bigstitcher.solver(xml_path, spark_root, log=logger, **solver_kwargs)

    monitor.finish(f"BigStitcher registration complete: {xml_path}")
    logger.info(f"BigStitcher registration completed; XML updated: {xml_path}")


# Registry
TASKS = {
    "save_as": task_save_as,
    "suite2p": task_suite2p,
    "isoview": task_isoview,
    "isoview_correct": task_correct_stack,
    "isoview_raw_projections": task_isoview_raw_projections,
    "isoview_fuse": task_multi_fuse,
    "isoview_bigstitcher": task_generate_bigstitcher,
    "isoview_bigstitcher_register": task_bigstitcher_register,
}
