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
from multiprocessing import Pool, cpu_count

from mbo_utilities import imread
from mbo_utilities.writer import imwrite
from mbo_utilities.util import load_npy
from mbo_utilities.arrays import register_zplanes_s3d, validate_s3d_registration
from mbo_utilities.metadata import get_param

logger = logging.getLogger("mbo.worker.tasks")


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
        self.log_dir = Path.home() / "mbo" / "logs"
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
            # Write atomically to avoid race conditions with readers
            # Write to temp file first, then rename (atomic on most filesystems)
            tmp_file = self.progress_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            tmp_file.replace(self.progress_file)
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
    rois = args.get("rois")
    num_timepoints = args.get("num_timepoints")
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

        # Prepare metadata for registration check
        # Merge existing array metadata with overrides
        combined_meta = getattr(arr, "metadata", {}).copy()
        combined_meta.update(metadata)

        # Helper to bridge progress callback
        def _reg_cb(progress, msg=""):
             monitor.update(0.05 + 0.05 * progress, f"Registration: {msg}")

        s3d_job_dir = None
        num_planes = get_param(combined_meta, "nplanes") or getattr(arr, "num_planes", 1)

        # Determine directory for registration outputs
        # If output_path has an extension (file .tif or wrapper .zarr), use its parent.
        # If output_path looks like a directory (no suffix), use it directly.
        reg_out_dir = output_path.parent if output_path.suffix else output_path

        # Check output path for existing registration
        job_id = combined_meta.get("job_id", "s3d-preprocessed")
        candidate = reg_out_dir / job_id

        if validate_s3d_registration(candidate, num_planes):
             s3d_job_dir = candidate
             logger.info(f"Found valid existing s3d-job: {s3d_job_dir}")
        else:
             logger.info("Running Suite3D registration...")
             try:
                 filenames = getattr(arr, "filenames", [])
                 if not filenames and hasattr(arr, "_files"):
                     filenames = arr._files

                 if filenames:
                     s3d_job_dir = register_zplanes_s3d(
                         filenames=filenames,
                         metadata=combined_meta,
                         outpath=reg_out_dir,
                         progress_callback=_reg_cb,
                     )
             except Exception as e:
                 logger.exception(f"Registration failed: {e}")

        if s3d_job_dir and validate_s3d_registration(s3d_job_dir, num_planes):
            metadata["apply_shift"] = True
            metadata["s3d-job"] = str(s3d_job_dir)
            monitor.update(0.1, "Z-registration ready.")
        else:
            logger.warning("Registration failed or invalid. Proceeding without shift.")
            metadata["apply_shift"] = False

    monitor.update(0.1, f"Saving to {output_path.name}...")

    # Define progress callback for imwrite
    def _progress_cb(current, total=None, **kwargs):
        if total is None:
            # Called from _writers.py with progress fraction as first arg
            p = 0.1 + 0.9 * float(current)
            msg = "Writing..."
        else:
            # Called from generic writer with (current, total)
            p = 0.1 + 0.9 * (current / min(total, 1)) if total > 0 else 0.5
            msg = f"Writing frame {current}/{total}"

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
                num_frames=num_timepoints,
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
                num_frames=num_timepoints,
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


def _suite2p_worker(args):
    """
    Worker function for multiprocessing pool.
    Runs suite2p on a single pre-extracted binary file.
    """
    from lbm_suite2p_python import run_plane

    bin_file = args["bin_file"]
    save_path = args["save_path"]
    ops = args["ops"]
    s2p_settings = args["s2p_settings"]

    try:
        run_plane(
            bin_file,  # input_data: first positional argument
            save_path=save_path,
            ops=ops,
            keep_raw=s2p_settings.get("keep_raw", False),
            keep_reg=s2p_settings.get("keep_reg", True),
            force_reg=s2p_settings.get("force_reg", False),
            force_detect=s2p_settings.get("force_detect", False),
            dff_window_size=s2p_settings.get("dff_window_size", 300),
            dff_percentile=s2p_settings.get("dff_percentile", 20),
            dff_smooth_window=s2p_settings.get("dff_smooth_window")
        )
        return {"status": "success", "plane": ops.get("plane")}
    except Exception as e:
        return {"status": "error", "plane": ops.get("plane"), "error": str(e)}


def task_suite2p(args: dict, logger: logging.Logger) -> None:
    """
    Suite2p Pipeline Task.

    Phase 1: Serial Safe Extraction
    - Reads Master TIFF (single process)
    - Writes data_raw.bin files for each plane

    Phase 2: Parallel Processing
    - Runs Suite2p on each .bin file using multiprocessing
    """
    monitor = TaskMonitor(args.get("output_dir", "."), uuid=args.get("_uuid"))
    monitor.update(0.01, "Initializing Suite2p pipeline...")

    input_path = args["input_path"]
    output_dir = Path(args["output_dir"])
    planes = args.get("planes", []) # List of plane indices (1-indexed)
    if not planes and "plane" in args:
        planes = [args["plane"]]
    roi = args.get("roi")
    num_timepoints = args.get("num_timepoints")
    ops = args.get("ops", {})
    s2p_settings = args.get("s2p_settings", {})

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Serial Extraction ---
    if isinstance(input_path, list):
        display_name = f"{len(input_path)} files ({Path(input_path[0]).name}...)"
    else:
        display_name = Path(input_path).name

    monitor.update(0.05, f"Opening source file(s): {display_name}...")
    logger.info(f"Opening {input_path}")

    arr = imread(input_path)

    # Sanity checks
    if hasattr(arr, "fix_phase"):
         monitor.update(0.05, "Applying phase correction settings...")
         # Assuming these args might be passed in ops or kwargs,
         # but usually encoded in the array defaults or metadata

    extracted_items = [] # List of dicts describing each extracted plane

    total_planes = len(planes)
    for i, p_idx in enumerate(sorted(planes)):
        monitor.update(0.1 + 0.4 * (i/total_planes), f"Extracting Plane {p_idx} ({i+1}/{total_planes})...")

        # Determine output folder
        if roi:
            plane_dir = output_dir / f"plane{p_idx:02d}_roi{roi:02d}"
        else:
            plane_dir = output_dir / f"plane{p_idx:02d}_stitched"

        plane_dir.mkdir(parents=True, exist_ok=True)
        raw_file = plane_dir / "data_raw.bin"
        ops_path = plane_dir / "ops.npy"

        # Prepare metadata/ops
        # We need to construct the specific ops for this plane
        # This mirrors logic from pipeline_widgets.py but simplified

        current_md = ops.copy()
        current_md.update({
            "plane": p_idx,
            "save_path": str(plane_dir),
            "raw_file": str(raw_file.resolve()),
            "ops_path": str(ops_path)
            # Add other necessary metadata if needed
        })

        # Extract and write
        # We use imwrite with 'planes' kwarg to handle the slicing efficiently
        logger.info(f"Extracting plane {p_idx} to {raw_file}")

        try:
             # Note: imwrite handles writing 'data_raw.bin' and 'ops.npy'
            if roi:
                arr.roi = roi

            # Handle Array Slicing for Extraction
            # We pass planes=p_idx to imwrite, which handles lazy chunked reading.
            # Avoid manual slicing (arr[:, p_idx-1, ...]) as it triggers a full data load.
            arr_to_write = arr
            write_planes_arg = p_idx # 1-indexed for imwrite

            # If array is 3D but we are asking for plane > 1, it implies interleaving
            # or concatenation. imwrite generic handles interleaving if we pass planes=p_idx.
            # But if it's concatenated files (ScanImageArray default), we rely on arr._imwrite
            # or generic writer to handle it.

            # Use imwrite to extract and write binary
            imwrite(
                arr_to_write,
                plane_dir,
                ext=".bin",
                overwrite=True,
                planes=write_planes_arg, # 1-indexed or None if pre-sliced
                num_frames=num_timepoints,
                output_name="data_raw.bin",
                metadata=current_md
            )

            # Reload ops from disk to get Lx, Ly, and other metadata added by imwrite
            updated_ops = load_npy(ops_path).item() if ops_path.exists() else current_md

            extracted_items.append({
                "bin_file": str(raw_file),
                "save_path": str(plane_dir),
                "ops": updated_ops,
                "s2p_settings": s2p_settings
            })

        except Exception as e:
            logger.exception(f"Failed to extract plane {p_idx}: {e}")
            logger.exception(traceback.format_exc())
            monitor.fail(f"Extraction failed for plane {p_idx}: {e}", details={"traceback": traceback.format_exc()})
            raise # Stop processing


    # --- Phase 2: Parallel Processing ---
    if not extracted_items:
        logger.error("No planes were extracted. Check if the selected planes exist in the source file.")
        monitor.fail("No planes were extracted. Check if the selected planes exist in the source file.")
        return

    monitor.update(0.5, "Starting parallel processing...")
    logger.info(f"Starting parallel suite2p processing for {len(extracted_items)} plane(s)")

    completed = 0
    errors = 0

    # Determine worker count
    max_workers = args.get("max_workers", max(1, cpu_count() - 2))

    with Pool(processes=max_workers) as pool:
        # Submit all jobs
        results = []
        for item in extracted_items:
            results.append(pool.apply_async(_suite2p_worker, (item,)))

        # Wait for results and update monitor
        while results:
            # Check remaining
            pending = [r for r in results if not r.ready()]
            finished = len(results) - len(pending)

            # Update progress
            current_progress = 0.5 + 0.5 * (finished / len(extracted_items))
            monitor.update(current_progress, f"Processing: {finished}/{len(extracted_items)} planes completed")

            if not pending:
                break

            time.sleep(0.5)

        # Collect final statuses
        for r in results:
            try:
                res = r.get()
                if res["status"] == "error":
                    errors += 1
                    logger.error(f"Error in plane {res['plane']}: {res['error']}")
                else:
                    completed += 1
            except Exception as e:
                errors += 1
                logger.exception(f"Worker crashed: {e}")

    if errors > 0:
        msg = f"Completed with {errors} errors. ({completed} success)"
        monitor.fail(msg)
        raise RuntimeError(msg)
    monitor.finish(f"All {completed} planes processed successfully.")

# Registry
TASKS = {
    "save_as": task_save_as,
    "suite2p": task_suite2p
}
