"""
Suite3D z-plane registration utilities.

This module provides functions for registering z-planes using Suite3D,
which computes rigid shifts to align planes spatially.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mbo_utilities import log

logger = log.get("arrays._registration")


def validate_s3d_registration(s3d_job_dir: Path, num_planes: int = None) -> bool:
    """
    Validate that Suite3D registration completed successfully.

    Parameters
    ----------
    s3d_job_dir : Path
        Path to the Suite3D job directory (e.g., 's3d-preprocessed')
    num_planes : int, optional
        Expected number of planes. If provided, validates that plane_shifts has correct length.

    Returns
    -------
    bool
        True if valid registration results exist, False otherwise.
    """
    if not s3d_job_dir or not Path(s3d_job_dir).is_dir():
        return False

    s3d_job_dir = Path(s3d_job_dir)
    summary_path = s3d_job_dir / "summary" / "summary.npy"

    if not summary_path.is_file():
        logger.warning(f"Suite3D summary file not found: {summary_path}.")
        return False

    try:
        summary = np.load(summary_path, allow_pickle=True).item()

        if not isinstance(summary, dict):
            logger.warning(f"Suite3D summary is not a dict: {type(summary)}")
            return False

        if "plane_shifts" not in summary:
            logger.warning("Suite3D summary missing 'plane_shifts' key")
            return False

        plane_shifts = summary["plane_shifts"]

        if not isinstance(plane_shifts, (list, np.ndarray)):
            logger.warning(f"plane_shifts has invalid type: {type(plane_shifts)}")
            return False

        plane_shifts = np.asarray(plane_shifts)

        if plane_shifts.ndim != 2 or plane_shifts.shape[1] != 2:
            logger.warning(
                f"plane_shifts has invalid shape: {plane_shifts.shape}, expected (n_planes, 2)"
            )
            return False

        if num_planes is not None and len(plane_shifts) != num_planes:
            logger.warning(
                f"plane_shifts length {len(plane_shifts)} doesn't match expected {num_planes} planes"
            )
            return False

        logger.debug(
            f"Valid Suite3D registration found with {len(plane_shifts)} plane shifts"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to validate Suite3D registration: {e}")
        return False


def register_zplanes_s3d(
    filenames, metadata, outpath=None, progress_callback=None
) -> Path | None:
    """
    Register z-planes using Suite3D.

    This function computes rigid shifts between z-planes to align them spatially.
    Requires Suite3D and CuPy to be installed.

    Parameters
    ----------
    filenames : list[Path]
        List of TIFF file paths to process.
    metadata : dict
        Metadata dictionary containing:
        - frame_rate : float (required)
        - num_planes : int (required)
        - tau : float (optional, default 1.3)
        - lbm : bool (optional, default True)
        - Other Suite3D parameters
    outpath : Path, optional
        Output directory for Suite3D job. If None, creates directory next to input.
    progress_callback : callable, optional
        Progress callback function.

    Returns
    -------
    Path | None
        Path to the Suite3D job directory, or None if registration failed.
    """
    # these are heavy imports, lazy import for now
    try:
        # https://github.com/MillerBrainObservatory/mbo_utilities/issues/35
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        from suite3d.job import Job  # noqa

        HAS_SUITE3D = True
    except ImportError:
        HAS_SUITE3D = False
        Job = None

    try:
        import cupy  # noqa

        HAS_CUPY = True
    except ImportError:
        HAS_CUPY = False

    if not HAS_SUITE3D:
        logger.warning(
            "Suite3D is not installed. Cannot preprocess. "
            "Set register_z = False in imwrite, or install Suite3D: "
            "`pip install mbo_utilities[suite3d, cuda12]` # CUDA 12.x or "
            "`pip install mbo_utilities[suite3d, cuda11]` # CUDA 11.x"
        )
        return None

    if not HAS_CUPY:
        logger.warning(
            "CuPy is not installed. Cannot preprocess. "
            "Set register_z = False in imwrite, or install CuPy: "
            "`pip install cupy-cuda12x` # CUDA 12.x or "
            "`pip install cupy-cuda11x` # CUDA 11.x"
        )
        return None

    if "frame_rate" not in metadata or "num_planes" not in metadata:
        logger.warning(
            "Missing required metadata for axial alignment: frame_rate / num_planes"
        )
        return None

    if outpath is not None:
        job_path = Path(outpath)
    else:
        job_path = Path(str(filenames[0].parent) + ".summary")

    job_id = metadata.get("job_id", "preprocessed")

    params = {
        "fs": metadata["frame_rate"],
        "planes": np.arange(metadata["num_planes"]),
        "n_ch_tif": metadata["num_planes"],
        "tau": metadata.get("tau", 1.3),
        "lbm": metadata.get("lbm", True),
        "fuse_strips": metadata.get("fuse_planes", False),
        "subtract_crosstalk": metadata.get("subtract_crosstalk", False),
        "init_n_frames": metadata.get("init_n_frames", 500),
        "n_init_files": metadata.get("n_init_files", 1),
        "n_proc_corr": metadata.get("n_proc_corr", 15),
        "max_rigid_shift_pix": metadata.get("max_rigid_shift_pix", 150),
        "3d_reg": metadata.get("3d_reg", True),
        "gpu_reg": metadata.get("gpu_reg", True),
        "block_size": metadata.get("block_size", [64, 64]),
    }

    if Job is None:
        logger.warning("Suite3D Job class not available.")
        return None

    job = Job(
        str(job_path),
        job_id,
        create=True,
        overwrite=True,
        verbosity=-1,
        tifs=filenames,
        params=params,
        progress_callback=progress_callback,
    )
    job._report(0.01, "Launching Suite3D job...")
    logger.debug("Running Suite3D job...")
    job.run_init_pass()
    out_dir = job_path / f"s3d-{job_id}"
    metadata["s3d-job"] = str(out_dir)
    metadata["s3d-params"] = params
    logger.info(f"Preprocessed data saved to {out_dir}")
    return out_dir
