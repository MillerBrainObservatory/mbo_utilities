"""
Registration utilities.

This module provides functions for registering z-planes using Suite3D,
which computes rigid shifts to align planes spatially.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.metadata import get_param
from mbo_utilities.file_io import load_npy

logger = log.get("arrays._registration")


def validate_s3d_registration(s3d_job_dir: Path, num_planes: int | None = None) -> bool:
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
        summary = load_npy(summary_path).item()

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
    filenames, metadata, outpath=None, progress_callback=None, s3d_params=None
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
        Metadata dictionary. Required keys: ``frame_rate``, ``num_planes``.
        Optional Suite3D tunables (any value here is overridden by an
        explicit ``s3d_params`` entry of the same name):

        - ``tau`` : float, default 1.3
        - ``lbm`` : bool, default True
        - ``init_n_frames`` : int, default 500 — frames pulled into the init pass
        - ``n_init_files`` : int, default 1 — TIFFs read in parallel during init
        - ``n_proc_corr`` : int, default 4 — multiprocessing workers used by
          ``load_and_stitch_full_tif_mp``. each worker reserves a shared-memory
          block sized to one tile; lower this if you hit Windows ``WinError 1455``
          (commitment limit) or run out of RAM.
        - ``max_rigid_shift_pix`` : int, default 150
        - ``3d_reg`` : bool, default True
        - ``gpu_reg`` : bool, default True
        - ``block_size`` : list[int], default [64, 64]
        - ``fuse_planes`` : bool, default False
        - ``subtract_crosstalk`` : bool, default False
    outpath : Path, optional
        Output directory for Suite3D job. If None, creates directory next to input.
    progress_callback : callable, optional
        Progress callback function.
    s3d_params : dict, optional
        Explicit overrides for the Suite3D ``params`` dict, applied last so
        they win over both metadata-derived values and built-in defaults.
        Use this when calling from code that wants to set resource knobs
        (e.g. ``{"n_proc_corr": 2, "init_n_frames": 200}``) without polluting
        ``metadata``.

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

    nplanes = get_param(metadata, "nplanes", default=1)
    # default n_proc_corr lowered from 15 -> 4 to avoid Windows commit-limit
    # blowups (WinError 1455). each worker reserves a tile-sized shared-memory
    # block via mmap; on windows that always counts against the pagefile commit
    # ceiling regardless of free ram. raise via metadata or s3d_params if you
    # know your machine can handle it.
    params = {
        "fs": get_param(metadata, "fs"),
        "planes": np.arange(nplanes),
        "n_ch_tif": nplanes,
        "tau": metadata.get("tau", 1.3),
        "lbm": metadata.get("lbm", True),
        "fuse_strips": metadata.get("fuse_planes", False),
        "subtract_crosstalk": metadata.get("subtract_crosstalk", False),
        # init_n_frames lowered from 500 -> 200 to keep the shared-memory
        # block within windows commit limits. the block size is
        # init_n_frames * n_ch_tif * page_h * page_w * 2, and each mp
        # worker attachment doubles the commit. with 14-plane LBM tiles
        # at 1116×224, 500 frames = 3.5 GB per mapping; 200 = 1.4 GB.
        "init_n_frames": metadata.get("init_n_frames", 200),
        "n_init_files": metadata.get("n_init_files", 1),
        "n_proc_corr": metadata.get("n_proc_corr", 2),
        "max_rigid_shift_pix": metadata.get("max_rigid_shift_pix", 150),
        "3d_reg": metadata.get("3d_reg", True),
        "gpu_reg": metadata.get("gpu_reg", True),
        "block_size": metadata.get("block_size", [64, 64]),
    }
    # explicit s3d_params overrides win last so api callers can tune
    # resource knobs without touching metadata.
    if s3d_params:
        params.update(s3d_params)
    logger.info(
        f"Suite3D init: n_proc_corr={params['n_proc_corr']} "
        f"init_n_frames={params['init_n_frames']} "
        f"n_init_files={params['n_init_files']}"
    )

    if Job is None:
        logger.warning("Suite3D Job class not available.")
        return None

    def _run_job(p):
        j = Job(
            str(job_path),
            job_id,
            create=True,
            overwrite=True,
            verbosity=-1,
            tifs=filenames,
            params=p,
            progress_callback=progress_callback,
        )
        j._report(0.01, "Launching Suite3D job...")
        j.run_init_pass()
        return j

    def _is_commit_limit_error(exc: Exception) -> bool:
        """Detect Windows WinError 1455 (paging file too small).

        The error may surface as an OSError with winerror=1455 directly,
        or as a re-raised exception from a multiprocessing worker where
        the winerror attribute didn't survive pickling. Fall back to
        substring matching on the message in that case.
        """
        if getattr(exc, "winerror", None) == 1455:
            return True
        return "1455" in str(exc) and "paging file" in str(exc).lower()

    # retry with reduced resource knobs on windows commit-limit blowup.
    # each mp worker mmaps a tile-sized shared-memory block; on windows
    # that always counts against the pagefile commit ceiling regardless
    # of free ram. back off n_proc_corr to 1 AND halve init_n_frames
    # so both the block size and the attachment count shrink.
    try:
        job = _run_job(params)
    except OSError as e:
        if _is_commit_limit_error(e):
            fallback_frames = max(50, params.get("init_n_frames", 200) // 2)
            logger.warning(
                f"WinError 1455 (paging file too small) with "
                f"n_proc_corr={params['n_proc_corr']}, "
                f"init_n_frames={params['init_n_frames']}. "
                f"Retrying with n_proc_corr=1, init_n_frames={fallback_frames}..."
            )
            params["n_proc_corr"] = 1
            params["init_n_frames"] = fallback_frames
            try:
                job = _run_job(params)
            except OSError as e2:
                if _is_commit_limit_error(e2):
                    logger.error(
                        f"WinError 1455 persists even at n_proc_corr=1, "
                        f"init_n_frames={fallback_frames}. The windows "
                        f"pagefile commit limit is too low for this dataset. "
                        f"Increase the pagefile or reduce the data size."
                    )
                raise
        else:
            raise
    out_dir = job_path / f"s3d-{job_id}"
    metadata["s3d-job"] = str(out_dir)
    metadata["s3d-params"] = params
    logger.info(f"Preprocessed data saved to {out_dir}")
    return out_dir
