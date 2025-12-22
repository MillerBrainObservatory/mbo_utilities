"""
imwrite - Write lazy imaging arrays to disk.

This module provides the imwrite() function for writing imaging data to
various file formats with support for ROI selection, z-plane registration,
chunked streaming, and format conversion.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities._writers import _try_generic_writers, add_processing_step
from mbo_utilities.arrays import (
    iter_rois,
    register_zplanes_s3d,
    supports_roi,
    validate_s3d_registration,
)
from mbo_utilities.metadata import get_param, RoiMode
from mbo_utilities.util import load_npy

logger = log.get("writer")


def imwrite(
    lazy_array,
    outpath: str | Path,
    ext: str = ".tiff",
    planes: list | tuple | None = None,
    num_frames: int | None = None,
    register_z: bool = False,
    roi_mode: RoiMode | str = RoiMode.concat_y,
    roi: int | Sequence[int] | None = None,
    metadata: dict | None = None,
    overwrite: bool = False,
    order: list | tuple = None,
    target_chunk_mb: int = 100,
    progress_callback: Callable | None = None,
    debug: bool = False,
    show_progress: bool = True,
    shift_vectors: np.ndarray | None = None,
    output_name: str | None = None,
    output_suffix: str | None = None,
    **kwargs,
):
    """
    Write a supported lazy imaging array to disk.

    This function handles writing multi-dimensional imaging data to various formats,
    with support for ROI selection, z-plane registration, chunked streaming, and
    format conversion. Use with `imread()` to load and convert imaging data.

    Parameters
    ----------
    lazy_array : object
        One of the supported lazy array readers providing `.shape`, `.metadata`,
        and `_imwrite()` methods:

        - `MboRawArray` : Raw ScanImage/ScanMultiROI TIFF files with phase correction
        - `Suite2pArray` : Memory-mapped binary (`data.bin` or `data_raw.bin`) + `ops.npy`
        - `MBOTiffArray` : Multi-file TIFF reader using Dask backend
        - `TiffArray` : Single or multi-TIFF reader
        - `H5Array` : HDF5 dataset wrapper (`h5py.File[dataset]`)
        - `ZarrArray` : Collection of z-plane `.zarr` stores
        - `NumpyArray` : Single `.npy` memory-mapped NumPy file
        - `NWBArray` : NWB file with "TwoPhotonSeries" acquisition dataset

    outpath : str or Path
        Target directory to write output files. Will be created if it doesn't exist.
        Files are named automatically based on plane/ROI (e.g., `plane01_roi1.tiff`).

    ext : str, default=".tiff"
        Output format extension. Supported formats:
        - `.tiff`, `.tif` : Multi-page TIFF (BigTIFF for >4GB)
        - `.bin` : Suite2p-compatible binary format with ops.npy metadata
        - `.zarr` : Zarr v3 array store
        - `.h5`, `.hdf5` : HDF5 format

    planes : list | tuple | int | None, optional
        Z-planes to export (1-based indexing). Options:
        - None (default) : Export all planes
        - int : Single plane, e.g. `planes=7` exports only plane 7
        - list/tuple : Specific planes, e.g. `planes=[1, 7, 14]`

    roi_mode : RoiMode | str, default=RoiMode.concat_y
        Mode for handling multi-ROI data. Options:
        - RoiMode.concat_y : Horizontally concatenate ROIs into single FOV (default)
        - RoiMode.separate : Write each ROI to separate files
        String values are accepted (case-insensitive): "concat_y", "separate".

    roi : int | Sequence[int] | None, optional
        Specific ROI(s) to export when roi_mode=RoiMode.separate. Options:
        - None (default) : Export all ROIs
        - int > 0 : Export specific ROI, e.g. `roi=1` exports only ROI 1
        - list/tuple : Export specific ROIs, e.g. `roi=[1, 3]`
        Note: When roi_mode=RoiMode.concat_y, this parameter is ignored.

    num_frames : int, optional
        Number of frames to export. If None (default), exports all frames.

    register_z : bool, default=False
        Perform z-plane registration using Suite3D before writing.

    shift_vectors : np.ndarray, optional
        Pre-computed z-shift vectors with shape (n_planes, 2) for [dy, dx] shifts.

    metadata : dict, optional
        Additional metadata to merge into output file headers/attributes.

    overwrite : bool, default=False
        Whether to overwrite existing output files.

    order : list | tuple, optional
        Reorder planes before writing. Must have same length as `planes`.

    target_chunk_mb : int, optional
        Target chunk size in MB for streaming writes. Default is 100 MB.

    progress_callback : Callable, optional
        Callback function for progress updates: `callback(progress, current_plane)`.

    debug : bool, default=False
        Enable verbose logging for troubleshooting.

    show_progress : bool, default=True
        Show tqdm progress bar during writing. Set to False in notebooks
        when you don't want progress output cluttering the display.

    output_name : str, optional
        Filename for binary output when ext=".bin".

    output_suffix : str, optional
        Custom suffix to append to output filenames. If None (default), files are
        named with "_stitched" for multi-ROI data when roi is None, or "_roiN"
        for specific ROIs. Examples: "_stitched", "_processed", "_session1".
        The suffix is automatically sanitized (illegal characters removed, double
        extensions prevented, underscore prefix added if missing).

    **kwargs
        Additional format-specific options passed to writer backends.

    Returns
    -------
    Path
        Path to the output directory containing written files.

    Examples
    --------
    >>> from mbo_utilities import imread, imwrite
    >>> data = imread("path/to/raw/*.tiff")
    >>> imwrite(data, "output/session1", roi=None)  # Stitch all ROIs

    >>> # Save specific planes
    >>> imwrite(data, "output/session1", planes=[1, 7, 14])

    >>> # Split ROIs
    >>> imwrite(data, "output/session1", roi=0)

    >>> # Z-plane registration
    >>> imwrite(data, "output/registered", register_z=True)

    >>> # Convert to Suite2p binary
    >>> imwrite(data, "output/suite2p", ext=".bin", roi=0)

    >>> # Save to Zarr
    >>> imwrite(data, "output/zarr_store", ext=".zarr")
    """
    if debug:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True
    else:
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    # normalize roi_mode to enum
    if isinstance(roi_mode, str):
        roi_mode = RoiMode.from_string(roi_mode)

    # save path
    if not isinstance(outpath, (str, Path)):
        raise TypeError(
            f"`outpath` must be a string or Path, got {type(outpath)} instead."
        )

    outpath = Path(outpath)
    if not outpath.parent.is_dir():
        raise ValueError(
            f"{outpath} is not inside a valid directory."
            f" Please create the directory first."
        )
    outpath.mkdir(exist_ok=True)

    # handle roi based on roi_mode
    if roi_mode == RoiMode.separate:
        # separate mode: set roi on array if specified
        if roi is not None and supports_roi(lazy_array):
            lazy_array.roi = roi
    elif roi_mode == RoiMode.concat_y:
        # concat mode: roi parameter is ignored, use None (stitch all)
        if roi is not None:
            logger.debug(
                f"roi={roi} ignored when roi_mode=concat_y. "
                f"All ROIs will be concatenated."
            )
        roi = None

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match "
                f"the number of planes ({len(planes)})."
            )
        if any(i < 0 or i >= len(planes) for i in order):
            raise ValueError(
                f"order indices must be in range [0, {len(planes) - 1}], got {order}"
            )
        planes = [planes[i] for i in order]

    existing_meta = getattr(lazy_array, "metadata", None)
    file_metadata = dict(existing_meta or {})

    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(metadata)}")
        file_metadata.update(metadata)

    # store roi_mode in metadata as string
    file_metadata["roi_mode"] = roi_mode.value

    if num_frames is not None:
        file_metadata["num_frames"] = int(num_frames)
        file_metadata["nframes"] = int(num_frames)

    if hasattr(lazy_array, "metadata"):
        try:
            lazy_array.metadata = file_metadata
        except AttributeError:
            pass

    s3d_job_dir = None
    if register_z:
        file_metadata["apply_shift"] = True
        num_planes = get_param(file_metadata, "nplanes")

        if shift_vectors is not None:
            file_metadata["shift_vectors"] = shift_vectors
            logger.info("Using provided shift_vectors for registration.")
        else:
            existing_s3d_dir = None

            if "s3d-job" in file_metadata:
                candidate = Path(file_metadata["s3d-job"])
                if validate_s3d_registration(candidate, num_planes):
                    logger.info(f"Found valid s3d-job in metadata: {candidate}")
                    existing_s3d_dir = candidate
                else:
                    logger.warning(
                        f"s3d-job in metadata exists but registration is invalid"
                    )

            if not existing_s3d_dir:
                job_id = file_metadata.get("job_id", "s3d-preprocessed")
                candidate = outpath / job_id
                if validate_s3d_registration(candidate, num_planes):
                    logger.info(f"Found valid existing s3d-job: {candidate}")
                    existing_s3d_dir = candidate

            if existing_s3d_dir:
                s3d_job_dir = existing_s3d_dir
                # notify callback that we're using cached registration
                if progress_callback:
                    progress_callback(1.0, "Using cached registration")

                if s3d_job_dir.joinpath("dirs.npy").is_file():
                    dirs = load_npy(s3d_job_dir / "dirs.npy").item()
                    for k, v in dirs.items():
                        if Path(v).is_dir():
                            file_metadata[k] = v
            else:
                logger.info("No valid s3d-job found, running Suite3D registration.")
                s3d_job_dir = register_zplanes_s3d(
                    filenames=lazy_array.filenames,
                    metadata=file_metadata,
                    outpath=outpath,
                    progress_callback=progress_callback,
                )

                if s3d_job_dir:
                    if validate_s3d_registration(s3d_job_dir, num_planes):
                        logger.info(f"Z-plane registration succeeded: {s3d_job_dir}")
                    else:
                        logger.error(
                            f"Suite3D job completed but validation failed. "
                            f"Proceeding without registration."
                        )
                        s3d_job_dir = None
                        file_metadata["apply_shift"] = False
                else:
                    logger.warning(
                        "Z-plane registration failed. Proceeding without registration."
                    )
                    file_metadata["apply_shift"] = False

        if s3d_job_dir:
            logger.info(f"Storing s3d-job path {s3d_job_dir} in metadata.")
            file_metadata["s3d-job"] = str(s3d_job_dir)

        if hasattr(lazy_array, "metadata"):
            try:
                lazy_array.metadata = file_metadata
            except AttributeError:
                pass
    else:
        file_metadata["apply_shift"] = False
        if hasattr(lazy_array, "metadata"):
            try:
                lazy_array.metadata = file_metadata
            except AttributeError:
                pass

    # Collect input files for processing history
    input_files = getattr(lazy_array, "filenames", None)
    if input_files:
        # handle single path or list of paths
        if isinstance(input_files, (str, Path)):
            input_files = [str(input_files)]
        else:
            input_files = [str(f) for f in input_files]

    # Extract scan-phase correction parameters if available (MboRawArray)
    scan_phase_params = {}
    if hasattr(lazy_array, "fix_phase"):
        scan_phase_params["fix_phase"] = getattr(lazy_array, "fix_phase", False)
    if hasattr(lazy_array, "use_fft"):
        scan_phase_params["use_fft"] = getattr(lazy_array, "use_fft", False)
    if hasattr(lazy_array, "phasecorr_method"):
        scan_phase_params["phasecorr_method"] = getattr(
            lazy_array, "phasecorr_method", None
        )

    # Start timing for processing history
    write_start_time = time.time()

    if hasattr(lazy_array, "_imwrite"):
        write_kwargs = kwargs.copy()
        if num_frames is not None:
            write_kwargs["num_frames"] = num_frames

        result = lazy_array._imwrite(
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            ext=ext,
            progress_callback=progress_callback,
            planes=planes,
            debug=debug,
            show_progress=show_progress,
            output_name=output_name,
            output_suffix=output_suffix,
            roi_mode=roi_mode,
            **write_kwargs,
        )
    else:
        logger.info(f"Falling back to generic writers for {type(lazy_array)}.")
        _try_generic_writers(
            lazy_array,
            outpath,
            overwrite=overwrite,
        )
        result = outpath

    # Record processing step in metadata
    write_duration = time.time() - write_start_time

    # Build extra info for processing history
    processing_extra = {
        "input_format": type(lazy_array).__name__,
        "output_format": ext,
        "num_frames": file_metadata.get("num_frames") or file_metadata.get("nframes"),
        "shape": list(lazy_array.shape) if hasattr(lazy_array, "shape") else None,
    }

    # Add scan-phase correction info if present
    if scan_phase_params:
        processing_extra["scan_phase_correction"] = scan_phase_params

    # Add z-registration info if used
    if register_z:
        processing_extra["z_registration"] = {
            "enabled": True,
            "s3d_job_dir": str(s3d_job_dir) if s3d_job_dir else None,
            "apply_shift": file_metadata.get("apply_shift", False),
        }
        if shift_vectors is not None:
            processing_extra["z_registration"]["shift_vectors_provided"] = True

    # Add ROI info if specified
    if roi is not None:
        processing_extra["roi"] = roi

    # Add planes info if specified
    if planes is not None:
        processing_extra["planes"] = list(planes) if hasattr(planes, "__iter__") else planes

    # Collect output files
    output_files = None
    if result and isinstance(result, Path):
        if result.is_dir():
            # List files in output directory
            out_files = list(result.glob(f"*{ext}"))
            if out_files:
                output_files = [str(f) for f in out_files[:20]]  # Limit to first 20
        else:
            output_files = [str(result)]

    add_processing_step(
        file_metadata,
        step_name="imwrite",
        input_files=input_files,
        output_files=output_files,
        duration_seconds=write_duration,
        extra=processing_extra,
    )

    # Update lazy_array metadata with processing history if possible
    if hasattr(lazy_array, "metadata"):
        try:
            lazy_array.metadata = file_metadata
        except AttributeError:
            pass

    logger.debug(
        f"Processing step recorded: imwrite to {ext} in {write_duration:.2f}s"
    )

    return result
