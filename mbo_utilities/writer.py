"""
imwrite - Write lazy imaging arrays to disk.

This module provides the imwrite() function for writing imaging data to
various file formats with support for ROI selection, z-plane registration,
chunked streaming, and format conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities._writers import _try_generic_writers
from mbo_utilities.arrays import (
    iter_rois,
    register_zplanes_s3d,
    supports_roi,
    validate_s3d_registration,
)

logger = log.get("writer")


def imwrite(
    lazy_array,
    outpath: str | Path,
    ext: str = ".tiff",
    planes: list | tuple | None = None,
    num_frames: int | None = None,
    register_z: bool = False,
    roi: int | Sequence[int] | None = None,
    metadata: dict | None = None,
    overwrite: bool = False,
    order: list | tuple = None,
    target_chunk_mb: int = 100,
    progress_callback: Callable | None = None,
    debug: bool = False,
    shift_vectors: np.ndarray | None = None,
    output_name: str | None = None,
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

    roi : int | Sequence[int] | None, optional
        ROI selection for multi-ROI data. Options:
        - None (default) : Stitch/fuse all ROIs horizontally into single FOV
        - 0 : Split all ROIs into separate files (one file per ROI per plane)
        - int > 0 : Export specific ROI, e.g. `roi=1` exports only ROI 1
        - list/tuple : Export specific ROIs, e.g. `roi=[1, 3]`

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

    output_name : str, optional
        Filename for binary output when ext=".bin".

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

    if roi is not None:
        if not supports_roi(lazy_array):
            logger.debug(
                f"{type(lazy_array).__name__} does not support ROIs. "
                f"Ignoring roi={roi}, defaulting to single ROI behavior."
            )
        else:
            lazy_array.roi = roi

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
        num_planes = file_metadata.get("num_planes")

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
                if s3d_job_dir.joinpath("dirs.npy").is_file():
                    dirs = np.load(s3d_job_dir / "dirs.npy", allow_pickle=True).item()
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

    if hasattr(lazy_array, "_imwrite"):
        write_kwargs = kwargs.copy()
        if num_frames is not None:
            write_kwargs["num_frames"] = num_frames

        return lazy_array._imwrite(
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            ext=ext,
            progress_callback=progress_callback,
            planes=planes,
            debug=debug,
            output_name=output_name,
            **write_kwargs,
        )
    else:
        logger.info(f"Falling back to generic writers for {type(lazy_array)}.")
        _try_generic_writers(
            lazy_array,
            outpath,
            overwrite=overwrite,
        )
        return outpath
