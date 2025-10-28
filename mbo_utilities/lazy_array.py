from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from . import log
from ._writers import _try_generic_writers
from .array_types import (
    Suite2pArray,
    H5Array,
    MBOTiffArray,
    TiffArray,
    MboRawArray,
    NpyArray,
    ZarrArray,
    register_zplanes_s3d,
    validate_s3d_registration,
    supports_roi,
)
from .file_io import derive_tag_from_filename
from .metadata import is_raw_scanimage, has_mbo_metadata

logger = log.get("lazy_array")


SUPPORTED_FTYPES = (
    ".npy",
    ".tif",
    ".tiff",
    ".bin",
    ".h5",
    ".zarr",
    ".json"
)

_ARRAY_TYPE_KWARGS = {
    MboRawArray: {
        "roi",
        "fix_phase",
        "phasecorr_method",
        "border",
        "upsample",
        "max_offset",
    },
    ZarrArray: {"filenames", "compressor", "rois"},
    MBOTiffArray: {"filenames", "_chunks"},
    Suite2pArray: set(),  # accepts no kwargs
    H5Array: {"dataset"},
    TiffArray: set(),
    NpyArray: set(),
    # DemixingResultsArray: set(),
}


def _filter_kwargs(cls, kwargs):
    allowed = _ARRAY_TYPE_KWARGS.get(cls, set())
    return {k: v for k, v in kwargs.items() if k in allowed}


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
        target_chunk_mb: int = 20,
        progress_callback: Callable | None = None,
        debug: bool = False,
        shift_vectors: np.ndarray | None = None,
        **kwargs,
):
    """
    Write a supported lazy imaging array (Suite2p, HDF5, TIFF, etc.) to disk.

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
        - `NpyArray` : Single `.npy` memory-mapped NumPy file
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
        ROI selection for multi-ROI data (e.g., MboRawArray from ScanImage). Options:
        - None (default) : Stitch/fuse all ROIs horizontally into single FOV
        - 0 : Split all ROIs into separate files (one file per ROI per plane)
        - int > 0 : Export specific ROI, e.g. `roi=1` exports only ROI 1
        - list/tuple : Export specific ROIs, e.g. `roi=[1, 3]` exports ROIs 1 and 3

    num_frames : int, optional
        Number of frames to export. If None (default), exports all frames.
        Useful for testing or exporting subsets: `num_frames=1000`

    register_z : bool, default=False
        Perform z-plane registration using Suite3D before writing. When True:
        - Computes rigid shifts between z-planes
        - Validates registration results (checks `summary.npy` for valid `plane_shifts`)
        - Applies shifts during write to align planes
        - Requires Suite3D and CuPy installed: `pip install mbo_utilities[suite3d,cuda12]`
        - Creates/reuses s3d job directory in outpath

    shift_vectors : np.ndarray, optional
        Pre-computed z-shift vectors with shape (n_planes, 2) for [dy, dx] shifts.
        Use this to apply previously computed registration without re-running Suite3D.
        Example: `shift_vectors=np.array([[0, 0], [2, -1], [1, 3]])`

    metadata : dict, optional
        Additional metadata to merge into output file headers/attributes.
        Merged with existing metadata from the source array.

    overwrite : bool, default=False
        Whether to overwrite existing output files. If False, skips existing files
        with a warning.

    order : list | tuple, optional
        Reorder planes before writing. Must have same length as `planes`.
        Example: `planes=[1,2,3], order=[2,0,1]` writes planes in order [3,1,2]

    target_chunk_mb : int, default=20
        Target chunk size in MB for streaming writes. Larger chunks may be faster
        but use more memory. Adjust based on available RAM.

    progress_callback : Callable, optional
        Callback function for progress updates: `callback(progress, current_plane)`.
        Receives progress as float 0-1 and current plane index.

    debug : bool, default=False
        Enable verbose logging to terminal for troubleshooting.

    **kwargs
        Additional format-specific options passed to writer backends.

    Returns
    -------
    Path
        Path to the output directory containing written files.

    Raises
    ------
    TypeError
        If lazy_array type is unsupported or incompatible with specified options.
    ValueError
        If outpath parent doesn't exist, metadata is malformed, or parameters are invalid.
    FileNotFoundError
        If expected companion files (e.g., `ops.npy`, `summary.npy`) are missing.
    KeyError
        If registration is requested but `plane_shifts` is missing from summary.

    Notes
    -----
    **File Naming Convention:**
    - Single ROI or stitched: `plane{Z:02d}_stitched.{ext}`
    - Multiple ROIs: `plane{Z:02d}_roi{R}.{ext}`
    - Binary format: `plane{Z:02d}_roi{R}/data_raw.bin` + `ops.npy`

    **Registration (register_z=True):**
    - Validates existing registration by checking `summary/summary.npy` for valid
      `plane_shifts` array with shape (n_planes, 2)
    - Only reruns Suite3D if validation fails or no existing job found
    - Registration shifts are applied during write to align planes spatially
    - Output files are padded to accommodate all shifts

    **Memory Management:**
    - Data is streamed in chunks (controlled by `target_chunk_mb`)
    - Only one chunk is held in memory at a time
    - Large files (>4GB) automatically use BigTIFF format

    **Phase Correction (MboRawArray only):**
    - Set `lazy_array.fix_phase = True` before calling imwrite
    - Corrects bidirectional scanning artifacts
    - Methods: 'mean', 'median', 'max' (set via `lazy_array.phasecorr_method`)

    Examples
    --------
    **Basic Usage - Stitch ROIs and save all planes as TIFF:**

    >>> from mbo_utilities import imread, imwrite
    >>> data = imread("path/to/raw/*.tiff")
    >>> imwrite(data, "output/session1", roi=None)  # Stitches all ROIs

    **Save specific planes only (first, middle, last for 14-plane volume):**

    >>> imwrite(data, "output/session1", planes=[1, 7, 14])
    # Creates: plane01_stitched.tiff, plane07_stitched.tiff, plane14_stitched.tiff

    **Split all ROIs into separate files:**

    >>> imwrite(data, "output/session1", roi=0)
    # Creates: plane01_roi1.tiff, plane01_roi2.tiff, ..., plane14_roi1.tiff, ...

    **Save specific ROIs only:**

    >>> imwrite(data, "output/session1", roi=[1, 3])  # Only ROIs 1 and 3
    >>> imwrite(data, "output/session1", roi=2)       # Only ROI 2

    **Z-plane registration with Suite3D:**

    >>> data = imread("path/to/raw/*.tiff")
    >>> imwrite(data, "output/registered", register_z=True, roi=None)
    # Computes and applies rigid shifts to align z-planes spatially

    **Use pre-computed registration shifts:**

    >>> shifts = np.load("previous_job/summary/summary.npy", allow_pickle=True).item()
    >>> shift_vectors = shifts['plane_shifts']  # shape: (n_planes, 2)
    >>> imwrite(data, "output/registered", shift_vectors=shift_vectors)

    **Convert to Suite2p binary format:**

    >>> data = imread("path/to/raw/*.tiff")
    >>> imwrite(data, "output/suite2p", ext=".bin", roi=0)
    # Creates: plane01_roi1/data_raw.bin, plane01_roi1/ops.npy, ...

    **Export subset of frames for testing:**

    >>> imwrite(data, "output/test", num_frames=1000, planes=[1, 7, 14])
    # Exports only first 1000 frames of planes 1, 7, and 14

    **Save to Zarr format with compression:**

    >>> imwrite(data, "output/zarr_store", ext=".zarr", roi=0)
    # Creates: output/zarr_store/plane01_roi1.zarr, ...

    **Enable phase correction (for raw ScanImage data):**

    >>> data = imread("path/to/raw/*.tiff")
    >>> data.fix_phase = True
    >>> data.phasecorr_method = "mean"  # or "median", "max"
    >>> data.use_fft = True  # Use FFT-based correction (faster)
    >>> imwrite(data, "output/corrected", roi=None)

    **Overwrite existing files:**

    >>> imwrite(data, "output/session1", planes=[1, 2, 3], overwrite=True)

    **Custom metadata:**

    >>> custom_meta = {"experimenter": "MBO-User", "Date": "2025-01-15"}
    >>> imwrite(data, "output/session1", metadata=custom_meta)

    **Reorder planes:**

    >>> imwrite(data, "output/session1", planes=[3, 2, 1], order=[2, 1, 0])
    # Writes plane 3 first, then plane 2, then plane 1

    **Progress callback reports per-zplane completion % for UIs:**

    >>> def progress_handler(progress, plane):
    ...     print(f"Plane {plane}: {progress*100:.1f}% complete")
    >>> imwrite(data, "output/session1", progress_callback=progress_handler)

    See Also
    --------
    imread : Load imaging data from various formats
    register_zplanes_s3d : Compute z-plane registration using Suite3D
    validate_s3d_registration : Validate Suite3D registration results
    """
    if debug:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True  # send to terminal
    else:
        logger.setLevel(logging.WARNING)
        logger.propagate = False  # don't send to terminal

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
            raise ValueError(
                f"{type(lazy_array)} does not support ROIs, but `roi` was provided."
            )
        lazy_array.roi = roi

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
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

    # Only assign back if object supports metadata
    if hasattr(lazy_array, "metadata"):
        lazy_array.metadata = file_metadata

    s3d_job_dir = None
    if register_z:
        lazy_array.metadata["apply_shift"] = True
        num_planes = file_metadata.get("num_planes")

        if shift_vectors is not None:
            lazy_array.metadata["shift_vectors"] = shift_vectors
            logger.info("Using provided shift_vectors for registration.")
        else:
            # Check if we already have a valid s3d-job directory
            existing_s3d_dir = None

            # Option 1: Check metadata for existing s3d-job
            if "s3d-job" in lazy_array.metadata:
                candidate = Path(lazy_array.metadata["s3d-job"])
                if validate_s3d_registration(candidate, num_planes):
                    logger.info(f"Found valid s3d-job in metadata: {candidate}")
                    existing_s3d_dir = candidate
                else:
                    logger.warning(f"s3d-job in metadata exists but registration is invalid: {candidate}")

            # Option 2: Check if outpath contains existing valid registration
            if not existing_s3d_dir:
                job_id = file_metadata.get("job_id", "s3d-preprocessed")
                candidate = outpath / job_id
                if validate_s3d_registration(candidate, num_planes):
                    logger.info(f"Found valid existing s3d-job: {candidate}")
                    existing_s3d_dir = candidate

            if existing_s3d_dir:
                s3d_job_dir = existing_s3d_dir
                # Load directory metadata if available
                if s3d_job_dir.joinpath("dirs.npy").is_file():
                    dirs = np.load(s3d_job_dir / "dirs.npy", allow_pickle=True).item()
                    for k, v in dirs.items():
                        if Path(v).is_dir():
                            lazy_array.metadata[k] = v
            else:
                # Need to run registration
                logger.info("No valid s3d-job found, running Suite3D registration.")
                s3d_job_dir = register_zplanes_s3d(
                    filenames=lazy_array.filenames,
                    metadata=file_metadata,
                    outpath=outpath,
                    progress_callback=progress_callback
                )

                if s3d_job_dir:
                    # Validate the registration actually succeeded
                    if validate_s3d_registration(s3d_job_dir, num_planes):
                        logger.info(f"Z-plane registration succeeded: {s3d_job_dir}")
                    else:
                        logger.error(
                            f"Suite3D job completed but registration validation failed. "
                            f"Check {s3d_job_dir}/summary/summary.npy for plane_shifts. "
                            f"Proceeding without registration."
                        )
                        s3d_job_dir = None
                        lazy_array.metadata["apply_shift"] = False
                else:
                    logger.warning(
                        "Z-plane registration failed. Proceeding without registration. "
                        "Check that Suite3D and CuPy are installed correctly."
                    )
                    lazy_array.metadata["apply_shift"] = False

        # Store s3d-job directory in metadata if available
        if s3d_job_dir:
            logger.info(f"Storing s3d-job path {s3d_job_dir} in metadata.")
            lazy_array.metadata["s3d-job"] = str(s3d_job_dir)
    else:
        # Registration not requested
        lazy_array.metadata["apply_shift"] = False

    if hasattr(lazy_array, "_imwrite"):
        return lazy_array._imwrite(  # noqa
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            ext=ext,
            progress_callback=progress_callback,
            planes=planes,
            debug=debug,
            **kwargs,
        )
    else:
        if isinstance(lazy_array, Suite2pArray):
            raise TypeError(
                "Attempting to write a Suite2pArray directly."
                " Is there an ops.npy file in a directory with a tiff file?"
                "Please make write these to separate directories."
            )
        logger.info(f"Falling back to generic writers for {type(lazy_array)}.")
        _try_generic_writers(
            lazy_array,
            outpath,
            overwrite=overwrite,
        )
        return outpath


def imread(
    inputs: str | Path | Sequence[str | Path],
    **kwargs,  # for the reader
):
    """
    Lazy load imaging data from supported file types.

    Currently supported file types:
    - .bin: Suite2p binary files (.bin + ops.npy)
    - .tif/.tiff: TIFF files (BigTIFF, OME-TIFF and raw ScanImage TIFFs)
    - .h5: HDF5 files
    - .zarr: Zarr v3

    Parameters
    ----------
    inputs : str, Path, ndarray, MboRawArray, or sequence of str/Path
        Input source. Can be:
        - Path to a file or directory
        - List/tuple of file paths
        - An existing lazy array
    **kwargs
        Extra keyword arguments passed to specific array readers.

    Returns
    -------
    array_like
        One of Suite2pArray, TiffArray, MboRawArray, MBOTiffArray, H5Array,
        or the input ndarray.

    Examples
    -------
    >>> from mbo_utilities import imread
    >>> arr = imread("/data/raw")  # directory with supported files, for full filename
    """
    if isinstance(inputs, np.ndarray):
        return inputs
    if isinstance(inputs, MboRawArray):
        return inputs

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")

        if p.suffix.lower() == ".zarr" and p.is_dir():
            paths = [p]
        elif p.is_dir():
            logger.debug(f"Input is a directory, searching for supported files in {p}")
            zarrs = list(p.glob("*.zarr"))
            if zarrs:
                logger.debug(f"Found {len(zarrs)} zarr stores in {p}, loading as ZarrArray.")
                paths = zarrs
            else:
                paths = [Path(f) for f in p.glob("*") if f.is_file()]
                logger.debug(f"Found {len(paths)} files in {p}")
        else:
            paths = [p]
    elif isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], np.ndarray):
            return inputs
        paths = [Path(p) for p in inputs if isinstance(p, (str, Path))]
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    if not paths:
        raise ValueError("No input files found.")

    filtered = [p for p in paths if p.suffix.lower() in SUPPORTED_FTYPES]
    if not filtered:
        raise ValueError(
            f"No supported files in {inputs}. \n"
            f"Supported file types are: {SUPPORTED_FTYPES}"
        )
    paths = filtered

    parent = paths[0].parent if paths else None
    ops_file = parent / "ops.npy" if parent else None

    # Suite2p ops file
    if ops_file and ops_file.exists():
        logger.debug(f"Ops.npy detected - reading {ops_file} from {ops_file}.")
        return Suite2pArray(parent / "ops.npy")

    exts = {p.suffix.lower() for p in paths}
    first = paths[0]

    if len(exts) > 1:
        if exts == {".bin", ".npy"}:
            npy_file = first.parent / "ops.npy"
            logger.debug(f"Reading {npy_file} from {npy_file}.")
            return Suite2pArray(npy_file)
        raise ValueError(f"Multiple file types found in input: {exts!r}")

    if first.suffix in [".tif", ".tiff"]:
        if is_raw_scanimage(first):
            logger.debug(f"Detected raw ScanImage TIFFs, loading as MboRawArray.")
            return MboRawArray(files=paths, **kwargs)
        if has_mbo_metadata(first):
            logger.debug(f"Detected MBO TIFFs, loading as MBOTiffArray.")
            return MBOTiffArray(paths, **kwargs)
        logger.debug(f"Loading TIFF files as TiffArray.")
        return TiffArray(paths)

    if first.suffix == ".bin":
        npy_file = first.parent / "ops.npy"
        if npy_file.exists():
            logger.debug(f"Reading Suite2p binary from {npy_file}.")
            return Suite2pArray(npy_file)
        raise NotImplementedError("BIN files without metadata are not yet supported.")

    if first.suffix == ".h5":
        logger.debug(f"Reading HDF5 files from {first}.")
        return H5Array(first)

    if first.suffix == ".zarr":
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info(f"Detected nested zarr stores, loading as ZarrArray.")
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))

        tag = derive_tag_from_filename
        # Case 2: flat zarr store with zarr.json
        if (first / "zarr.json").exists():
            logger.info(f"Detected zarr.json, loading as ZarrArray.")
            return ZarrArray(paths, **_filter_kwargs(ZarrArray, kwargs))

        raise ValueError(
            f"Zarr path {first} is not a valid store. "
            "Expected nested *.zarr dirs or a zarr.json inside."
        )

    if first.suffix == ".json":
        logger.debug(f"Reading JSON files from {first}.")
        return ZarrArray(first.parent, **_filter_kwargs(ZarrArray, kwargs))

    if first.suffix == ".npy" and (first.parent / "pmd_demixer.npy").is_file():
        raise NotImplementedError("PMD Arrays are not yet supported.")
        # return DemixingResultsArray(first.parent)

    raise TypeError(f"Unsupported file type: {first.suffix}")
