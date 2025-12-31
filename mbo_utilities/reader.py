"""
imread - Lazy load imaging data from supported file types.

This module provides the imread() function for loading imaging data from
various file formats as lazy arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays import (
    BinArray,
    H5Array,
    IsoviewArray,
    MBOTiffArray,
    MboRawArray,
    NumpyArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
    _extract_tiff_plane_number,
    open_scanimage,
)
from mbo_utilities.metadata import has_mbo_metadata, is_raw_scanimage

logger = log.get("reader")

MBO_SUPPORTED_FTYPES = [".tiff", ".tif", ".zarr", ".bin", ".h5", ".npy"]

# Re-export PIPELINE_TAGS for backward compatibility (canonical location is file_io.py)
from mbo_utilities.file_io import PIPELINE_TAGS

_ARRAY_TYPE_KWARGS = {
    MboRawArray: {
        "roi",
        "fix_phase",
        "phasecorr_method",
        "border",
        "upsample",
        "max_offset",
        "use_fft",
    },
    ZarrArray: {"filenames", "compressor", "rois"},
    MBOTiffArray: {"filenames", "_chunks"},
    Suite2pArray: set(),
    BinArray: {"shape"},
    H5Array: {"dataset"},
    TiffArray: set(),
    NumpyArray: {"metadata"},
}


def _filter_kwargs(cls, kwargs):
    allowed = _ARRAY_TYPE_KWARGS.get(cls, set())
    return {k: v for k, v in kwargs.items() if k in allowed}


def imread(
    inputs: str | Path | np.ndarray | Sequence[str | Path],
    **kwargs,
):
    """
    Lazy load imaging data from supported file types.

    Currently supported file types:
    - .bin: Suite2p binary files (.bin + ops.npy)
    - .tif/.tiff: TIFF files (BigTIFF, OME-TIFF and raw ScanImage TIFFs)
    - .h5: HDF5 files
    - .zarr: Zarr v3
    - .npy: NumPy arrays
    - np.ndarray: In-memory numpy arrays (wrapped as NumpyArray)

    Parameters
    ----------
    inputs : str, Path, ndarray, or sequence of str/Path
        Input source. Can be:
        - Path to a file or directory
        - List/tuple of file paths
        - A numpy array (will be wrapped as NumpyArray for full imwrite support)
        - An existing lazy array (passed through unchanged)
    **kwargs
        Extra keyword arguments passed to specific array readers.

    Returns
    -------
    array_like
        A lazy array appropriate for the input format. Use `mbo formats` CLI
        command to list all supported formats and their array types.

    Examples
    --------
    >>> from mbo_utilities import imread, imwrite
    >>> arr = imread("/data/raw")  # directory with supported files
    >>> arr = imread("data.tiff")  # single file
    >>> arr = imread(["file1.tiff", "file2.tiff"])  # multiple files

    >>> # Wrap numpy array for imwrite compatibility
    >>> data = np.random.randn(100, 512, 512)
    >>> arr = imread(data)  # Returns NumpyArray
    >>> imwrite(arr, "output", ext=".zarr")  # Full write support
    """
    # Wrap numpy arrays in NumpyArray for full imwrite/protocol support
    if isinstance(inputs, np.ndarray):
        logger.debug(f"Wrapping numpy array with shape {inputs.shape} as NumpyArray")
        return NumpyArray(inputs, **_filter_kwargs(NumpyArray, kwargs))
    # Pass through already-loaded lazy arrays (has _imwrite method)
    if hasattr(inputs, "_imwrite") and hasattr(inputs, "shape"):
        return inputs

    if "isoview" in kwargs.items():
        print("Isoview!")
        return IsoviewArray(inputs)

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")

        if p.suffix.lower() == ".zarr" and p.is_dir():
            paths = [p]
        elif p.is_dir():
            logger.debug(f"Input is a directory, searching for supported files in {p}")

            # Check for Isoview structure: TM* subfolders with .zarr files
            tm_folders = [
                d for d in p.iterdir() if d.is_dir() and d.name.startswith("TM")
            ]
            if tm_folders:
                logger.info(
                    f"Detected Isoview structure with {len(tm_folders)} TM folders."
                )
                return IsoviewArray(p)

            # Check if this IS a TM folder (single timepoint)
            if p.name.startswith("TM"):
                zarrs = list(p.glob("*.zarr"))
                if zarrs:
                    logger.info(
                        f"Detected single TM folder with {len(zarrs)} zarr files."
                    )
                    return IsoviewArray(p)

            zarrs = list(p.glob("*.zarr"))
            if zarrs:
                logger.debug(
                    f"Found {len(zarrs)} zarr stores in {p}, loading as ZarrArray."
                )
                paths = zarrs
            else:
                # Check for Suite2p structure (ops.npy or plane subdirs)
                # unified Suite2pArray handles both single plane and volume
                ops_file = p / "ops.npy"
                if ops_file.exists():
                    logger.info(f"Detected Suite2p directory at {p}")
                    return Suite2pArray(p)

                # Check for plane subdirectories (volumetric suite2p)
                plane_subdirs = [d for d in p.iterdir() if d.is_dir() and (d / "ops.npy").exists()]
                if plane_subdirs:
                    logger.info(f"Detected Suite2p volume with {len(plane_subdirs)} planes in {p}")
                    return Suite2pArray(p)

                # Check for TIFF volume structure (planeXX.tiff files)
                # unified TiffArray handles both single files and plane volumes
                plane_tiffs = sorted(p.glob("plane*.tif*"))
                if plane_tiffs:
                    logger.info(f"Detected TIFF volume with {len(plane_tiffs)} planes in {p}")
                    return TiffArray(p)

                paths = [Path(f) for f in p.glob("*") if f.is_file()]
                logger.debug(f"Found {len(paths)} files in {p}")
        else:
            paths = [p]
    elif isinstance(inputs, (list, tuple)):
        if not inputs:
            raise ValueError("Input list is empty")

        # Check if all items are ndarrays
        if all(isinstance(item, np.ndarray) for item in inputs):
            return inputs

        # Check if all items are paths
        if not all(isinstance(item, (str, Path)) for item in inputs):
            raise TypeError(
                f"Mixed input types in list. Expected all paths or all ndarrays. "
                f"Got: {[type(item).__name__ for item in inputs]}"
            )

        paths = [Path(p) for p in inputs]
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    if not paths:
        raise ValueError("No input files found.")

    filtered = [p for p in paths if p.suffix.lower() in MBO_SUPPORTED_FTYPES]
    if not filtered:
        raise ValueError(
            f"No supported files in {inputs}. \n"
            f"Supported file types are: {MBO_SUPPORTED_FTYPES}"
        )
    paths = filtered

    parent = paths[0].parent if paths else None
    ops_file = parent / "ops.npy" if parent else None

    # Suite2p ops file
    if ops_file and ops_file.exists():
        if len(paths) == 1 and paths[0].suffix.lower() == ".bin":
            logger.debug(f"Ops.npy detected - reading specific binary {paths[0]}.")
            return Suite2pArray(paths[0])
        else:
            logger.debug(f"Ops.npy detected - reading from {ops_file}.")
            return Suite2pArray(ops_file)

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
            logger.debug(f"Detected raw ScanImage TIFFs, using open_scanimage for auto-detection.")
            return open_scanimage(files=paths, **_filter_kwargs(MboRawArray, kwargs))

        # Check if list of files represents multiple distinct planes
        if len(paths) > 1:
            plane_nums = { _extract_tiff_plane_number(p.name) for p in paths }
            plane_nums.discard(None)
            if len(plane_nums) > 1:
                logger.debug(f"Detected multiple planes in file list, loading as volumetric TiffArray.")
                # We use TiffArray because it handles Z-plane assembly from grouped files
                return TiffArray(paths, **_filter_kwargs(TiffArray, kwargs))

        if has_mbo_metadata(first):
            logger.debug(f"Detected MBO TIFFs, loading as MBOTiffArray.")
            return MBOTiffArray(paths, **_filter_kwargs(MBOTiffArray, kwargs))
        logger.debug(f"Loading TIFF files as TiffArray.")
        return TiffArray(paths, **_filter_kwargs(TiffArray, kwargs))

    if first.suffix == ".bin":
        if isinstance(inputs, (str, Path)) and Path(inputs).suffix == ".bin":
            logger.debug(f"Reading binary file as BinArray: {first}")
            return BinArray(first, **_filter_kwargs(BinArray, kwargs))

        npy_file = first.parent / "ops.npy"
        if npy_file.exists():
            logger.debug(f"Reading Suite2p directory from {npy_file}.")
            return Suite2pArray(npy_file)

        raise ValueError(
            f"Cannot read .bin file without ops.npy or shape parameter. "
            f"Provide shape=(nframes, Ly, Lx) as kwarg or ensure ops.npy exists."
        )

    if first.suffix == ".h5":
        logger.debug(f"Reading HDF5 files from {first}.")
        return H5Array(first, **_filter_kwargs(H5Array, kwargs))

    if first.suffix == ".zarr":
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info(f"Detected nested zarr stores, loading as ZarrArray.")
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))

        # Case 2: flat zarr store with zarr.json
        if (first / "zarr.json").exists():
            # Check if this is an isoview consolidated zarr (has camera_N groups)
            camera_dirs = [d for d in first.iterdir() if d.is_dir() and d.name.startswith("camera_")]
            if camera_dirs:
                logger.info(f"Detected isoview consolidated zarr with {len(camera_dirs)} cameras.")
                # For a single consolidated zarr, find the parent TM folder
                # or use the zarr's parent as the isoview root
                parent = first.parent
                if parent.name.startswith("TM"):
                    # Single TM folder
                    return IsoviewArray(parent)
                else:
                    # The zarr file IS the isoview structure (single timepoint)
                    return IsoviewArray(parent)

            logger.info(f"Detected zarr.json, loading as ZarrArray.")
            return ZarrArray(paths, **_filter_kwargs(ZarrArray, kwargs))

        raise ValueError(
            f"Zarr path {first} is not a valid store. "
            "Expected nested *.zarr dirs or a zarr.json inside."
        )

    if first.suffix == ".json":
        logger.debug(f"Reading JSON files from {first}.")
        return ZarrArray(first.parent, **_filter_kwargs(ZarrArray, kwargs))

    if first.suffix == ".npy":
        # Check for PMD demixer arrays
        if (first.parent / "pmd_demixer.npy").is_file():
            raise NotImplementedError("PMD Arrays are not yet supported.")

        logger.debug(f"Loading .npy file as NumpyArray: {first}")
        return NumpyArray(first, **_filter_kwargs(NumpyArray, kwargs))

    raise TypeError(f"Unsupported file type: {first.suffix}")


