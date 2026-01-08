"""
imread - Lazy load imaging data from supported file types.

This module provides the imread() function for loading imaging data from
various file formats as lazy arrays.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays import (
    BinArray,
    CalibrationArray,
    H5Array,
    IsoviewArray,
    LBMArray,
    MBOTiffArray,
    NumpyArray,
    PiezoArray,
    ScanImageArray,
    SinglePlaneArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
    _extract_tiff_plane_number,
    open_scanimage,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = log.get("reader")

MBO_SUPPORTED_FTYPES = [".tiff", ".tif", ".zarr", ".bin", ".h5", ".npy"]

# Re-export PIPELINE_TAGS for backward compatibility (canonical location is file_io.py)


@lru_cache(maxsize=32)
def _get_init_params(cls: type) -> set[str]:
    """
    Get the set of parameter names accepted by a class's __init__.

    Uses inspect.signature for dynamic introspection rather than hardcoded
    mappings. Results are cached for performance.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    set[str]
        Set of parameter names (excluding 'self').
    """
    try:
        sig = inspect.signature(cls.__init__)
        # Exclude 'self' and collect all parameter names
        return {
            name
            for name, param in sig.parameters.items()
            if name != "self" and param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
    except (ValueError, TypeError):
        # Fallback for classes without inspectable __init__
        return set()


def _filter_kwargs(cls, kwargs):
    """
    Filter kwargs to only those accepted by cls.__init__.

    Uses dynamic introspection via _get_init_params rather than
    hardcoded mappings.
    """
    allowed = _get_init_params(cls)
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

    # filter out pollen calibration result files (*_pollen.h5)
    # these are output files, not source data
    paths = [p for p in paths if not p.name.endswith("_pollen.h5")]
    if not paths:
        raise ValueError(
            f"No source data files found in {inputs}. "
            f"Only pollen calibration result files (*_pollen.h5) were found."
        )

    parent = paths[0].parent if paths else None
    ops_file = parent / "ops.npy" if parent else None

    # Suite2p ops file
    if ops_file and ops_file.exists():
        if len(paths) == 1 and paths[0].suffix.lower() == ".bin":
            logger.debug(f"Ops.npy detected - reading specific binary {paths[0]}.")
            return Suite2pArray(paths[0])
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
        # Check if list of files represents multiple distinct planes
        # (this takes priority over type detection - it's a structural choice)
        if len(paths) > 1:
            plane_nums = {_extract_tiff_plane_number(p.name) for p in paths}
            plane_nums.discard(None)
            if len(plane_nums) > 1:
                logger.debug("Detected multiple planes in file list, loading as volumetric TiffArray.")
                return TiffArray(paths, **_filter_kwargs(TiffArray, kwargs))

        # Try array classes in priority order (most specific first)
        # Each class's can_open() checks if it can handle the file
        TIFF_ARRAY_CLASSES = [
            # Specialized ScanImage subclasses (most specific)
            (LBMArray, "LBM stack"),
            (PiezoArray, "piezo stack"),
            (CalibrationArray, "calibration/pollen stack"),
            (SinglePlaneArray, "single-plane ScanImage"),
            # Generic ScanImage (raw acquisition data)
            (ScanImageArray, "raw ScanImage"),
            # Processed MBO TIFFs
            (MBOTiffArray, "MBO processed TIFF"),
            # Fallback (any TIFF)
            (TiffArray, "generic TIFF"),
        ]

        for array_cls, description in TIFF_ARRAY_CLASSES:
            if array_cls.can_open(first):
                logger.debug(f"Detected {description}, loading as {array_cls.__name__}.")
                return array_cls(paths, **_filter_kwargs(array_cls, kwargs))

    if first.suffix == ".bin":
        if isinstance(inputs, (str, Path)) and Path(inputs).suffix == ".bin":
            logger.debug(f"Reading binary file as BinArray: {first}")
            return BinArray(first, **_filter_kwargs(BinArray, kwargs))

        npy_file = first.parent / "ops.npy"
        if npy_file.exists():
            logger.debug(f"Reading Suite2p directory from {npy_file}.")
            return Suite2pArray(npy_file)

        raise ValueError(
            "Cannot read .bin file without ops.npy or shape parameter. "
            "Provide shape=(nframes, Ly, Lx) as kwarg or ensure ops.npy exists."
        )

    if first.suffix == ".h5":
        logger.debug(f"Reading HDF5 files from {first}.")
        return H5Array(first, **_filter_kwargs(H5Array, kwargs))

    if first.suffix == ".zarr":
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info("Detected nested zarr stores, loading as ZarrArray.")
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
                # The zarr file IS the isoview structure (single timepoint)
                return IsoviewArray(parent)

            logger.info("Detected zarr.json, loading as ZarrArray.")
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


