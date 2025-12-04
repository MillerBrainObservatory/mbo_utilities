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
    Suite2pVolumeArray,
    TiffArray,
    TiffVolumeArray,
    ZarrArray,
    find_suite2p_plane_dirs,
    find_tiff_plane_files,
)
from mbo_utilities.metadata import has_mbo_metadata, is_raw_scanimage

logger = log.get("reader")

MBO_SUPPORTED_FTYPES = [".tiff", ".zarr", ".bin", ".h5", ".npy"]
PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")

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
    inputs: str | Path | Sequence[str | Path],
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
        ZarrArray, NumpyArray, IsoviewArray, or the input ndarray.

    Examples
    --------
    >>> from mbo_utilities import imread
    >>> arr = imread("/data/raw")  # directory with supported files
    >>> arr = imread("data.tiff")  # single file
    >>> arr = imread(["file1.tiff", "file2.tiff"])  # multiple files
    """
    if isinstance(inputs, np.ndarray):
        return inputs
    if isinstance(inputs, MboRawArray):
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
                # Check for Suite2p volume structure (subdirs with ops.npy)
                plane_dirs = find_suite2p_plane_dirs(p)
                if plane_dirs:
                    logger.info(
                        f"Detected Suite2p volume with {len(plane_dirs)} planes in {p}"
                    )
                    return Suite2pVolumeArray(p, plane_dirs=plane_dirs)

                # Check for TIFF volume structure (planeXX.tiff files)
                tiff_plane_files = find_tiff_plane_files(p)
                if tiff_plane_files:
                    logger.info(
                        f"Detected TIFF volume with {len(tiff_plane_files)} planes in {p}"
                    )
                    return TiffVolumeArray(p, plane_files=tiff_plane_files)

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
            logger.debug(f"Detected raw ScanImage TIFFs, loading as MboRawArray.")
            return MboRawArray(files=paths, **kwargs)
        if has_mbo_metadata(first):
            logger.debug(f"Detected MBO TIFFs, loading as MBOTiffArray.")
            return MBOTiffArray(paths, **kwargs)
        logger.debug(f"Loading TIFF files as TiffArray.")
        return TiffArray(paths)

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
        return H5Array(first)

    if first.suffix == ".zarr":
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info(f"Detected nested zarr stores, loading as ZarrArray.")
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))

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

    if first.suffix == ".npy":
        # Check for PMD demixer arrays
        if (first.parent / "pmd_demixer.npy").is_file():
            raise NotImplementedError("PMD Arrays are not yet supported.")

        logger.debug(f"Loading .npy file as NumpyArray: {first}")
        return NumpyArray(first)

    raise TypeError(f"Unsupported file type: {first.suffix}")


