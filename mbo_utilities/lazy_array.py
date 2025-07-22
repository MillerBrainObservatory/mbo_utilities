from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from . import log
from .array_types import (
    DemixingResultsArray,
    Suite2pArray,
    H5Array,
    MBOTiffArray,
    TiffArray,
    MboRawArray,
    NpyArray,
)
from .file_io import get_files
from .metadata import is_raw_scanimage, has_mbo_metadata
from .roi import supports_roi

logger = log.get("lazy_array")


SUPPORTED_FTYPES = (
    ".npy",
    ".tif",
    ".tiff",
    ".bin",
    ".h5",
    ".zarr",
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
    MBOTiffArray: set(),  # accepts no kwargs
    Suite2pArray: set(),  # accepts no kwargs
    H5Array: {"dataset"},
    TiffArray: set(),
    NpyArray: set(),
    DemixingResultsArray: set(),
}


def _filter_kwargs(cls, kwargs):
    allowed = _ARRAY_TYPE_KWARGS.get(cls, set())
    return {k: v for k, v in kwargs.items() if k in allowed}


def imwrite(
    lazy_array,
    outpath: str | Path,
    planes: list | tuple = None,
    roi: int | Sequence[int] | None = None,
    metadata: dict = None,
    overwrite: bool = True,
    ext: str = ".tiff",
    order: list | tuple = None,
    target_chunk_mb: int = 20,
    progress_callback: Callable = None,
    debug: bool = False,
):
    # Logging
    if debug:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True  # send to terminal
    else:
        logger.setLevel(logging.WARNING)
        logger.info("Debug mode disabled; setting log level to WARNING.")
        logger.propagate = False  # don't send to terminal

    # save path
    outpath = Path(outpath)
    if not outpath.parent.is_dir():
        raise ValueError(f"{outpath} is not inside a valid directory.")

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

    # Handle metadata
    file_metadata = lazy_array.metadata or {}
    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Provided metadata must be a dictionary, got {type(metadata)} instead."
            )
        file_metadata.update(metadata)

    file_metadata["save_path"] = str(outpath.resolve())
    if hasattr(lazy_array, "metadata"):
        lazy_array.metadata.update(file_metadata)

    if hasattr(lazy_array, "_imwrite"):
        return lazy_array._imwrite(  # noqa
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            ext=ext,
            progress_callback=progress_callback,
            planes=planes,
            debug=debug,
        )
    else:
        if isinstance(lazy_array, Suite2pArray):
            raise TypeError("Attempting to write a Suite2pArray directly."
                            " Is there an ops.npy file in a directory with a tiff file?"
                            "Please make write these to separate directories.")
        raise TypeError(f"{type(lazy_array)} does not implement an `imwrite()` method.")


def imread(
    inputs: str | Path | Sequence[str | Path],
    **kwargs,  # for the reader
):
    if isinstance(inputs, np.ndarray):
        return inputs
    if isinstance(inputs, MboRawArray):
        return inputs

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")
        paths = [Path(f) for f in get_files(p)] if p.is_dir() else [p]
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
        raise ValueError(f"No supported files in {inputs}")
    paths = filtered

    parent = paths[0].parent if paths else None
    ops_file = parent / "ops.npy" if parent else None

    if ops_file and ops_file.exists():
        md = np.load(str(ops_file), allow_pickle=True).item()
        if (parent / "ops.npy").exists():
            return Suite2pArray(md)
        elif (parent / "reg_tif").is_dir():
            tifs = sorted((parent / "reg_tif").glob("*.tif"))
            if tifs:
                return TiffArray(tifs)
        else:
            raise ValueError(
                f"Unable to open suite2p data from {parent}.\n"
                "Please ensure the directory structure is correct."
            )

    exts = {p.suffix.lower() for p in paths}
    first = paths[0]

    if len(exts) > 1:
        if exts == {".bin", ".npy"}:
            npy_file = first.parent / "ops.npy"
            bin_file = first.parent / "data_raw.bin"
            md = np.load(str(npy_file), allow_pickle=True).item()
            return Suite2pArray(bin_file, md)
        raise ValueError(f"Multiple file types found in input: {exts!r}")

    if first.suffix in [".tif", ".tiff"]:
        if is_raw_scanimage(first):
            return MboRawArray(files=paths, **_filter_kwargs(MboRawArray, kwargs))
        if has_mbo_metadata(first):
            return MBOTiffArray(paths)
        return TiffArray(paths)

    if first.suffix == ".bin":
        npy_file = first.parent / "ops.npy"
        bin_file = first.parent / "data_raw.bin"
        if npy_file.exists():
            md = np.load(str(npy_file), allow_pickle=True).item()
            return Suite2pArray(bin_file, md)
        raise NotImplementedError("BIN files with metadata are not yet supported.")

    if first.suffix == ".h5":
        return H5Array(first)

    if first.suffix == ".npy" and (first.parent / "pmd_demixer.npy").is_file():
        return DemixingResultsArray(first.parent)

    raise TypeError(f"Unsupported file type: {first.suffix}")
