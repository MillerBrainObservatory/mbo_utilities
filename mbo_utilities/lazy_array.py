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
)
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
    # DemixingResultsArray: set(),
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
    register_z: bool = False,
    debug: bool = False,
    shift_vectors: np.ndarray = None,
    **kwargs,  # for specific array writers
):
    # Logging
    if debug:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True  # send to terminal
    else:
        logger.setLevel(logging.WARNING)
        # logger.info("Debug mode disabled; setting log level to WARNING.")
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

    # Handle metadata
    if hasattr(lazy_array, "metadata") and lazy_array.metadata is not None:
        file_metadata = dict(lazy_array.metadata)
    else:
        file_metadata = {}

    # Always ensure save_path is recorded
    file_metadata["save_path"] = str(outpath.resolve())

    # Merge in user-supplied metadata
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError(f"Provided metadata must be a dictionary, got {type(metadata)} instead.")
        file_metadata.update(metadata)

    # propagate merged metadata back to array
    if hasattr(lazy_array, "metadata"):
        lazy_array.metadata.update(file_metadata)

    s3d_job_dir = None
    if register_z:
        lazy_array.metadata["apply_shift"] = True

        if shift_vectors is not None:
            lazy_array.metadata["shift_vectors"] = shift_vectors
        else:
            # check metadata for s3d-job dir
            if (
                "s3d-job" in lazy_array.metadata
                and Path(lazy_array.metadata["s3d-job"]).is_dir()
            ):
                print("Detected s3d-job in metadata, moving data to s3d output path.")
                s3d_job_dir = Path(lazy_array.metadata["s3d-job"])
            else:  # check if the input is in a s3d-job folder
                job_id = lazy_array.metadata.get("job_id", "s3d-preprocessed")
                s3d_job_dir = outpath / job_id

            if s3d_job_dir.joinpath("dirs.npy").is_file():
                dirs = np.load(s3d_job_dir / "dirs.npy", allow_pickle=True).item()
                for k, v in dirs.items():
                    if Path(v).is_dir():
                        lazy_array.metadata[k] = v
            else:
                # check if outpath contains an s3d job
                npy_files = outpath.rglob("*.npy")
                if "dirs.npy" in [f.name for f in npy_files]:
                    print(
                        f"Detected existing s3d-job in outpath {outpath}, skipping preprocessing."
                    )
                    s3d_job_dir = outpath
                else:
                    print(f"No s3d-job detected, preprocessing data.")
                    s3d_job_dir = register_zplanes_s3d(
                        lazy_array.filenames, file_metadata, outpath
                    )
                    print(f"Registered z-planes, results saved to {s3d_job_dir}.")

    if s3d_job_dir:
        lazy_array.metadata["s3d-job"] = s3d_job_dir
    else:
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
            # see if its a directory of .zarr dirs, which are treated differently
            zarrs = list(p.glob("*.zarr"))
            if zarrs:
                paths = zarrs
            else:
                paths = [Path(f) for f in p.glob("*") if f.is_file()]
            # paths = [Path(f) for f in get_files(p)]
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
        return Suite2pArray(parent / "ops.npy")

    exts = {p.suffix.lower() for p in paths}
    first = paths[0]

    if len(exts) > 1:
        if exts == {".bin", ".npy"}:
            npy_file = first.parent / "ops.npy"
            return Suite2pArray(npy_file)
        raise ValueError(f"Multiple file types found in input: {exts!r}")

    if first.suffix in [".tif", ".tiff"]:
        if is_raw_scanimage(first):
            return MboRawArray(files=paths, **kwargs)
        if has_mbo_metadata(first):
            return MBOTiffArray(paths, **kwargs)
        return TiffArray(paths)

    if first.suffix == ".bin":
        npy_file = first.parent / "ops.npy"
        if npy_file.exists():
            return Suite2pArray(npy_file)
        raise NotImplementedError("BIN files without metadata are not yet supported.")

    if first.suffix == ".h5":
        return H5Array(first)

    if first.suffix == ".zarr":
        # TODO: benchmark - save as volumetric in a single .zarr store?
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))

        # Case 2: flat zarr store with zarr.json
        if (first / "zarr.json").exists():
            return ZarrArray(paths, **_filter_kwargs(ZarrArray, kwargs))

        raise ValueError(
            f"Zarr path {first} is not a valid store. "
            "Expected nested *.zarr dirs or a zarr.json inside."
        )

    if first.suffix == ".npy" and (first.parent / "pmd_demixer.npy").is_file():
        raise NotImplementedError("PMD Arrays are not yet supported.")
        # return DemixingResultsArray(first.parent)

    raise TypeError(f"Unsupported file type: {first.suffix}")
