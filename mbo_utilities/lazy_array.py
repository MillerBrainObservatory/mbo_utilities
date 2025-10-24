from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Callable

import numpy as np

from . import log
from ._writers import _try_generic_writers, _write_plane
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
from .file_io import derive_tag_from_filename
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

    Users likely will want to use `mbo.imread` to load data as input to this function.

    Parameters
    ----------
    lazy_array : object
        One of the supported lazy array readers providing `.shape`, `.metadata`,
        and `_imwrite()` methods:

        - `Suite2pArray` : memory-mapped binary (`data.bin` or `data_raw.bin`)
          paired with an `ops.npy`. Can write to TIFF or binary targets.
        - `H5Array` : HDF5 dataset wrapper (`h5py.File[dataset]`).
        - `MBOTiffArray` : multi-file TIFF reader using Dask/memmap backend.
        - `TiffArray` : single or multi-TIFF reader.
        - `MboRawArray` : raw ScanImage/ScanMultiROI acquisition object.
        - `NpyArray` : single `.npy` memory-mapped NumPy file.
        - `ZarrArray` : collection of z-plane `.zarr` stores.
        - `NWBArray` : NWB file with “TwoPhotonSeries” acquisition dataset.

    outpath : str or Path
        Target directory or file path to write output into. Must exist or be creatable.
    planes : list or tuple of int, optional
        Specific z-planes to export (1-based indexing for consistency with Suite2p).
        Defaults to all planes.
    num_frames : list or tuple of int, optional
        The number of frames to export. Defaults to all frames.
    roi : int or sequence of int, optional
        ROI index(es) to restrict output for multi-ROI data (e.g. `MboRawArray`).
    metadata : dict, optional
        Additional metadata to merge into the written file header.
    overwrite : bool, default=False
        Overwrite existing output files if True.
    ext : str, default=".tiff"
        Output format extension. Supports ".tiff", ".tif", ".bin", etc.
    order : list or tuple of int, optional
        Re-ordering of `planes` before writing, e.g. `[2, 0, 1]`.
    target_chunk_mb : int, default=20
        Approximate target chunk size in MB for streamed writes.
    progress_callback : callable, optional
        Function to receive progress updates during writing.
    register_z : bool, default=False
        If True, perform z-plane registration via Suite3D preprocessing
        (`register_zplanes_s3d`) before writing.
    debug : bool, default=False
        Enable verbose logging.
    shift_vectors : np.ndarray, optional
        Pre-computed z-shift vectors to embed into metadata.

    Returns
    -------
    Path
        Path to the written output directory or file.

    Raises
    ------
    TypeError
        If the input array type is unsupported or incompatible with options.
    ValueError
        If `outpath` is invalid or metadata is malformed.
    FileNotFoundError
        If expected companion files (e.g. `ops.npy`) are missing.

    Notes
    -----
    - Metadata from the source array is merged with `metadata` and recorded in
      the written output (e.g. TIFF tags, Zarr attributes, or sidecar JSON).
    - When `register_z=True`, the function attempts to detect or generate a
      Suite3D job directory (`s3d-job`) and stores registration parameters there.
    - The writing backend (`_write_plane`) supports efficient chunked I/O for
      large 3-D or 4-D volumes.

    Examples
    --------
    >>> from mbo_utilities import imread, imwrite
    >>> mbo_data = imread("data/session1")  # load data from supported files

    # Tile ScanImagee multi-ROI's and write all planes to a single multi-page TIFF.
    >>> imwrite(mbo_data, ext="tif", outpath="output/session1/extracted", roi=None)

    Write axially registered data to a new folder:
    >>> imwrite(mbo_data, ext="tif", outpath="output/session1/extracted", register_z=True)

    Write only the first two planes, overwriting existing TIFFs:
        >>> imwrite(mbo_data, "output/session1", planes=[1, 2], overwrite=True, roi=None)

    Write ALL roi's to Zarr format:
        >>> imwrite(mbo_data, "output/rois", roi=0, ext=".zarr")

    Write ROI 1 to Suite2p-compatible binary format:
        >>> imwrite("data/session1", "output/bin_output", roi=1, ext=".bin")
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

        if shift_vectors is not None:
            lazy_array.metadata["shift_vectors"] = shift_vectors
        else:
            # check metadata for s3d-job dir
            if (
                "s3d-job" in lazy_array.metadata
                and Path(lazy_array.metadata["s3d-job"]).is_dir()
            ):
                logger.debug("Detected s3d-job in metadata, moving data to s3d output path.")
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
                    logger.info(
                        f"Detected existing s3d-job in outpath {outpath}, skipping preprocessing."
                    )
                    s3d_job_dir = outpath
                else:
                    logger.info(f"No s3d-job detected, preprocessing data.")
                    # s3d_params = kwargs.get("s3d_params", {})
                    s3d_job_dir = register_zplanes_s3d(
                        filenames=lazy_array.filenames,
                        metadata=file_metadata,
                        outpath=outpath,
                        progress_callback=progress_callback
                    )
                    logger.info(f"Registered z-planes, results saved to {s3d_job_dir}.")

    if s3d_job_dir:
        logger.info(f"Storing s3d-job path {s3d_job_dir} in metadata.")
        lazy_array.metadata["s3d-job"] = s3d_job_dir
    else:
        logger.info("No s3d-job directory used or created.")
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
