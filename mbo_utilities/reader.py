"""
imread - Lazy load imaging data from supported file types.

This module provides the imread() function for loading imaging data from
various file formats as lazy arrays.
"""

from __future__ import annotations

import importlib.util
import inspect
from functools import lru_cache
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays import (
    BinArray,
    LBMPiezoArray,
    H5Array,
    LBMArray,
    MP4Array,
    NumpyArray,
    PiezoArray,
    ScanImageArray,
    SinglePlaneArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
    _extract_tiff_plane_number,
)
from mbo_utilities.arrays.isoview import (
    IsoviewArray,
    detect_isoview_kind,
)
from mbo_utilities.lazy_array import _dispatch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = log.get("reader")

# UI dropdown shows these formats (excludes .tif to avoid duplication)
MBO_SUPPORTED_FTYPES = [".tiff", ".zarr", ".bin", ".h5", ".klb", ".mp4"]
# reading accepts .tif as alias for .tiff
MBO_READABLE_FTYPES = [".tiff", ".tif", ".zarr", ".bin", ".h5", ".npy", ".klb", ".mp4"]

# extensions that require an optional third-party package. when the package
# isn't importable we drop the extension from the GUI dropdown so the user
# can't pick a format that would fail at write time.
_OPTIONAL_PKG_BY_EXT = {".klb": "pyklb"}

MBO_AVAILABLE_FTYPES = [
    ext for ext in MBO_SUPPORTED_FTYPES
    if _OPTIONAL_PKG_BY_EXT.get(ext) is None
    or importlib.util.find_spec(_OPTIONAL_PKG_BY_EXT[ext]) is not None
]

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
    channel : int, optional
        Zero-based color-channel index. When given, the returned array
        is wrapped as a 4D TZYX view of that single channel — useful for
        feeding multi-channel sources into pipelines that expect TZYX
        input. Subprocess workers can re-create the same view by passing
        ``reader_kwargs={"channel": N}`` to ``imread``.
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
    # Pull single-channel wrapping out so every return path benefits.
    # Stored as ``channel`` (zero-based) — picklable through reader_kwargs
    # so subprocess workers can re-create the wrap after their own imread.
    channel = kwargs.pop("channel", None)
    squeeze = kwargs.pop("squeeze", False)
    arr = _imread_impl(inputs, **kwargs)
    if channel is not None:
        from mbo_utilities.arrays._channel_view import _ChannelView
        if not hasattr(arr, "shape") or len(arr.shape) < 5:
            logger.debug(
                "imread(channel=%r): underlying array is %dD, returning unwrapped",
                channel, getattr(arr, "ndim", "?"),
            )
        else:
            arr = _ChannelView(arr, int(channel))
    if squeeze and hasattr(arr, "squeeze"):
        arr = arr.squeeze()
    return arr


def _imread_impl(
    inputs: str | Path | np.ndarray | Sequence[str | Path],
    **kwargs,
):
    """Internal imread that returns the raw lazy array without channel wrapping."""
    # Wrap numpy arrays in NumpyArray for full imwrite/protocol support
    if isinstance(inputs, np.ndarray):
        logger.debug(f"Wrapping numpy array with shape {inputs.shape} as NumpyArray")
        arr = NumpyArray(inputs, **_filter_kwargs(NumpyArray, kwargs))
        if arr._dims_inferred and inputs.ndim >= 3 and arr.input_dims:
            # root mbo logger carries the stream handler; child loggers are
            # silent in plain scripts, and this is a user-facing hint.
            log.get().info(
                "numpy %s read as %s; pass dims= to override",
                tuple(inputs.shape), "".join(arr.input_dims),
            )
        return arr
    # A SqueezedView is a display lens over a canonical 5D array; normalize
    # back to that base so imread()/pipeline() operate on the real 5D array
    # (the view drops axes the writer/pipeline rely on).
    from mbo_utilities.squeeze import SqueezedView
    if isinstance(inputs, SqueezedView):
        return inputs.base
    # Pass through already-loaded lazy arrays (has _imwrite method)
    if hasattr(inputs, "_imwrite") and hasattr(inputs, "shape"):
        return inputs

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")

        # redirect an inner-zarr path (zarr.json / chunk file) to the store
        # root so can_open() sees the store directory, matching the legacy
        # redirect further down.
        if not p.is_dir():
            for ancestor in p.parents:
                if ancestor.suffix.lower() == ".zarr" and ancestor.is_dir():
                    logger.debug(f"Redirecting {p.name} -> parent zarr store {ancestor}")
                    p = ancestor
                    break

        # v4 dispatch: the highest-PRIORITY registered class (built-in or a
        # third-party plugin) whose can_open() accepts the path wins. Falls
        # through to the legacy detection below for inputs no class claims
        # (multi-file lists, .bin/.klb/.mp4, reg_tif folders, mixed dirs).
        cls = _dispatch(p)
        if cls is not None:
            logger.debug(f"Dispatch selected {cls.__name__} for {p}")
            return cls(p, **_filter_kwargs(cls, kwargs))

        # Suite2p outputs take priority over isoview ancestor matching.
        # detect_isoview_kind walks up parents looking for `.corrected` /
        # `.fused` ancestors, so a suite2p plane folder sitting inside
        # `<root>.corrected.registered/` would otherwise be returned as
        # an IsoviewArray over the 5D source. ops.npy / plane subdirs are
        # unambiguous suite2p markers — if present, the user pointed at
        # the suite2p folder, not the source tree.
        if p.is_dir():
            if (p / "ops.npy").exists():
                logger.info(f"Detected Suite2p directory at {p}")
                return Suite2pArray(p)
            if p.name == "reg_tif" and (p.parent / "ops.npy").exists():
                logger.info(f"Detected Suite2p reg_tif folder at {p}")
                return Suite2pArray(p.parent / "ops.npy", use_reg_tif=True)
            plane_subdirs = [d for d in p.iterdir() if d.is_dir() and (d / "ops.npy").exists()]
            if plane_subdirs:
                logger.info(f"Detected Suite2p volume with {len(plane_subdirs)} planes in {p}")
                return Suite2pArray(p)

        # Isoview detection runs before file-vs-dir dispatch so "Open File"
        # on `<root>.corrected/SPM##/TM######/*.zarr`, "Open Folder" on
        # `<root>.corrected/SPM##`, and `mbo <root>.corrected/` all resolve
        # to the enclosing stack root via parent-walk.
        iso_kind = detect_isoview_kind(p)
        if iso_kind is not None:
            logger.info(f"Detected isoview-{iso_kind} tree at {p}")
            return IsoviewArray(p, kind=iso_kind)

        # if the user (or a file dialog) handed us a path inside a .zarr v3
        # store, redirect to the store root. file pickers default to selecting
        # the inner zarr.json/zgroup.json/zarray rather than the .zarr folder
        # itself, and that previously failed with "no supported files".
        if not p.is_dir():
            for ancestor in p.parents:
                if ancestor.suffix.lower() == ".zarr" and ancestor.is_dir():
                    logger.debug(
                        f"Redirecting {p.name} -> parent zarr store {ancestor}"
                    )
                    p = ancestor
                    break

        if p.suffix.lower() == ".zarr" and p.is_dir():
            paths = [p]
        elif p.is_dir():
            logger.debug(f"Input is a directory, searching for supported files in {p}")

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

                # Pointed at a suite2p reg_tif/ folder directly. Route to the
                # parent plane via Suite2pArray with use_reg_tif=True so files
                # are sorted by numeric frame_start, not lexicographically
                # (suite2p uses variable-width filenames).
                if p.name == "reg_tif" and (p.parent / "ops.npy").exists():
                    logger.info(f"Detected Suite2p reg_tif folder at {p}")
                    return Suite2pArray(p.parent / "ops.npy", use_reg_tif=True)

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

    filtered = [p for p in paths if p.suffix.lower() in MBO_READABLE_FTYPES]
    if not filtered:
        raise ValueError(
            f"No supported files in {inputs}. \n"
            f"Supported file types are: {MBO_READABLE_FTYPES}"
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
            (LBMPiezoArray, "LBM+piezo stack"),
            (SinglePlaneArray, "single-plane ScanImage"),
            # Generic ScanImage (raw acquisition data)
            (ScanImageArray, "raw ScanImage"),
            # Fallback: TiffArray handles both standard TIFFs and ImageJ hyperstacks
            (TiffArray, "TIFF"),
        ]

        for array_cls, description in TIFF_ARRAY_CLASSES:
            if array_cls.can_open(first):
                # filter to only files this class can open, so saved
                # outputs or unrelated tiffs in the same folder are excluded
                if len(paths) > 1:
                    valid = [p for p in paths if array_cls.can_open(p)]
                    if not valid:
                        continue
                    if len(valid) < len(paths):
                        logger.info(
                            f"Filtered {len(paths) - len(valid)} non-{description} "
                            f"file(s) from directory load"
                        )
                    paths = valid
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

    if first.suffix == ".mp4":
        logger.debug(f"Reading MP4 file as MP4Array: {first}")
        return MP4Array(first, **_filter_kwargs(MP4Array, kwargs))

    if first.suffix == ".zarr":
        # Case 1: nested zarrs inside
        sub_zarrs = list(first.glob("*.zarr"))
        if sub_zarrs:
            logger.info("Detected nested zarr stores, loading as ZarrArray.")
            return ZarrArray(sub_zarrs, **_filter_kwargs(ZarrArray, kwargs))

        # Case 2: flat zarr store with zarr.json
        if (first / "zarr.json").exists():
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

    if first.suffix == ".klb":
        import pyklb

        # if only one KLB file specified, read it directly
        if len(paths) == 1:
            logger.info(f"Loading KLB file: {first}")
            data = pyklb.readfull(str(first))
            return NumpyArray(data)

        # clusterPT tree: TM######/SPM##_TM######_CM##_CHN##.klb
        parent = first.parent
        if parent.name.startswith("TM"):
            logger.info("Detected KLB file in TM folder, loading as isoview-clusterpt.")
            return IsoviewArray(parent.parent, kind="clusterpt")

        tm_folders = [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("TM")]
        if tm_folders:
            logger.info("Detected KLB file in clusterPT root, loading as isoview-clusterpt.")
            return IsoviewArray(parent, kind="clusterpt")

        logger.info(f"Loading standalone KLB file: {first}")
        data = pyklb.readfull(str(first))
        return NumpyArray(data)

    raise TypeError(f"Unsupported file type: {first.suffix}")


