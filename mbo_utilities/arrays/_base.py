"""
Common helpers and base utilities for array types.

This module contains shared functionality used across all array implementations:
- ROI handling (supports_roi, normalize_roi, iter_rois)
- Plane normalization (_normalize_planes)
- Output path building (_build_output_path)
- Common write implementation (_imwrite_base)
- Axis/dimension utilities (_to_tzyx, _axes_or_guess)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dask import array as da

from mbo_utilities import log
from mbo_utilities.arrays.features._dim_labels import get_dims, get_num_planes
from mbo_utilities._writers import _write_plane
from mbo_utilities.metadata import RoiMode
from numpy.exceptions import AxisError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = log.get("arrays._base")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}


def supports_roi(obj):
    """
    Check if object supports ROI operations.

    .. deprecated::
        Use ``hasattr(obj, 'roi_mode')`` for duck typing instead.
        This function is kept for backward compatibility.
    """
    # Modern check: duck typing via roi_mode attribute
    if hasattr(obj, "roi_mode"):
        return True
    # Legacy check for backwards compatibility
    return hasattr(obj, "roi") and hasattr(obj, "num_rois")


def normalize_roi(value):
    """Return ROI as None, int, or list[int] with consistent semantics."""
    if value in (None, (), [], False):
        return None
    if value is True:
        return 0  # "split ROIs" GUI flag
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def iter_rois(obj):
    """
    Yield ROI indices based on MBO semantics.

    .. deprecated::
        Use ``arr.iter_rois()`` method on arrays with RoiFeatureMixin instead.
        Check support with ``hasattr(arr, 'roi_mode')``.
        This function is kept for backward compatibility.

    - roi=None -> yield None (stitched full-FOV image)
    - roi=0 -> yield each ROI index from 1..num_rois (split all)
    - roi=int > 0 -> yield that ROI only
    - roi=list/tuple -> yield each element (as given)
    """
    # Modern approach: use object's own iter_rois method if available
    if hasattr(obj, "roi_mode") and hasattr(obj, "iter_rois"):
        yield from obj.iter_rois()
        return

    # Legacy fallback
    if not supports_roi(obj):
        yield None
        return

    roi = getattr(obj, "roi", None)
    num_rois = getattr(obj, "num_rois", 1)

    if roi is None:
        yield None
    elif roi == 0:
        yield from range(1, num_rois + 1)
    elif isinstance(roi, int):
        yield roi
    elif isinstance(roi, (list, tuple)):
        for r in roi:
            if r == 0:
                yield from range(1, num_rois + 1)
            else:
                yield r


def _normalize_planes(planes, num_planes: int) -> list[int]:
    """
    Normalize planes argument to 0-indexed list.

    Parameters
    ----------
    planes : int | list | tuple | None
        Planes to write (1-based indexing from user).
    num_planes : int
        Total number of planes available.

    Returns
    -------
    list[int]
        0-indexed plane indices.
    """
    if planes is None:
        return list(range(num_planes))
    if isinstance(planes, int):
        return [planes - 1]  # 1-based to 0-based
    return [p - 1 for p in planes]


def _sanitize_suffix(suffix: str) -> str:
    """
    Sanitize a filename suffix to prevent invalid filenames.

    Parameters
    ----------
    suffix : str
        Raw suffix string from user input.

    Returns
    -------
    str
        Sanitized suffix safe for use in filenames.
    """
    if not suffix:
        return ""

    # Remove illegal characters for Windows/Unix filenames
    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        suffix = suffix.replace(char, "")

    # Remove any file extension patterns (e.g., ".bin", ".tiff")
    # This prevents issues like "plane01_stitched.bin.bin"
    import re

    suffix = re.sub(r"\.[a-zA-Z0-9]+$", "", suffix)

    # Ensure suffix starts with underscore if not empty and doesn't already
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    # Remove any double underscores
    while "__" in suffix:
        suffix = suffix.replace("__", "_")

    # Strip trailing underscores
    return suffix.rstrip("_")



def _build_output_path(
    outpath: Path,
    plane_idx: int,
    roi: int | None,
    ext: str,
    output_name: str | None = None,
    structural: bool = False,
    has_multiple_rois: bool = False,
    output_suffix: str | None = None,
    **kwargs,
) -> Path:
    """
    Build output file path for a single plane.

    Parameters
    ----------
    outpath : Path
        Base output directory.
    plane_idx : int
        0-indexed plane number.
    roi : int | None
        ROI index (1-based) or None for stitched/single ROI.
    ext : str
        File extension (without dot).
    output_name : str | None
        Override output filename (for .bin files).
    structural : bool
        If True, use data_chan2.bin naming for structural channel.
    has_multiple_rois : bool
        If True and roi is None, use "_stitched" suffix by default.
    output_suffix : str | None
        Custom suffix to append to filenames. If None, uses "_stitched" for
        multi-ROI data when roi is None, or "_roiN" for specific ROIs.
        The suffix is sanitized to remove illegal characters and prevent
        double extensions.

    Returns
    -------
    Path
        Full output file path.
    """
    plane_num = plane_idx + 1  # Convert to 1-based for filenames

    # Determine suffix based on ROI and custom output_suffix
    if roi is None:
        if output_suffix is not None:
            # Use custom suffix (sanitized)
            roi_suffix = _sanitize_suffix(output_suffix)
        elif has_multiple_rois:
            # Default to "_stitched" for multi-ROI data
            roi_suffix = "_stitched"
        else:
            roi_suffix = ""
    else:
        roi_suffix = f"_roi{roi}"

    if ext == "bin":
        if output_name:
            # Caller specified exact output - use it directly
            if structural:
                return outpath / "data_chan2.bin"
            return outpath / output_name

        # Build subdirectory structure
        subdir = f"plane{plane_num:02d}{roi_suffix}"
        plane_dir = outpath / subdir
        plane_dir.mkdir(parents=True, exist_ok=True)

        if structural:
            return plane_dir / "data_chan2.bin"
        return plane_dir / "data_raw.bin"
    # Non-binary formats: single file per plane
    return outpath / f"plane{plane_num:02d}{roi_suffix}.{ext}"


def _sanitize_value(v):
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, (Path, np.dtype, type)):
        return str(v)
    if hasattr(v, "dtype") or hasattr(v, "item"):  # numpy scalars/arrays
        if hasattr(v, "ndim") and v.ndim == 0:
            return v.item()
        if hasattr(v, "tolist"):
            return v.tolist()
        if hasattr(v, "item"):
            return v.item()
    return v


def _sanitize_metadata(md: dict) -> dict:
    """Recursively sanitize metadata for JSON serialization."""
    return _sanitize_value(md)


def _imwrite_base(
    arr,
    outpath: Path | str,
    planes: int | list | tuple | None = None,
    ext: str = ".tiff",
    overwrite: bool = False,
    target_chunk_mb: int = 50,
    progress_callback: Callable | None = None,
    debug: bool = False,
    show_progress: bool = True,
    roi_iterator=None,
    output_suffix: str | None = None,
    roi_mode: RoiMode | str | None = None,
    **kwargs,
) -> Path:
    """
    Common implementation for array _imwrite() methods.

    This function handles the common pattern of:
    1. Normalizing planes argument (1-based to 0-based)
    2. Iterating over ROIs (if applicable)
    3. Building output paths
    4. Calling _write_plane() for each plane

    Parameters
    ----------
    arr : LazyArrayProtocol
        Array to write. Must have shape, metadata, and support indexing.
    outpath : Path | str
        Output directory.
    planes : int | list | tuple | None
        Planes to write (1-based indexing). None means all planes.
    ext : str
        Output format extension (e.g., '.tiff', '.bin', '.zarr').
    overwrite : bool
        Whether to overwrite existing files.
    target_chunk_mb : int
        Target chunk size in MB for streaming writes.
    progress_callback : callable | None
        Progress callback function.
    debug : bool
        Enable debug output.
    roi_iterator : iterator | None
        Custom ROI iterator for arrays with ROI support.
        If None, uses iter_rois(arr) which yields [None] for arrays without ROIs.
    output_suffix : str | None
        Custom suffix to append to output filenames. If None, defaults to
        "_stitched" for multi-ROI data when roi is None.
        Examples: "_stitched", "_processed", "_mydata"
    **kwargs
        Additional arguments passed to _write_plane().

    Returns
    -------
    Path
        Output directory path.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    ext_clean = ext.lower().lstrip(".")

    # Get metadata
    md = dict(arr.metadata) if arr.metadata else {}

    # Merge metadata overrides if present
    if "metadata_overrides" in kwargs:
        overrides = kwargs.get("metadata_overrides")
        if overrides and isinstance(overrides, dict):
            md.update(overrides)

    # Sanitize metadata for serialization (e.g. JSON in Zarr/Tiff)
    md = _sanitize_metadata(md)

    # Get dimensions using protocol helpers
    dims = get_dims(arr)
    num_planes = get_num_planes(arr)

    # Extract shape info
    # for 4D arrays, first dim is always the iteration dimension (frames/volumes)
    # for 3D arrays, check if there's a time dimension
    if len(arr.shape) == 4:
        nframes = arr.shape[0]
    elif len(arr.shape) == 3 and dims[0] in {"T", "timepoints"}:
        nframes = arr.shape[0]
    else:
        nframes = 1
    Ly, Lx = arr.shape[-2], arr.shape[-1]

    # validate num_planes against actual shape (metadata may not match data)
    if len(arr.shape) == 4:
        actual_z_size = arr.shape[1]  # TZYX format
        if num_planes > actual_z_size:
            logger.debug(
                f"num_planes ({num_planes}) > actual Z dim ({actual_z_size}), using shape"
            )
            num_planes = actual_z_size
    elif len(arr.shape) == 3:
        # 3D data (TYX) has no Z dimension, treat as single plane
        num_planes = 1

    # Update metadata
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["num_timepoints"] = nframes
    md["nframes"] = nframes  # suite2p alias
    md["num_frames"] = nframes  # legacy alias

    # normalize and store roi_mode
    if roi_mode is not None:
        if isinstance(roi_mode, str):
            roi_mode = RoiMode.from_string(roi_mode)
        md["roi_mode"] = roi_mode.value

    # Normalize planes to 0-indexed list
    planes_list = _normalize_planes(planes, num_planes)

    # Use provided ROI iterator or detect via duck typing
    # Arrays with roi_mode attribute support multi-ROI operations
    if roi_iterator is not None:
        roi_iter = roi_iterator
    elif hasattr(arr, "roi_mode") and hasattr(arr, "iter_rois"):
        roi_iter = arr.iter_rois()
    else:
        roi_iter = iter([None])  # No ROI support, single iteration

    # Check if array has multiple ROIs (for "_stitched" suffix)
    has_multiple_rois = getattr(arr, "num_rois", 1) > 1

    for roi in roi_iter:
        # Update array's ROI if it supports ROI operations
        if roi is not None and hasattr(arr, "roi_mode"):
            arr.roi = roi

        for plane_idx in planes_list:
            target = _build_output_path(
                outpath,
                plane_idx,
                roi,
                ext_clean,
                output_name=kwargs.get("output_name"),
                structural=kwargs.get("structural", False),
                has_multiple_rois=has_multiple_rois,
                output_suffix=output_suffix,
            )

            if target.exists() and not overwrite:
                logger.warning(f"File {target} already exists. Skipping write.")
                continue

            # Build plane-specific metadata
            plane_md = md.copy()
            plane_md["plane"] = plane_idx + 1  # 1-based in metadata
            if roi is not None:
                plane_md["roi"] = roi
                plane_md["mroi"] = roi  # alias

            _write_plane(
                arr,
                target,
                overwrite=overwrite,
                target_chunk_mb=target_chunk_mb,
                metadata=plane_md,
                progress_callback=progress_callback,
                debug=debug,
                show_progress=show_progress,
                dshape=(nframes, Ly, Lx),
                plane_index=plane_idx,
                **kwargs,
            )

    # signal completion
    if progress_callback:
        progress_callback(1.0, len(planes_list))

    return outpath


def _to_tzyx(a: da.Array, axes: str) -> da.Array:
    """Convert dask array to TZYX dimension order."""
    order = [ax for ax in ["T", "Z", "C", "S", "Y", "X"] if ax in axes]
    perm = [axes.index(ax) for ax in order]
    a = da.transpose(a, axes=perm)
    have_T = "T" in order
    pos = {ax: i for i, ax in enumerate(order)}
    tdim = a.shape[pos["T"]] if have_T else 1
    merge_dims = [d for d, ax in enumerate(order) if ax in ("Z", "C", "S")]
    if merge_dims:
        front = []
        if have_T:
            front.append(pos["T"])
        rest = [d for d in range(a.ndim) if d not in front]
        a = da.transpose(a, axes=front + rest)
        newshape = [
            tdim if have_T else 1,
            int(np.prod([a.shape[i] for i in rest[:-2]])),
            a.shape[-2],
            a.shape[-1],
        ]
        a = a.reshape(newshape)
    else:
        if have_T:
            if a.ndim == 3:
                a = da.expand_dims(a, 1)
        else:
            a = da.expand_dims(a, 0)
            a = da.expand_dims(a, 1)
        if order[-2:] != ["Y", "X"]:
            yx_pos = [order.index("Y"), order.index("X")]
            keep = [i for i in range(len(order)) if i not in yx_pos]
            a = da.transpose(a, axes=keep + yx_pos)
    return a


def _axes_or_guess(arr_ndim: int) -> str:
    """Guess axis labels from array dimensionality."""
    if arr_ndim == 2:
        return "YX"
    if arr_ndim == 3:
        return "ZYX"
    if arr_ndim == 4:
        return "TZYX"
    return "Unknown"


def _safe_get_metadata(path: Path) -> dict:
    """Safely get metadata from a file path."""
    try:
        from mbo_utilities.metadata import get_metadata

        return get_metadata(path)
    except Exception:
        return {}


class ReductionMixin:
    """
    Mixin providing numpy-compatible reduction methods for arrays.

    Adds mean, max, min, std, sum methods that work like numpy does. For lazy
    arrays, uses chunked processing to avoid loading everything into memory.
    For numpy arrays or memmaps, delegates directly to numpy.

    The API matches numpy exactly - same parameters, same behavior. No custom
    defaults for axis (axis=None reduces over entire array, just like numpy).

    Required attributes (duck-typed):
        shape : tuple[int, ...]
            Array dimensions
        dtype : np.dtype
            Data type
        __getitem__ : callable
            Slicing support

    Usage
    -----
    class MyArray(ReductionMixin):
        # ... array implementation with shape, dtype, __getitem__ ...
        pass

    arr = MyArray(path)
    scalar = arr.mean()           # Mean of all elements (like numpy)
    volume = arr.mean(axis=0)     # Mean over first axis
    """

    def _is_numpy_like(self) -> bool:
        """Check if this array can be reduced directly by numpy."""
        # numpy arrays, memmaps, and anything with __array__ that's small enough
        return isinstance(self, (np.ndarray, np.memmap))

    def _get_array_info(self) -> tuple[tuple, np.dtype, int]:
        """
        Get shape, dtype, ndim from array using duck typing.

        Returns
        -------
        tuple
            (shape, dtype, ndim)

        Raises
        ------
        TypeError
            If required attributes are missing
        """
        shape = getattr(self, "shape", None)
        dtype = getattr(self, "dtype", None)

        if shape is None:
            raise TypeError(
                f"{type(self).__name__} missing required 'shape' attribute for reductions"
            )

        ndim = len(shape)
        if dtype is None:
            dtype = np.float64  # fallback

        return shape, dtype, ndim

    def _reduce(
        self,
        func: str,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
        out: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply reduction function with numpy-compatible semantics.

        Parameters
        ----------
        func : str
            Reduction function name: 'mean', 'max', 'min', 'std', 'sum', 'var'
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce over. None reduces over all elements.
        dtype : np.dtype | None
            Output dtype.
        keepdims : bool
            If True, reduced axes are left with size 1.
        out : np.ndarray | None
            Output array (numpy compatibility, used if provided).
        **kwargs
            Additional arguments passed to numpy (e.g., ddof for std/var).

        Returns
        -------
        np.ndarray or scalar
            Reduced result.
        """
        shape, _arr_dtype, ndim = self._get_array_info()

        # For numpy arrays/memmaps, just delegate directly
        if self._is_numpy_like():
            np_func = getattr(np, func)
            # Not all numpy functions accept dtype (max/min don't)
            if func in ("max", "min"):
                return np_func(self, axis=axis, keepdims=keepdims, out=out, **kwargs)
            return np_func(self, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs)

        # Normalize axis
        if axis is not None:
            if isinstance(axis, int):
                if axis < 0:
                    axis = ndim + axis
                if axis < 0 or axis >= ndim:
                    raise AxisError(axis, ndim)
            else:
                # tuple of axes
                axis = tuple(ax if ax >= 0 else ndim + ax for ax in axis)
                for ax in axis:
                    if ax < 0 or ax >= ndim:
                        raise AxisError(ax, ndim)

        # Determine if we can just load and compute (small arrays)
        total_elements = int(np.prod(shape))
        chunk_threshold = 100_000_000  # ~100M elements, ~800MB for float64

        if total_elements <= chunk_threshold:
            # Small enough to load entirely - use explicit slicing, not np.asarray
            # (np.asarray only returns single frame for fast preview)
            data = self[:]
            np_func = getattr(np, func)
            # Not all numpy functions accept dtype (max/min don't)
            if func in ("max", "min"):
                result = np_func(data, axis=axis, keepdims=keepdims, out=out, **kwargs)
            else:
                result = np_func(data, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs)
            return result

        # Large array - use chunked reduction
        return self._chunked_reduce(
            func, axis=axis, dtype=dtype, keepdims=keepdims, out=out, **kwargs
        )

    def _chunked_reduce(
        self,
        func: str,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        keepdims: bool = False,
        out: np.ndarray | None = None,
        chunk_size: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """Apply reduction function over axis, processing in chunks for large arrays."""
        from tqdm.auto import tqdm

        shape, arr_dtype, ndim = self._get_array_info()

        # axis=None means reduce over all - load in chunks along axis 0
        if axis is None:
            # Reduce everything to a scalar
            n = shape[0]
            if func == "mean":
                total_sum = 0.0
                total_count = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    total_sum += np.sum(chunk, dtype=np.float64)
                    total_count += chunk.size
                result = total_sum / total_count
            elif func == "sum":
                result = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    result += np.sum(chunk, dtype=dtype)
            elif func == "max":
                result = None
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    chunk_max = np.max(chunk)
                    result = chunk_max if result is None else max(result, chunk_max)
            elif func == "min":
                result = None
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end])
                    chunk_min = np.min(chunk)
                    result = chunk_min if result is None else min(result, chunk_min)
            elif func in ("std", "var"):
                # Two-pass for numerical stability
                mean_val = self._chunked_reduce("mean", axis=None, chunk_size=chunk_size)
                variance_sum = 0.0
                total_count = 0
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[start:end]).astype(np.float64)
                    variance_sum += np.sum((chunk - mean_val) ** 2)
                    total_count += chunk.size
                ddof = kwargs.get("ddof", 0)
                variance = variance_sum / (total_count - ddof)
                result = np.sqrt(variance) if func == "std" else variance
            else:
                raise ValueError(f"Unknown reduction function: {func}")

            if dtype is not None:
                result = np.dtype(dtype).type(result)
            if keepdims:
                result = np.array(result).reshape((1,) * ndim)
            if out is not None:
                out[...] = result
                return out
            return result

        # Single axis reduction
        if isinstance(axis, int):
            ax = axis
            n = shape[ax]

            # Determine output shape
            out_shape = list(shape)
            out_shape.pop(ax)
            out_shape = tuple(out_shape)

            # Determine dtype
            if dtype is None:
                if func in ("mean", "std", "var"):
                    reduce_dtype = np.float64
                else:
                    reduce_dtype = arr_dtype
            else:
                reduce_dtype = dtype

            # Build slicing helper
            def make_slice(start, end):
                slices = [slice(None)] * ndim
                slices[ax] = slice(start, end)
                return tuple(slices)

            if func == "mean":
                accumulator = np.zeros(out_shape, dtype=np.float64)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[make_slice(start, end)])
                    accumulator += np.sum(chunk, axis=ax, dtype=np.float64)
                result = (accumulator / n).astype(reduce_dtype)

            elif func == "sum":
                accumulator = np.zeros(out_shape, dtype=reduce_dtype)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[make_slice(start, end)])
                    accumulator += np.sum(chunk, axis=ax)
                result = accumulator

            elif func == "max":
                chunk = np.asarray(self[make_slice(0, min(chunk_size, n))])
                accumulator = np.max(chunk, axis=ax)
                for start in tqdm(range(chunk_size, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[make_slice(start, end)])
                    accumulator = np.maximum(accumulator, np.max(chunk, axis=ax))
                result = accumulator.astype(reduce_dtype)

            elif func == "min":
                chunk = np.asarray(self[make_slice(0, min(chunk_size, n))])
                accumulator = np.min(chunk, axis=ax)
                for start in tqdm(range(chunk_size, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[make_slice(start, end)])
                    accumulator = np.minimum(accumulator, np.min(chunk, axis=ax))
                result = accumulator.astype(reduce_dtype)

            elif func in ("std", "var"):
                mean_val = self._chunked_reduce("mean", axis=ax, chunk_size=chunk_size)
                variance = np.zeros(out_shape, dtype=np.float64)
                for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                    end = min(start + chunk_size, n)
                    chunk = np.asarray(self[make_slice(start, end)]).astype(np.float64)
                    mean_expanded = np.expand_dims(mean_val, axis=ax)
                    variance += np.sum((chunk - mean_expanded) ** 2, axis=ax)
                ddof = kwargs.get("ddof", 0)
                variance = variance / (n - ddof)
                result = (np.sqrt(variance) if func == "std" else variance).astype(reduce_dtype)

            else:
                raise ValueError(f"Unknown reduction function: {func}")

            if keepdims:
                result = np.expand_dims(result, axis=ax)
            if out is not None:
                out[...] = result
                return out
            return result

        # Multi-axis reduction - reduce one at a time
        # Sort axes in descending order so indices don't shift
        axes = sorted(axis, reverse=True)
        result = self
        for ax in axes:
            result = result._chunked_reduce(func, axis=ax, dtype=dtype, **kwargs) if hasattr(result, "_chunked_reduce") else getattr(np, func)(result, axis=ax, dtype=dtype, **kwargs)
        if keepdims:
            for ax in sorted(axis):
                result = np.expand_dims(result, axis=ax)
        if out is not None:
            out[...] = result
            return out
        return result

    def mean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute mean along axis.

        Identical to numpy.mean() - axis=None reduces over all elements.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        dtype : np.dtype | None
            Output dtype. Defaults to float64.
        out : np.ndarray | None
            Output array.
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Mean value(s).
        """
        return self._reduce("mean", axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kwargs)

    def max(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute maximum along axis.

        Identical to numpy.max() - axis=None finds global maximum.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        out : np.ndarray | None
            Output array.
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Maximum value(s).
        """
        return self._reduce("max", axis=axis, out=out, keepdims=keepdims, **kwargs)

    def min(
        self,
        axis: int | tuple[int, ...] | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute minimum along axis.

        Identical to numpy.min() - axis=None finds global minimum.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        out : np.ndarray | None
            Output array.
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Minimum value(s).
        """
        return self._reduce("min", axis=axis, out=out, keepdims=keepdims, **kwargs)

    def std(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute standard deviation along axis.

        Identical to numpy.std() - axis=None computes over all elements.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        dtype : np.dtype | None
            Output dtype. Defaults to float64.
        out : np.ndarray | None
            Output array.
        ddof : int
            Delta degrees of freedom (divisor is N - ddof).
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Standard deviation.
        """
        return self._reduce("std", axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, **kwargs)

    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute variance along axis.

        Identical to numpy.var() - axis=None computes over all elements.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        dtype : np.dtype | None
            Output dtype. Defaults to float64.
        out : np.ndarray | None
            Output array.
        ddof : int
            Delta degrees of freedom (divisor is N - ddof).
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Variance.
        """
        return self._reduce("var", axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, **kwargs)

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        out: np.ndarray | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute sum along axis.

        Identical to numpy.sum() - axis=None sums all elements.

        Parameters
        ----------
        axis : int | tuple[int, ...] | None
            Axis or axes to reduce. None reduces all elements.
        dtype : np.dtype | None
            Output dtype.
        out : np.ndarray | None
            Output array.
        keepdims : bool
            Keep reduced dimensions as size 1.

        Returns
        -------
        np.ndarray or scalar
            Sum.
        """
        return self._reduce("sum", axis=axis, dtype=dtype, out=out, keepdims=keepdims, **kwargs)
