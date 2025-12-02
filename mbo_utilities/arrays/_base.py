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

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from dask import array as da

from mbo_utilities import log
from mbo_utilities._protocols import get_dims, get_num_planes
from mbo_utilities._writers import _write_plane

if TYPE_CHECKING:
    from typing import Callable

logger = log.get("arrays._base")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}


def supports_roi(obj):
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
    """Yield ROI indices based on MBO semantics.

    - roi=None -> yield None (stitched full-FOV image)
    - roi=0 -> yield each ROI index from 1..num_rois (split all)
    - roi=int > 0 -> yield that ROI only
    - roi=list/tuple -> yield each element (as given)
    """
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


def _build_output_path(
    outpath: Path,
    plane_idx: int,
    roi: int | None,
    ext: str,
    output_name: str | None = None,
    structural: bool = False,
    has_multiple_rois: bool = False,
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
        If True and roi is None, use "_stitched" suffix.

    Returns
    -------
    Path
        Full output file path.
    """
    plane_num = plane_idx + 1  # Convert to 1-based for filenames

    # Determine suffix based on ROI
    if roi is None:
        roi_suffix = "_stitched" if has_multiple_rois else ""
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
    else:
        # Non-binary formats: single file per plane
        return outpath / f"plane{plane_num:02d}{roi_suffix}.{ext}"


def _imwrite_base(
    arr,
    outpath: Path | str,
    planes: int | list | tuple | None = None,
    ext: str = ".tiff",
    overwrite: bool = False,
    target_chunk_mb: int = 50,
    progress_callback: "Callable | None" = None,
    debug: bool = False,
    roi_iterator=None,
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

    # Get dimensions using protocol helpers
    dims = get_dims(arr)
    num_planes = get_num_planes(arr)

    # Extract shape info
    nframes = arr.shape[0] if "T" in dims else 1
    Ly, Lx = arr.shape[-2], arr.shape[-1]

    # Update metadata
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["nframes"] = nframes
    md["num_frames"] = nframes  # alias for backwards compatibility

    # Normalize planes to 0-indexed list
    planes_list = _normalize_planes(planes, num_planes)

    # Use provided ROI iterator or default
    roi_iter = roi_iterator if roi_iterator is not None else iter_rois(arr)

    # Check if array has multiple ROIs (for "_stitched" suffix)
    has_multiple_rois = getattr(arr, "num_rois", 1) > 1

    for roi in roi_iter:
        # Update array's ROI if it supports it
        if roi is not None and hasattr(arr, "roi"):
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
                dshape=(nframes, Ly, Lx),
                plane_index=plane_idx,
                **kwargs,
            )

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
    elif arr_ndim == 3:
        return "ZYX"
    elif arr_ndim == 4:
        return "TZYX"
    else:
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
    Mixin providing numpy-like reduction methods for lazy arrays.

    Adds mean, max, min, std, sum methods that work efficiently with lazy arrays
    by processing data in chunks. For 4D arrays (T, Z, Y, X), reduces over time
    to produce 3D volumes (Z, Y, X) by default.

    Usage
    -----
    class MyArray(ReductionMixin):
        # ... array implementation ...
        pass

    arr = MyArray(path)
    volume = arr.mean(axis=0)  # Mean over time -> (Z, Y, X)
    """

    def _chunked_reduce(
        self,
        func: str,
        axis: int | None = None,
        chunk_size: int = 100,
        dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """
        Apply reduction function over axis, processing in chunks.

        Parameters
        ----------
        func : str
            Reduction function name: 'mean', 'max', 'min', 'std', 'sum'
        axis : int | None
            Axis to reduce over. If None and ndim > 2, defaults to 0 (time).
            For 2D arrays, reduces over entire array.
        chunk_size : int
            Number of frames to process at once along the reduction axis.
        dtype : np.dtype | None
            Output dtype. If None, uses float64 for mean/std, input dtype otherwise.

        Returns
        -------
        np.ndarray
            Reduced array.
        """
        from tqdm.auto import tqdm

        # Default axis to 0 for time series data
        if axis is None:
            if self.ndim > 2:
                axis = 0
            else:
                # For 2D, reduce over everything
                return getattr(np.asarray(self), func)()

        # Validate axis
        if axis < 0:
            axis = self.ndim + axis
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis {axis} out of bounds for {self.ndim}D array")

        n = self.shape[axis]

        # Determine output shape (remove the reduction axis)
        out_shape = list(self.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

        # Determine dtype
        if dtype is None:
            if func in ('mean', 'std'):
                dtype = np.float64
            else:
                dtype = self.dtype

        # For small arrays, just compute directly
        if n <= chunk_size * 2:
            data = np.asarray(self)
            return getattr(np, func)(data, axis=axis).astype(dtype)

        # Chunked reduction
        if func == 'mean':
            # Running mean: accumulate sum and count
            accumulator = np.zeros(out_shape, dtype=np.float64)
            for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                end = min(start + chunk_size, n)
                # Build slice for the reduction axis
                slices = [slice(None)] * self.ndim
                slices[axis] = slice(start, end)
                chunk = np.asarray(self[tuple(slices)])
                accumulator += np.sum(chunk, axis=axis, dtype=np.float64)
            return (accumulator / n).astype(dtype)

        elif func == 'sum':
            accumulator = np.zeros(out_shape, dtype=dtype)
            for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                end = min(start + chunk_size, n)
                slices = [slice(None)] * self.ndim
                slices[axis] = slice(start, end)
                chunk = np.asarray(self[tuple(slices)])
                accumulator += np.sum(chunk, axis=axis)
            return accumulator

        elif func == 'max':
            # Initialize with first chunk
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(0, min(chunk_size, n))
            accumulator = np.max(np.asarray(self[tuple(slices)]), axis=axis)
            for start in tqdm(range(chunk_size, n, chunk_size), desc=f"Computing {func}"):
                end = min(start + chunk_size, n)
                slices[axis] = slice(start, end)
                chunk = np.asarray(self[tuple(slices)])
                accumulator = np.maximum(accumulator, np.max(chunk, axis=axis))
            return accumulator.astype(dtype)

        elif func == 'min':
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(0, min(chunk_size, n))
            accumulator = np.min(np.asarray(self[tuple(slices)]), axis=axis)
            for start in tqdm(range(chunk_size, n, chunk_size), desc=f"Computing {func}"):
                end = min(start + chunk_size, n)
                slices[axis] = slice(start, end)
                chunk = np.asarray(self[tuple(slices)])
                accumulator = np.minimum(accumulator, np.min(chunk, axis=axis))
            return accumulator.astype(dtype)

        elif func == 'std':
            # Two-pass: first compute mean, then variance
            mean_val = self._chunked_reduce('mean', axis=axis, chunk_size=chunk_size)
            variance = np.zeros(out_shape, dtype=np.float64)
            for start in tqdm(range(0, n, chunk_size), desc=f"Computing {func}"):
                end = min(start + chunk_size, n)
                slices = [slice(None)] * self.ndim
                slices[axis] = slice(start, end)
                chunk = np.asarray(self[tuple(slices)]).astype(np.float64)
                # Expand mean for broadcasting
                mean_expanded = np.expand_dims(mean_val, axis=axis)
                variance += np.sum((chunk - mean_expanded) ** 2, axis=axis)
            return np.sqrt(variance / n).astype(dtype)

        else:
            raise ValueError(f"Unknown reduction function: {func}")

    def mean(self, axis: int | None = None, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Compute mean along axis.

        For 4D arrays (T, Z, Y, X), defaults to axis=0 to produce a 3D volume.

        Parameters
        ----------
        axis : int | None
            Axis to reduce. Defaults to 0 (time) for 3D+ arrays.
        dtype : np.dtype | None
            Output dtype. Defaults to float64.

        Returns
        -------
        np.ndarray
            Mean projection.
        """
        return self._chunked_reduce('mean', axis=axis, dtype=dtype)

    def max(self, axis: int | None = None, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Compute maximum along axis.

        For 4D arrays (T, Z, Y, X), defaults to axis=0 to produce a 3D volume.

        Parameters
        ----------
        axis : int | None
            Axis to reduce. Defaults to 0 (time) for 3D+ arrays.
        dtype : np.dtype | None
            Output dtype. Defaults to input dtype.

        Returns
        -------
        np.ndarray
            Max projection.
        """
        return self._chunked_reduce('max', axis=axis, dtype=dtype)

    def min(self, axis: int | None = None, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Compute minimum along axis.

        For 4D arrays (T, Z, Y, X), defaults to axis=0 to produce a 3D volume.

        Parameters
        ----------
        axis : int | None
            Axis to reduce. Defaults to 0 (time) for 3D+ arrays.
        dtype : np.dtype | None
            Output dtype. Defaults to input dtype.

        Returns
        -------
        np.ndarray
            Min projection.
        """
        return self._chunked_reduce('min', axis=axis, dtype=dtype)

    def std(self, axis: int | None = None, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Compute standard deviation along axis.

        For 4D arrays (T, Z, Y, X), defaults to axis=0 to produce a 3D volume.

        Parameters
        ----------
        axis : int | None
            Axis to reduce. Defaults to 0 (time) for 3D+ arrays.
        dtype : np.dtype | None
            Output dtype. Defaults to float64.

        Returns
        -------
        np.ndarray
            Standard deviation.
        """
        return self._chunked_reduce('std', axis=axis, dtype=dtype)

    def sum(self, axis: int | None = None, dtype: np.dtype | None = None) -> np.ndarray:
        """
        Compute sum along axis.

        For 4D arrays (T, Z, Y, X), defaults to axis=0 to produce a 3D volume.

        Parameters
        ----------
        axis : int | None
            Axis to reduce. Defaults to 0 (time) for 3D+ arrays.
        dtype : np.dtype | None
            Output dtype. Defaults to input dtype.

        Returns
        -------
        np.ndarray
            Sum.
        """
        return self._chunked_reduce('sum', axis=axis, dtype=dtype)
