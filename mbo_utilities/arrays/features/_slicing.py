"""
unified slicing and chunk iteration for lazy arrays.

provides base functions for:
- parsing start:stop:step selections into index lists
- calculating optimal chunk sizes based on memory constraints
- iterating over non-spatial dimensions (T, Z, C, etc.)

all functions use 0-based indexing internally, with optional 1-based input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mbo_utilities.arrays.features._dim_tags import DIM_ALIASES

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

# spatial dims that we don't iterate over
SPATIAL_DIMS = {"Y", "X"}


def normalize_dim_key(dim_name: str) -> str:
    """
    normalize dimension name to canonical single-letter key.

    parameters
    ----------
    dim_name : str
        dimension name (e.g., "timepoints", "z-planes", "T", "Z")

    returns
    -------
    str
        canonical key (T, Z, C, V, R, B, A) or uppercase input
    """
    if len(dim_name) == 1:
        return dim_name.upper()
    return DIM_ALIASES.get(dim_name.lower(), dim_name.upper())


def parse_selection(
    selection: int | slice | list | Sequence[int] | str | None,
    dim_size: int,
    one_based: bool = True,
) -> list[int]:
    """
    parse a selection into a list of 0-based indices.

    parameters
    ----------
    selection : int | slice | list | Sequence[int] | str | None
        selection specification:
        - None: all indices
        - int: single index
        - slice: start:stop:step
        - str: "start:stop:step" or "start:stop" or "start" format
        - list/sequence: explicit indices
    dim_size : int
        size of the dimension
    one_based : bool
        if True, input indices are 1-based and will be converted to 0-based

    returns
    -------
    list[int]
        0-based indices

    examples
    --------
    >>> parse_selection("1:10", 100, one_based=True)  # frames 1-10 -> indices 0-9
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> parse_selection("1:10:2", 100, one_based=True)  # every other frame
    [0, 2, 4, 6, 8]
    >>> parse_selection([1, 5, 10], 100, one_based=True)  # specific frames
    [0, 4, 9]
    """
    if selection is None:
        return list(range(dim_size))

    # handle string format "start:stop:step"
    if isinstance(selection, str):
        parts = selection.split(":")
        if len(parts) == 1:
            # single index as string
            return parse_selection(int(parts[0]), dim_size, one_based)
        elif len(parts) == 2:
            # start:stop
            start = int(parts[0]) if parts[0] else (1 if one_based else 0)
            stop = int(parts[1]) if parts[1] else dim_size
            return parse_selection(slice(start, stop), dim_size, one_based)
        elif len(parts) == 3:
            # start:stop:step
            start = int(parts[0]) if parts[0] else (1 if one_based else 0)
            stop = int(parts[1]) if parts[1] else dim_size
            step = int(parts[2]) if parts[2] else 1
            return parse_selection(slice(start, stop, step), dim_size, one_based)
        else:
            raise ValueError(f"invalid selection string format: {selection}")

    if isinstance(selection, int):
        idx = selection - 1 if one_based else selection
        if idx < 0:
            idx = dim_size + idx + (1 if one_based else 0)
        return [max(0, min(idx, dim_size - 1))]

    if isinstance(selection, slice):
        start = selection.start or (1 if one_based else 0)
        stop = selection.stop or dim_size
        step = selection.step or 1

        if one_based:
            start = start - 1
            # stop is inclusive in 1-based, exclusive in 0-based
            # so stop stays the same when converting
        else:
            # 0-based slice, stop is already exclusive
            pass

        return list(range(max(0, start), min(stop, dim_size), step))

    # list/sequence of indices
    indices = list(selection)
    if one_based:
        indices = [i - 1 for i in indices]
    # filter to valid range instead of clamping (clamping causes duplicate frames)
    valid_indices = [i for i in indices if 0 <= i < dim_size]
    if len(valid_indices) < len(indices):
        # log warning about out-of-range indices being filtered
        import logging
        logging.getLogger("mbo_utilities").warning(
            f"Selection contains {len(indices) - len(valid_indices)} out-of-range indices "
            f"(max valid index: {dim_size - 1}). These will be skipped."
        )
    return valid_indices


@dataclass
class TimeSelection:
    """parsed timepoint selection with include/exclude ranges."""

    include_indices: list[int]  # 0-based indices to include
    exclude_indices: list[int]  # 0-based indices that were excluded
    include_str: str  # original include string (e.g., "1:100:1")
    exclude_str: str | None  # original exclude string (e.g., "50:60")

    @property
    def final_indices(self) -> list[int]:
        """indices after removing excluded."""
        exclude_set = set(self.exclude_indices)
        return [i for i in self.include_indices if i not in exclude_set]

    @property
    def count(self) -> int:
        """number of final indices."""
        return len(self.final_indices)

    def has_overlap(self) -> bool:
        """check if exclude overlaps with include (valid exclusion)."""
        if not self.exclude_indices:
            return False
        include_set = set(self.include_indices)
        return any(i in include_set for i in self.exclude_indices)

    def to_metadata(self) -> dict:
        """convert to metadata dict for saving."""
        result = {
            "include": self.include_str,
            "include_indices_0based": self.final_indices,
        }
        if self.exclude_str:
            result["exclude"] = self.exclude_str
            result["exclude_indices_0based"] = self.exclude_indices
        return result


def parse_timepoint_selection(
    selection_str: str,
    dim_size: int,
    one_based: bool = True,
) -> TimeSelection:
    """
    parse timepoint selection string with optional exclude range.

    format: "include_range" or "include_range,exclude_range"
    where each range is "start:stop:step" or "start:stop" (step defaults to 1)

    parameters
    ----------
    selection_str : str
        selection string, e.g., "1:100", "1:100:2", "1:100,50:60"
    dim_size : int
        total size of dimension
    one_based : bool
        if True, input indices are 1-based

    returns
    -------
    TimeSelection
        parsed selection with include/exclude indices

    examples
    --------
    >>> sel = parse_timepoint_selection("1:100", 1000)  # frames 1-100
    >>> sel = parse_timepoint_selection("1:100:2", 1000)  # every other frame
    >>> sel = parse_timepoint_selection("1:100,50:60", 1000)  # 1-100 excluding 50-60

    raises
    ------
    ValueError
        if exclude range doesn't overlap with include range (invalid exclusion)
    """
    selection_str = selection_str.strip()

    # split on comma for include,exclude
    parts = selection_str.split(",")
    include_str = parts[0].strip()
    exclude_str = parts[1].strip() if len(parts) > 1 else None

    # parse include range
    include_indices = parse_selection(include_str, dim_size, one_based=one_based)

    # parse exclude range if present
    exclude_indices = []
    if exclude_str:
        exclude_indices = parse_selection(exclude_str, dim_size, one_based=one_based)

    result = TimeSelection(
        include_indices=include_indices,
        exclude_indices=exclude_indices,
        include_str=include_str,
        exclude_str=exclude_str,
    )

    # validate: exclude must overlap with include
    if exclude_str and not result.has_overlap():
        raise ValueError(
            f"exclude range '{exclude_str}' has no overlap with include range '{include_str}'"
        )

    return result


@dataclass
class DimSelection:
    """selection state for a single dimension."""

    dim_key: str  # canonical key (T, Z, C, etc.)
    dim_index: int  # position in shape tuple
    dim_size: int  # total size of dimension
    indices: list[int]  # 0-based selected indices

    @property
    def count(self) -> int:
        """number of selected indices."""
        return len(self.indices)

    @property
    def is_full(self) -> bool:
        """True if all indices are selected."""
        return self.count == self.dim_size

    def to_slice_or_list(self):
        """
        convert to slice if contiguous with step=1, else return list.

        returns
        -------
        slice | list[int]
            slice for contiguous ranges, list otherwise
        """
        if not self.indices:
            return slice(0, 0)

        if len(self.indices) == 1:
            return self.indices[0]

        # check if contiguous with uniform step
        step = self.indices[1] - self.indices[0]
        if step <= 0:
            return self.indices

        is_uniform = all(
            self.indices[i + 1] - self.indices[i] == step
            for i in range(len(self.indices) - 1)
        )

        if is_uniform:
            return slice(self.indices[0], self.indices[-1] + 1, step if step > 1 else None)

        return self.indices


@dataclass
class ArraySlicing:
    """
    manages slicing state for a lazy array.

    tracks selections for all non-spatial dimensions and provides
    utilities for chunked iteration.
    """

    dims: tuple[str, ...]  # normalized dimension labels
    shape: tuple[int, ...]  # array shape
    selections: dict[str, DimSelection]  # dim_key -> selection
    dtype: np.dtype

    @classmethod
    def from_array(
        cls,
        arr,
        selections: dict[str, Any] | None = None,
        one_based: bool = True,
    ) -> "ArraySlicing":
        """
        create slicing state from array and optional selections.

        parameters
        ----------
        arr : array-like
            array with shape, dtype, and dims
        selections : dict | None
            mapping of dim key to selection (e.g., {"T": [1,3,5], "Z": slice(1,5)})
        one_based : bool
            if True, selections use 1-based indexing

        returns
        -------
        ArraySlicing
            slicing state
        """
        from mbo_utilities.arrays.features._dim_labels import get_dims

        raw_dims = get_dims(arr)
        dims = tuple(normalize_dim_key(d) for d in raw_dims)
        shape = arr.shape
        dtype = np.dtype(arr.dtype)

        dim_selections = {}
        for i, dim_key in enumerate(dims):
            if dim_key in SPATIAL_DIMS:
                continue

            sel = None
            if selections and dim_key in selections:
                sel = selections[dim_key]

            indices = parse_selection(sel, shape[i], one_based=one_based)
            dim_selections[dim_key] = DimSelection(
                dim_key=dim_key,
                dim_index=i,
                dim_size=shape[i],
                indices=indices,
            )

        return cls(dims=dims, shape=shape, selections=dim_selections, dtype=dtype)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """(Y, X) dimensions."""
        return self.shape[-2], self.shape[-1]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """shape after applying selections."""
        result = []
        for i, dim_key in enumerate(self.dims):
            if dim_key in SPATIAL_DIMS:
                result.append(self.shape[i])
            elif dim_key in self.selections:
                result.append(self.selections[dim_key].count)
            else:
                result.append(self.shape[i])
        return tuple(result)

    def bytes_per_frame(self) -> int:
        """bytes for one frame (all spatial dims, one index per non-spatial dim)."""
        Ly, Lx = self.spatial_shape
        n_planes = 1
        if "Z" in self.selections:
            n_planes = self.selections["Z"].count
        return n_planes * Ly * Lx * self.dtype.itemsize

    def calculate_chunk_size(self, target_mb: float = 50.0) -> int:
        """
        calculate number of frames per chunk for target memory.

        parameters
        ----------
        target_mb : float
            target chunk size in megabytes

        returns
        -------
        int
            number of frames (T dimension) per chunk
        """
        target_bytes = target_mb * 1024 * 1024
        bytes_per = self.bytes_per_frame()
        if bytes_per <= 0:
            return 1
        return max(1, int(target_bytes // bytes_per))

    def iter_chunks(
        self,
        chunk_dim: str = "T",
        chunk_size: int | None = None,
        target_mb: float = 50.0,
    ) -> Iterator["ChunkInfo"]:
        """
        iterate over chunks along a dimension.

        parameters
        ----------
        chunk_dim : str
            dimension to chunk along (default "T")
        chunk_size : int | None
            explicit chunk size, or None to auto-calculate
        target_mb : float
            target chunk size in MB (used if chunk_size is None)

        yields
        ------
        ChunkInfo
            chunk metadata with indices for each dimension
        """
        chunk_dim = normalize_dim_key(chunk_dim)

        if chunk_dim not in self.selections:
            # dimension not in array, yield single chunk with all selections
            yield ChunkInfo(
                chunk_index=0,
                total_chunks=1,
                selections={k: v.indices for k, v in self.selections.items()},
            )
            return

        sel = self.selections[chunk_dim]
        indices = sel.indices

        if chunk_size is None:
            chunk_size = self.calculate_chunk_size(target_mb)

        total_chunks = (len(indices) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(indices))
            chunk_indices = indices[start:end]

            # build selections dict for this chunk
            chunk_selections = {}
            for k, v in self.selections.items():
                if k == chunk_dim:
                    chunk_selections[k] = chunk_indices
                else:
                    chunk_selections[k] = v.indices

            yield ChunkInfo(
                chunk_index=chunk_idx,
                total_chunks=total_chunks,
                selections=chunk_selections,
            )


@dataclass
class ChunkInfo:
    """metadata for a single chunk during iteration."""

    chunk_index: int
    total_chunks: int
    selections: dict[str, list[int]]  # dim_key -> 0-based indices for this chunk

    @property
    def progress(self) -> float:
        """progress fraction (0.0 to 1.0)."""
        if self.total_chunks <= 0:
            return 1.0
        return (self.chunk_index + 1) / self.total_chunks


def read_chunk(
    arr,
    chunk_info: ChunkInfo,
    dims: tuple[str, ...],
) -> np.ndarray:
    """
    read a chunk from a lazy array.

    reads frame-by-frame to avoid np.ix_ which lazy arrays don't support.
    handles TZYX, TYX, ZYX dimension orderings.

    parameters
    ----------
    arr : array-like
        lazy array supporting indexing
    chunk_info : ChunkInfo
        chunk metadata with selections (0-based indices)
    dims : tuple[str, ...]
        normalized dimension labels (e.g., ("T", "Z", "Y", "X"))

    returns
    -------
    np.ndarray
        chunk data with shape (T_chunk, Z_sel, Y, X) or (T_chunk, Y, X)
    """
    t_indices = chunk_info.selections.get("T")
    z_indices = chunk_info.selections.get("Z")

    ndim = len(dims)

    # handle ZYX (no T dimension)
    if "T" not in dims:
        if z_indices is not None:
            # ZYX with z selection
            if ndim == 3 and dims[0] == "Z":
                frames = []
                for z_idx in z_indices:
                    frame = arr[z_idx, :, :]
                    frames.append(np.asarray(frame))
                result = np.stack(frames, axis=0)
                # add T=1 dimension at front
                return result[np.newaxis, :, :, :]
            else:
                # unknown ordering, try direct slice
                result = np.asarray(arr[z_indices, :, :])
                return result[np.newaxis, :, :, :]
        else:
            # ZYX full selection
            result = np.asarray(arr[:, :, :])
            return result[np.newaxis, :, :, :]

    # handle TYX and TZYX
    if t_indices is None:
        t_indices = [0]

    frames = []
    for t_idx in t_indices:
        if ndim == 4:
            # TZYX
            if z_indices is not None:
                # select specific z planes
                z_frames = []
                for z_idx in z_indices:
                    z_frame = arr[t_idx, z_idx, :, :]
                    z_frames.append(np.asarray(z_frame))
                frame = np.stack(z_frames, axis=0)  # (Z_sel, Y, X)
            else:
                # all z planes
                frame = np.asarray(arr[t_idx, :, :, :])  # (Z, Y, X)
        elif ndim == 3:
            if dims[0] == "T":
                # TYX
                frame = np.asarray(arr[t_idx, :, :])  # (Y, X)
            else:
                # unknown 3D ordering
                frame = np.asarray(arr[t_idx])
        else:
            # 2D or other
            frame = np.asarray(arr[t_idx])

        frames.append(frame)

    # stack along T axis
    result = np.stack(frames, axis=0)

    # ensure output is at least 4D (T, Z, Y, X) for consistency
    if result.ndim == 3:
        # (T, Y, X) -> (T, 1, Y, X)
        result = result[:, np.newaxis, :, :]

    return result
