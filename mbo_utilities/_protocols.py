from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

ARRAY_LIKE_ATTRS = ["shape", "ndim", "__getitem__"]

# re-export from features module for backwards compatibility
from mbo_utilities.arrays.features._dim_labels import (
    get_dims,
    get_num_planes,
    infer_dims,
)


@runtime_checkable
class ArrayProtocol(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key): ...


@runtime_checkable
class LazyArrayProtocol(Protocol):
    """
    Protocol for lazy array types used in mbo_utilities.

    All array types (MboRawArray, Suite2pArray, ZarrArray, etc.) should implement
    this protocol to ensure compatibility with imread/imwrite and downstream
    processing pipelines like Suite2p.

    Required Properties
    -------------------
    shape : tuple[int, ...]
        Array dimensions. For imaging data, typically (T, Y, X) or (T, Z, Y, X).
    dtype : numpy.dtype
        Data type of array elements.
    ndim : int
        Number of dimensions.
    metadata : dict
        Metadata dictionary. Must include 'nframes' for Suite2p compatibility.
        Use 'num_frames' as an alias for backwards compatibility with old datasets.
    filenames : list[Path]
        Source file path(s) for the array data.
    min : float
        Minimum value in the array (used for display scaling).
    max : float
        Maximum value in the array (used for display scaling).

    Required Methods
    ----------------
    __getitem__(key) -> np.ndarray
        Indexing support. Should return numpy arrays for slices.
    __len__() -> int
        Number of frames (first dimension).

    Optional Properties
    -------------------
    dims : tuple[str, ...] | None
        Dimension labels, e.g., ('T', 'Z', 'Y', 'X') or ('T', 'Y', 'X').
        If None, dimensions are inferred from ndim:
        - 2D: ('Y', 'X')
        - 3D: ('T', 'Y', 'X')
        - 4D: ('T', 'Z', 'Y', 'X')
        Z and C (channel) are treated equivalently as the "plane" dimension.
    num_planes : int | None
        Number of Z-planes. If None, inferred from shape at Z dimension.
        Alias: num_channels (for backwards compatibility with ScanImage terminology).

    Optional Methods
    ----------------
    __array__() -> np.ndarray
        Convert to numpy array. May load subset for large arrays.
    _imwrite(outpath, **kwargs)
        Write array to disk. Called by mbo.imwrite().
    imshow(**kwargs)
        Display array using fastplotlib ImageWidget.
    close()
        Release resources (file handles, memmaps).

    Metadata Keys
    -------------
    The following metadata keys are used by Suite2p and should be populated:

    - nframes : int (required) - Number of frames/timepoints
    - num_frames : int (alias) - Backwards compatibility alias for nframes
    - Ly : int - Height in pixels
    - Lx : int - Width in pixels
    - fs : float - Frame rate in Hz
    - plane : int - Z-plane index (1-based)

    Example
    -------
    >>> class MyArray:
    ...     def __init__(self, path):
    ...         self._data = np.load(path, mmap_mode='r')
    ...         self._path = Path(path)
    ...
    ...     @property
    ...     def shape(self) -> tuple[int, ...]:
    ...         return self._data.shape
    ...
    ...     @property
    ...     def dtype(self):
    ...         return self._data.dtype
    ...
    ...     @property
    ...     def ndim(self) -> int:
    ...         return self._data.ndim
    ...
    ...     @property
    ...     def metadata(self) -> dict:
    ...         return {"nframes": self.shape[0], "num_frames": self.shape[0]}
    ...
    ...     @property
    ...     def filenames(self) -> list[Path]:
    ...         return [self._path]
    ...
    ...     @property
    ...     def min(self) -> float:
    ...         return float(self._data[0].min())
    ...
    ...     @property
    ...     def max(self) -> float:
    ...         return float(self._data[0].max())
    ...
    ...     def __getitem__(self, key):
    ...         return self._data[key]
    ...
    ...     def __len__(self) -> int:
    ...         return self.shape[0]
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> "DTypeLike": ...

    @property
    def ndim(self) -> int: ...

    @property
    def metadata(self) -> dict: ...

    @property
    def filenames(self) -> list[Path]: ...

    @property
    def min(self) -> float: ...

    @property
    def max(self) -> float: ...

    def __getitem__(self, key: int | slice | tuple) -> np.ndarray: ...

    def __len__(self) -> int: ...
