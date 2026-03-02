"""
Chunk size feature for arrays.

Provides chunking configuration for lazy/dask arrays.
"""

from __future__ import annotations


import numpy as np

from mbo_utilities.arrays.features._base import ArrayFeature, ArrayFeatureEvent


# default chunk patterns
CHUNKS_2D = (-1, -1)  # full spatial
CHUNKS_3D = (1, -1, -1)  # single frame, full spatial
CHUNKS_4D = (1, 1, -1, -1)  # single frame/plane, full spatial


def normalize_chunks(
    chunks: tuple | dict | None,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Normalize chunk specification to actual sizes.

    Parameters
    ----------
    chunks : tuple | dict | None
        chunk specification:
        - None: use defaults based on ndim
        - tuple: chunk sizes (-1 means full dimension, "auto" means auto-tune)
        - dict: {axis: size} mapping
    shape : tuple[int, ...]
        array shape

    Returns
    -------
    tuple[int, ...]
        normalized chunk sizes
    """
    ndim = len(shape)

    if chunks is None:
        if ndim == 2:
            chunks = CHUNKS_2D
        elif ndim == 3:
            chunks = CHUNKS_3D
        elif ndim == 4:
            chunks = CHUNKS_4D
        else:
            chunks = tuple(1 if i == 0 else -1 for i in range(ndim))

    if isinstance(chunks, dict):
        result = list(shape)
        for axis, size in chunks.items():
            if size == -1:
                result[axis] = shape[axis]
            elif size == "auto":
                result[axis] = min(shape[axis], 256)  # reasonable default
            else:
                result[axis] = min(size, shape[axis])
        return tuple(result)

    result = []
    for i, c in enumerate(chunks):
        if c == -1:
            result.append(shape[i])
        elif c == "auto":
            result.append(min(shape[i], 256))
        else:
            result.append(min(c, shape[i]))
    return tuple(result)


def estimate_chunk_memory(chunks: tuple[int, ...], dtype: np.dtype) -> int:
    """
    Estimate memory for a single chunk in bytes.

    Parameters
    ----------
    chunks : tuple[int, ...]
        chunk sizes
    dtype : np.dtype
        data type

    Returns
    -------
    int
        bytes per chunk
    """
    return int(np.prod(chunks) * dtype.itemsize)


class ChunkSizeFeature(ArrayFeature):
    """
    Chunk size feature for arrays.

    Manages chunking configuration for lazy arrays.

    Parameters
    ----------
    chunks : tuple | dict | None
        chunk specification
    shape : tuple[int, ...]
        array shape for normalization

    Examples
    --------
    >>> cs = ChunkSizeFeature((1, 1, 512, 512), shape=(1000, 10, 512, 512))
    >>> cs.value
    (1, 1, 512, 512)
    >>> cs.memory_per_chunk(np.dtype('int16'))
    524288
    """

    def __init__(
        self,
        chunks: tuple | dict | None = None,
        shape: tuple[int, ...] = (),
        property_name: str = "chunks",
    ):
        super().__init__(property_name=property_name)
        self._shape = shape
        if shape:
            self._chunks = normalize_chunks(chunks, shape)
        else:
            self._chunks = chunks if chunks else ()

    @property
    def value(self) -> tuple[int, ...]:
        """Chunk sizes."""
        return self._chunks

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._chunks)

    def set_value(self, array, value: tuple | dict | None) -> None:
        """
        Set chunk sizes.

        Parameters
        ----------
        array : array-like
            the array this feature belongs to
        value : tuple | dict | None
            new chunk specification
        """
        old_value = self._chunks

        if hasattr(array, "shape"):
            self._shape = array.shape
        self._chunks = normalize_chunks(value, self._shape)

        if old_value != self._chunks:
            event = ArrayFeatureEvent(
                type=self._property_name,
                info={"value": self._chunks, "old_value": old_value},
            )
            self._call_event_handlers(event)

    def memory_per_chunk(self, dtype: np.dtype) -> int:
        """
        Estimate memory for a single chunk.

        Parameters
        ----------
        dtype : np.dtype
            data type

        Returns
        -------
        int
            bytes per chunk
        """
        return estimate_chunk_memory(self._chunks, np.dtype(dtype))

    def memory_per_chunk_mb(self, dtype: np.dtype) -> float:
        """
        Estimate memory for a single chunk in MB.

        Parameters
        ----------
        dtype : np.dtype
            data type

        Returns
        -------
        float
            MB per chunk
        """
        return self.memory_per_chunk(dtype) / (1024 * 1024)

    def num_chunks(self) -> int:
        """
        Calculate total number of chunks.

        Returns
        -------
        int
            number of chunks
        """
        if not self._shape or not self._chunks:
            return 0
        n = 1
        for s, c in zip(self._shape, self._chunks, strict=False):
            n *= (s + c - 1) // c
        return n

    def optimize_for_memory(
        self,
        target_mb: float = 50.0,
        dtype: np.dtype = np.dtype("int16"),
    ) -> tuple[int, ...]:
        """
        Suggest chunk sizes for target memory budget.

        Parameters
        ----------
        target_mb : float
            target chunk size in MB
        dtype : np.dtype
            data type

        Returns
        -------
        tuple[int, ...]
            suggested chunk sizes
        """
        if not self._shape:
            return ()

        target_bytes = target_mb * 1024 * 1024
        itemsize = np.dtype(dtype).itemsize

        # start with full spatial, single frame/plane
        ndim = len(self._shape)
        chunks = list(self._shape)

        # always chunk time dimension to 1
        if ndim >= 3:
            chunks[0] = 1

        # chunk z dimension to 1 for 4D
        if ndim >= 4:
            chunks[1] = 1

        # check if within budget
        current_bytes = np.prod(chunks) * itemsize
        if current_bytes <= target_bytes:
            return tuple(chunks)

        # reduce spatial dims if needed
        while current_bytes > target_bytes:
            # find largest spatial dim
            spatial_start = min(2, ndim)
            largest_idx = spatial_start
            for i in range(spatial_start, ndim):
                if chunks[i] > chunks[largest_idx]:
                    largest_idx = i

            # halve it
            chunks[largest_idx] = max(1, chunks[largest_idx] // 2)
            current_bytes = np.prod(chunks) * itemsize

        return tuple(chunks)

    def to_dask_chunks(self) -> dict[int, int | str]:
        """
        Convert to dask chunks dict format.

        Returns
        -------
        dict
            {axis: size} mapping
        """
        return dict(enumerate(self._chunks))

    def __repr__(self) -> str:
        if not self._chunks:
            return "ChunkSizeFeature(None)"
        return f"ChunkSizeFeature{self._chunks}"
