"""
Chunk size feature for arrays.

Provides chunking configuration for lazy/dask arrays.
"""

from __future__ import annotations


import numpy as np


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
