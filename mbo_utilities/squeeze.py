"""
Opt-in squeezed view of a 5D LazyArray.

v4 arrays are always 5D TCZYX. `SqueezedView` presents a size-1-dims-dropped
view for notebook/display ergonomics (`arr.squeeze()` or
`imread(path, squeeze=True)`). Indexing on the view is translated back to the
canonical 5D index on the underlying array; the base stays 5D for writers and
the viewer pipeline.
"""

from __future__ import annotations

import numpy as np


def _expand_squeezed_key(key, kept):
    """Map a squeezed-space `key` to a full 5D TCZYX key.

    Dropped (size-1) axes are filled with 0; kept axes default to full
    slices and receive the user's index in squeezed order.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if Ellipsis in key:
        idx = key.index(Ellipsis)
        n_missing = len(kept) - (len(key) - 1)
        key = key[:idx] + (slice(None),) * max(n_missing, 0) + key[idx + 1:]
    full = [0] * 5
    for i in kept:
        full[i] = slice(None)
    for slot, k in zip(kept, key):
        full[slot] = k
    return tuple(full)


class SqueezedView:
    """Lightweight view of a 5D LazyArray with size-1 dims dropped.

    Indexing translates back to the canonical 5D index on the underlying
    array, so integer-indexed and dropped axes squeeze consistently.
    """

    def __init__(self, base):
        self._base = base
        self._kept = tuple(i for i, s in enumerate(base.shape) if s > 1)

    @property
    def base(self):
        """The underlying 5D LazyArray."""
        return self._base

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._base.shape[i] for i in self._kept)

    @property
    def ndim(self) -> int:
        return len(self._kept)

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def metadata(self):
        """Metadata of the underlying 5D array (describes the canonical source)."""
        return getattr(self._base, "metadata", {})

    @metadata.setter
    def metadata(self, value):
        self._base.metadata = value

    @property
    def dims(self) -> tuple[str, ...] | None:
        base_dims = getattr(self._base, "dims", None)
        if base_dims:
            return tuple(d for i, d in enumerate(base_dims) if i in self._kept)
        return None

    def __len__(self) -> int:
        return self.shape[0] if self._kept else 1

    def __getitem__(self, key):
        return self._base[_expand_squeezed_key(key, self._kept)]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._base, dtype=dtype)

    def _imwrite(self, outpath, **kwargs):
        # writers operate on the canonical 5D array, not the squeezed view
        return self._base._imwrite(outpath, **kwargs)

    def __repr__(self) -> str:
        return f"SqueezedView(shape={self.shape}, dtype={self.dtype}, base={type(self._base).__name__})"
