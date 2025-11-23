from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

ARRAY_LIKE_ATTRS = ["shape", "ndim", "__getitem__"]


@runtime_checkable
class ArrayProtocol(Protocol):
    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key): ...


class LazyArrayProtocol:
    """
    Protocol for lazy array types.

    Must implement:
    - __getitem__    (method)
    - __len__        (method)
    - min            (property)
    - max            (property)
    - ndim           (property)
    - shape          (property)
    - dtype          (property)
    - metadata       (property)

    Optionally implement:
    - __array__      (method)
    - imshow         (method)
    - _imwrite       (method)
    - close          (method)
    - chunks         (property)
    - dask           (property)
    """

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __array__(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def min(self) -> float:
        raise NotImplementedError

    @property
    def max(self) -> float:
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError
