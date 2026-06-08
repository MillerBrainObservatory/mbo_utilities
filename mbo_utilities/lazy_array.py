"""
v4 pluggable LazyArray base class and imread dispatch registry.

Dependency-light by design: imports only the standard library so it can be
imported early (e.g. for ``isinstance(obj, LazyArray)`` checks in downstream
packages such as lbm_suite2p_python) without pulling in numpy/tifffile/zarr.

Phase 1 of the v4 rollout: this base carries the always-5D size accessors
(folded in from the former ``Shape5DMixin``) plus the registration and
dispatch plumbing. Per-class ``.shape``/``.ndim`` are locked to 5D and
``imread()`` is switched to ``_dispatch()`` in later phases; nothing here
changes existing behavior yet.
"""

from __future__ import annotations

from os.path import commonpath
from pathlib import Path
from typing import ClassVar


class LazyArray:
    """Base class for every array ``imread()`` can return.

    Subclasses implement ``_shape5d()`` returning a 5-tuple (T, C, Z, Y, X)
    with singletons for unused dims, plus ``__getitem__``, ``dtype``,
    ``metadata`` and ``can_open()``. This base provides the named 5D size
    accessors and the registry hooks used by ``imread()`` dispatch.
    """

    PRIORITY: ClassVar[int] = 50

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        """return the 5D TCZYX shape. subclasses must implement this."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """shape as 5D TCZYX. subclasses may override (e.g. squeezed views)."""
        return self._shape5d()

    @property
    def ndim(self) -> int:
        """always 5 for a canonical LazyArray."""
        return 5

    @property
    def nt(self) -> int:
        """number of timepoints."""
        return self._shape5d()[0]

    @property
    def nc(self) -> int:
        """number of channels."""
        return self._shape5d()[1]

    @property
    def nz(self) -> int:
        """number of z-planes."""
        return self._shape5d()[2]

    @property
    def ny(self) -> int:
        """spatial height."""
        return self._shape5d()[3]

    @property
    def nx(self) -> int:
        """spatial width."""
        return self._shape5d()[4]

    @property
    def source_path(self) -> Path | None:
        """canonical path `imread()` uses to reconstruct this array.

        default implementation derives it from `self.filenames` (list or
        single path) or falls back to `self.path`. subclasses whose files
        span per-plane subdirectories (e.g. volumetric Suite2p) must
        override — a file path like `.../plane01/data.bin` is not
        equivalent to its parent directory once passed through imread.
        """
        filenames = getattr(self, "filenames", None)
        if filenames is None:
            # fallback for arrays that use a different attribute name
            # (NumpyArray → `.path`, IsoView* → `.base_path`).
            path = getattr(self, "path", None) or getattr(self, "base_path", None)
            return Path(path) if path else None
        if isinstance(filenames, (str, Path)):
            return Path(filenames)
        paths = [str(p) for p in filenames]
        if not paths:
            return None
        if len(paths) == 1:
            return Path(paths[0])
        try:
            return Path(commonpath(paths))
        except ValueError:
            return Path(paths[0]).parent

    @classmethod
    def can_open(cls, path: Path) -> bool:
        """return True if this class can open `path`. default: no."""
        return False

    def squeeze(self):
        """Return a view with size-1 T/C/Z axes dropped (opt-in ergonomics)."""
        from mbo_utilities.squeeze import SqueezedView
        return SqueezedView(self)


_ENTRY_POINT_GROUP = "mbo_utilities.lazy_arrays"
_REGISTRY: list[type[LazyArray]] = []
_ENTRY_POINTS_LOADED = False


def register_array_class(cls: type[LazyArray], priority: int | None = None) -> None:
    """Register a LazyArray subclass for imread() dispatch.

    If `priority` is given, it overrides `cls.PRIORITY` for this class.
    Idempotent: re-registering the same class is a no-op.
    """
    if not (isinstance(cls, type) and issubclass(cls, LazyArray)):
        raise TypeError(f"{cls!r} is not a LazyArray subclass")
    if priority is not None:
        cls.PRIORITY = int(priority)
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)


def _load_entry_point_arrays() -> None:
    """Load classes from the 'mbo_utilities.lazy_arrays' entry-point group.

    Cached after the first call.
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except TypeError:
        # importlib.metadata < 3.10 has no group= selector
        eps = entry_points().get(_ENTRY_POINT_GROUP, [])
    for ep in eps:
        try:
            cls = ep.load()
        except Exception:
            continue
        if isinstance(cls, type) and issubclass(cls, LazyArray):
            register_array_class(cls)


def _dispatch(path) -> type[LazyArray] | None:
    """Return the registered class with highest PRIORITY whose can_open() wins."""
    _load_entry_point_arrays()
    p = Path(path)
    for cls in sorted(_REGISTRY, key=lambda c: c.PRIORITY, reverse=True):
        try:
            if cls.can_open(p):
                return cls
        except Exception:
            continue
    return None
