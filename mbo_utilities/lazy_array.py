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

# canonical dims by reported rank (OME-NGFF 0.5: time -> channel -> space)
_DEFAULT_DIMS_BY_NDIM: dict[int, tuple[str, ...]] = {
    2: ("Y", "X"),
    3: ("T", "Y", "X"),
    4: ("T", "Z", "Y", "X"),
    5: ("T", "C", "Z", "Y", "X"),
}


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

    _metadata: dict | None = None
    _declared_dims: tuple[str, ...] | None = None
    _dimension_specs = None

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension labels; declared order if set, else canonical by rank."""
        if self._declared_dims is not None:
            return self._declared_dims
        return _DEFAULT_DIMS_BY_NDIM.get(self.ndim, ("T", "C", "Z", "Y", "X"))

    @dims.setter
    def dims(self, value) -> None:
        from mbo_utilities import log
        from mbo_utilities.arrays.features._dim_labels import parse_dims

        # never raise on a rank mismatch: warn and chain-guess by rank
        self._declared_dims = parse_dims(value, self.ndim, strict=False)
        self.invalidate_dimension_specs()
        log.get().info("dims %s  shape %s", "".join(self._declared_dims), self.shape)

    @property
    def metadata(self) -> dict:
        """Metadata dict (never None). Subclasses may override the getter."""
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        # `dims` is kept as plain metadata here; reported dims come from the
        # array's rank. Classes that re-permute on dims (NumpyArray) override
        # this setter. A stored `dims` of a different length must not raise.
        self._metadata = dict(value)

    @property
    def dimension_specs(self):
        """Reactive DimensionSpecs built from dims/shape/metadata."""
        if self._dimension_specs is None:
            from mbo_utilities.arrays.features._dim_spec import DimensionSpecs

            self._dimension_specs = DimensionSpecs.from_array(self)
        return self._dimension_specs

    def invalidate_dimension_specs(self) -> None:
        self._dimension_specs = None

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        return self.dimension_specs.spatial_dims

    @property
    def iteratable_dims(self) -> tuple[str, ...]:
        return self.dimension_specs.iteratable_dims

    @property
    def batch_dims(self) -> tuple[str, ...]:
        return self.dimension_specs.batch_dims

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints (T), name-keyed via dimension_specs."""
        return self.dimension_specs.num_timepoints

    @property
    def num_zplanes(self) -> int:
        """Number of z-planes (Z), name-keyed via dimension_specs."""
        return self.dimension_specs.num_zplanes

    @property
    def num_planes(self) -> int:
        """Z-plane count; alias of num_zplanes (picks up subclass overrides)."""
        return self.num_zplanes

    @property
    def dx(self) -> float:
        """Pixel size in X from metadata (1.0 if unknown)."""
        return self.dimension_specs.dx

    @property
    def dy(self) -> float:
        """Pixel size in Y from metadata (1.0 if unknown)."""
        return self.dimension_specs.dy

    @property
    def dz(self) -> float | None:
        """Z-step size from metadata (None if no Z dim)."""
        return self.dimension_specs.dz

    @property
    def fs(self) -> float | None:
        """Frame rate in Hz from metadata (None if unknown)."""
        return self.dimension_specs.fs

    @property
    def finterval(self) -> float | None:
        """Frame interval in seconds (None if unknown)."""
        return self.dimension_specs.finterval

    @property
    def slider_dims(self) -> tuple[str, ...] | None:
        from mbo_utilities.arrays.features._dim_labels import get_slider_dims

        return get_slider_dims(self)

    def summary_stats_dim_role(self, name: str):
        """How a scrollable dim is treated by the summary-stats tab.

        Returns ``(is_series_candidate, off_role)`` where ``off_role`` is the
        role the dim takes when it is not the chosen series axis. ``name`` is a
        canonical axis label (``"Z"``/``"T"``/``"C"``/…). Override per array to
        reclassify a dim (e.g. an array whose ``T`` axis holds spatial tiles
        should return ``(False, GROUP)`` so tiles are broken out, not averaged).
        See `mbo_utilities.arrays.features._summary_stats`.
        """
        from mbo_utilities.arrays.features._summary_stats import default_dim_role

        return default_dim_role(name)

    def summary_stats_metrics(self):
        """Metric columns shown by the summary-stats tab (default: mean/std/SNR).

        Override to add or replace metrics; each is a
        `mbo_utilities.arrays.features._summary_stats.StatsMetric`.
        """
        from mbo_utilities.arrays.features._summary_stats import DEFAULT_METRICS

        return DEFAULT_METRICS

    def summary_stats_spec(
        self, *, dims=None, shape=None, series_pref=None, labels=None
    ):
        """Resolve this array's summary-stats layout into a ``SummaryStatsSpec``.

        ``dims``/``shape`` default to the array's own; the GUI passes the
        rendered (possibly singleton-squeezed) view so axis positions match
        what it indexes. ``series_pref`` ("z"/"t") and ``labels`` (canonical ->
        display label) come from the GUI. The spec states exactly what the tab
        will show via ``spec.describe()``.
        """
        from mbo_utilities.arrays.features._summary_stats import (
            build_summary_stats_spec,
        )

        return build_summary_stats_spec(
            tuple(dims) if dims is not None else tuple(self.dims),
            tuple(shape) if shape is not None else tuple(self.shape),
            dim_role=self.summary_stats_dim_role,
            metrics=self.summary_stats_metrics(),
            series_pref=series_pref,
            labels=labels,
        )

    def _summary_stats_store_path(self) -> str | None:
        """Path to the zarr store that caches this array's summary stats.

        Returns None when the array is not single-zarr-backed (the stats are
        then kept only in memory for the session). Disk-backed subclasses
        (ZarrArray, IsoviewArray) override this with their store path so the
        GUI can persist/reload stats without recomputing every open.
        """
        return None

    def save_summary_stats(self, payload: dict, means=None) -> bool:
        """Persist a computed summary-stats payload (+ optional mean stack).

        Format-agnostic: writes into the backing zarr store via
        `_summary_stats_store_path`. Returns False (no-op) when not disk-backed
        or on any write error, so the caller never has to guard the format.
        """
        path = self._summary_stats_store_path()
        if path is None:
            return False
        from mbo_utilities.arrays.features._summary_stats import write_stats_store

        try:
            write_stats_store(path, payload, means)
            return True
        except Exception:
            return False

    def load_summary_stats(self):
        """Return cached ``(payload, means|None)`` from the store, or None."""
        path = self._summary_stats_store_path()
        if path is None:
            return None
        from mbo_utilities.arrays.features._summary_stats import read_stats_store

        try:
            return read_stats_store(path)
        except Exception:
            return None

    def dim_index(self, label: str) -> int | None:
        try:
            return self.dims.index(label.upper())
        except ValueError:
            return None

    def has_dim(self, label: str) -> bool:
        return self.dim_index(label) is not None

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
