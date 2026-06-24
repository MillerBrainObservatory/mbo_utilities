"""
Summary-stats descriptor: how an array's dimensions map onto the mbo-studio
Signal Quality / summary-stats tab.

Every dimension is either an *image* dim (the 2D display plane, Y/X — stats
are computed over these) or a *scrollable* dim. Each scrollable dim is
classified for stats as one of:

- ``SERIES`` — forms the stats x-axis (table rows / plot points), e.g. Zplane.
  More than one candidate -> the UI offers a "pick one".
- ``GROUP``  — produces a separate series per index, followed by its slider,
  e.g. Tile, Camera, View.
- ``REDUCE`` — collapsed into each point by the metric aggregator (mean over
  the axis), e.g. Timepoints.

An array declares intent by overriding two ``LazyArray`` hooks:

- ``summary_stats_dim_role(name)`` -> ``(is_series_candidate, off_role)`` for a
  scrollable dim. ``off_role`` is the role the dim takes when it is *not* the
  chosen series axis. Default: ``Z -> (True, GROUP)``, ``T -> (True, REDUCE)``,
  everything else ``-> (False, GROUP)``.
- ``summary_stats_metrics()`` -> the metric columns. Default: mean, std, SNR.

The GUI is a pure renderer of the resolved :class:`SummaryStatsSpec`; it never
classifies dimensions itself. ``spec.describe()`` states exactly what the tab
will show, so the contract is introspectable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from mbo_utilities.arrays.features._dim_labels import _SLIDER_NAME_ALIASES


class StatsDimRole(str, Enum):
    """How a scrollable dimension is treated by the summary-stats tab."""

    SERIES = "series"
    GROUP = "group"
    REDUCE = "reduce"


# canonical display labels and the descriptive aliases not covered by
# `_SLIDER_NAME_ALIASES`.
_STAT_LABELS = {"Z": "Zplane", "T": "Timepoint", "C": "Channel"}
_EXTRA_CANON = {
    "zslices": "Z", "zslice": "Z", "volumes": "Z",
    "frames": "T", "frame": "T", "time": "T",
}

SNR_TOOLTIP = (
    "SNR = (mean_foreground - mean_background) / std_background, "
    "foreground = top 20% brightest pixels, background = bottom 50%"
)


def canonical_axis(dim: str) -> str | None:
    """Map a dim label to canonical ``Z``/``T``/``C``/``Y``/``X`` (or None)."""
    dl = "".join(ch for ch in str(dim).lower() if ch.isalpha())
    if dl in ("y", "x"):
        return dl.upper()
    for canon, aliases in _SLIDER_NAME_ALIASES.items():
        if dl in aliases:
            return canon.upper()
    return _EXTRA_CANON.get(dl)


def subsample_indices(n: int, max_count: int) -> list[int]:
    """Evenly-spaced 0-based indices, at most ``max_count`` of them."""
    n = int(n)
    if n <= 1:
        return [0]
    if n <= max_count:
        return list(range(n))
    stride = -(-n // max_count)  # ceil division
    return list(range(0, n, stride))


@dataclass
class SubsampleBudget:
    """Sampling caps that bound stats compute and stored memory.

    Tunable code-level defaults (no UI). Spatial binning only applies when the
    series is broken out into groups (many combos to store); the single-series
    path keeps full resolution.
    """

    max_series_points: int = 40   # max sampled points on the x-axis
    max_per_group: int = 12       # max sampled indices per group (selector) axis
    spatial_bin: int = 4          # Y/X stride for stored stats / mean-images
    avg_max_samples: int = 30     # max frames averaged per point over a reduce axis


DEFAULT_BUDGET = SubsampleBudget()


# metric reducers: (stack(N,Yb,Xb), mean_img(Yb,Xb), has_reduce: bool) -> float
def _metric_mean(stack: np.ndarray, mean_img: np.ndarray, has_reduce: bool) -> float:
    return float(np.mean(mean_img))


def _metric_std(stack: np.ndarray, mean_img: np.ndarray, has_reduce: bool) -> float:
    # mean per-pixel temporal std when there is a reduce axis to average over
    # (the noise measure for time-series data); with a single frame per point
    # (e.g. one image per tile/camera/zplane, where temporal std is identically
    # zero) fall back to the spatial std of the plane so it stays meaningful.
    if has_reduce and stack.shape[0] > 1:
        return float(np.mean(np.std(stack, axis=0)))
    return float(np.std(mean_img))


def _metric_snr(stack: np.ndarray, mean_img: np.ndarray, has_reduce: bool) -> float:
    p80 = np.percentile(mean_img, 80)
    p50 = np.percentile(mean_img, 50)
    fg = mean_img >= p80
    bg = mean_img <= p50
    fg_mean = float(mean_img[fg].mean()) if fg.any() else 0.0
    bg_mean = float(mean_img[bg].mean()) if bg.any() else 0.0
    bg_std = float(mean_img[bg].std()) if bg.any() else 1.0
    return (fg_mean - bg_mean) / bg_std if bg_std > 0 else 0.0


@dataclass
class StatsMetric:
    """One summary-stats column: a scalar reduced from a point's pixel stack."""

    key: str               # stored key, e.g. "mean"
    label: str             # full row label, e.g. "Mean Fluorescence"
    unit: str              # display unit, e.g. "a.u."
    reducer: Callable      # (stack, mean_img, has_reduce) -> float
    short: str = ""        # compact column header, e.g. "Mean" (defaults to label)
    tooltip: str = ""      # optional hover help (e.g. the SNR formula)

    @property
    def header(self) -> str:
        """Compact column header for the multi-point table."""
        return self.short or self.label.split()[0]


DEFAULT_METRICS: tuple[StatsMetric, ...] = (
    StatsMetric("mean", "Mean Fluorescence", "a.u.", _metric_mean, short="Mean"),
    StatsMetric("std", "Std. Deviation", "a.u.", _metric_std, short="Std"),
    StatsMetric("snr", "Signal-to-Noise", "ratio", _metric_snr, short="SNR",
                tooltip=SNR_TOOLTIP),
)


@dataclass
class StatsDim:
    """A scrollable dim's resolved place in the summary-stats layout."""

    name: str              # canonical "Z"/"T"/"C"
    label: str             # display label, e.g. "Zplane"/"Tile"
    size: int
    axis: int              # position in dims / shape
    candidate: bool        # may be chosen as the SERIES axis
    role: StatsDimRole     # resolved role given the current series choice
    indices: list[int]     # sampled 0-based indices (empty for REDUCE)


@dataclass
class SummaryStatsSpec:
    """Resolved descriptor the GUI renders. Built by `build_summary_stats_spec`."""

    image_dims: tuple[str, ...]            # ("Y", "X")
    series: StatsDim | None                # the x-axis dim (None -> single point)
    series_candidates: tuple[StatsDim, ...]  # all dims that could be the series
    groups: tuple[StatsDim, ...]           # per-series selectors (follow sliders)
    reduce: tuple[StatsDim, ...]           # collapsed per point
    metrics: tuple[StatsMetric, ...]
    dims: tuple                            # the dims the spec was built over
    shape: tuple
    spatial_bin: int
    budget: SubsampleBudget

    @property
    def both_series(self) -> bool:
        """True when more than one dim could be the series axis (offer a pick)."""
        return len(self.series_candidates) > 1

    @property
    def has_reduce(self) -> bool:
        return bool(self.reduce)

    def describe(self) -> str:
        """One-line statement of exactly what the tab will show."""
        s = self.series.label if self.series else "—"
        g = ", ".join(d.label for d in self.groups) or "—"
        r = ", ".join(d.label for d in self.reduce) or "—"
        m = ", ".join(d.label for d in self.metrics)
        return f"Series: {s}  ·  Grouped: {g}  ·  Collapsed: {r}  ·  Metrics: {m}"


def default_dim_role(name: str) -> tuple[bool, StatsDimRole]:
    """Default ``(is_series_candidate, off_role)`` for a scrollable dim.

    Z is a series candidate that otherwise groups; T is a series candidate that
    otherwise collapses (real timepoints are averaged); everything else (cameras,
    views, channels) groups.
    """
    n = name.upper()
    if n == "Z":
        return (True, StatsDimRole.GROUP)
    if n == "T":
        return (True, StatsDimRole.REDUCE)
    return (False, StatsDimRole.GROUP)


def _label_for(canon: str, labels: dict | None) -> str:
    # prefer a descriptive slider label (e.g. "Tile", "Cam", "Zplane"); a bare
    # single-letter label ("z"/"t"/"c") reads worse than the canonical name.
    lab = labels.get(canon) if labels else None
    if lab and len(str(lab)) > 1:
        return str(lab)
    return _STAT_LABELS.get(canon, canon)


def _choose_series(candidates: list[str], pref: str | None) -> str | None:
    """Pick the series canonical name: honor ``pref``, else Z, else T, else first."""
    if not candidates:
        return None
    if pref:
        p = str(pref).upper()
        if p in candidates:
            return p
    for c in ("Z", "T"):
        if c in candidates:
            return c
    return candidates[0]


def build_summary_stats_spec(
    dims,
    shape,
    *,
    dim_role: Callable[[str], tuple[bool, StatsDimRole]] | None = None,
    metrics: tuple[StatsMetric, ...] | None = None,
    series_pref: str | None = None,
    labels: dict | None = None,
    budget: SubsampleBudget = DEFAULT_BUDGET,
) -> SummaryStatsSpec:
    """Resolve image/scrollable dims + roles + budget into a `SummaryStatsSpec`.

    ``dim_role`` classifies each scrollable dim (default `default_dim_role`);
    ``labels`` optionally overrides display labels by canonical name (the GUI
    passes the live slider labels here).
    """
    dims = tuple(dims)
    shape = tuple(shape)
    dim_role = dim_role or default_dim_role
    metrics = tuple(metrics) if metrics is not None else DEFAULT_METRICS

    image_dims = tuple(d for d in dims if canonical_axis(d) in ("Y", "X"))

    # scrollable (non-spatial) dims with size > 1
    scroll = []  # (canon, axis, size)
    for ax, d in enumerate(dims):
        canon = canonical_axis(d)
        if canon in ("Y", "X"):
            continue
        size = int(shape[ax]) if ax < len(shape) else 1
        if size <= 1:
            continue
        scroll.append((canon or str(d).upper(), ax, size))

    roles = {canon: dim_role(canon) for canon, _, _ in scroll}  # canon -> (cand, off)
    cand_names = [canon for canon, _, _ in scroll if roles[canon][0]]
    series_canon = _choose_series(cand_names, series_pref)

    series = None
    groups: list[StatsDim] = []
    reduce: list[StatsDim] = []
    candidates: list[StatsDim] = []
    for canon, ax, size in scroll:
        cand, off = roles[canon]
        role = StatsDimRole.SERIES if canon == series_canon else off
        if role == StatsDimRole.SERIES:
            idxs = subsample_indices(size, budget.max_series_points)
        elif role == StatsDimRole.GROUP:
            idxs = subsample_indices(size, budget.max_per_group)
        else:
            idxs = []
        dim = StatsDim(canon, _label_for(canon, labels), size, ax, cand, role, idxs)
        if role == StatsDimRole.SERIES:
            series = dim
        elif role == StatsDimRole.GROUP:
            groups.append(dim)
        else:
            reduce.append(dim)
        if cand:
            candidates.append(dim)

    return SummaryStatsSpec(
        image_dims=image_dims,
        series=series,
        series_candidates=tuple(candidates),
        groups=tuple(groups),
        reduce=tuple(reduce),
        metrics=metrics,
        dims=dims,
        shape=shape,
        # bin only when grouped (many combos to store); the single-series path
        # keeps full resolution so mean-subtraction is unchanged.
        spatial_bin=budget.spatial_bin if groups else 1,
        budget=budget,
    )
