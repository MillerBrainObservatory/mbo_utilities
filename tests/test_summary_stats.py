"""Tests for the array-owned summary-stats descriptor (Signal Quality tab).

Covers the classification contract (image vs scrollable dims; series / group /
reduce roles), the per-array override hook, metric reducers, and the resolved
``SummaryStatsSpec`` the GUI renders.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.arrays.features import (
    DEFAULT_METRICS,
    StatsDimRole,
    SummaryStatsSpec,
    build_summary_stats_spec,
    default_dim_role,
)
from mbo_utilities.lazy_array import LazyArray


def _names(dims):
    return [d.name for d in dims]


class TestDefaultDimRole:
    def test_z_is_series_candidate_else_group(self):
        assert default_dim_role("Z") == (True, StatsDimRole.GROUP)

    def test_t_is_series_candidate_else_reduce(self):
        assert default_dim_role("T") == (True, StatsDimRole.REDUCE)

    def test_other_dims_group(self):
        assert default_dim_role("C") == (False, StatsDimRole.GROUP)
        assert default_dim_role("V") == (False, StatsDimRole.GROUP)


class TestBuildSpec:
    def test_classic_tzyx_series_z_reduce_t(self):
        spec = build_summary_stats_spec(("T", "Z", "Y", "X"), (100, 8, 64, 64))
        assert spec.series.name == "Z"
        assert spec.groups == ()
        assert _names(spec.reduce) == ["T"]
        assert spec.both_series is True
        assert spec.image_dims == ("Y", "X")
        # single series -> full-resolution mean images
        assert spec.spatial_bin == 1

    def test_dual_channel_groups_channel(self):
        spec = build_summary_stats_spec(("T", "C", "Z", "Y", "X"), (50, 2, 8, 64, 64))
        assert spec.series.name == "Z"
        assert _names(spec.groups) == ["C"]
        assert _names(spec.reduce) == ["T"]

    def test_pick_timepoint_series(self):
        spec = build_summary_stats_spec(
            ("T", "C", "Z", "Y", "X"), (10, 2, 8, 64, 64), series_pref="t"
        )
        assert spec.series.name == "T"
        assert set(_names(spec.groups)) == {"C", "Z"}
        assert spec.reduce == ()

    def test_tiled_t_grouped_not_reduced(self):
        # array declares T as a spatial tile axis: group, never collapse
        def role(name):
            if name == "T":
                return (False, StatsDimRole.GROUP)
            return default_dim_role(name)

        spec = build_summary_stats_spec(
            ("T", "C", "Z", "Y", "X"), (4, 2, 8, 64, 64), dim_role=role
        )
        assert spec.series.name == "Z"
        assert _names(spec.groups) == ["T", "C"]
        assert spec.reduce == ()
        # no timepoint series candidate -> no "pick one"
        assert spec.both_series is False
        # grouped -> stored mean-images are spatially binned
        assert spec.spatial_bin > 1

    def test_singleton_dims_dropped(self):
        spec = build_summary_stats_spec(("T", "C", "Z", "Y", "X"), (100, 1, 8, 64, 64))
        assert "C" not in _names(spec.groups)  # singleton channel not scrollable

    def test_series_candidates_list_all_candidates(self):
        spec = build_summary_stats_spec(("T", "C", "Z", "Y", "X"), (10, 2, 8, 64, 64))
        assert set(_names(spec.series_candidates)) == {"Z", "T"}

    def test_subsample_caps_series_and_groups(self):
        spec = build_summary_stats_spec(
            ("T", "C", "Z", "Y", "X"), (5, 64, 4000, 64, 64), series_pref="z"
        )
        assert len(spec.series.indices) <= spec.budget.max_series_points
        for g in spec.groups:
            assert len(g.indices) <= spec.budget.max_per_group

    def test_label_override_prefers_descriptive(self):
        spec = build_summary_stats_spec(
            ("T", "Z", "Y", "X"), (50, 8, 64, 64), labels={"Z": "z", "T": "t"}
        )
        # a bare single-letter slider label falls back to the descriptive name
        assert spec.series.label == "Zplane"
        assert spec.reduce[0].label == "Timepoint"

    def test_describe_is_introspectable(self):
        spec = build_summary_stats_spec(("T", "C", "Z", "Y", "X"), (50, 2, 8, 64, 64))
        text = spec.describe()
        assert "Series: Zplane" in text
        assert "Grouped: Channel" in text
        assert "Collapsed: Timepoint" in text


class TestMetrics:
    def test_std_temporal_when_reduced(self):
        rng = np.random.default_rng(0)
        stack = rng.normal(100, 5, size=(20, 32, 32)).astype(np.float32)
        mean_img = stack.mean(axis=0)
        std = next(m for m in DEFAULT_METRICS if m.key == "std")
        assert 3.0 < std.reducer(stack, mean_img, True) < 7.0

    def test_std_spatial_when_single_frame(self):
        # one image per point (e.g. a tile/camera/zplane): temporal std is 0,
        # so the metric falls back to the spatial spread of the plane.
        rng = np.random.default_rng(1)
        img = rng.normal(100, 15, size=(1, 32, 32)).astype(np.float32)
        std = next(m for m in DEFAULT_METRICS if m.key == "std")
        assert std.reducer(img, img[0], False) > 5.0

    def test_metric_header_short(self):
        std = next(m for m in DEFAULT_METRICS if m.key == "std")
        assert std.header == "Std"


class _MiniArray(LazyArray):
    """Minimal concrete LazyArray to exercise the base hooks."""

    def __init__(self, shape):
        self._shp = tuple(shape)

    def _shape5d(self):
        return self._shp

    def __getitem__(self, key):
        return np.zeros(self._shp)[key]


class _TiledArray(_MiniArray):
    def summary_stats_dim_role(self, name):
        if name.upper() == "T":
            return (False, StatsDimRole.GROUP)
        return super().summary_stats_dim_role(name)


class TestLazyArrayHooks:
    def test_base_array_yields_spec(self):
        arr = _MiniArray((100, 1, 8, 64, 64))  # TCZYX, singleton C
        spec = arr.summary_stats_spec()
        assert isinstance(spec, SummaryStatsSpec)
        assert spec.series.name == "Z"
        assert _names(spec.reduce) == ["T"]

    def test_override_reclassifies_t(self):
        arr = _TiledArray((4, 2, 8, 64, 64))  # TCZYX
        spec = arr.summary_stats_spec()
        assert spec.series.name == "Z"
        assert _names(spec.groups) == ["T", "C"]
        assert spec.reduce == ()
