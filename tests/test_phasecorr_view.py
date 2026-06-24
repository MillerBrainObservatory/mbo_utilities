"""Tests for PhaseCorrectedView / with_phasecorr.

Covers the transparent-wrapper contract (passthrough when disabled, corrects
when enabled, reversible), the per-(t, c, z) offset cache the GUI reads, and
that the view exposes the same phase_correction feature the GUI duck-types on.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.arrays._phasecorr_view import PhaseCorrectedView, with_phasecorr
from mbo_utilities.arrays.features import PhaseCorrectionFeature
from mbo_utilities.arrays.numpy import NumpyArray


def _offset_movie(nt=400, ny=64, nx=256, shift=3, seed=0):
    """(T, Y, X) movie with sharp shared row structure and an injected
    bidirectional offset of `shift` px on the odd rows."""
    rng = np.random.default_rng(seed)
    xprof = rng.standard_normal(nx).astype(np.float32)
    base = np.broadcast_to(xprof, (ny, nx))
    movie = (base[None] + 0.05 * rng.standard_normal((nt, ny, nx))).astype(np.float32)
    movie[:, 1::2, :] = np.roll(movie[:, 1::2, :], shift, axis=-1)
    return movie


@pytest.fixture
def arr():
    return NumpyArray(_offset_movie())


class TestSurface:
    def test_shape_dims_passthrough(self, arr):
        v = with_phasecorr(arr)
        assert v.shape == arr.shape
        assert v.ndim == 5
        assert tuple(v.dims) == ("T", "C", "Z", "Y", "X")
        assert v.dtype == arr.dtype

    def test_exposes_phase_correction_feature(self, arr):
        v = with_phasecorr(arr)
        assert isinstance(v.phase_correction, PhaseCorrectionFeature)
        assert v.fix_phase is False  # disabled by default

    def test_forwards_unknown_attrs_to_source(self, arr):
        v = with_phasecorr(arr)
        # num_planes is defined on the source, not the view
        assert v.num_planes == arr.num_planes


class TestCorrection:
    def test_passthrough_when_disabled(self, arr):
        v = with_phasecorr(arr)
        assert np.array_equal(np.asarray(v[10, 0, 0]), np.asarray(arr[10, 0, 0]))

    def test_corrects_when_enabled(self, arr):
        v = with_phasecorr(arr)
        src = np.asarray(arr[10, 0, 0])
        v.fix_phase = True
        assert not np.array_equal(np.asarray(v[10, 0, 0]), src)

    def test_recovers_injected_offset(self, arr):
        v = with_phasecorr(arr, enabled=True)
        np.asarray(v[10, 0, 0])
        # injected +3 px roll on odd rows -> estimator recovers ~ -3
        assert v.get_offset_at(10, 0, 0) == pytest.approx(-3.0, abs=0.1)

    def test_reversible(self, arr):
        v = with_phasecorr(arr, enabled=True)
        np.asarray(v[10, 0, 0])
        v.fix_phase = False
        assert np.array_equal(np.asarray(v[10, 0, 0]), np.asarray(arr[10, 0, 0]))

    def test_window_read_single_offset(self, arr):
        v = with_phasecorr(arr, enabled=True)
        chunk = np.asarray(v[0:200, 0, 0])
        assert chunk.shape == (200, 64, 256)
        # method="mean" -> one offset for the window
        assert v.get_offset_at(0, 0, 0) == pytest.approx(-3.0, abs=0.1)


class TestCache:
    def test_offset_cache_miss_returns_none(self, arr):
        v = with_phasecorr(arr, enabled=True)
        assert v.get_offset_at(123, 0, 0) is None

    def test_toggling_clears_cache(self, arr):
        v = with_phasecorr(arr, enabled=True)
        np.asarray(v[10, 0, 0])
        assert v.get_offset_at(10, 0, 0) is not None
        v.fix_phase = False
        assert v.get_offset_at(10, 0, 0) is None


def test_with_phasecorr_returns_view(arr):
    assert isinstance(with_phasecorr(arr), PhaseCorrectedView)
