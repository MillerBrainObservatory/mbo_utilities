"""
Tests for the opt-in SqueezedView layer (v4 phase 4).

Arrays are always 5D; squeezing drops size-1 T/C/Z axes for ergonomics
while indexing delegates back to the canonical 5D array.
"""

import numpy as np
import pytest
import tifffile

import mbo_utilities
from mbo_utilities.arrays import NumpyArray
from mbo_utilities.squeeze import SqueezedView


def test_squeeze_returns_view():
    a = NumpyArray(np.zeros((4, 8, 6), dtype="uint16"))
    s = a.squeeze()
    assert isinstance(s, SqueezedView)
    assert s.base is a


def test_squeeze_2d():
    data = np.arange(8 * 6, dtype="uint16").reshape(8, 6)
    a = NumpyArray(data)  # (1, 1, 1, 8, 6)
    s = a.squeeze()
    assert s.shape == (8, 6)
    assert s.ndim == 2
    assert np.array_equal(np.asarray(s[:]), data)


def test_squeeze_tyx_indexing():
    data = np.arange(5 * 8 * 6, dtype="uint16").reshape(5, 8, 6)
    a = NumpyArray(data)  # (5, 1, 1, 8, 6)
    s = a.squeeze()
    assert s.shape == (5, 8, 6)
    assert s.ndim == 3
    assert s.dims == ("T", "Y", "X")
    frame = np.asarray(s[2])
    assert frame.shape == (8, 6)
    assert np.array_equal(frame, data[2])


def test_squeeze_keeps_nonsingleton_channel():
    data = np.zeros((4, 2, 1, 8, 6), dtype="uint16")  # Z=1, C=2
    a = NumpyArray(data)
    s = a.squeeze()
    assert s.shape == (4, 2, 8, 6)
    assert s.dims == ("T", "C", "Y", "X")


def test_imread_squeeze_kwarg(tmp_path):
    p = tmp_path / "ts.tif"
    # avoid 3/4-length axes (tifffile would treat them as RGB samples)
    tifffile.imwrite(str(p), np.zeros((10, 16, 12), dtype="uint16"))

    full = mbo_utilities.imread(p)
    assert full.shape == (10, 1, 1, 16, 12)

    squeezed = mbo_utilities.imread(p, squeeze=True)
    assert squeezed.shape == (10, 16, 12)
    assert squeezed.ndim == 3


def test_squeezed_view_imwrite_delegates(tmp_path):
    data = np.random.RandomState(0).randint(0, 255, (4, 8, 6)).astype("uint16")
    a = NumpyArray(data)
    s = a.squeeze()
    out = tmp_path / "out"
    s._imwrite(out, ext=".tiff")  # writes via the canonical 5D base
    assert any(out.rglob("*.tif*"))
