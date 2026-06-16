"""LazyArray base contract: dims/metadata/dimension_specs come from the base.

Pins the consolidated contract so a new array class needs only `_shape5d`,
`__getitem__`, `dtype`, `can_open`, plus `self._metadata`. Natural-rank
classes opt out via `ndim`/`shape`. BinArray (suite2p input) stays 3D.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.lazy_array import LazyArray


class _Minimal(LazyArray):
    """Smallest possible 5D reader — everything else is inherited."""

    def __init__(self, shape, metadata=None):
        self._s = shape
        self._metadata = metadata or {}

    def _shape5d(self):
        return self._s

    @property
    def dtype(self):
        return np.dtype("uint16")

    def __getitem__(self, key):
        return np.zeros(self._s, dtype="uint16")[key]


class TestBaseContract:
    def test_shape_ndim_from_base(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.shape == (4, 2, 3, 8, 8)
        assert arr.ndim == 5
        assert (arr.nt, arr.nc, arr.nz, arr.ny, arr.nx) == arr.shape

    def test_dims_default_canonical(self):
        assert _Minimal((4, 2, 3, 8, 8)).dims == ("T", "C", "Z", "Y", "X")

    def test_metadata_default_dict(self):
        arr = _Minimal((1, 1, 1, 8, 8))
        assert arr.metadata == {}
        arr.metadata = {"foo": 1}
        assert arr.metadata["foo"] == 1

    def test_metadata_stores_dims_plainly(self):
        # base keeps `dims` as plain metadata; reported dims come from rank.
        arr = _Minimal((4, 2, 3, 8, 8))
        arr.metadata = {"dims": ("T", "C", "Z", "Y", "X"), "foo": 1}
        assert arr.metadata["foo"] == 1
        assert arr.metadata["dims"] == ("T", "C", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_metadata_rank_mismatched_dims_does_not_raise(self):
        # a 4-element dims stored on a 5D array (e.g. round-tripped from a
        # single-channel write) must store cleanly, not raise.
        arr = _Minimal((4, 1, 3, 8, 8))
        arr.metadata = {"dims": ("T", "Z", "Y", "X")}
        assert arr.metadata["dims"] == ("T", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")  # still rank-derived

    def test_explicit_dims_setter(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        arr.dims = ("T", "C", "Z", "Y", "X")
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_dimension_specs_reactive(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.dimension_specs.num_channels == 2
        assert arr.dimension_specs.num_zplanes == 3

    def test_slider_dims_drop_singletons(self):
        # T=4 has a slider; C=1/Z=1 singletons are dropped; Y/X never sliders
        arr = _Minimal((4, 1, 1, 8, 8))
        assert arr.slider_dims == ("t",)

    def test_dim_index_and_has_dim(self):
        arr = _Minimal((4, 2, 3, 8, 8))
        assert arr.dim_index("Z") == 2
        assert arr.has_dim("C")
        assert not arr.has_dim("R")


class TestBinArrayStays3D:
    """suite2p input reader must stay natural 3D (T, Y, X)."""

    def test_bin_array_is_3d(self, tmp_path):
        from mbo_utilities.arrays.bin import BinArray

        data = np.zeros((5, 8, 8), dtype=np.int16)
        path = tmp_path / "data_raw.bin"
        data.tofile(path)
        arr = BinArray(str(path), shape=(5, 8, 8), dtype=np.int16)
        assert arr.ndim == 3
        assert arr.shape == (5, 8, 8)
        # internal 5D accessor still pads for the writers
        assert arr._shape5d() == (5, 1, 1, 8, 8)
