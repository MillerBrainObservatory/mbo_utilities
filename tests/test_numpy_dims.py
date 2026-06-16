"""Tests for declarable, reactive, OME-compatible dims on NumpyArray.

`imread(np.ndarray)` wraps the array as a NumpyArray that is always 5D
TCZYX. The wrapped array's axes can be declared (`dims=`/`dim_order=`,
`arr.dims = ...`, or `arr.metadata = {"dims": ...}`) so that downstream
reactive metadata (dimension_specs, OME axes) and the zarr writer's
`dimension_names` label the data correctly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

import tifffile  # noqa: F401  (import-order wiring, see test_roundtrip)

import mbo_utilities as mbo
from mbo_utilities.arrays import NumpyArray
from mbo_utilities.arrays.features import get_dims


def _make_tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="mbo_numpy_dims_"))


def _dimension_names(zarr_path: Path) -> list[str]:
    z = zarr.open(str(zarr_path), mode="r")
    arr = z["0"] if "0" in z else z
    return list(arr.metadata.dimension_names)


class TestDefaultInference:
    """Default dim inference from raw rank: 3D=TYX, 4D=TZYX, 5D=TCZYX."""

    def test_3d_is_tyx(self):
        arr = NumpyArray(np.zeros((10, 32, 32), dtype=np.uint16))
        assert arr.input_dims == ("T", "Y", "X")
        assert arr.shape == (10, 1, 1, 32, 32)
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_4d_is_tzyx(self):
        arr = NumpyArray(np.zeros((10, 3, 32, 32), dtype=np.uint16))
        assert arr.input_dims == ("T", "Z", "Y", "X")
        assert arr.shape == (10, 1, 3, 32, 32)  # the 3 lands on Z

    def test_5d_is_tczyx(self):
        arr = NumpyArray(np.zeros((10, 2, 3, 32, 32), dtype=np.uint16))
        assert arr.input_dims == ("T", "C", "Z", "Y", "X")
        assert arr.shape == (10, 2, 3, 32, 32)

    def test_inferred_flag(self):
        assert NumpyArray(np.zeros((10, 3, 32, 32)))._dims_inferred is True
        assert NumpyArray(np.zeros((10, 3, 32, 32)), dims="TCYX")._dims_inferred is False


class TestExplicitDims:
    """Declaring dims maps each source axis onto the canonical 5D layout."""

    def test_ctor_dims_places_channel(self):
        # 4D two-channel acquisition: the 2 is C, not Z.
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16), dims="TCYX")
        assert arr.shape == (10, 2, 1, 32, 32)  # C=2, Z=1
        assert get_dims(arr) == ("T", "C", "Z", "Y", "X")

    def test_dim_order_alias(self):
        a = NumpyArray(np.zeros((10, 2, 32, 32)), dim_order="TCYX")
        b = NumpyArray(np.zeros((10, 2, 32, 32)), dims="TCYX")
        assert a.shape == b.shape == (10, 2, 1, 32, 32)

    def test_permutation(self):
        # ZTYX source -> canonical TZYX placement.
        arr = NumpyArray(np.zeros((3, 10, 32, 32), dtype=np.uint16), dims="ZTYX")
        assert arr.shape == (10, 1, 3, 32, 32)

    def test_imread_routes_dims(self):
        arr = mbo.imread(np.zeros((10, 2, 32, 32), dtype=np.uint16), dims="TCYX")
        assert isinstance(arr, NumpyArray)
        assert arr.shape == (10, 2, 1, 32, 32)

    @pytest.mark.parametrize("bad", ["TZY", "TTYX", "TQYX"])
    def test_invalid_dims_warn_and_chain_guess(self, bad):
        # unusable dims must NOT raise: warn and fall back to the rank guess
        arr = NumpyArray(np.zeros((10, 3, 32, 32)), dims=bad)
        assert arr.shape == (10, 1, 3, 32, 32)  # inferred TZYX
        assert arr.input_dims == ("T", "Z", "Y", "X")

    def test_setter_mismatch_does_not_raise(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16))
        arr.dims = "TZY"  # wrong length -> warn + chain guess, no raise
        assert arr.shape == (10, 1, 2, 32, 32)


class TestReactivity:
    """Setting dims after construction remaps and recomputes reactive metadata."""

    def test_setter_remaps_shape(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16))
        assert arr.shape == (10, 1, 2, 32, 32)  # default: Z=2
        arr.dims = "TCYX"
        assert arr.shape == (10, 2, 1, 32, 32)  # now C=2

    def test_setter_updates_dimension_specs(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16))
        assert arr.dimension_specs.num_channels == 1
        assert arr.dimension_specs.num_zplanes == 2
        arr.dims = "TCYX"
        # cache must be invalidated -> recomputed against new layout
        assert arr.dimension_specs.num_channels == 2
        assert arr.dimension_specs.num_zplanes == 1

    def test_metadata_dims_setter(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16))
        arr.metadata = {"dims": "TCYX", "experiment": "x"}
        assert arr.shape == (10, 2, 1, 32, 32)
        assert arr.metadata["experiment"] == "x"
        # getter reports canonical 5D dims, never the loose raw order
        assert arr.metadata["dims"] == ("T", "C", "Z", "Y", "X")

    def test_metadata_roundtrip_is_noop(self):
        # reading metadata then writing it back must not raise (canonical echo)
        arr = NumpyArray(np.zeros((10, 32, 32), dtype=np.uint16))
        arr.metadata = dict(arr.metadata)
        assert arr.shape == (10, 1, 1, 32, 32)

    def test_ctor_dims_from_metadata(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16),
                         metadata={"dims": "TCYX"})
        assert arr.shape == (10, 2, 1, 32, 32)

    def test_ctor_dimension_names_alias(self):
        # NGFF-style lowercase dimension_names is accepted as a dims alias
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16),
                         metadata={"dimension_names": ["t", "c", "y", "x"]})
        assert arr.shape == (10, 2, 1, 32, 32)

    def test_metadata_dimension_names_setter(self):
        arr = NumpyArray(np.zeros((10, 2, 32, 32), dtype=np.uint16))
        arr.metadata = {"dimension_names": ["t", "c", "y", "x"]}
        assert arr.shape == (10, 2, 1, 32, 32)

    def test_metadata_exposes_dimension_names(self):
        arr = NumpyArray(np.zeros((10, 2, 3, 32, 32), dtype=np.uint16))
        assert arr.metadata["dimension_names"] == ["t", "c", "z", "y", "x"]


class TestOmeWriterLabeling:
    """Declared dims drive the OME-Zarr dimension_names via get_dims."""

    def test_untagged_4d_two_channel_labels_z(self):
        out = _make_tmp()
        try:
            data = np.random.randint(0, 100, (8, 2, 16, 16), dtype=np.uint16)
            mbo.imwrite(NumpyArray(data), out, ext=".zarr", ome=True, overwrite=True)
            written = next(p for p in out.iterdir() if p.suffix == ".zarr")
            # default inference treats the 2 as Z -> 4D TZYX output
            assert _dimension_names(written) == ["t", "z", "y", "x"]
        finally:
            import shutil
            shutil.rmtree(out, ignore_errors=True)

    def test_tagged_4d_two_channel_labels_c(self):
        out = _make_tmp()
        try:
            data = np.random.randint(0, 100, (8, 2, 16, 16), dtype=np.uint16)
            mbo.imwrite(NumpyArray(data, dims="TCYX"), out, ext=".zarr",
                        ome=True, overwrite=True)
            written = next(p for p in out.iterdir() if p.suffix == ".zarr")
            names = _dimension_names(written)
            assert "c" in names
            z = zarr.open(str(written), mode="r")
            level0 = z["0"] if "0" in z else z
            assert level0.shape[names.index("c")] == 2
        finally:
            import shutil
            shutil.rmtree(out, ignore_errors=True)
