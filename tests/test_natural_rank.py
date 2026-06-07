"""
5D-shape tests for TiffArray (v4).

Every array `imread()` returns is always 5D TCZYX (`ndim == 5`), with
singleton T/C/Z axes kept, not squeezed. User-facing squeezing is opt-in
(`arr.squeeze()` / `imread(squeeze=True)`).

These tests pin that behavior:

1. `shape`/`ndim`/`dims` are always 5D TCZYX
2. `nt`/`nc`/`nz`/`ny`/`nx` match `shape`
3. `__getitem__` follows numpy 5D semantics (integer axes squeeze out)
4. `np.asarray(arr)` returns a representative (Y, X) frame
5. `arr.vmin`/`arr.vmax` work on every rank pattern and array class
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from mbo_utilities.arrays import TiffArray


def _write_tiff(path, data):
    tifffile.imwrite(str(path), data)
    return path


@pytest.fixture
def tiff_2d(tmp_path):
    """Plain 2D image (Y, X)."""
    rng = np.random.RandomState(0)
    data = rng.randint(0, 4096, size=(64, 48), dtype=np.uint16)
    p = _write_tiff(tmp_path / "img2d.tif", data)
    return p, data


@pytest.fixture
def tiff_3d_tyx(tmp_path):
    """Time series (T, Y, X)."""
    rng = np.random.RandomState(1)
    data = rng.randint(0, 4096, size=(20, 64, 48), dtype=np.uint16)
    p = _write_tiff(tmp_path / "ts.tif", data)
    return p, data


@pytest.fixture
def tiff_3d_zyx(tmp_path):
    """Z-stack (Z, Y, X) — written as planeXX volume so TiffArray
    detects it as multi-plane with T=1."""
    rng = np.random.RandomState(2)
    nz = 8
    vol_dir = tmp_path / "zstack"
    vol_dir.mkdir()
    for z in range(nz):
        plane = rng.randint(0, 4096, size=(64, 48), dtype=np.uint16)
        # write each plane as a single 2D image (T=1)
        _write_tiff(vol_dir / f"plane{z:02d}.tif", plane)
    return vol_dir, nz


@pytest.fixture
def tiff_4d_tzyx(tmp_path):
    """Multi-plane time series (T, Z, Y, X)."""
    rng = np.random.RandomState(3)
    nt, nz, ny, nx = 12, 4, 64, 48
    data = rng.randint(0, 4096, size=(nt, nz, ny, nx), dtype=np.uint16)
    vol_dir = tmp_path / "vol"
    vol_dir.mkdir()
    for z in range(nz):
        _write_tiff(vol_dir / f"plane{z:02d}.tif", data[:, z])
    return vol_dir, data


class TestTiffArray5DShape:
    """shape, ndim, dims are always 5D TCZYX."""

    def test_2d_image(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        ny, nx = data.shape
        assert arr.shape == (1, 1, 1, ny, nx)
        assert arr.ndim == 5
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_3d_time_series(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        nt, ny, nx = data.shape
        assert arr.shape == (nt, 1, 1, ny, nx)
        assert arr.ndim == 5

    def test_3d_zstack_no_time(self, tiff_3d_zyx):
        path, nz = tiff_3d_zyx
        arr = TiffArray(path)
        # T=1, C=1, Z=nz
        assert arr.ndim == 5
        assert arr.shape[0] == 1
        assert arr.shape[2] == nz
        assert arr.dims == ("T", "C", "Z", "Y", "X")

    def test_4d_volume_time_series(self, tiff_4d_tzyx):
        path, data = tiff_4d_tzyx
        arr = TiffArray(path)
        nt, nz, ny, nx = data.shape
        assert arr.shape == (nt, 1, nz, ny, nx)
        assert arr.ndim == 5


class TestTiffArray5DInvariants:
    """nt / nc / nz / ny / nx match shape and shape is always length 5."""

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_shape_always_length_5(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        assert len(arr.shape) == 5
        assert arr.ndim == 5

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_named_accessors_match_shape(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        assert (arr.nt, arr.nc, arr.nz, arr.ny, arr.nx) == arr.shape


class TestTiffArray5DIndexing:
    """__getitem__ follows numpy 5D semantics: integer axes squeeze out."""

    def test_2d_full_frame(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        frame = np.asarray(arr[0, 0, 0])
        assert frame.shape == data.shape
        np.testing.assert_array_equal(frame.astype(data.dtype), data)

    def test_2d_spatial_subregion(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        out = np.asarray(arr[0, 0, 0, 10:30, 5:25])
        assert out.shape == (20, 20)
        np.testing.assert_array_equal(out.astype(data.dtype), data[10:30, 5:25])

    def test_int_on_one_axis_squeezes_only_that_axis(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        # arr[5] indexes T → (C, Z, Y, X)
        out = np.asarray(arr[5])
        assert out.shape == (1, 1, *data.shape[1:])

    def test_tyx_frame(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        out = np.asarray(arr[5, 0, 0])
        assert out.shape == data.shape[1:]
        np.testing.assert_array_equal(out.astype(data.dtype), data[5])

    def test_zyx_plane(self, tiff_3d_zyx):
        path, nz = tiff_3d_zyx
        arr = TiffArray(path)
        plane = np.asarray(arr[0, 0, 3])
        assert plane.ndim == 2
        assert plane.shape == (arr.ny, arr.nx)

    def test_tzyx_frame(self, tiff_4d_tzyx):
        path, data = tiff_4d_tzyx
        arr = TiffArray(path)
        out = np.asarray(arr[3, 0, 1])
        assert out.shape == data.shape[2:]
        np.testing.assert_array_equal(out.astype(data.dtype), data[3, 1])

    def test_full_slice_keeps_5d(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        out = np.asarray(arr[:])
        assert out.shape == arr.shape


class TestTiffArrayNumpyProtocol:
    """np.asarray returns a representative (Y, X) frame; vmin/vmax never crash."""

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_vmin_vmax_no_crash(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        vmin = arr.vmin
        vmax = arr.vmax
        assert np.isfinite(vmin)
        assert np.isfinite(vmax)
        assert vmin <= vmax

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_asarray_returns_yx_frame(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        materialized = np.asarray(arr)
        # __array__ returns a single (Y, X) representative frame
        assert materialized.ndim == 2
        assert materialized.shape == (arr.ny, arr.nx)


class TestVminVmaxAcrossArrayClasses:
    """`vmin`/`vmax` must work on every lazy-array class.

    Regression: each non-TIFF class (`ZarrArray`, `H5Array`, `NumpyArray`,
    `Suite2pArray`) used to define its own `_compute_frame_vminmax` with a
    hardcoded `self[0, 0, 0]` index. The methods now live on `ReductionMixin`,
    so this test pins coverage across every class that inherits from it.
    """

    def test_zarr_array(self, tmp_path):
        import zarr
        from mbo_utilities.arrays import ZarrArray
        rng = np.random.RandomState(0)
        data = rng.randint(0, 4096, size=(8, 3, 32, 32), dtype=np.uint16)
        z = zarr.open(str(tmp_path / "a.zarr"), mode="w",
                      shape=data.shape, dtype=data.dtype)
        z[:] = data
        arr = ZarrArray(tmp_path / "a.zarr")
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)
        assert arr.vmin <= arr.vmax

    def test_h5_array(self, tmp_path):
        import h5py
        from mbo_utilities.arrays import H5Array
        rng = np.random.RandomState(1)
        data = rng.randint(0, 4096, size=(2, 1, 3, 32, 32), dtype=np.uint16)
        with h5py.File(tmp_path / "a.h5", "w") as f:
            f.create_dataset("mov", data=data)
        arr = H5Array(tmp_path / "a.h5")
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)
        assert arr.vmin <= arr.vmax

    def test_numpy_array(self):
        from mbo_utilities.arrays import NumpyArray
        rng = np.random.RandomState(2)
        data = rng.randint(0, 4096, size=(4, 1, 3, 32, 32), dtype=np.uint16)
        arr = NumpyArray(data)
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)
        assert arr.vmin <= arr.vmax

    def test_numpy_array_2d(self):
        from mbo_utilities.arrays import NumpyArray
        rng = np.random.RandomState(3)
        data = rng.randint(0, 4096, size=(64, 48), dtype=np.uint16)
        arr = NumpyArray(data)
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)

    def test_tiff_array_2d(self, tiff_2d):
        path, _ = tiff_2d
        arr = TiffArray(path)
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)

    def test_tiff_array_zyx(self, tiff_3d_zyx):
        path, _ = tiff_3d_zyx
        arr = TiffArray(path)
        assert np.isfinite(arr.vmin) and np.isfinite(arr.vmax)
