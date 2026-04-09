"""
Natural-rank contract tests for TiffArray.

These tests pin the contracts the viewer pipeline depends on:

1. `shape`/`ndim`/`dims` report natural rank (singletons in T/C/Z dropped)
2. `shape5d`/`nt`/`nc`/`nz`/`ny`/`nx` always report 5D for writers
3. `__getitem__` accepts both natural-rank keys and legacy 5-length keys
4. `np.asarray(arr)` honors natural rank (not a row of the 5D inflation)
5. `arr.vmin`/`arr.vmax` work without crashing on every rank pattern

Each bug we shipped between the 5D refactor and now would have been
caught by one of these. New patterns belong here.
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


class TestTiffArrayNaturalRankShape:
    """shape, ndim, dims report natural rank with singletons squeezed."""

    def test_2d_image(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        assert arr.shape == data.shape
        assert arr.ndim == 2
        assert arr.dims == ("Y", "X")
        # 5D contract still preserved
        assert arr.shape5d == (1, 1, 1, *data.shape)

    def test_3d_time_series(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        assert arr.shape == data.shape
        assert arr.ndim == 3
        assert arr.dims == ("T", "Y", "X")
        assert arr.shape5d == (data.shape[0], 1, 1, data.shape[1], data.shape[2])

    def test_3d_zstack_no_time(self, tiff_3d_zyx):
        path, nz = tiff_3d_zyx
        arr = TiffArray(path)
        # T=1, C=1 dropped → (Z, Y, X)
        assert arr.ndim == 3
        assert arr.shape[0] == nz
        assert arr.dims == ("Z", "Y", "X")
        assert arr.shape5d[0] == 1  # T squeezed in natural, kept in shape5d
        assert arr.shape5d[2] == nz

    def test_4d_volume_time_series(self, tiff_4d_tzyx):
        path, data = tiff_4d_tzyx
        arr = TiffArray(path)
        assert arr.shape == data.shape
        assert arr.ndim == 4
        assert arr.dims == ("T", "Z", "Y", "X")
        assert arr.shape5d == (data.shape[0], 1, data.shape[1], data.shape[2], data.shape[3])


class TestTiffArrayShape5DInvariants:
    """shape5d / nt / nc / nz / ny / nx are stable regardless of natural rank."""

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_shape5d_always_length_5(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        assert len(arr.shape5d) == 5

    @pytest.mark.parametrize("fixture_name", [
        "tiff_2d", "tiff_3d_tyx", "tiff_3d_zyx", "tiff_4d_tzyx"
    ])
    def test_named_accessors_match_shape5d(self, request, fixture_name):
        fixture = request.getfixturevalue(fixture_name)
        path = fixture[0]
        arr = TiffArray(path)
        s5 = arr.shape5d
        assert (arr.nt, arr.nc, arr.nz, arr.ny, arr.nx) == s5


class TestTiffArrayNaturalRankIndexing:
    """__getitem__ supports both natural-rank and legacy 5D keys."""

    def test_2d_full_slice(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        out = np.asarray(arr[:])
        assert out.shape == data.shape
        np.testing.assert_array_equal(out.astype(data.dtype), data)

    def test_2d_subregion(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        out = np.asarray(arr[10:30, 5:25])
        assert out.shape == (20, 20)
        np.testing.assert_array_equal(out.astype(data.dtype), data[10:30, 5:25])

    def test_2d_legacy_5d_key_returns_frame(self, tiff_2d):
        """`arr[0, 0, 0]` from 5D-era callers must still return a (Y, X) frame.

        Regression: my first natural-rank __getitem__ silently dropped
        the third index, mis-routing arr[0,0,0] to a single pixel and
        crashing _compute_frame_vminmax.
        """
        path, data = tiff_2d
        arr = TiffArray(path)
        frame = np.asarray(arr[0, 0, 0])
        assert frame.shape == data.shape

    def test_2d_legacy_5d_key_with_spatial_slice(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        out = np.asarray(arr[0, 0, 0, 10:20, 5:15])
        assert out.shape == (10, 10)

    def test_3d_tyx_index_t(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        out = np.asarray(arr[5])
        assert out.shape == data.shape[1:]
        np.testing.assert_array_equal(out.astype(data.dtype), data[5])

    def test_3d_zyx_index_z(self, tiff_3d_zyx):
        path, nz = tiff_3d_zyx
        arr = TiffArray(path)
        # natural rank (Z, Y, X) — first index is Z
        plane = np.asarray(arr[0])
        assert plane.ndim == 2
        assert plane.shape == arr.shape[1:]

    def test_4d_tzyx_natural_index(self, tiff_4d_tzyx):
        path, data = tiff_4d_tzyx
        arr = TiffArray(path)
        out = np.asarray(arr[3, 1])
        assert out.shape == data.shape[2:]
        np.testing.assert_array_equal(out.astype(data.dtype), data[3, 1])


class TestTiffArrayNumpyProtocol:
    """np.asarray / vmin / vmax / astype must honor natural rank.

    Regression: __array__ returned self[0] (a single row) for natural 2D
    arrays. _compute_frame_vminmax used self[0, 0, 0] hardcoded to 5D.
    Both broke when TiffArray started reporting natural rank.
    """

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

    def test_2d_asarray_returns_full_image(self, tiff_2d):
        path, data = tiff_2d
        arr = TiffArray(path)
        materialized = np.asarray(arr)
        # __array__ for 2D should return the whole image, not a row
        assert materialized.ndim == 2
        assert materialized.shape == data.shape

    def test_3d_tyx_asarray_returns_first_frame(self, tiff_3d_tyx):
        path, data = tiff_3d_tyx
        arr = TiffArray(path)
        materialized = np.asarray(arr)
        # 3D+: __array__ returns first slice along outer dim → (Y, X)
        assert materialized.shape == data.shape[1:]


class TestVminVmaxAcrossArrayClasses:
    """`vmin`/`vmax` must work on every lazy-array class.

    Regression: each non-TIFF class (`ZarrArray`, `H5Array`, `NumpyArray`,
    `Suite2pArray`) used to define its own `_compute_frame_vminmax` with a
    hardcoded `self[0, 0, 0]` index. The TiffArray fix to `__array__`
    didn't propagate to those copies, and the user hit a zarr crash with
    "cannot select an axis to squeeze out which has size not equal to one".
    The methods now live on `ReductionMixin`, so this test pins coverage
    across every class that inherits from it.
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
