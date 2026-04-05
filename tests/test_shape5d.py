"""
tests for shape5d protocol (phase 1).

validates that shape5d returns correct 5D (T, C, Z, Y, X) for each array
type, and that nt/nc/nz/ny/nx match expected dimension sizes.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from mbo_utilities.arrays import (
    BinArray,
    NumpyArray,
    TiffArray,
    ZarrArray,
    DIMS,
)
from mbo_utilities.arrays._base import Shape5DMixin


class TestShape5DProtocol:
    """verify the Shape5DMixin interface."""

    def test_dims_constant(self):
        assert DIMS == ("T", "C", "Z", "Y", "X")

    def test_mixin_requires_implementation(self):
        class Incomplete(Shape5DMixin):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete().shape5d


class TestNumpyArrayShape5D:
    """NumpyArray shape5d for various input ndims."""

    def test_3d_tyx(self, synthetic_3d_data):
        arr = NumpyArray(synthetic_3d_data)
        s5 = arr.shape5d
        assert len(s5) == 5
        assert s5 == (20, 1, 1, 128, 128)
        assert arr.nt == 20
        assert arr.nc == 1
        assert arr.nz == 1
        assert arr.ny == 128
        assert arr.nx == 128

    def test_4d_tzyx(self, synthetic_4d_data):
        arr = NumpyArray(synthetic_4d_data)
        s5 = arr.shape5d
        assert len(s5) == 5
        assert s5 == (10, 1, 3, 64, 64)
        assert arr.nt == 10
        assert arr.nc == 1
        assert arr.nz == 3

    def test_5d_tczyx(self):
        data = np.zeros((5, 2, 3, 32, 32), dtype=np.int16)
        arr = NumpyArray(data)
        s5 = arr.shape5d
        assert s5 == (5, 2, 3, 32, 32)

    def test_2d_yx(self):
        data = np.zeros((64, 64), dtype=np.int16)
        arr = NumpyArray(data)
        s5 = arr.shape5d
        assert s5 == (1, 1, 1, 64, 64)


class TestBinArrayShape5D:
    """BinArray is always 3D (T, Y, X)."""

    def test_shape5d(self, synthetic_3d_data):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "test.bin"
            shape = synthetic_3d_data.shape
            arr = BinArray(path, shape=shape, dtype=synthetic_3d_data.dtype)
            arr._file[:] = synthetic_3d_data

            s5 = arr.shape5d
            assert len(s5) == 5
            assert s5 == (shape[0], 1, 1, shape[1], shape[2])
            assert arr.nt == shape[0]
            assert arr.nc == 1
            assert arr.nz == 1
            arr.close()


class TestTiffArrayShape5D:
    """TiffArray shape5d for single-file and volume data."""

    def test_single_file_3d(self, synthetic_3d_data):
        """single tiff file: shape is (T, 1, Y, X), shape5d adds C=1."""
        import tifffile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "test.tif"
            tifffile.imwrite(str(path), synthetic_3d_data)

            arr = TiffArray(path)
            s5 = arr.shape5d
            assert len(s5) == 5
            # T=20, C=1, Z=1, Y=128, X=128
            assert s5[0] == 20
            assert s5[1] == 1  # C
            assert s5[2] == 1  # Z
            assert s5[3] == 128
            assert s5[4] == 128

    def test_volume_dir(self, synthetic_4d_data):
        """volume dir with plane files: shape5d includes Z."""
        import tifffile

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            vol_dir = Path(tmp) / "volume"
            vol_dir.mkdir()
            nz = synthetic_4d_data.shape[1]
            for z in range(nz):
                plane_path = vol_dir / f"plane{z:02d}.tif"
                tifffile.imwrite(str(plane_path), synthetic_4d_data[:, z])

            arr = TiffArray(vol_dir)
            s5 = arr.shape5d
            assert len(s5) == 5
            assert s5[0] == 10  # T
            assert s5[1] == 1   # C
            assert s5[2] == 3   # Z
            assert s5[3] == 64  # Y
            assert s5[4] == 64  # X
            assert arr.nz == 3


class TestZarrArrayShape5D:
    """ZarrArray is always 4D TZYX, shape5d inserts C=1."""

    def test_shape5d(self, synthetic_4d_data):
        import zarr

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "test.zarr"
            store = zarr.open(str(path), mode="w", shape=synthetic_4d_data.shape, dtype=synthetic_4d_data.dtype)
            store[:] = synthetic_4d_data

            arr = ZarrArray(path)
            s5 = arr.shape5d
            assert len(s5) == 5
            assert s5 == (10, 1, 3, 64, 64)
            assert arr.nc == 1


class TestShape5DConsistency:
    """cross-array consistency checks."""

    def test_shape5d_length_always_5(self, synthetic_3d_data, synthetic_4d_data):
        """all array types return exactly 5 elements."""
        arr3 = NumpyArray(synthetic_3d_data)
        arr4 = NumpyArray(synthetic_4d_data)

        assert len(arr3.shape5d) == 5
        assert len(arr4.shape5d) == 5

    def test_spatial_dims_preserved(self, synthetic_3d_data):
        """Y, X dimensions match original data."""
        arr = NumpyArray(synthetic_3d_data)
        assert arr.ny == synthetic_3d_data.shape[-2]
        assert arr.nx == synthetic_3d_data.shape[-1]

    def test_named_accessors_match_shape5d(self, synthetic_4d_data):
        """nt/nc/nz/ny/nx match shape5d tuple elements."""
        arr = NumpyArray(synthetic_4d_data)
        s5 = arr.shape5d
        assert arr.nt == s5[0]
        assert arr.nc == s5[1]
        assert arr.nz == s5[2]
        assert arr.ny == s5[3]
        assert arr.nx == s5[4]
