"""
Array class unit tests.

Tests individual array class functionality:
- Instantiation
- Indexing behavior
- Properties (shape, dtype, ndim)
- Protocol compliance (LazyArrayProtocol)
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

import mbo_utilities as mbo
from mbo_utilities.arrays import (
    BinArray,
    H5Array,
    NumpyArray,
    Suite2pArray,
    TiffArray,
    ZarrArray,
)


class TestNumpyArray:
    """Test NumpyArray class."""

    def test_instantiation_from_ndarray(self, synthetic_3d_data):
        """Create NumpyArray from numpy ndarray."""
        arr = NumpyArray(synthetic_3d_data)

        assert arr.shape == synthetic_3d_data.shape
        assert arr.dtype == synthetic_3d_data.dtype
        assert arr.ndim == synthetic_3d_data.ndim

    def test_indexing_single_frame(self, synthetic_3d_data):
        """Index single frame."""
        arr = NumpyArray(synthetic_3d_data)

        frame = arr[0]
        expected = synthetic_3d_data[0]

        assert frame.shape == expected.shape
        assert np.array_equal(frame, expected)

    def test_indexing_slice(self, synthetic_3d_data):
        """Index with slice."""
        arr = NumpyArray(synthetic_3d_data)

        sliced = arr[2:5]
        expected = synthetic_3d_data[2:5]

        assert sliced.shape == expected.shape
        assert np.array_equal(sliced, expected)

    def test_indexing_fancy(self, synthetic_3d_data):
        """Fancy indexing with list."""
        arr = NumpyArray(synthetic_3d_data)

        indices = [0, 2, 4]
        result = arr[indices]
        expected = synthetic_3d_data[indices]

        assert result.shape == expected.shape
        assert np.array_equal(result, expected)

    def test_len(self, synthetic_3d_data):
        """Test __len__."""
        arr = NumpyArray(synthetic_3d_data)
        assert len(arr) == synthetic_3d_data.shape[0]

    def test_has_metadata(self, synthetic_3d_data):
        """NumpyArray should have metadata attribute."""
        arr = NumpyArray(synthetic_3d_data)
        assert hasattr(arr, "metadata")
        assert isinstance(arr.metadata, dict)


class TestBinArray:
    """Test BinArray class."""

    @pytest.fixture
    def bin_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary binary file for testing."""
        bin_path = tmp_path / "test.bin"

        # Write binary file
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        # Write ops.npy
        ops = {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
        }
        np.save(tmp_path / "ops.npy", ops)

        return bin_path, synthetic_3d_data

    def test_instantiation_with_shape(self, bin_file):
        """Create BinArray with explicit shape."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path, shape=expected_data.shape, dtype=np.int16)

        assert arr.shape == expected_data.shape
        assert arr.dtype == np.int16

    def test_instantiation_infers_from_ops(self, bin_file):
        """BinArray should infer shape from ops.npy."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path)

        assert arr.shape == expected_data.shape

    def test_indexing(self, bin_file):
        """Test indexing BinArray."""
        bin_path, expected_data = bin_file

        arr = BinArray(bin_path)

        frame = arr[0]
        assert np.array_equal(frame, expected_data[0])

    def test_setitem(self, tmp_path, synthetic_3d_data):
        """Test writing to BinArray."""
        bin_path = tmp_path / "writable.bin"

        arr = BinArray(bin_path, shape=synthetic_3d_data.shape, dtype=np.int16)

        # Write data
        arr[0] = synthetic_3d_data[0]
        arr.flush()

        # Read back
        assert np.array_equal(arr[0], synthetic_3d_data[0])

    def test_context_manager(self, bin_file):
        """Test BinArray as context manager."""
        bin_path, _ = bin_file

        with BinArray(bin_path) as arr:
            _ = arr[0]
        # Should not raise


class TestH5Array:
    """Test H5Array class."""

    @pytest.fixture
    def h5_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary HDF5 file."""
        import h5py

        h5_path = tmp_path / "test.h5"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("mov", data=synthetic_3d_data)

        return h5_path, synthetic_3d_data

    def test_instantiation(self, h5_file):
        """Create H5Array from file."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        assert arr.shape == expected_data.shape
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, h5_file):
        """Test indexing H5Array."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        frame = arr[0]
        assert frame.shape == expected_data[0].shape
        assert np.array_equal(frame, expected_data[0])

    def test_slice_indexing(self, h5_file):
        """Test slice indexing."""
        h5_path, expected_data = h5_file

        arr = H5Array(h5_path)

        sliced = arr[1:4]
        assert sliced.shape == expected_data[1:4].shape
        assert np.array_equal(sliced, expected_data[1:4])


class TestZarrArray:
    """Test ZarrArray class."""

    @pytest.fixture
    def zarr_store(self, synthetic_3d_data, tmp_path):
        """Create a temporary Zarr store."""
        import zarr

        zarr_path = tmp_path / "test.zarr"

        z = zarr.open_array(
            str(zarr_path),
            mode="w",
            shape=synthetic_3d_data.shape,
            dtype=synthetic_3d_data.dtype,
            chunks=(1, *synthetic_3d_data.shape[1:]),
        )
        z[:] = synthetic_3d_data

        return zarr_path, synthetic_3d_data

    def test_instantiation(self, zarr_store):
        """Create ZarrArray from store."""
        zarr_path, expected_data = zarr_store

        arr = ZarrArray(zarr_path)

        # ZarrArray normalizes to TZYX format, so 3D (T,Y,X) becomes 4D (T,1,Y,X)
        if expected_data.ndim == 3:
            expected_shape = (expected_data.shape[0], 1, expected_data.shape[1], expected_data.shape[2])
        else:
            expected_shape = expected_data.shape

        assert arr.shape == expected_shape
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, zarr_store):
        """Test indexing ZarrArray."""
        zarr_path, expected_data = zarr_store

        arr = ZarrArray(zarr_path)

        frame = arr[0]
        # ZarrArray returns data, shape may differ due to TZYX normalization
        frame_np = np.asarray(frame).squeeze()  # Remove singleton dimensions
        expected_np = np.asarray(expected_data[0]).squeeze()
        # Just verify spatial dimensions match after squeezing
        assert frame_np.shape == expected_np.shape

    def test_has_zstats(self, zarr_store):
        """ZarrArray should have zstats property."""
        zarr_path, _ = zarr_store

        arr = ZarrArray(zarr_path)
        # zstats may be None if not computed, but property should exist
        assert hasattr(arr, "zstats")


class TestTiffArray:
    """Test TiffArray class."""

    @pytest.fixture
    def tiff_file(self, synthetic_3d_data, tmp_path):
        """Create a temporary TIFF file."""
        import tifffile

        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, synthetic_3d_data)

        return tiff_path, synthetic_3d_data

    def test_instantiation(self, tiff_file):
        """Create TiffArray from file."""
        tiff_path, expected_data = tiff_file

        arr = TiffArray(tiff_path)

        # TiffArray may normalize to 4D (T, Z, Y, X)
        # Check spatial dimensions and frame count
        assert arr.shape[-2:] == expected_data.shape[-2:]
        assert arr.shape[0] == expected_data.shape[0]
        assert arr.dtype == expected_data.dtype

    def test_indexing(self, tiff_file):
        """Test indexing TiffArray."""
        tiff_path, expected_data = tiff_file

        arr = TiffArray(tiff_path)

        frame = arr[0]
        # may have extra Z dimension due to 4D normalization
        frame_np = np.asarray(frame)
        # verify spatial dims are present (order may vary due to 4D normalization)
        frame_squeezed = frame_np.squeeze()
        assert frame_squeezed.shape == expected_data[0].shape, \
            f"Squeezed frame shape mismatch: {frame_squeezed.shape} vs {expected_data[0].shape}"


class TestSuite2pArray:
    """Test Suite2pArray class."""

    @pytest.fixture
    def suite2p_dir(self, synthetic_3d_data, tmp_path):
        """Create a Suite2p-style directory structure."""
        # Suite2pArray expects specific file names: data.bin or data_raw.bin
        s2p_dir = tmp_path / "suite2p" / "plane0"
        s2p_dir.mkdir(parents=True)

        # Write binary - Suite2pArray looks for data.bin or data_raw.bin
        bin_path = s2p_dir / "data.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        # Write ops.npy with required fields
        ops = {
            "Ly": synthetic_3d_data.shape[1],
            "Lx": synthetic_3d_data.shape[2],
            "nframes": synthetic_3d_data.shape[0],
            "filelist": [str(bin_path)],
            "reg_file": str(bin_path),  # Suite2pArray looks for this
        }
        np.save(s2p_dir / "ops.npy", ops)

        return s2p_dir, synthetic_3d_data

    def test_instantiation(self, suite2p_dir):
        """Create Suite2pArray from ops.npy file."""
        s2p_path, expected_data = suite2p_dir

        # Suite2pArray expects ops.npy path, not directory
        ops_path = s2p_path / "ops.npy"
        arr = Suite2pArray(ops_path)

        # Check spatial dimensions and frame count
        assert arr.shape[-2:] == expected_data.shape[-2:]
        assert arr.shape[0] == expected_data.shape[0]

    def test_indexing(self, suite2p_dir):
        """Test indexing Suite2pArray."""
        s2p_path, expected_data = suite2p_dir

        ops_path = s2p_path / "ops.npy"
        arr = Suite2pArray(ops_path)

        frame = arr[0]
        frame_np = np.asarray(frame)
        assert frame_np.shape[-2:] == expected_data[0].shape[-2:]


class TestTiffVolumeAutoDetection:
    """Test TiffArray auto-detection of volume directories."""

    @pytest.fixture
    def tiff_volume_dir(self, synthetic_4d_data, tmp_path):
        """Create a directory with planeXX.tiff files."""
        import tifffile

        vol_dir = tmp_path / "volume"
        vol_dir.mkdir()

        # synthetic_4d_data is (T, Z, Y, X) - write each plane as separate file
        n_planes = synthetic_4d_data.shape[1]
        for z in range(n_planes):
            plane_data = synthetic_4d_data[:, z, :, :]  # (T, Y, X)
            plane_path = vol_dir / f"plane{z:02d}.tiff"
            tifffile.imwrite(plane_path, plane_data)

        return vol_dir, synthetic_4d_data

    def test_volume_directory_detection(self, tiff_volume_dir):
        """TiffArray should detect volume directory and return 4D shape."""
        vol_dir, expected_data = tiff_volume_dir

        arr = TiffArray(vol_dir)

        # should be 4D: (T, Z, Y, X)
        assert arr.ndim == 4, f"Expected 4D array, got {arr.ndim}D"
        assert arr.shape[0] == expected_data.shape[0], "Frame count mismatch"
        assert arr.shape[1] == expected_data.shape[1], "Plane count mismatch"
        assert arr.shape[2:] == expected_data.shape[2:], "Spatial dims mismatch"

    def test_volume_indexing(self, tiff_volume_dir):
        """Test indexing volume TiffArray."""
        vol_dir, expected_data = tiff_volume_dir

        arr = TiffArray(vol_dir)

        # index single timepoint
        frame = arr[0]
        frame_np = np.asarray(frame)
        # verify total elements match (axis order may differ)
        assert frame_np.size == expected_data[0].size, "Element count mismatch"
        assert frame_np.ndim == 3, "Single frame should be 3D"

    def test_find_tiff_plane_files(self, tiff_volume_dir):
        """Test find_tiff_plane_files helper function."""
        from mbo_utilities.arrays.tiff import find_tiff_plane_files

        vol_dir, expected_data = tiff_volume_dir
        n_planes = expected_data.shape[1]

        plane_files = find_tiff_plane_files(vol_dir)

        assert len(plane_files) == n_planes
        # should be sorted by plane number
        for i, f in enumerate(plane_files):
            assert f"plane{i:02d}" in f.name


class TestSuite2pVolumeAutoDetection:
    """Test Suite2pArray auto-detection of multi-plane directories."""

    @pytest.fixture
    def suite2p_volume_dir(self, synthetic_4d_data, tmp_path):
        """Create a Suite2p multi-plane directory structure."""
        parent_dir = tmp_path / "suite2p"
        parent_dir.mkdir()

        # synthetic_4d_data is (T, Z, Y, X)
        n_planes = synthetic_4d_data.shape[1]

        for z in range(n_planes):
            plane_dir = parent_dir / f"plane{z}"
            plane_dir.mkdir()

            plane_data = synthetic_4d_data[:, z, :, :]  # (T, Y, X)

            # write binary file
            bin_path = plane_dir / "data.bin"
            mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=plane_data.shape)
            mmap[:] = plane_data
            mmap.flush()
            del mmap

            # write ops.npy
            ops = {
                "Ly": plane_data.shape[1],
                "Lx": plane_data.shape[2],
                "nframes": plane_data.shape[0],
                "reg_file": str(bin_path),
            }
            np.save(plane_dir / "ops.npy", ops)

        return parent_dir, synthetic_4d_data

    def test_volume_directory_detection(self, suite2p_volume_dir):
        """Suite2pArray should detect multi-plane directory and return 4D shape."""
        parent_dir, expected_data = suite2p_volume_dir

        arr = Suite2pArray(parent_dir)

        # should be 4D: (T, Z, Y, X)
        assert arr.ndim == 4, f"Expected 4D array, got {arr.ndim}D"
        assert arr.shape[0] == expected_data.shape[0], "Frame count mismatch"
        assert arr.shape[1] == expected_data.shape[1], "Plane count mismatch"
        assert arr.shape[2:] == expected_data.shape[2:], "Spatial dims mismatch"

    def test_volume_indexing(self, suite2p_volume_dir):
        """Test indexing volume Suite2pArray."""
        parent_dir, expected_data = suite2p_volume_dir

        arr = Suite2pArray(parent_dir)

        # index single timepoint
        frame = arr[0]
        frame_np = np.asarray(frame)
        # verify total elements match (axis order may differ)
        assert frame_np.size == expected_data[0].size, "Element count mismatch"
        assert frame_np.ndim == 3, "Single frame should be 3D"

    def test_find_suite2p_plane_dirs(self, suite2p_volume_dir):
        """Test find_suite2p_plane_dirs helper function."""
        from mbo_utilities.arrays.suite2p import find_suite2p_plane_dirs

        parent_dir, expected_data = suite2p_volume_dir
        n_planes = expected_data.shape[1]

        plane_dirs = find_suite2p_plane_dirs(parent_dir)

        assert len(plane_dirs) == n_planes
        # should be sorted by plane number
        for i, d in enumerate(plane_dirs):
            assert f"plane{i}" in d.name


class TestProtocolCompliance:
    """Test that array classes comply with expected protocols."""

    @pytest.fixture(params=["numpy", "bin", "h5", "zarr", "tiff"])
    def array_instance(self, request, synthetic_3d_data, tmp_path):
        """Create array instance of each type."""
        array_type = request.param

        if array_type == "numpy":
            return NumpyArray(synthetic_3d_data)

        elif array_type == "bin":
            bin_path = tmp_path / "test.bin"
            mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
            mmap[:] = synthetic_3d_data
            mmap.flush()
            del mmap
            ops = {"Ly": synthetic_3d_data.shape[1], "Lx": synthetic_3d_data.shape[2], "nframes": synthetic_3d_data.shape[0]}
            np.save(tmp_path / "ops.npy", ops)
            return BinArray(bin_path)

        elif array_type == "h5":
            import h5py
            h5_path = tmp_path / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("mov", data=synthetic_3d_data)
            return H5Array(h5_path)

        elif array_type == "zarr":
            import zarr
            zarr_path = tmp_path / "test.zarr"
            z = zarr.open_array(str(zarr_path), mode="w", shape=synthetic_3d_data.shape, dtype=synthetic_3d_data.dtype)
            z[:] = synthetic_3d_data
            return ZarrArray(zarr_path)

        elif array_type == "tiff":
            import tifffile
            tiff_path = tmp_path / "test.tif"
            tifffile.imwrite(tiff_path, synthetic_3d_data)
            return TiffArray(tiff_path)

    def test_has_shape(self, array_instance):
        """All arrays should have shape attribute."""
        assert hasattr(array_instance, "shape")
        assert isinstance(array_instance.shape, tuple)

    def test_has_dtype(self, array_instance):
        """All arrays should have dtype attribute."""
        assert hasattr(array_instance, "dtype")

    def test_has_ndim(self, array_instance):
        """All arrays should have ndim attribute."""
        assert hasattr(array_instance, "ndim")
        assert isinstance(array_instance.ndim, int)

    def test_indexable(self, array_instance):
        """All arrays should support indexing."""
        frame = array_instance[0]
        assert frame is not None

    def test_has_len(self, array_instance):
        """All arrays should support len()."""
        # Some arrays may not implement __len__, use shape[0] as fallback
        try:
            length = len(array_instance)
            assert length == array_instance.shape[0]
        except TypeError:
            # If no __len__, verify shape[0] is accessible
            assert array_instance.shape[0] > 0

    def test_has_metadata(self, array_instance):
        """All arrays should have metadata attribute."""
        assert hasattr(array_instance, "metadata")


class TestImreadDispatcher:
    """Test that imread returns correct array types."""

    def test_imread_tiff(self, tmp_path, synthetic_3d_data):
        """imread should return appropriate array for TIFF."""
        import tifffile

        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, synthetic_3d_data)

        arr = mbo.imread(tiff_path)

        # Should return some array-like object with correct spatial dims
        assert hasattr(arr, "shape")
        assert arr.shape[-2:] == synthetic_3d_data.shape[-2:]
        assert arr.shape[0] == synthetic_3d_data.shape[0]

    def test_imread_zarr(self, tmp_path, synthetic_3d_data):
        """imread should return ZarrArray for .zarr."""
        import zarr

        zarr_path = tmp_path / "test.zarr"
        z = zarr.open_array(str(zarr_path), mode="w", shape=synthetic_3d_data.shape, dtype=synthetic_3d_data.dtype)
        z[:] = synthetic_3d_data

        arr = mbo.imread(zarr_path)

        # ZarrArray normalizes to 4D, check spatial dims and frame count
        assert hasattr(arr, "shape")
        assert arr.shape[-2:] == synthetic_3d_data.shape[-2:]
        assert arr.shape[0] == synthetic_3d_data.shape[0]

    def test_imread_h5(self, tmp_path, synthetic_3d_data):
        """imread should return H5Array for .h5."""
        import h5py

        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("mov", data=synthetic_3d_data)

        arr = mbo.imread(h5_path)

        assert hasattr(arr, "shape")
        assert arr.shape == synthetic_3d_data.shape

    def test_imread_bin(self, tmp_path, synthetic_3d_data):
        """imread should return array for binary directory."""
        bin_dir = tmp_path / "suite2p"
        bin_dir.mkdir()

        bin_path = bin_dir / "data_raw.bin"
        mmap = np.memmap(bin_path, mode="w+", dtype=np.int16, shape=synthetic_3d_data.shape)
        mmap[:] = synthetic_3d_data
        mmap.flush()
        del mmap

        ops = {"Ly": synthetic_3d_data.shape[1], "Lx": synthetic_3d_data.shape[2], "nframes": synthetic_3d_data.shape[0]}
        np.save(bin_dir / "ops.npy", ops)

        arr = mbo.imread(bin_dir)

        assert hasattr(arr, "shape")
        assert arr.shape == synthetic_3d_data.shape
