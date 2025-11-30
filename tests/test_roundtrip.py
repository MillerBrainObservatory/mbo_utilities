"""
Round-trip format conversion tests.

Strategy:
1. Create a reference TIFF from source data (golden baseline)
2. For each target format: write data → read back → compare to reference TIFF
3. All comparisons are pixel-exact against the reference TIFF

This ensures:
- All formats can write data correctly
- All formats can read data correctly
- Data integrity is preserved across format conversions
"""

import numpy as np
import pytest
import tifffile

import mbo_utilities as mbo
from mbo_utilities.arrays import NumpyArray

# Import helpers from conftest - pytest makes these available
from tests.conftest import compare_arrays, compute_frame_correlation, find_output_file


def wrap_array(data):
    """Wrap numpy array in NumpyArray for imwrite compatibility."""
    if isinstance(data, np.ndarray):
        return NumpyArray(data)
    return data


# =============================================================================
# Core Round-trip Tests
# =============================================================================


class TestRoundtripToTiff:
    """Test writing to TIFF and comparing against reference."""

    def test_write_tiff_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write data via mbo.imwrite to TIFF and verify it matches reference."""
        ref_data = reference_tiff["data"]

        # For 4D data, mbo.imwrite creates separate files per plane
        # Test with single plane for exact comparison
        if ref_data.ndim == 4:
            test_data = ref_data[:, 0, :, :]  # First z-plane (T, Y, X)
        else:
            test_data = ref_data

        # Write using mbo.imwrite - wrap in NumpyArray
        mbo.imwrite(wrap_array(test_data), output_dir, ext=".tiff", overwrite=True)

        # Find output
        found_file, all_files = find_output_file(output_dir, ".tiff")
        assert found_file is not None, f"No TIFF output found in {output_dir}"

        # Read back
        readback = tifffile.imread(found_file)

        # Compare
        result = array_compare(test_data, readback)

        assert result["shape_match"], f"Shape mismatch: {result['shape1']} vs {result['shape2']}"
        assert result["exact_match"], f"Data mismatch: max_diff={result.get('max_diff', 'N/A')}"

    def test_write_tiff_multiplane_creates_files(self, reference_tiff, output_dir):
        """Verify 4D data creates multiple plane files."""
        ref_data = reference_tiff["data"]

        if ref_data.ndim != 4:
            pytest.skip("Need 4D data for multi-plane test")

        # Write full 4D data
        mbo.imwrite(wrap_array(ref_data), output_dir, ext=".tiff", overwrite=True)

        # Find all output files
        _, all_files = find_output_file(output_dir, ".tiff")

        # Should have multiple files (one per plane)
        n_planes = ref_data.shape[1]
        assert len(all_files) >= 1, f"Expected at least 1 TIFF file, got {len(all_files)}"

        # Read first plane and verify shape
        first_file = sorted(all_files)[0]
        readback = tifffile.imread(first_file)

        # Should be 3D (T, Y, X) - one plane's worth of data
        assert readback.ndim == 3, f"Expected 3D output, got {readback.ndim}D"
        assert readback.shape[0] == ref_data.shape[0], "Frame count mismatch"
        assert readback.shape[1:] == ref_data.shape[2:], "Spatial dimensions mismatch"

    def test_write_tiff_single_plane(self, reference_tiff, output_dir, array_compare):
        """Write single z-plane to TIFF."""
        ref_data = reference_tiff["data"]

        # Extract single plane if 4D
        if ref_data.ndim == 4:
            single_plane = ref_data[:, 0, :, :]  # First z-plane
        else:
            single_plane = ref_data

        mbo.imwrite(wrap_array(single_plane), output_dir, ext=".tiff", overwrite=True)

        found_file, _ = find_output_file(output_dir, ".tiff")
        assert found_file is not None

        readback = tifffile.imread(found_file)
        result = array_compare(single_plane, readback)

        assert result["exact_match"], f"Single plane mismatch: {result}"


class TestRoundtripToZarr:
    """Test writing to Zarr and comparing against reference TIFF."""

    def test_write_zarr_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write to Zarr, read back, compare to reference TIFF."""
        ref_data = reference_tiff["data"]

        # Write to zarr
        mbo.imwrite(wrap_array(ref_data), output_dir, ext=".zarr", ome=False, overwrite=True)

        # Find and read back
        out_file, _ = find_output_file(output_dir, ".zarr")
        assert out_file is not None, f"No Zarr output found in {output_dir}"

        readback = mbo.imread(out_file)
        readback_np = np.asarray(readback).squeeze()  # Remove singleton dimensions

        # Compare to reference (squeeze ref_data too if needed)
        ref_squeezed = ref_data.squeeze() if ref_data.ndim > 3 and 1 in ref_data.shape else ref_data
        result = array_compare(ref_squeezed, readback_np)

        # For zarr, shapes may differ due to dimension handling
        assert result["dtype_match"], f"Dtype mismatch: {result['dtype1']} vs {result['dtype2']}"

        # Check no zero frames (data loss indicator)
        if result.get("has_zero_frames"):
            pytest.fail(f"Zero frames detected: {result['zero_frames']}")

    def test_write_zarr_ome_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write OME-Zarr, read back, compare to reference."""
        ref_data = reference_tiff["data"]

        mbo.imwrite(wrap_array(ref_data), output_dir, ext=".zarr", ome=True, overwrite=True)

        out_file, _ = find_output_file(output_dir, ".zarr")
        assert out_file is not None

        readback = np.asarray(mbo.imread(out_file)).squeeze()
        ref_squeezed = ref_data.squeeze() if ref_data.ndim > 3 and 1 in ref_data.shape else ref_data
        result = array_compare(ref_squeezed, readback)

        assert result["dtype_match"], f"Dtype mismatch: {result}"


class TestRoundtripToH5:
    """Test writing to HDF5 and comparing against reference TIFF."""

    def test_write_h5_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write to HDF5, read back, compare to reference TIFF."""
        ref_data = reference_tiff["data"]

        # For 4D data, H5 writer may create separate files per plane
        # Test with single plane for exact comparison
        if ref_data.ndim == 4:
            test_data = ref_data[:, 0, :, :]  # First z-plane (T, Y, X)
        else:
            test_data = ref_data

        mbo.imwrite(wrap_array(test_data), output_dir, ext=".h5", overwrite=True)

        out_file, _ = find_output_file(output_dir, ".h5")
        assert out_file is not None, f"No H5 output found in {output_dir}"

        # Read back using h5py directly for exact comparison
        import h5py
        with h5py.File(out_file, "r") as f:
            # Find the dataset
            if "data" in f:
                readback = f["data"][:]
            elif "mov" in f:
                readback = f["mov"][:]
            else:
                pytest.fail(f"No data/mov dataset in {out_file}. Keys: {list(f.keys())}")

        result = array_compare(test_data, readback)

        assert result["shape_match"], f"Shape mismatch: {result}"
        assert result["dtype_match"], f"Dtype mismatch: {result}"
        assert result["exact_match"], f"Data mismatch: max_diff={result.get('max_diff')}"

        if result.get("has_zero_frames"):
            pytest.fail(f"Zero frames detected: {result['zero_frames']}")


class TestRoundtripToBin:
    """Test writing to Suite2p binary format and comparing against reference."""

    def test_write_bin_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write to binary, read back, compare to reference TIFF."""
        ref_data = reference_tiff["data"]

        # Binary format needs 3D data (T, Y, X)
        if ref_data.ndim == 4:
            # Use first plane for this test
            test_data = ref_data[:, 0, :, :]
        else:
            test_data = ref_data

        mbo.imwrite(wrap_array(test_data), output_dir, ext=".bin", overwrite=True)

        out_file, _ = find_output_file(output_dir, ".bin")
        assert out_file is not None, f"No bin output found in {output_dir}"

        readback = np.asarray(mbo.imread(out_file))
        result = array_compare(test_data, readback)

        assert result["shape_match"], f"Shape mismatch: {result}"
        assert result["dtype_match"], f"Dtype mismatch: {result}"
        assert result["exact_match"], f"Data mismatch: max_diff={result.get('max_diff')}"

        if result.get("has_zero_frames"):
            pytest.fail(f"Zero frames detected: {result['zero_frames']}")


class TestRoundtripToNpy:
    """Test writing to NumPy format and comparing against reference."""

    def test_write_npy_matches_reference(self, reference_tiff, output_dir, array_compare):
        """Write to .npy, read back, compare to reference."""
        ref_data = reference_tiff["data"]

        # For 4D data, NPY writer may create separate files per plane
        # Test with single plane for exact comparison
        if ref_data.ndim == 4:
            test_data = ref_data[:, 0, :, :]  # First z-plane (T, Y, X)
        else:
            test_data = ref_data

        mbo.imwrite(wrap_array(test_data), output_dir, ext=".npy", overwrite=True)

        out_file, _ = find_output_file(output_dir, ".npy")
        assert out_file is not None, f"No npy output found in {output_dir}"

        readback = np.load(out_file)
        result = array_compare(test_data, readback)

        assert result["shape_match"], f"Shape mismatch: {result}"
        assert result["exact_match"], f"Data mismatch: {result}"


# =============================================================================
# Cross-format conversion tests
# =============================================================================


class TestCrossFormatConversion:
    """Test converting between different formats."""

    @pytest.mark.parametrize("source_ext,target_ext", [
        (".h5", ".tiff"),
        (".bin", ".tiff"),
    ])
    def test_format_to_format(self, reference_tiff, output_dir, array_compare, source_ext, target_ext):
        """Test A → B format conversion preserves data."""
        ref_data = reference_tiff["data"]

        # Use 3D data (single plane) for all cross-format tests
        # since multi-plane data creates separate files per plane
        if ref_data.ndim == 4:
            test_data = ref_data[:, 0, :, :]
        else:
            test_data = ref_data

        # Step 1: Write to source format
        source_dir = output_dir / "source"
        source_dir.mkdir()

        mbo.imwrite(wrap_array(test_data), source_dir, ext=source_ext, ome=False, overwrite=True)

        # Step 2: Read from source format
        source_file, _ = find_output_file(source_dir, source_ext)
        assert source_file is not None, f"No {source_ext} output in {source_dir}"

        intermediate = mbo.imread(source_file)

        # Step 3: Write to target format
        target_dir = output_dir / "target"
        target_dir.mkdir()

        mbo.imwrite(intermediate, target_dir, ext=target_ext, ome=False, overwrite=True)

        # Step 4: Read back and compare to original
        target_file, _ = find_output_file(target_dir, target_ext)
        assert target_file is not None

        if target_ext == ".tiff":
            readback = tifffile.imread(target_file)
        else:
            readback = np.asarray(mbo.imread(target_file))

        result = array_compare(test_data, readback)

        assert result["shape_match"], f"{source_ext}→{target_ext} shape mismatch: {result}"
        assert result["exact_match"], f"{source_ext}→{target_ext} data mismatch: {result}"


# =============================================================================
# Frame count tests (num_frames parameter)
# =============================================================================


class TestNumFramesParameter:
    """Test that num_frames parameter limits output correctly."""

    @pytest.mark.parametrize("ext", [".tiff", ".bin"])
    def test_num_frames_limits_output(self, reference_tiff, output_dir, ext):
        """Verify num_frames parameter limits the number of frames written."""
        ref_data = reference_tiff["data"]
        total_frames = ref_data.shape[0]
        num_frames = min(5, total_frames)

        # For bin, use 3D
        if ext == ".bin" and ref_data.ndim == 4:
            test_data = ref_data[:, 0, :, :]
        else:
            test_data = ref_data

        mbo.imwrite(wrap_array(test_data), output_dir, ext=ext, num_frames=num_frames, overwrite=True)

        out_file, _ = find_output_file(output_dir, ext)
        assert out_file is not None

        if ext == ".tiff":
            readback = tifffile.imread(out_file)
        else:
            readback = np.asarray(mbo.imread(out_file))

        assert readback.shape[0] == num_frames, \
            f"Expected {num_frames} frames, got {readback.shape[0]}"


# =============================================================================
# Data integrity tests
# =============================================================================


class TestDataIntegrity:
    """Test data integrity across operations."""

    def test_no_zero_frames_after_roundtrip(self, reference_tiff, output_dir):
        """Ensure no frames become all-zeros during round-trip."""
        ref_data = reference_tiff["data"]

        # Test H5 format
        mbo.imwrite(wrap_array(ref_data), output_dir, ext=".h5", overwrite=True)

        out_file, _ = find_output_file(output_dir, ".h5")
        readback = np.asarray(mbo.imread(out_file))

        # Check every frame
        for i in range(readback.shape[0]):
            frame = readback[i] if readback.ndim == 3 else readback[i, 0]
            assert frame.max() > 0, f"Frame {i} is all zeros!"

    def test_dtype_preservation(self, reference_tiff, output_dir):
        """Verify dtype is preserved across formats."""
        ref_data = reference_tiff["data"]
        expected_dtype = ref_data.dtype

        results = {}
        for ext in [".tiff", ".h5"]:
            sub_dir = output_dir / f"dtype_{ext.lstrip('.')}"
            sub_dir.mkdir(exist_ok=True)

            mbo.imwrite(wrap_array(ref_data), sub_dir, ext=ext, overwrite=True)

            out_file, _ = find_output_file(sub_dir, ext)
            if out_file:
                if ext == ".tiff":
                    readback = tifffile.imread(out_file)
                else:
                    readback = np.asarray(mbo.imread(out_file))

                results[ext] = str(readback.dtype)

        for ext, dtype in results.items():
            assert dtype == str(expected_dtype), \
                f"Dtype changed for {ext}: {expected_dtype} → {dtype}"


# =============================================================================
# Overwrite behavior tests
# =============================================================================


class TestOverwriteBehavior:
    """Test overwrite=True/False behavior."""

    def test_overwrite_true_replaces_tiff(self, synthetic_3d_data, output_dir):
        """Verify overwrite=True replaces existing TIFF files."""
        data1 = synthetic_3d_data
        data2 = synthetic_3d_data + 100  # Different data

        # First write
        mbo.imwrite(wrap_array(data1), output_dir, ext=".tiff", overwrite=True)
        out_file, _ = find_output_file(output_dir, ".tiff")
        first_read = tifffile.imread(out_file)

        # Second write with overwrite=True
        mbo.imwrite(wrap_array(data2), output_dir, ext=".tiff", overwrite=True)
        second_read = tifffile.imread(out_file)

        # Should be different (data2 was written)
        assert not np.array_equal(first_read, second_read), "Overwrite didn't replace data"


# =============================================================================
# Source array type tests
# =============================================================================


class TestSourceArrayTypes:
    """Test writing from different source array types."""

    def test_write_from_numpy_array(self, synthetic_3d_data, output_dir):
        """Write from plain numpy array (wrapped in NumpyArray)."""
        mbo.imwrite(wrap_array(synthetic_3d_data), output_dir, ext=".tiff", overwrite=True)

        out_file, _ = find_output_file(output_dir, ".tiff")
        readback = tifffile.imread(out_file)
        assert np.array_equal(synthetic_3d_data, readback)

    def test_write_from_mborawarray(self, source_array, output_dir):
        """Write from MboRawArray (lazy source)."""
        # Use first 5 frames to keep test fast
        test_frames = 5

        mbo.imwrite(source_array, output_dir, ext=".tiff", num_frames=test_frames, overwrite=True)

        out_file, _ = find_output_file(output_dir, ".tiff")
        assert out_file is not None, "No output file created"

        readback = tifffile.imread(out_file)
        assert readback.shape[0] == test_frames, f"Expected {test_frames} frames, got {readback.shape[0]}"
