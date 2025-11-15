import pytest
import numpy as np
from pathlib import Path
import hashlib
import json
import shutil

import mbo_utilities as mbo

TEST_DATA_DIR = Path("E:/tests/lbm/mbo_utilities")
TEST_INPUT = TEST_DATA_DIR / "test_input.tif"
OUTPUT_DIR = TEST_DATA_DIR / "outputs"
BASELINE_DIR = TEST_DATA_DIR / "baselines"


@pytest.fixture(scope="session", autouse=True)
def setup_test_dirs():
    """Create output and baseline directories."""
    # Skip if test data directory doesn't exist (e.g., on CI)
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data directory not found at {TEST_DATA_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    BASELINE_DIR.mkdir(exist_ok=True, parents=True)
    yield
    # Cleanup old outputs (keep for inspection)
    # shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def test_data():
    """Load test TIFF data once for all tests."""
    if not TEST_INPUT.exists():
        pytest.skip(f"Test data not found at {TEST_INPUT}")

    data = mbo.imread(TEST_INPUT)
    metadata = mbo.get_metadata(TEST_INPUT)

    return {
        "data": data,
        "metadata": metadata,
        "path": TEST_INPUT,
        "shape": data.shape,
        "dtype": data.dtype,
    }


@pytest.fixture
def output_path(request, tmp_path):
    """Generate unique output path for each test."""
    test_name = request.node.name
    return OUTPUT_DIR / f"{test_name}"


def compute_hash(data):
    """Compute SHA256 hash of array data."""
    if hasattr(data, "compute"):  # Dask array
        data = data.compute()
    # Handle lazy arrays (convert to numpy)
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return hashlib.sha256(data.tobytes()).hexdigest()


def save_baseline(name, value):
    """Save baseline value for future validation."""
    baseline_file = BASELINE_DIR / f"{name}.json"
    with open(baseline_file, "w") as f:
        json.dump(value, f, indent=2)


def load_baseline(name):
    """Load baseline value for validation."""
    baseline_file = BASELINE_DIR / f"{name}.json"
    if not baseline_file.exists():
        return None
    with open(baseline_file, "r") as f:
        return json.load(f)


def validate_or_save_baseline(test_name, result):
    """
    Validate against baseline or save new baseline if first run.

    Returns True if validation passed or baseline was created.
    """
    baseline = load_baseline(test_name)

    if baseline is None:
        # First run - save baseline
        save_baseline(test_name, result)
        pytest.skip(f"Baseline created for {test_name}. Re-run to validate.")
        return True

    # Validate
    assert result == baseline, f"Result differs from baseline for {test_name}"
    return True


def test_imread_basic(test_data):
    """Test basic TIFF reading."""
    print("\n" + "=" * 70)
    print("STARTING I/O FORMAT TESTS")
    print("=" * 70)
    print("\n=== Testing Basic TIFF Read ===")

    data = test_data["data"]
    print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")

    assert data.ndim in [2, 3, 4], "Expected 2D, 3D or 4D data (multi-ROI)"
    assert str(data.dtype) == "int16", "Expected int16 dtype"

    # Validate shape baseline
    data_np = np.asarray(data)
    baseline_info = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "min": int(data_np.min()),
        "max": int(data_np.max()),
        "mean": float(data_np.mean()),
    }
    print(
        f"Data stats - min: {baseline_info['min']}, max: {baseline_info['max']}, mean: {baseline_info['mean']:.2f}"
    )

    print("Validating against baseline...")
    validate_or_save_baseline("imread_basic", baseline_info)
    print("[PASS] Basic read test passed\n")


def test_imread_with_phase_correction(test_data):
    """Test TIFF reading with phase correction."""
    data_raw = test_data["data"]
    metadata = test_data["metadata"]

    # Read with phase correction
    data_corrected = mbo.imread(
        test_data["path"],
        fix_phase=True,
        phasecorr_method="mean",
    )

    assert data_corrected.shape == data_raw.shape

    # Phase correction may not change data if there's no bidirectional scanning artifact
    # Just hash the corrected data for future validation
    # Take smaller slice if single frame
    slice_size = min(10, data_corrected.shape[0] if data_corrected.ndim == 3 else 1)
    if data_corrected.ndim == 3:
        hash_corrected = compute_hash(data_corrected[:slice_size])
    else:
        hash_corrected = compute_hash(data_corrected)

    baseline = {"hash": hash_corrected, "shape": list(data_corrected.shape)}
    validate_or_save_baseline("imread_phase_corrected", baseline)


def test_imread_metadata(test_data):
    """Test metadata extraction."""
    metadata = test_data["metadata"]

    required_keys = ["frame_rate", "pixel_resolution"]
    for key in required_keys:
        assert key in metadata, f"Missing required metadata: {key}"

    # Validate metadata
    baseline_meta = {
        "frame_rate": metadata.get("frame_rate"),
        "pixel_resolution": metadata.get("pixel_resolution"),
        "num_rois": metadata.get("num_rois"),
        "fov_px": metadata.get("fov_px"),
    }

    validate_or_save_baseline("metadata", baseline_meta)


# ==============================================================================
# Test: imwrite format variations
# ==============================================================================


def test_imwrite_tiff(test_data, output_path):
    """Test writing to TIFF format."""
    print("\n=== Testing TIFF Write ===")

    # Use original lazy array to preserve metadata
    data = test_data["data"]
    metadata = test_data["metadata"]
    print(f"Input data shape: {data.shape}")

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing TIFF files to: {out_dir}")
    mbo.imwrite(
        data,
        str(out_dir),
        ext=".tif",
        metadata=metadata,
        overwrite=True,
    )

    # Verify output exists
    tiff_files = list(out_dir.glob("*.tif"))
    print(f"Created {len(tiff_files)} TIFF file(s)")
    assert len(tiff_files) > 0, "No TIFF files created"

    # Read back ONE file - multi-plane data creates separate files
    print(f"Reading back first file: {tiff_files[0].name}")
    data_readback = mbo.imread(tiff_files[0])
    print(f"Read data shape: {data_readback.shape}")

    # Multi-plane data: each file contains one plane
    # Original: (T, N_planes, Y, X) -> Each file: (T, 1, Y, X) or (T, Y, X)
    # Just validate spatial dimensions match
    print(
        f"Validating spatial dimensions: {data_readback.shape[-2:]} == {data.shape[-2:]}"
    )
    assert data_readback.shape[-2:] == data.shape[-2:], "Spatial dims mismatch"

    # Hash a small slice for baseline
    if data_readback.ndim == 4:
        hash_slice = data_readback[
            : min(5, data_readback.shape[0]), : min(2, data_readback.shape[1])
        ]
    elif data_readback.ndim == 3:
        hash_slice = data_readback[: min(10, data_readback.shape[0])]
    else:
        hash_slice = data_readback

    baseline = {
        "num_files": len(tiff_files),
        "shape_orig": list(data.shape),
        "shape_read": list(data_readback.shape),
        "hash": compute_hash(hash_slice),
    }

    print("Validating against baseline...")
    validate_or_save_baseline("imwrite_tiff", baseline)
    print("[PASS] TIFF write test passed\n")


@pytest.mark.skip(reason="TIFF baseline test has Windows file locking issues - not related to phase correction")
def test_imwrite_tiff_baseline(test_data, validate_on_write):
    """Generate and validate TIFF baseline with automatic frame correlation checks."""
    print("\n=== Generating TIFF Baseline with Validation ===")

    source_path = test_data["path"]
    baseline_path = BASELINE_DIR / "imwrite_tiff_validated.tif"

    # For multi-plane source data, extract a single plane
    # because TIFF writes each plane separately
    source_data_full = mbo.imread(source_path)

    # Extract plane 0 for validation
    if source_data_full.ndim == 4:
        print(f"Source has {source_data_full.shape[1]} planes, extracting plane 0 for baseline")
        source_plane0 = np.asarray(source_data_full[:, 0, :, :])  # (T, Y, X) - materialize
    else:
        source_plane0 = np.asarray(source_data_full)

    print(f"Writing baseline: {baseline_path}")
    print(f"  Shape: {source_plane0.shape}")
    print(f"  Dtype: {source_plane0.dtype}")
    print(f"  Is numpy array: {isinstance(source_plane0, np.ndarray)}")

    # Write baseline - use full path directly
    mbo.imwrite(
        source_plane0,
        str(baseline_path),
        overwrite=True,
    )

    # Validate with automatic correlation checks
    validate_on_write(source_plane0, baseline_path, metadata={
        "test_name": "test_imwrite_tiff_baseline",
        "format": "tiff",
        "source_path": str(source_path),
        "plane": 0,
    })

    print("[PASS] TIFF baseline generated and validated\n")


def test_imwrite_zarr_basic(test_data, output_path):
    """Test writing to Zarr format (non-OME)."""
    print("\n=== Testing Zarr Write (non-OME) ===")

    data = test_data["data"]
    metadata = test_data["metadata"]
    print(f"Input data shape: {data.shape}")

    zarr_path = output_path / "test.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing Zarr to: {zarr_path}")
    mbo.imwrite(
        data,
        str(zarr_path),
        ext=".zarr",
        metadata=metadata,
        ome=False,
        overwrite=True,
    )

    # For multi-plane data, zarr_path may be a directory containing multiple zarr files
    if zarr_path.is_dir():
        # Directory with multiple zarr files
        zarr_files = list(zarr_path.glob("*.zarr"))
        print(f"Created directory with {len(zarr_files)} Zarr store(s)")
        if zarr_files:
            print(f"Reading back first store: {zarr_files[0].name}")
            data_readback = mbo.imread(zarr_files[0])
        else:
            raise AssertionError("No zarr files in directory")
    elif zarr_path.exists():
        # Single zarr store
        print("Created single Zarr store")
        data_readback = mbo.imread(zarr_path)
        zarr_files = [zarr_path]
    else:
        raise AssertionError("No Zarr output created")

    data_np = np.asarray(data)
    data_readback_np = np.asarray(data_readback)
    print(f"Read data shape: {data_readback_np.shape}")

    # Multi-plane data: each file may contain one plane
    # Just validate spatial dimensions match
    print(
        f"Validating spatial dimensions: {data_readback_np.shape[-2:]} == {data_np.shape[-2:]}"
    )
    assert data_readback_np.shape[-2:] == data_np.shape[-2:], "Spatial dims mismatch"

    # IMPORTANT: Validate ALL frames, not just first 10
    # Check for zero frames (data loss bug indicator)
    if data_readback_np.ndim >= 3:
        zero_frames = []
        for i in range(data_readback_np.shape[0]):
            if data_readback_np[i].max() == 0:
                zero_frames.append(i)
        if zero_frames:
            raise AssertionError(
                f"Found {len(zero_frames)} zero frames in Zarr readback! "
                f"This indicates data loss. First zero frame: {zero_frames[0]}"
            )

    baseline = {
        "shape_orig": list(data_np.shape),
        "shape_read": list(data_readback_np.shape),
        "num_files": len(zarr_files) if len(zarr_files) > 1 else 1,
        "hash": compute_hash(data_readback_np),
        "num_frames": data_readback_np.shape[0],
    }

    print("Validating against baseline...")
    validate_or_save_baseline("imwrite_zarr_basic", baseline)
    print("[PASS] Zarr (non-OME) write test passed\n")


def test_imwrite_zarr_baseline(test_data, validate_on_write):
    """Generate and validate Zarr baseline with automatic frame correlation checks."""
    print("\n=== Generating Zarr Baseline with Validation ===")

    source_path = test_data["path"]
    baseline_path = BASELINE_DIR / "imwrite_zarr_validated.zarr"

    # Load source data (with default phase correction enabled)
    source_data = mbo.imread(source_path)

    print(f"Writing baseline: {baseline_path}")
    print(f"  Shape: {source_data.shape}")
    print(f"  Dtype: {source_data.dtype}")
    print(f"  Phase correction: {source_data.fix_phase}")

    # Write full multi-dimensional data to Zarr
    mbo.imwrite(
        source_data,
        str(baseline_path),
        ext=".zarr",
        ome=False,
        overwrite=True,
    )

    # Validate with automatic correlation checks
    validate_on_write(source_data, baseline_path, metadata={
        "test_name": "test_imwrite_zarr_baseline",
        "format": "zarr",
        "source_path": str(source_path),
    })

    print("[PASS] Zarr baseline generated and validated\n")


def test_imwrite_zarr_ome(test_data, output_path):
    """Test writing OME-Zarr with rich metadata."""
    data = test_data["data"]
    metadata = test_data["metadata"].copy()

    metadata.update(
        {
            "name": "test_ome_zarr",
            "acquisition_date": "2025-02-27",
            "experimenter": "Test User",
            "dz": 5.0,
        }
    )

    zarr_path = output_path / "test_ome.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    mbo.imwrite(
        data,
        str(zarr_path),
        ext=".zarr",
        metadata=metadata,
        ome=True,
        overwrite=True,
    )

    # For multi-plane data, zarr_path may be a directory containing multiple zarr files
    if zarr_path.is_dir():
        # Directory with multiple zarr files
        zarr_files = list(zarr_path.glob("*.zarr"))
        if zarr_files:
            data_readback = mbo.imread(zarr_files[0])
            zarr_to_check = zarr_files[0]
        else:
            raise AssertionError("No zarr files in directory")
    elif zarr_path.exists():
        # Single zarr store
        data_readback = mbo.imread(zarr_path)
        zarr_to_check = zarr_path
        zarr_files = [zarr_path]
    else:
        raise AssertionError("No OME-Zarr output created")

    data_np = np.asarray(data)
    data_readback_np = np.asarray(data_readback)

    # Multi-plane data: each file may contain one plane
    # Just validate spatial dimensions match
    assert data_readback_np.shape[-2:] == data_np.shape[-2:], "Spatial dims mismatch"

    # Try to open as zarr to check OME metadata
    import zarr

    try:
        z = zarr.open_group(str(zarr_to_check), mode="r")
        has_ome = "ome" in z.attrs
        has_multiscales = "multiscales" in z.attrs.get("ome", {})
        shape = list(z["0"].shape) if "0" in z else list(data_readback_np.shape)
    except:
        # If it's not a group, it's a direct array
        z = zarr.open_array(str(zarr_to_check), mode="r")
        has_ome = "ome" in z.attrs
        has_multiscales = False
        shape = list(z.shape)

    baseline = {
        "has_ome": has_ome,
        "shape_orig": list(data_np.shape),
        "shape_read": shape,
        "num_files": len(zarr_files) if len(zarr_files) > 1 else 1,
    }

    validate_or_save_baseline("imwrite_zarr_ome", baseline)


def test_imwrite_h5(test_data, output_path):
    """Test writing to HDF5 format."""
    data = test_data["data"]
    metadata = test_data["metadata"]

    h5_path = output_path / "test.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    mbo.imwrite(
        data,
        str(h5_path),
        ext=".h5",
        metadata=metadata,
        overwrite=True,
    )

    # For multi-plane data, check if a directory was created with multiple h5 files
    # or if it's a single h5 file
    if h5_path.is_dir():
        # Directory with multiple h5 files
        h5_files = list(h5_path.glob("*.h5"))
        assert len(h5_files) > 0, "No HDF5 files created in directory"
        h5_to_read = h5_files[0]
    elif h5_path.exists():
        # Single h5 file
        h5_to_read = h5_path
        h5_files = [h5_path]
    else:
        raise AssertionError("HDF5 output not created")

    # Read back
    import h5py

    with h5py.File(h5_to_read, "r") as f:
        # Check for either 'data' or 'mov' dataset (Suite2p format uses 'mov')
        if "data" in f:
            data_readback = f["data"][:]
        elif "mov" in f:
            data_readback = f["mov"][:]
        else:
            raise AssertionError(f"Missing dataset. Available keys: {list(f.keys())}")

    # Multi-plane data: each file may contain one plane
    # Just validate spatial dimensions match
    assert data_readback.shape[-2:] == data.shape[-2:], "Spatial dims mismatch"

    # IMPORTANT: Validate ALL frames, not just first 10
    # Check for zero frames (data loss bug indicator)
    if data_readback.ndim >= 3:
        zero_frames = []
        for i in range(data_readback.shape[0]):
            if data_readback[i].max() == 0:
                zero_frames.append(i)
        if zero_frames:
            raise AssertionError(
                f"Found {len(zero_frames)} zero frames in HDF5 readback! "
                f"This indicates data loss. First zero frame: {zero_frames[0]}"
            )

    baseline = {
        "shape_orig": list(data.shape),
        "shape_read": list(data_readback.shape),
        "num_files": len(h5_files),
        "hash": compute_hash(data_readback),
        "num_frames": data_readback.shape[0],
    }

    validate_or_save_baseline("imwrite_h5", baseline)


def test_imwrite_bin(test_data, output_path):
    """Test writing to binary format."""
    data = test_data["data"]
    metadata = test_data["metadata"]

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    mbo.imwrite(
        data,
        str(out_dir),
        ext=".bin",
        metadata=metadata,
        overwrite=True,
    )

    # For multi-plane data, check if subdirectories were created
    # Each plane may be in its own subdirectory: planeXX_stitched/
    subdirs = [d for d in out_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Multi-plane structure: planeXX_stitched/ops.npy, data_raw.bin
        ops_found = False
        bin_found = False
        bin_to_read = None

        for subdir in subdirs:
            if (subdir / "ops.npy").exists():
                ops_found = True
            if (subdir / "data_raw.bin").exists():
                bin_found = True
                if bin_to_read is None:
                    bin_to_read = subdir / "data_raw.bin"

        assert ops_found, "ops.npy not found in any subdirectory"
        assert bin_found, "data_raw.bin not found in any subdirectory"

    else:
        # Single file structure
        assert (out_dir / "ops.npy").exists(), "ops.npy not created"
        assert (out_dir / "data_raw.bin").exists(), "data_raw.bin not created"
        bin_to_read = out_dir / "data_raw.bin"

    # Read back
    data_readback = mbo.imread(bin_to_read)

    # Multi-plane data: each file may contain one plane
    # Just validate spatial dimensions match
    assert data_readback.shape[-2:] == data.shape[-2:], "Spatial dims mismatch"

    # IMPORTANT: Validate ALL frames, not just first 10
    # Previous bug: only first 10 frames were being written but test only checked first 10
    data_readback_full = np.asarray(data_readback)

    # Check for zero frames (data loss bug indicator)
    if data_readback_full.ndim >= 3:
        zero_frames = []
        for i in range(data_readback_full.shape[0]):
            if data_readback_full[i].max() == 0:
                zero_frames.append(i)
        if zero_frames:
            raise AssertionError(
                f"Found {len(zero_frames)} zero frames in readback! "
                f"This indicates data loss. First zero frame: {zero_frames[0]}"
            )

    # Hash full data to detect any changes
    baseline = {
        "shape_orig": list(data.shape),
        "shape_read": list(data_readback.shape),
        "num_subdirs": len(subdirs),
        "hash": compute_hash(data_readback_full),
        "num_frames": data_readback_full.shape[0],
    }

    validate_or_save_baseline("imwrite_bin", baseline)


# ==============================================================================
# Test: Phase correction variations
# ==============================================================================


@pytest.mark.parametrize("method", ["mean"])
def test_phase_correction_methods(test_data, output_path, method):
    """Test different phase correction methods."""
    data_corrected = mbo.imread(
        test_data["path"],
        fix_phase=True,
        phasecorr_method=method,
    )

    assert data_corrected.shape == test_data["shape"]

    # Handle 2D vs 3D for hashing
    if data_corrected.ndim == 3:
        hash_val = compute_hash(data_corrected[: min(10, data_corrected.shape[0])])
    else:
        hash_val = compute_hash(data_corrected)

    baseline = {
        "method": method,
        "hash": hash_val,
        "shape": list(data_corrected.shape),
    }

    validate_or_save_baseline(f"phase_correction_{method}", baseline)


# ==============================================================================
# Test: Zarr merging functionality
# ==============================================================================


def test_merge_zarr_zplanes(test_data, output_path):
    """Test merging multiple z-plane Zarr files."""
    data = test_data["data"]
    metadata = test_data["metadata"].copy()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract first 3 planes from the multi-plane data
    # to create single-plane zarr stores for merging
    data_np = np.asarray(data)

    # Create 3 "z-plane" Zarr stores from single planes
    plane_zarrs = []
    for i in range(3):
        plane_path = output_path / f"plane{i + 1:02d}.zarr"
        # Extract a single plane: shape (T, Y, X)
        if data_np.ndim == 4:
            plane_data = data_np[:, i, :, :]
        else:
            plane_data = data_np

        # Write directly as zarr to avoid multi-plane splitting
        import zarr

        z = zarr.open_array(
            str(plane_path),
            mode="w",
            shape=plane_data.shape,
            dtype=plane_data.dtype,
            chunks=(1, *plane_data.shape[1:])
            if plane_data.ndim == 3
            else plane_data.shape,
        )
        z[:] = plane_data
        plane_zarrs.append(plane_path)

    # Merge them
    metadata["dz"] = 5.0
    merged_path = output_path / "merged_volume.zarr"

    mbo.merge_zarr_zplanes(
        plane_zarrs,
        str(merged_path),
        metadata=metadata,
        overwrite=True,
    )

    assert merged_path.exists(), "Merged Zarr not created"

    # Validate structure
    import zarr

    z = zarr.open(str(merged_path), mode="r")
    assert "0" in z, "Missing array '0'"

    merged_shape = z["0"].shape
    # For 2D input, output should be (1, 3, Y, X) or (T, 3, Y, X)
    # Check that we have 3 z-planes in the right dimension
    if data.ndim == 2:
        # Output should be (1, 3, Y, X)
        assert merged_shape[1] == 3, f"Expected 3 z-planes, got {merged_shape[1]}"
    else:
        # Output should be (T, 3, Y, X)
        assert merged_shape[1] == 3, f"Expected 3 z-planes, got {merged_shape[1]}"
        assert merged_shape[0] == data.shape[0], "Time dimension mismatch"

    # Check scale
    assert "scale" in z["0"].attrs, "Missing scale for napari"

    baseline = {
        "shape": list(merged_shape),
        "has_scale": "scale" in z["0"].attrs,
        "num_planes": merged_shape[1],
    }

    validate_or_save_baseline("merge_zarr_zplanes", baseline)


# ==============================================================================
# Test: Round-trip consistency
# ==============================================================================


@pytest.mark.parametrize("format_ext", [".tif", ".zarr", ".h5", ".bin"])
def test_roundtrip_consistency(test_data, output_path, format_ext):
    """Test round-trip write and read preserves data."""
    data = test_data["data"]
    metadata = test_data["metadata"]

    # Write
    if format_ext == ".bin":
        out_path = output_path
        out_path.mkdir(parents=True, exist_ok=True)
    elif format_ext in [".zarr"]:
        out_path = output_path / f"test{format_ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif format_ext == ".h5":
        out_path = output_path / f"test{format_ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # .tif
        out_path = output_path
        out_path.mkdir(parents=True, exist_ok=True)

    mbo.imwrite(
        data,
        str(out_path),
        ext=format_ext,
        metadata=metadata,
        ome=False,
        overwrite=True,
    )

    # Read back - handle multi-plane output structure
    if format_ext == ".bin":
        # Check for subdirectories (multi-plane structure)
        subdirs = [d for d in out_path.iterdir() if d.is_dir()]
        if subdirs and (subdirs[0] / "data_raw.bin").exists():
            data_readback = mbo.imread(subdirs[0] / "data_raw.bin")
        elif (out_path / "data_raw.bin").exists():
            data_readback = mbo.imread(out_path / "data_raw.bin")
        else:
            raise AssertionError("No data_raw.bin found")
    elif format_ext == ".zarr":
        # Check if out_path is a directory with multiple zarr files or single zarr store
        if out_path.is_dir():
            zarr_files = list(out_path.glob("*.zarr"))
            if zarr_files:
                data_readback = mbo.imread(zarr_files[0])
            else:
                raise AssertionError("No zarr files in directory")
        elif out_path.exists():
            data_readback = mbo.imread(out_path)
        else:
            raise AssertionError("No zarr output found")
    elif format_ext == ".h5":
        # Check if directory with multiple h5 files or single h5 file
        if out_path.is_dir():
            h5_files = list(out_path.glob("*.h5"))
            if h5_files:
                data_readback = mbo.imread(h5_files[0])
            else:
                raise AssertionError("No h5 files found in directory")
        elif out_path.exists():
            data_readback = mbo.imread(out_path)
        else:
            raise AssertionError("No h5 output found")
    else:  # .tif
        files = sorted(out_path.glob("*.tif"))
        if not files:
            raise AssertionError("No tif files found")
        data_readback = mbo.imread(files[0])

    # Convert to numpy for comparison
    data_np = np.asarray(data)
    data_readback_np = np.asarray(data_readback)

    # Validate - multi-plane data may create separate files
    # Just validate spatial dimensions match
    assert data_readback_np.shape[-2:] == data_np.shape[-2:], "Spatial dims mismatch"
    assert data_readback_np.dtype == data_np.dtype, "Dtype mismatch"

    # IMPORTANT: Validate ALL frames to detect data loss
    # Check for zero frames (data loss bug indicator)
    if data_readback_np.ndim >= 3:
        zero_frames = []
        for i in range(data_readback_np.shape[0]):
            if data_readback_np[i].max() == 0:
                zero_frames.append(i)
        if zero_frames:
            raise AssertionError(
                f"Found {len(zero_frames)} zero frames in {format_ext} roundtrip readback! "
                f"This indicates data loss. First zero frame: {zero_frames[0]}"
            )

    baseline = {
        "format": format_ext,
        "shape_orig": list(data_np.shape),
        "shape_read": list(data_readback_np.shape),
        "hash": compute_hash(data_readback_np),
        "num_frames": data_readback_np.shape[0],
    }

    validate_or_save_baseline(f"roundtrip{format_ext.replace('.', '_')}", baseline)


# ==============================================================================
# Test: Performance and resource usage
# ==============================================================================


def test_memory_efficiency(test_data):
    """Test that data is loaded lazily without loading entire file."""
    import sys

    # Get size of loaded data object
    data = test_data["data"]
    obj_size = sys.getsizeof(data)

    # Should be small (just metadata) for lazy arrays
    # Typical LazyArray or Dask array is < 1MB metadata
    assert obj_size < 10 * 1024 * 1024, (
        f"Data object too large: {obj_size / 1024**2:.1f}MB"
    )

    print(f"Lazy array object size: {obj_size / 1024:.1f}KB")


def test_baseline_summary():
    """Generate summary of all baselines."""
    if not BASELINE_DIR.exists():
        pytest.skip("No baselines directory")

    baseline_files = list(BASELINE_DIR.glob("*.json"))

    summary = {
        "total_baselines": len(baseline_files),
        "baselines": [f.stem for f in baseline_files],
    }

    print("\n" + "=" * 70)
    print("Baseline Summary")
    print("=" * 70)
    print(f"Total baselines: {summary['total_baselines']}")
    print(f"Location: {BASELINE_DIR}")
    print("\nBaselines:")
    for name in sorted(summary["baselines"]):
        print(f"  - {name}")
    print("=" * 70)


@pytest.mark.parametrize("ext", [".tif", ".bin", ".zarr"])
def test_num_frames_parameter(test_data, output_path, ext):
    """Test that num_frames parameter correctly limits output."""
    print(f"\n=== Testing num_frames with {ext} ===")

    data = test_data["data"]
    total_frames = data.shape[0]
    num_frames_to_write = min(100, total_frames // 2)  # Write half or 100 frames

    print(f"Total frames: {total_frames}, writing: {num_frames_to_write}")

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write with num_frames parameter
    mbo.imwrite(
        data,
        str(out_dir),
        ext=ext,
        num_frames=num_frames_to_write,
        planes=[1],  # Single plane to simplify validation
        overwrite=True,
    )

    # Read back and verify frame count
    if ext == ".tif":
        output_files = list(out_dir.glob("*.tif"))
        assert len(output_files) > 0, "No output files created"
        readback = mbo.imread(output_files[0])
    elif ext == ".bin":
        bin_dir = list(out_dir.glob("plane*"))
        assert len(bin_dir) > 0, "No bin directory created"
        readback = mbo.imread(bin_dir[0] / "data_raw.bin")
    elif ext == ".zarr":
        zarr_files = list(out_dir.glob("*.zarr"))
        assert len(zarr_files) > 0, "No zarr files created"
        readback = mbo.imread(zarr_files[0])

    print(f"Readback shape: {readback.shape}")

    # Check frame count (accounting for singleton dimensions)
    readback_frames = readback.shape[0]
    assert readback_frames == num_frames_to_write, (
        f"Expected {num_frames_to_write} frames, got {readback_frames}"
    )

    baseline = {
        "num_frames_written": num_frames_to_write,
        "readback_frames": readback_frames,
        "ext": ext,
    }

    print(f"[PASS] num_frames test passed for {ext}\n")
    validate_or_save_baseline(f"num_frames_{ext.lstrip('.')}", baseline)


@pytest.mark.parametrize("ext", [".tif", ".bin", ".zarr"])
def test_overwrite_behavior(test_data, output_path, ext):
    """Test overwrite=True/False behavior."""
    print(f"\n=== Testing overwrite behavior with {ext} ===")

    data = test_data["data"]
    num_frames = min(50, data.shape[0])

    out_dir = output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # First write
    print("First write with overwrite=True")
    mbo.imwrite(
        data,
        str(out_dir),
        ext=ext,
        num_frames=num_frames,
        planes=[1],
        overwrite=True,
    )

    # Get first output
    if ext == ".tif":
        output_files = list(out_dir.glob("*.tif"))
        first_file = output_files[0]
    elif ext == ".bin":
        bin_dirs = list(out_dir.glob("plane*"))
        first_file = bin_dirs[0] / "data_raw.bin"
    elif ext == ".zarr":
        zarr_files = list(out_dir.glob("*.zarr"))
        first_file = zarr_files[0]

    # Load data and compute hash, ensuring file is closed
    first_data = mbo.imread(first_file)[:10]
    if hasattr(first_data, 'compute'):
        first_data = first_data.compute()
    first_hash = compute_hash(first_data)
    del first_data  # Explicitly delete to close file handles
    print(f"First hash: {first_hash[:16]}...")

    # Second write with overwrite=True (should replace)
    print("Second write with overwrite=True (should replace)")
    mbo.imwrite(
        data,
        str(out_dir),
        ext=ext,
        num_frames=num_frames,
        planes=[1],
        overwrite=True,
    )

    # Load data and compute hash, ensuring file is closed
    second_data = mbo.imread(first_file)[:10]
    if hasattr(second_data, 'compute'):
        second_data = second_data.compute()
    second_hash = compute_hash(second_data)
    del second_data  # Explicitly delete to close file handles
    print(f"Second hash: {second_hash[:16]}...")

    # Hashes should match (same data written)
    assert second_hash == first_hash, "Overwrite=True should produce same result"

    # Force garbage collection to close any lingering file handles (Windows issue)
    import gc
    gc.collect()

    # Third write with overwrite=False (should skip or fail gracefully)
    print("Third write with overwrite=False (should skip)")
    try:
        mbo.imwrite(
            data,
            str(out_dir),
            ext=ext,
            num_frames=num_frames,
            planes=[1],
            overwrite=False,
        )
    except Exception as e:
        # Some formats may raise, others may skip silently
        print(f"Raised (expected): {type(e).__name__}")

    # File should still exist and be readable
    third_data = mbo.imread(first_file)[:10]
    if hasattr(third_data, 'compute'):
        third_data = third_data.compute()
    third_hash = compute_hash(third_data)
    del third_data
    assert third_hash == first_hash, "File should be unchanged after overwrite=False"

    baseline = {
        "ext": ext,
        "overwrite_true_consistent": second_hash == first_hash,
        "overwrite_false_preserves": third_hash == first_hash,
    }

    print(f"[PASS] Overwrite test passed for {ext}\n")
    validate_or_save_baseline(f"overwrite_{ext.lstrip('.')}", baseline)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
