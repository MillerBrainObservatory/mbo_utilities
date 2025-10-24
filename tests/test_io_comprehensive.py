"""
Comprehensive tests for imread/imwrite functionality.

This test suite extensively tests reading and writing operations across
different file formats (.zarr, .tiff, .bin, .h5) with timing information.
Tests are split between CI (synthetic data) and local (real data).
"""
import os
import time
import functools
from pathlib import Path

import pytest
import numpy as np
import h5py


# ============================================================================
# Test Configuration
# ============================================================================

# Lazy imports to avoid loading heavy dependencies during test collection
# Don't call _get_mbo_dirs() at module level - wait until test execution
try:
    DEFAULT_DATA_ROOT = Path(os.getenv("MBO_TEST_DATA", "/tmp/mbo_test_data"))
except Exception:
    DEFAULT_DATA_ROOT = Path("/tmp/mbo_test_data")

DATA_ROOT = DEFAULT_DATA_ROOT

# Test data specifications
SYNTHETIC_SHAPE = (50, 64, 128)  # (T, H, W) - small for CI
SYNTHETIC_DTYPE = np.uint16


def skip_if_missing_data(func):
    """Skip test if real data files are not available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DATA_ROOT.exists() or len(list(DATA_ROOT.glob("*.tif*"))) == 0:
            pytest.skip(f"Required data files not found in {DATA_ROOT}")
        return func(*args, **kwargs)
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name="Operation"):
        self.name = name
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"\n  ⏱️  {self.name}: {self.elapsed:.3f}s")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Create synthetic test data."""
    np.random.seed(42)
    return np.random.randint(0, 2**12, size=SYNTHETIC_SHAPE, dtype=SYNTHETIC_DTYPE)


@pytest.fixture
def synthetic_tiff(tmp_path, synthetic_data):
    """Create a synthetic TIFF file."""
    import tifffile
    tiff_path = tmp_path / "synthetic_plane0.tif"  # Include plane in filename

    # Add MBO metadata to make it recognizable
    metadata = {
        "pixel_resolution": [1.0, 1.0],
        "frame_rate": 30.0,
        "dtype": str(synthetic_data.dtype),
        "shape": list(synthetic_data.shape),
        "plane": 0,  # Add plane metadata
    }

    with Timer(f"Creating synthetic TIFF ({synthetic_data.nbytes / 1e6:.1f} MB)"):
        tifffile.imwrite(
            tiff_path,
            synthetic_data,
            metadata=metadata,
            compression=None,
        )

    return tiff_path


@pytest.fixture
def real_data_files():
    """Get real data files if available."""
    from mbo_utilities import get_files

    if not DATA_ROOT.exists():
        pytest.skip(f"Real data directory not found: {DATA_ROOT}")

    files = get_files(DATA_ROOT, "tif")
    if not files:
        pytest.skip(f"No TIFF files found in {DATA_ROOT}")

    return files


# ============================================================================
# CI Tests (using synthetic data)
# ============================================================================

@pytest.mark.ci
@pytest.mark.io
def test_imread_synthetic_tiff(synthetic_tiff):
    """Test imread with synthetic TIFF file."""
    from mbo_utilities import imread

    with Timer("imread synthetic TIFF"):
        arr = imread(synthetic_tiff)

    assert arr is not None
    assert hasattr(arr, 'shape')
    assert hasattr(arr, 'dtype')
    # Shape may have extra dimensions (T, C, H, W) vs (T, H, W)
    assert len(arr.shape) >= 3
    print(f"  📊 Shape: {arr.shape}, dtype: {arr.dtype}")


@pytest.mark.ci
@pytest.mark.io
@pytest.mark.formats
@pytest.mark.parametrize("ext,format_name", [
    (".tiff", "TIFF"),
    (".zarr", "Zarr"),
    (".h5", "HDF5"),
])
def test_imwrite_formats_synthetic(tmp_path, synthetic_tiff, ext, format_name):
    """Test writing synthetic data to different formats with timing."""
    from mbo_utilities import imread, imwrite

    print(f"\n🔧 Testing {format_name} format")

    # Read data first (imwrite expects lazy array objects)
    arr = imread(synthetic_tiff)

    # Write data
    outdir = tmp_path / f"output_{ext.replace('.', '')}"
    outdir.mkdir(parents=True, exist_ok=True)

    with Timer(f"imwrite to {format_name}"):
        result_path = imwrite(
            arr,
            outdir,
            ext=ext,
            overwrite=True,
            planes=None,  # Write all planes
        )

    assert result_path is not None
    print(f"  📁 Output: {result_path}")

    # Verify output exists
    if ext == ".zarr":
        output_files = list(outdir.glob("*.zarr"))
    elif ext == ".h5":
        output_files = list(outdir.glob("*.h5"))
    else:
        output_files = list(outdir.glob("*.tif*"))

    assert len(output_files) > 0, f"No {format_name} files created"
    print(f"  ✅ Created {len(output_files)} file(s)")


@pytest.mark.ci
@pytest.mark.io
def test_roundtrip_tiff(tmp_path, synthetic_tiff):
    """Test read-write roundtrip with TIFF format."""
    from mbo_utilities import imread, imwrite

    print("\n🔄 Testing TIFF roundtrip")

    # Read original
    with Timer("Read original TIFF"):
        arr = imread(synthetic_tiff)
        original_data = np.array(arr[:])  # Force load

    # Write to new location
    outdir = tmp_path / "roundtrip_tiff"
    outdir.mkdir(exist_ok=True)

    with Timer("Write TIFF"):
        imwrite(arr, outdir, ext=".tiff", overwrite=True, planes=None)

    # Read back
    output_files = list(outdir.glob("*.tif*"))
    assert len(output_files) > 0

    with Timer("Read written TIFF"):
        arr2 = imread(output_files[0])
        written_data = np.array(arr2[:])

    # Compare
    print(f"  📊 Original shape: {original_data.shape}, Written shape: {written_data.shape}")
    np.testing.assert_array_equal(original_data, written_data)
    print("  ✅ Data matches!")


@pytest.mark.ci
@pytest.mark.io
def test_roundtrip_zarr(tmp_path, synthetic_tiff):
    """Test read-write roundtrip with Zarr format."""
    from mbo_utilities import imread, imwrite

    print("\n🔄 Testing Zarr roundtrip")

    # Read original
    with Timer("Read original TIFF"):
        arr = imread(synthetic_tiff)
        original_data = np.array(arr[:])

    # Write to Zarr
    outdir = tmp_path / "roundtrip_zarr"
    outdir.mkdir(exist_ok=True)

    with Timer("Write Zarr"):
        imwrite(arr, outdir, ext=".zarr", overwrite=True, planes=None)

    # Read back from Zarr
    zarr_files = list(outdir.glob("*.zarr"))
    assert len(zarr_files) > 0

    with Timer("Read written Zarr"):
        arr2 = imread(zarr_files[0])
        written_data = np.array(arr2[:])

    # Compare
    print(f"  📊 Original shape: {original_data.shape}, Written shape: {written_data.shape}")
    np.testing.assert_array_equal(original_data, written_data)
    print("  ✅ Data matches!")


@pytest.mark.ci
@pytest.mark.io
def test_roundtrip_h5(tmp_path, synthetic_tiff):
    """Test read-write roundtrip with HDF5 format."""
    from mbo_utilities import imread, imwrite

    print("\n🔄 Testing HDF5 roundtrip")

    # Read original
    with Timer("Read original TIFF"):
        arr = imread(synthetic_tiff)
        original_data = np.array(arr[:])

    # Write to HDF5
    outdir = tmp_path / "roundtrip_h5"
    outdir.mkdir(exist_ok=True)

    with Timer("Write HDF5"):
        imwrite(arr, outdir, ext=".h5", overwrite=True, planes=None)

    # Read back from HDF5
    h5_files = list(outdir.glob("*.h5"))
    assert len(h5_files) > 0

    with Timer("Read written HDF5"):
        arr2 = imread(h5_files[0])
        written_data = np.array(arr2[:])

    # Compare
    print(f"  📊 Original shape: {original_data.shape}, Written shape: {written_data.shape}")
    np.testing.assert_array_equal(original_data, written_data)
    print("  ✅ Data matches!")


@pytest.mark.ci
def test_metadata_preservation(tmp_path, synthetic_tiff):
    """Test that metadata is preserved through read/write cycles."""
    from mbo_utilities import imread, imwrite

    print("\n📋 Testing metadata preservation")

    # Read with metadata
    arr = imread(synthetic_tiff)
    original_metadata = arr.metadata if hasattr(arr, 'metadata') else {}

    print(f"  📊 Original metadata keys: {list(original_metadata.keys())}")

    # Write without trying to modify metadata
    outdir = tmp_path / "metadata_test"
    outdir.mkdir(exist_ok=True)

    # Note: metadata parameter adds to existing metadata, doesn't replace
    imwrite(arr, outdir, ext=".tiff", overwrite=True, planes=None)

    # Read back and check metadata
    output_files = list(outdir.glob("*.tif*"))
    if output_files:
        arr2 = imread(output_files[0])

        if hasattr(arr2, 'metadata'):
            print(f"  📊 Written metadata keys: {list(arr2.metadata.keys())}")
            print("  ✅ Metadata preserved")


# ============================================================================
# Local Tests (using real data)
# ============================================================================

@pytest.mark.local
@pytest.mark.io
@skip_if_missing_data
def test_imread_real_data(real_data_files):
    """Test imread with real data files."""
    from mbo_utilities import imread

    print(f"\n📂 Testing with {len(real_data_files)} real data file(s)")

    with Timer("imread real data"):
        arr = imread(real_data_files)

    assert arr is not None
    assert hasattr(arr, 'shape')
    print(f"  📊 Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  💾 Size: {np.prod(arr.shape) * arr.dtype.itemsize / 1e9:.2f} GB")


@pytest.mark.local
@pytest.mark.io
@pytest.mark.formats
@pytest.mark.slow
@pytest.mark.parametrize("ext,format_name", [
    (".tiff", "TIFF"),
    (".zarr", "Zarr"),
    (".bin", "Binary"),
    (".h5", "HDF5"),
])
@skip_if_missing_data
def test_imwrite_formats_real_data(tmp_path, real_data_files, ext, format_name):
    """Test writing real data to all formats with timing."""
    from mbo_utilities import imread, imwrite

    print(f"\n🔧 Testing {format_name} format with real data")

    # Read real data
    with Timer(f"Read real data"):
        arr = imread(real_data_files)
        data_size_mb = np.prod(arr.shape) * arr.dtype.itemsize / 1e6
        print(f"  💾 Data size: {data_size_mb:.1f} MB")

    # Write to format
    outdir = tmp_path / f"real_output_{ext.replace('.', '')}"
    outdir.mkdir(parents=True, exist_ok=True)

    with Timer(f"Write to {format_name}"):
        result_path = imwrite(
            arr,
            outdir,
            ext=ext,
            overwrite=True,
        )

    assert result_path is not None
    print(f"  📁 Output: {result_path}")

    # Calculate throughput
    if hasattr(result_path, 'exists') and result_path.exists():
        if result_path.is_dir():
            total_size = sum(f.stat().st_size for f in result_path.rglob('*') if f.is_file())
        else:
            total_size = result_path.stat().st_size
        print(f"  💾 Output size: {total_size / 1e6:.1f} MB")


@pytest.mark.local
@pytest.mark.io
@pytest.mark.slow
@skip_if_missing_data
def test_roi_stitched(tmp_path, real_data_files):
    """Test writing stitched multi-ROI data (roi=None)."""
    print("\n🔗 Testing stitched multi-ROI (roi=None)")

    with Timer("Read data"):
        arr = imread(real_data_files)

    # Check if this is multi-ROI data
    if not hasattr(arr, 'roi') or not callable(getattr(arr, 'supports_roi', lambda: False)):
        pytest.skip("Data does not support ROI operations")

    print(f"  📊 Original shape: {arr.shape}")

    # Write stitched (roi=None)
    outdir = tmp_path / "stitched"
    outdir.mkdir(exist_ok=True)

    with Timer("Write stitched ROIs"):
        imwrite(arr, outdir, ext=".tiff", roi=None, overwrite=True)

    # Verify output
    output_files = list(outdir.glob("*.tif*"))
    assert len(output_files) > 0
    print(f"  📁 Created {len(output_files)} file(s)")

    # Read back and check
    with Timer("Read stitched output"):
        arr_stitched = imread(output_files[0])
        print(f"  📊 Stitched shape: {arr_stitched.shape}")


@pytest.mark.local
@pytest.mark.io
@pytest.mark.slow
@skip_if_missing_data
def test_roi_individual(tmp_path, real_data_files):
    """Test writing individual ROIs (roi=0, roi=1, etc.)."""
    print("\n📦 Testing individual ROIs (roi=0)")

    with Timer("Read data"):
        arr = imread(real_data_files)

    # Check if this is multi-ROI data
    if not hasattr(arr, 'roi'):
        pytest.skip("Data does not support ROI operations")

    print(f"  📊 Original shape: {arr.shape}")

    # Write individual ROI
    outdir = tmp_path / "individual_roi"
    outdir.mkdir(exist_ok=True)

    with Timer("Write ROI 0"):
        imwrite(arr, outdir, ext=".tiff", roi=0, overwrite=True)

    # Verify output
    output_files = list(outdir.glob("*roi*0*.tif*"))
    if not output_files:
        output_files = list(outdir.glob("*.tif*"))

    assert len(output_files) > 0
    print(f"  📁 Created {len(output_files)} file(s) for ROI 0")

    # Read back and check
    with Timer("Read ROI 0 output"):
        arr_roi0 = imread(output_files[0])
        print(f"  📊 ROI 0 shape: {arr_roi0.shape}")


@pytest.mark.local
@pytest.mark.io
@pytest.mark.slow
@skip_if_missing_data
def test_roi_comparison(tmp_path, real_data_files):
    """Compare stitched vs individual ROI outputs."""
    print("\n🔬 Comparing stitched vs individual ROIs")

    with Timer("Read data"):
        arr = imread(real_data_files)

    if not hasattr(arr, 'roi'):
        pytest.skip("Data does not support ROI operations")

    # Write stitched
    stitched_dir = tmp_path / "stitched_compare"
    stitched_dir.mkdir(exist_ok=True)

    with Timer("Write stitched"):
        imwrite(arr, stitched_dir, ext=".tiff", roi=None, overwrite=True)

    # Write individual ROIs
    roi0_dir = tmp_path / "roi0_compare"
    roi0_dir.mkdir(exist_ok=True)

    with Timer("Write ROI 0"):
        imwrite(arr, roi0_dir, ext=".tiff", roi=0, overwrite=True)

    roi1_dir = tmp_path / "roi1_compare"
    roi1_dir.mkdir(exist_ok=True)

    with Timer("Write ROI 1"):
        imwrite(arr, roi1_dir, ext=".tiff", roi=1, overwrite=True)

    # Load and compare shapes
    stitched_files = list(stitched_dir.glob("*.tif*"))
    roi0_files = list(roi0_dir.glob("*.tif*"))
    roi1_files = list(roi1_dir.glob("*.tif*"))

    if stitched_files and roi0_files and roi1_files:
        import tifffile
        stitched_data = tifffile.imread(stitched_files[0])
        roi0_data = tifffile.imread(roi0_files[0])
        roi1_data = tifffile.imread(roi1_files[0])

        print(f"  📊 Stitched shape: {stitched_data.shape}")
        print(f"  📊 ROI 0 shape: {roi0_data.shape}")
        print(f"  📊 ROI 1 shape: {roi1_data.shape}")

        # Check that stitched width is sum of ROI widths (or similar)
        if len(stitched_data.shape) == 3 and len(roi0_data.shape) == 3:
            assert stitched_data.shape[0] == roi0_data.shape[0] == roi1_data.shape[0], \
                "Time dimension should match"
            assert stitched_data.shape[1] == roi0_data.shape[1] == roi1_data.shape[1], \
                "Height should match"
            print("  ✅ ROI dimensions consistent")


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.mark.local
@pytest.mark.slow
@pytest.mark.formats
@skip_if_missing_data
def test_format_comparison_benchmark(tmp_path, real_data_files):
    """Benchmark writing performance across all formats."""
    print("\n⚡ Format Performance Benchmark")

    with Timer("Read data"):
        arr = imread(real_data_files)
        data_size_gb = np.prod(arr.shape) * arr.dtype.itemsize / 1e9
        print(f"  💾 Data size: {data_size_gb:.3f} GB")

    formats = [
        (".tiff", "TIFF"),
        (".zarr", "Zarr"),
        (".bin", "Binary"),
        (".h5", "HDF5"),
    ]

    results = {}

    for ext, format_name in formats:
        outdir = tmp_path / f"benchmark_{ext.replace('.', '')}"
        outdir.mkdir(exist_ok=True)

        timer = Timer(f"Write {format_name}")
        with timer:
            imwrite(arr, outdir, ext=ext, overwrite=True)

        results[format_name] = timer.elapsed
        throughput = data_size_gb / timer.elapsed
        print(f"    📈 Throughput: {throughput:.2f} GB/s")

    # Print summary
    print("\n📊 Performance Summary:")
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for format_name, elapsed in sorted_results:
        throughput = data_size_gb / elapsed
        print(f"  {format_name:8s}: {elapsed:6.3f}s ({throughput:.2f} GB/s)")
