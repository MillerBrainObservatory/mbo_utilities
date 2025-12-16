"""
Shared pytest fixtures for mbo_utilities tests.

Test data location: E:/tests/lbm/mbo_utilities/

Usage:
    pytest tests/ -v                    # Run all tests
    pytest tests/test_roundtrip.py -v   # Run specific test file
    KEEP_TEST_OUTPUT=1 pytest tests/    # Keep output files for inspection
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

import mbo_utilities as mbo

TEST_DATA_ROOT = Path("E:/tests/lbm/mbo_utilities")
TEST_INPUT_TIFF = TEST_DATA_ROOT / "test_input.tif"
BASELINE_DIR = TEST_DATA_ROOT / "baselines"
OUTPUT_DIR = TEST_DATA_ROOT / "test_outputs"


@pytest.fixture(scope="session")
def test_data_root():
    """Root directory for test data."""
    if not TEST_DATA_ROOT.exists():
        pytest.skip(f"Test data directory not found: {TEST_DATA_ROOT}")
    return TEST_DATA_ROOT


@pytest.fixture(scope="session")
def source_tiff_path(test_data_root):
    """Path to the source test TIFF file."""
    if not TEST_INPUT_TIFF.exists():
        pytest.skip(f"Test input not found: {TEST_INPUT_TIFF}")
    return TEST_INPUT_TIFF


@pytest.fixture(scope="session")
def source_array(source_tiff_path):
    """
    Load the source test array once per session.

    Returns the MboRawArray with phase correction disabled for consistent baseline.
    """
    arr = mbo.imread(source_tiff_path, fix_phase=False)
    return arr


@pytest.fixture(scope="session")
def source_metadata(source_tiff_path):
    """Load metadata from source TIFF."""
    return mbo.get_metadata(source_tiff_path)


@pytest.fixture(scope="session")
def source_data_subset(source_array):
    """
    A smaller subset of source data for faster tests.

    Returns (data, shape_info) where shape_info describes the subset.
    """
    arr = source_array

    # Take first 10 frames, all planes
    if arr.ndim == 4:
        n_frames = min(10, arr.shape[0])
        n_planes = arr.shape[1]
        subset = np.asarray(arr[:n_frames])
        shape_info = {
            "n_frames": n_frames,
            "n_planes": n_planes,
            "height": arr.shape[2],
            "width": arr.shape[3],
            "original_shape": arr.shape,
        }
    elif arr.ndim == 3:
        n_frames = min(10, arr.shape[0])
        subset = np.asarray(arr[:n_frames])
        shape_info = {
            "n_frames": n_frames,
            "n_planes": 1,
            "height": arr.shape[1],
            "width": arr.shape[2],
            "original_shape": arr.shape,
        }
    else:
        subset = np.asarray(arr)
        shape_info = {"original_shape": arr.shape}

    return subset, shape_info


@pytest.fixture(scope="session")
def baseline_dir(test_data_root):
    """Directory for baseline reference files."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINE_DIR


@pytest.fixture(scope="session")
def reference_tiff_path(baseline_dir):
    """Path to the reference TIFF (golden baseline)."""
    return baseline_dir / "reference.tiff"


@pytest.fixture(scope="session")
def reference_tiff(reference_tiff_path, source_data_subset):
    """
    The reference TIFF file - canonical source of truth.

    If it doesn't exist, creates it from source_data_subset.
    All other format tests compare against this.
    """
    import tifffile

    subset, shape_info = source_data_subset

    if not reference_tiff_path.exists():
        print(f"\nCreating reference TIFF: {reference_tiff_path}")
        print(f"  Shape: {subset.shape}, dtype: {subset.dtype}")

        # Write reference TIFF directly using tifffile for exact control
        tifffile.imwrite(
            reference_tiff_path,
            subset,
            photometric='minisblack',
            metadata={'axes': 'TZYX' if subset.ndim == 4 else 'TYX'},
        )
        print(f"  Created: {reference_tiff_path}")

    # Load and return the reference data
    ref_data = tifffile.imread(reference_tiff_path)

    return {
        "path": reference_tiff_path,
        "data": ref_data,
        "shape": ref_data.shape,
        "dtype": ref_data.dtype,
        "shape_info": shape_info,
    }


@pytest.fixture
def output_dir(request, test_data_root):
    """
    Unique output directory for each test.

    Creates: E:/tests/lbm/mbo_utilities/test_outputs/<test_name>/
    Cleaned up after test unless KEEP_TEST_OUTPUT=1
    """
    import os

    test_name = request.node.name
    out_dir = OUTPUT_DIR / test_name

    # Clean previous run
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    yield out_dir

    # Cleanup unless explicitly kept
    if os.environ.get("KEEP_TEST_OUTPUT", "0") != "1":
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.fixture
def temp_array(source_data_subset):
    """
    Fresh copy of test data subset for modification.

    Use this when the test might modify the data.
    """
    subset, shape_info = source_data_subset
    return subset.copy(), shape_info


@pytest.fixture
def synthetic_3d_data():
    """
    Synthetic 3D test data (T, Y, X) - single z-plane.

    Deterministic, reproducible, fast to generate.
    """
    rng = np.random.RandomState(42)
    shape = (20, 128, 128)

    # Base noise
    data = rng.normal(500, 100, shape).astype(np.int16)

    # Add some structure (simulated cells)
    for i in range(5):
        cy, cx = rng.randint(20, 108, size=2)
        yy, xx = np.ogrid[:128, :128]
        mask = ((yy - cy)**2 + (xx - cx)**2) < 100

        # Temporal signal
        for t in range(20):
            signal = 200 + 100 * np.sin(2 * np.pi * t / 10 + i)
            data[t][mask] += int(signal)

    return data.clip(0, 4095).astype(np.int16)


@pytest.fixture
def synthetic_4d_data():
    """
    Synthetic 4D test data (T, Z, Y, X) - multi z-plane.

    Deterministic, reproducible, fast to generate.
    """
    rng = np.random.RandomState(42)
    shape = (10, 3, 64, 64)  # 10 frames, 3 planes, 64x64

    # Base noise
    data = rng.normal(500, 100, shape).astype(np.int16)

    # Add structure per plane
    for z in range(3):
        for i in range(3):
            cy, cx = rng.randint(10, 54, size=2)
            yy, xx = np.ogrid[:64, :64]
            mask = ((yy - cy)**2 + (xx - cx)**2) < 50

            for t in range(10):
                # Z-dependent signal
                z_factor = 1.0 - 0.2 * abs(z - 1)  # Peak at middle plane
                signal = (200 + 100 * np.sin(2 * np.pi * t / 5 + i)) * z_factor
                data[t, z][mask] += int(signal)

    return data.clip(0, 4095).astype(np.int16)


@pytest.fixture
def sample_metadata():
    """Sample metadata dict for testing metadata preservation."""
    return {
        "frame_rate": 30.0,
        "pixel_resolution": [0.5, 0.5],
        "num_rois": 1,
        "num_planes": 14,
        "fov_px": [512, 512],
        "experimenter": "test_user",
        "acquisition_date": "2025-01-15",
        "dz": 5.0,
    }


@pytest.fixture
def array_compare():
    """Fixture to access compare_arrays helper."""
    return compare_arrays


@pytest.fixture
def correlation_check():
    """Fixture to access compute_frame_correlation helper."""
    return compute_frame_correlation


def compare_arrays(arr1, arr2, rtol=1e-5, atol=0.5):
    """
    Compare two arrays, handling lazy arrays and different shapes.

    Returns dict with comparison results.
    """
    # Materialize if needed
    if hasattr(arr1, 'compute'):
        arr1 = arr1.compute()
    if hasattr(arr2, 'compute'):
        arr2 = arr2.compute()

    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    result = {
        "shape1": arr1.shape,
        "shape2": arr2.shape,
        "dtype1": str(arr1.dtype),
        "dtype2": str(arr2.dtype),
        "shape_match": arr1.shape == arr2.shape,
        "dtype_match": arr1.dtype == arr2.dtype,
    }

    if result["shape_match"]:
        diff = np.abs(arr1.astype(np.float64) - arr2.astype(np.float64))
        result["max_diff"] = float(diff.max())
        result["mean_diff"] = float(diff.mean())
        result["exact_match"] = np.array_equal(arr1, arr2)
        result["close_match"] = np.allclose(arr1, arr2, rtol=rtol, atol=atol)

        # Check for zero frames (data loss indicator)
        if arr2.ndim >= 3:
            zero_frames = [i for i in range(arr2.shape[0]) if arr2[i].max() == 0]
            result["zero_frames"] = zero_frames
            result["has_zero_frames"] = len(zero_frames) > 0

    return result


def compute_frame_correlation(arr1, arr2, num_frames=10):
    """
    Compute per-frame correlation between two arrays.

    Returns dict with correlation statistics.
    """
    from scipy.stats import pearsonr

    arr1 = np.asarray(arr1).astype(np.float32)
    arr2 = np.asarray(arr2).astype(np.float32)

    if arr1.shape != arr2.shape:
        return {"error": f"Shape mismatch: {arr1.shape} vs {arr2.shape}"}

    # Select frames to test
    n_frames = min(num_frames, arr1.shape[0])
    indices = np.linspace(0, arr1.shape[0] - 1, n_frames, dtype=int)

    correlations = []
    for idx in indices:
        if arr1.ndim == 4:
            # Test middle z-plane
            z = arr1.shape[1] // 2
            f1 = arr1[idx, z].flatten()
            f2 = arr2[idx, z].flatten()
        else:
            f1 = arr1[idx].flatten()
            f2 = arr2[idx].flatten()

        corr, _ = pearsonr(f1, f2)
        correlations.append(corr)

    correlations = np.array(correlations)

    return {
        "mean": float(correlations.mean()),
        "min": float(correlations.min()),
        "max": float(correlations.max()),
        "all_above_99": bool(correlations.min() > 0.99),
        "all_above_999": bool(correlations.min() > 0.999),
        "num_tested": n_frames,
    }


def find_output_file(output_dir, ext):
    """
    Find the output file(s) in a directory after imwrite.

    Handles the various output structures:
    - Single file: output.zarr, output.h5
    - Directory with files: planeXX/*.tif, planeXX/data_raw.bin
    - Nested zarr: output/*.zarr

    Returns (primary_file_path, all_files_list)
    """
    output_dir = Path(output_dir)
    ext_clean = ext.lower().lstrip(".")

    if ext_clean in ("tif", "tiff"):
        # Look for TIFF files
        files = list(output_dir.rglob("*.tif")) + list(output_dir.rglob("*.tiff"))
        if files:
            return files[0], files
        return None, []

    elif ext_clean == "zarr":
        # Look for zarr stores
        zarr_dirs = list(output_dir.rglob("*.zarr"))
        if zarr_dirs:
            return zarr_dirs[0], zarr_dirs
        # Maybe it's output_dir itself
        if (output_dir / ".zarray").exists() or (output_dir / "zarr.json").exists():
            return output_dir, [output_dir]
        return None, []

    elif ext_clean == "bin":
        # Look for binary files with ops.npy
        bin_files = list(output_dir.rglob("data_raw.bin"))
        if bin_files:
            # Return parent directory (the plane directory)
            return bin_files[0].parent, bin_files
        return None, []

    elif ext_clean in ("h5", "hdf5"):
        files = list(output_dir.rglob("*.h5")) + list(output_dir.rglob("*.hdf5"))
        if files:
            return files[0], files
        return None, []

    elif ext_clean == "npy":
        files = list(output_dir.rglob("*.npy"))
        # Filter out ops.npy
        files = [f for f in files if f.name != "ops.npy"]
        if files:
            return files[0], files
        return None, []

    return None, []


# Make find_output_file available as a fixture too
@pytest.fixture
def find_output():
    """Fixture to access find_output_file helper."""
    return find_output_file


_test_results = []


def pytest_sessionfinish(session, exitstatus):
    """Print summary after all tests complete."""
    if _test_results:
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in _test_results if r.get("passed", False))
        failed = len(_test_results) - passed

        print(f"Total tests with results: {len(_test_results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in _test_results:
                if not r.get("passed", False):
                    print(f"  - {r.get('name', 'unknown')}: {r.get('error', 'unknown error')}")

        print("=" * 70)
