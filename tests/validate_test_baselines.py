"""
Auto-detect and validate test baselines.

Finds baseline JSON files and validates the data they reference.
"""

import numpy as np
from pathlib import Path
import mbo_utilities as mbo
from scipy.stats import pearsonr
import json


def find_baseline_files(baselines_dir):
    """Find all baseline JSON files."""
    if not baselines_dir.exists():
        print(f"Baselines directory not found: {baselines_dir}")
        return []

    json_files = list(baselines_dir.glob("*.json"))
    print(f"Found {len(json_files)} baseline files in {baselines_dir}")
    return json_files


def load_baseline_metadata(json_path):
    """Load baseline metadata from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def validate_frame_correlation(source_data, written_data, num_test_frames=50):
    """
    Check correlation between source and written frames.

    Handles different dimensionalities:
    - 3D: (T, Y, X) - single plane
    - 4D: (T, Z, Y, X) - multi-plane, validates each plane separately
    """
    print(f"\n  Frame correlation check:")
    print(f"    Source shape: {source_data.shape}")
    print(f"    Written shape: {written_data.shape}")

    # Ensure shapes are compatible
    if source_data.shape != written_data.shape:
        print(f"    FAIL Shape mismatch - cannot validate")
        return {"status": "shape_mismatch"}

    # Determine test strategy based on dimensionality
    if len(source_data.shape) == 3:
        # Single plane: (T, Y, X)
        num_frames = min(num_test_frames, source_data.shape[0])
        test_indices = np.random.choice(source_data.shape[0], size=num_frames, replace=False)

        correlations = []
        for idx in test_indices:
            src = source_data[idx].astype(np.float32).flatten()
            dst = written_data[idx].astype(np.float32).flatten()
            corr, _ = pearsonr(src, dst)
            correlations.append(corr)

        correlations = np.array(correlations)

        print(f"    Tested {num_frames} frames")
        print(f"    Correlation: mean={correlations.mean():.6f}, min={correlations.min():.6f}")

        if correlations.min() > 0.99:
            print(f"    All frames have high correlation (>0.99)")
            status = "passed"
        else:
            print(f"    Low correlation detected!")
            status = "failed"

        return {
            "status": status,
            "correlation_mean": float(correlations.mean()),
            "correlation_min": float(correlations.min()),
        }

    elif len(source_data.shape) == 4:
        # Multi-plane: (T, Z, Y, X)
        num_frames_per_plane = min(10, source_data.shape[0])  # Test fewer per plane
        all_correlations = []

        for z_idx in range(source_data.shape[1]):
            test_indices = np.random.choice(source_data.shape[0], size=num_frames_per_plane, replace=False)

            for idx in test_indices:
                src = source_data[idx, z_idx].astype(np.float32).flatten()
                dst = written_data[idx, z_idx].astype(np.float32).flatten()
                corr, _ = pearsonr(src, dst)
                all_correlations.append(corr)

        all_correlations = np.array(all_correlations)

        print(f"    Tested {len(all_correlations)} frames across {source_data.shape[1]} planes")
        print(f"    Correlation: mean={all_correlations.mean():.6f}, min={all_correlations.min():.6f}")

        if all_correlations.min() > 0.99:
            print(f"    All frames have high correlation (>0.99)")
            status = "passed"
        else:
            print(f"    Low correlation detected!")
            status = "failed"

        return {
            "status": status,
            "correlation_mean": float(all_correlations.mean()),
            "correlation_min": float(all_correlations.min()),
        }

    else:
        print(f"    FAIL Unsupported dimensionality: {len(source_data.shape)}")
        return {"status": "unsupported"}


def validate_baseline_data(baseline_path, metadata):
    """
    Validate a baseline data file.

    Args:
        baseline_path: Path to the baseline data file
        metadata: Metadata dict from JSON
    """
    print(f"\n{'='*60}")
    print(f"Validating: {baseline_path.name}")
    print(f"{'='*60}")

    results = {
        "file": str(baseline_path),
        "checks": {}
    }

    # Check if file exists
    if not baseline_path.exists():
        print(f"  File does not exist!")
        results["status"] = "missing"
        return results

    # Load the data
    try:
        print(f"\n  Loading baseline data...")
        data = mbo.imread(baseline_path)
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
    except Exception as e:
        print(f"  Failed to load: {e}")
        results["status"] = "load_failed"
        results["error"] = str(e)
        return results

    # Check shape
    expected_shape = tuple(metadata.get("shape", []))
    if expected_shape:
        if data.shape == expected_shape:
            print(f"  Shape matches expected: {expected_shape}")
            results["checks"]["shape"] = "passed"
        else:
            print(f"  Shape mismatch!")
            print(f"    Expected: {expected_shape}")
            print(f"    Got: {data.shape}")
            results["checks"]["shape"] = "failed"
    else:
        print(f"  No expected shape in metadata")
        results["checks"]["shape"] = "skipped"

    # Check dtype
    expected_dtype = metadata.get("dtype")
    if expected_dtype:
        # Handle numpy dtype string representations
        actual_dtype_str = str(data.dtype)
        if actual_dtype_str == expected_dtype or np.dtype(actual_dtype_str) == np.dtype(expected_dtype):
            print(f"  Dtype matches expected: {expected_dtype}")
            results["checks"]["dtype"] = "passed"
        else:
            print(f"  FAIL Dtype mismatch!")
            print(f"    Expected: {expected_dtype}")
            print(f"    Got: {actual_dtype_str}")
            results["checks"]["dtype"] = "failed"
    else:
        print(f"  WARNING No expected dtype in metadata")
        results["checks"]["dtype"] = "skipped"

    # If source_path is available in metadata, validate frame correlation
    source_path_str = metadata.get("source_path")
    if source_path_str:
        source_path = Path(source_path_str)
        if source_path.exists():
            print(f"\n  Loading source data for correlation check: {source_path}")
            try:
                source_data = mbo.imread(source_path)
                corr_results = validate_frame_correlation(source_data, data)
                results["checks"]["correlation"] = corr_results
            except Exception as e:
                print(f"  FAIL Correlation check failed: {e}")
                results["checks"]["correlation"] = {"status": "error", "error": str(e)}
        else:
            print(f"  WARNING Source path not found: {source_path}")
            results["checks"]["correlation"] = {"status": "source_missing"}
    else:
        print(f"  WARNING No source path in metadata - skipping correlation check")
        results["checks"]["correlation"] = {"status": "skipped"}

    # Overall status
    all_passed = all(
        v == "passed" or (isinstance(v, dict) and v.get("status") == "passed")
        for v in results["checks"].values()
        if v not in ["skipped", {"status": "skipped"}]
    )

    if all_passed:
        results["status"] = "passed"
        print(f"\n  PASS ALL CHECKS PASSED")
    else:
        results["status"] = "failed"
        print(f"\n  FAIL SOME CHECKS FAILED")

    return results


def main():
    """Main validation routine."""
    import sys

    print("="*80)
    print("BASELINE VALIDATION")
    print("="*80)

    # Allow custom baseline path via command line argument
    if len(sys.argv) > 1:
        baselines_dir = Path(sys.argv[1])
        print(f"\nUsing custom baselines directory: {baselines_dir}")
    else:
        # Try common locations
        possible_paths = [
            Path("E:/tests/lbm/mbo_utilities/baselines"),
            Path(__file__).parent / "lbm" / "mbo_utilities" / "baselines",
        ]

        baselines_dir = None
        for path in possible_paths:
            if path.exists():
                baselines_dir = path
                print(f"\nFound baselines directory: {baselines_dir}")
                break

        if baselines_dir is None:
            print(f"\nBaselines directory not found in common locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nUsage: python validate_test_baselines.py [path_to_baselines]")
            return

    if not baselines_dir.exists():
        print(f"\nBaselines directory not found: {baselines_dir}")
        print("Please update the path in this script.")
        return

    # Find all baseline JSON files
    json_files = find_baseline_files(baselines_dir)

    if not json_files:
        print("\nNo baseline files found!")
        return

    # Validate each baseline
    all_results = []

    for json_file in json_files:
        print(f"\n\nProcessing: {json_file.name}")

        # Load metadata
        try:
            metadata = load_baseline_metadata(json_file)
        except Exception as e:
            print(f"  FAIL Failed to load metadata: {e}")
            all_results.append({
                "file": str(json_file),
                "status": "metadata_error",
                "error": str(e)
            })
            continue

        # Determine baseline data file path
        # Assume baseline data has same name as JSON but different extension
        baseline_name = json_file.stem

        # Try to find the actual data file
        # Common extensions: .tif, .tiff, .zarr, .h5, .bin
        data_file = None
        for ext in ['.tif', '.tiff', '.zarr', '.h5', '.bin']:
            candidate = baselines_dir / f"{baseline_name}{ext}"
            if candidate.exists():
                data_file = candidate
                break

        if data_file is None:
            # Check if metadata has a path
            data_path_str = metadata.get("path") or metadata.get("baseline_path")
            if data_path_str:
                data_file = Path(data_path_str)

        if data_file is None or not data_file.exists():
            print(f"  FAIL Could not find baseline data file for {baseline_name}")
            all_results.append({
                "file": str(json_file),
                "status": "data_file_not_found"
            })
            continue

        # Validate the baseline
        results = validate_baseline_data(data_file, metadata)
        all_results.append(results)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for r in all_results if r.get("status") == "passed")
    failed = sum(1 for r in all_results if r.get("status") == "failed")
    other = len(all_results) - passed - failed

    print(f"\nTotal baselines: {len(all_results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Other: {other}")

    if failed > 0:
        print(f"\nWARNING {failed} baseline(s) failed validation!")
        print("  Review the output above for details.")
        print("  DO NOT regenerate baselines until issues are fixed.")
    elif passed == len(all_results):
        print(f"\nPASS All baselines passed validation!")
        print("  It is safe to regenerate baselines if needed.")
    else:
        print(f"\nWARNING Some baselines could not be validated.")
        print("  Review the output above for details.")

    # Save detailed results
    results_file = baselines_dir.parent / "baseline_validation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
