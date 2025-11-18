"""
Validate test baseline data integrity.

This script checks that:
1. Raw data is properly loaded and shaped
2. Data written to baselines has correct deinterleaving (TZ -> T, Z)
3. Frames are in the correct order (high correlation with source)
4. Dimensions match expected values when reading back
5. No frames are shuffled, duplicated, or missing

Run this before regenerating baselines to ensure the pipeline is correct.
"""

import numpy as np
from pathlib import Path
import mbo_utilities as mbo
from scipy.stats import pearsonr
import json


def validate_shape_deinterleaving(source_shape, written_shape, num_planes):
    """
    Validate that TZ data was properly deinterleaved to T, Z format.

    Source should be (T*Z, Y, X) or (T*Z, Z, Y, X)
    Written should be (T, Y, X) for single plane or (T, Z, Y, X) for multi-plane
    """
    print(f"\nShape validation:")
    print(f"  Source shape: {source_shape}")
    print(f"  Written shape: {written_shape}")
    print(f"  Num planes: {num_planes}")

    # Check if source is multi-plane
    if len(source_shape) == 4:
        # Source is (T, Z, Y, X) - already deinterleaved
        expected_t = source_shape[0]
        expected_z = source_shape[1]
        print(f"  Source format: (T={expected_t}, Z={expected_z}, Y, X)")
    else:
        # Source is (T*Z, Y, X) - needs deinterleaving
        expected_t = source_shape[0] // num_planes
        expected_z = num_planes
        print(f"  Source format: (T*Z={source_shape[0]}, Y, X) -> T={expected_t}, Z={expected_z}")

    # Check written shape
    if len(written_shape) == 4:
        # Multi-plane output
        assert written_shape[0] == expected_t, f"T mismatch: {written_shape[0]} != {expected_t}"
        assert written_shape[1] == expected_z, f"Z mismatch: {written_shape[1]} != {expected_z}"
        print(f"  PASS Multi-plane output correct: (T={written_shape[0]}, Z={written_shape[1]}, Y, X)")
    elif len(written_shape) == 3:
        # Single plane output
        assert written_shape[0] == expected_t, f"T mismatch: {written_shape[0]} != {expected_t}"
        print(f"  PASS Single plane output correct: (T={written_shape[0]}, Y, X)")
    else:
        raise ValueError(f"Unexpected written shape: {written_shape}")

    return True


def validate_frame_order(source_data, written_data, num_planes=None, plane_idx=0, num_test_frames=50):
    """
    Validate that frames are in correct order by checking correlations.

    Args:
        source_data: Original data array
        written_data: Written data array
        num_planes: Number of z-planes (if source is interleaved TZ format)
        plane_idx: Which plane to validate (for multi-plane data)
        num_test_frames: Number of frames to test
    """
    print(f"\nFrame order validation:")
    print(f"  Testing {num_test_frames} random frames...")

    # Determine how to index source data
    if len(source_data.shape) == 4:
        # Source is (T, Z, Y, X) - already deinterleaved
        source_frames = source_data[:, plane_idx]
        print(f"  Source format: (T, Z, Y, X) - using plane {plane_idx}")
    elif len(source_data.shape) == 3:
        if num_planes is not None and num_planes > 1:
            # Source is (T*Z, Y, X) - interleaved, need to extract every Zth frame
            # For plane_idx, get frames at indices: plane_idx, plane_idx + Z, plane_idx + 2*Z, ...
            num_frames = source_data.shape[0] // num_planes
            frame_indices = [plane_idx + i * num_planes for i in range(num_frames)]
            source_frames = source_data[frame_indices]
            print(f"  Source format: (T*Z, Y, X) - extracting every {num_planes}th frame starting at {plane_idx}")
            print(f"  Extracted {len(frame_indices)} frames for plane {plane_idx}")
        else:
            # Single plane data
            source_frames = source_data
            print(f"  Source format: (T, Y, X) - single plane")
    else:
        raise ValueError(f"Unexpected source shape: {source_data.shape}")

    # Determine how to index written data
    if len(written_data.shape) == 4:
        # Written is (T, Z, Y, X)
        written_frames = written_data[:, plane_idx]
        print(f"  Written format: (T, Z, Y, X) - using plane {plane_idx}")
    elif len(written_data.shape) == 3:
        # Written is (T, Y, X)
        written_frames = written_data
        print(f"  Written format: (T, Y, X)")
    else:
        raise ValueError(f"Unexpected written shape: {written_data.shape}")

    # Validate matching number of frames
    assert source_frames.shape[0] == written_frames.shape[0], \
        f"Frame count mismatch: {source_frames.shape[0]} != {written_frames.shape[0]}"

    # Test random frames
    num_frames = min(num_test_frames, source_frames.shape[0])
    test_indices = np.random.choice(source_frames.shape[0], size=num_frames, replace=False)
    test_indices = sorted(test_indices)

    correlations = []
    mae_values = []

    for idx in test_indices:
        src_frame = source_frames[idx].astype(np.float32)
        written_frame = written_frames[idx].astype(np.float32)

        # Correlation
        corr, _ = pearsonr(src_frame.flatten(), written_frame.flatten())
        correlations.append(corr)

        # Mean absolute error
        mae = np.mean(np.abs(src_frame - written_frame))
        mae_values.append(mae)

    correlations = np.array(correlations)
    mae_values = np.array(mae_values)

    print(f"\n  Correlation statistics:")
    print(f"    Mean: {correlations.mean():.6f}")
    print(f"    Min:  {correlations.min():.6f}")
    print(f"    Max:  {correlations.max():.6f}")

    print(f"\n  MAE statistics:")
    print(f"    Mean: {mae_values.mean():.2f}")
    print(f"    Min:  {mae_values.min():.2f}")
    print(f"    Max:  {mae_values.max():.2f}")

    # Check thresholds
    if correlations.min() > 0.999:
        print(f"\n  PASS EXCELLENT: All frames have correlation > 0.999")
        status = "excellent"
    elif correlations.min() > 0.99:
        print(f"\n  PASS GOOD: All frames have correlation > 0.99")
        status = "good"
    elif correlations.min() > 0.95:
        print(f"\n  WARNING WARNING: Some frames have correlation < 0.99")
        print(f"    Frames with low correlation: {np.sum(correlations < 0.99)}")
        status = "warning"
    else:
        print(f"\n  FAIL FAILED: Some frames have correlation < 0.95")
        print(f"    This suggests frames are shuffled or corrupted!")
        low_corr_indices = test_indices[correlations < 0.95]
        print(f"    Problem frame indices: {low_corr_indices}")
        status = "failed"

    return {
        "status": status,
        "correlation_mean": float(correlations.mean()),
        "correlation_min": float(correlations.min()),
        "correlation_max": float(correlations.max()),
        "mae_mean": float(mae_values.mean()),
        "mae_min": float(mae_values.min()),
        "mae_max": float(mae_values.max()),
        "num_frames_tested": num_frames,
    }


def validate_imread_dimensions(file_path, expected_shape, expected_dtype):
    """
    Validate that imread returns expected dimensions and dtype.
    """
    print(f"\nimread validation:")
    print(f"  Loading: {file_path}")

    data = mbo.imread(file_path)

    print(f"  Shape: {data.shape}")
    print(f"  Expected: {expected_shape}")
    print(f"  Dtype: {data.dtype}")
    print(f"  Expected dtype: {expected_dtype}")

    # Check shape
    if data.shape == expected_shape:
        print(f"  PASS Shape matches")
        shape_match = True
    else:
        print(f"  FAIL Shape mismatch!")
        shape_match = False

    # Check dtype
    if data.dtype == expected_dtype:
        print(f"  PASS Dtype matches")
        dtype_match = True
    else:
        print(f"  FAIL Dtype mismatch!")
        dtype_match = False

    return {
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "actual_shape": data.shape,
        "actual_dtype": str(data.dtype),
    }


def validate_baseline(baseline_name, source_path, written_path, config):
    """
    Validate a single baseline.

    Args:
        baseline_name: Name of the baseline test
        source_path: Path to source data
        written_path: Path to written baseline data
        config: Dict with validation parameters:
            - num_planes: Number of z-planes
            - expected_shape: Expected shape after reading
            - expected_dtype: Expected dtype
            - plane_idx: Which plane to test (default 0)
    """
    print("=" * 80)
    print(f"Validating baseline: {baseline_name}")
    print("=" * 80)

    results = {"baseline": baseline_name}

    # Load source data
    print(f"\nLoading source data: {source_path}")
    source_data = mbo.imread(source_path)
    print(f"  Source shape: {source_data.shape}")
    print(f"  Source dtype: {source_data.dtype}")

    # Load written data
    print(f"\nLoading written data: {written_path}")
    written_data = mbo.imread(written_path)
    print(f"  Written shape: {written_data.shape}")
    print(f"  Written dtype: {written_data.dtype}")

    # Validate shape/deinterleaving
    try:
        num_planes = config.get("num_planes", 1)
        validate_shape_deinterleaving(source_data.shape, written_data.shape, num_planes)
        results["shape_validation"] = "passed"
    except Exception as e:
        print(f"\n  FAIL Shape validation failed: {e}")
        results["shape_validation"] = f"failed: {e}"
        return results

    # Validate frame order
    try:
        plane_idx = config.get("plane_idx", 0)
        frame_results = validate_frame_order(
            source_data,
            written_data,
            num_planes=num_planes,
            plane_idx=plane_idx,
            num_test_frames=config.get("num_test_frames", 50)
        )
        results["frame_validation"] = frame_results
    except Exception as e:
        print(f"\n  FAIL Frame validation failed: {e}")
        results["frame_validation"] = f"failed: {e}"
        return results

    # Validate imread dimensions
    try:
        expected_shape = config.get("expected_shape", written_data.shape)
        expected_dtype = config.get("expected_dtype", written_data.dtype)
        imread_results = validate_imread_dimensions(written_path, expected_shape, expected_dtype)
        results["imread_validation"] = imread_results
    except Exception as e:
        print(f"\n  FAIL imread validation failed: {e}")
        results["imread_validation"] = f"failed: {e}"
        return results

    return results


def main():
    """Run validation on all baselines."""

    # Define your test baselines here
    # Adjust paths and configurations based on your actual test setup

    baselines_dir = Path(__file__).parent / "lbm" / "mbo_utilities" / "baselines"

    # Example configuration - adjust to match your actual tests
    baselines = [
        {
            "name": "imwrite_tiff",
            "source": "path/to/raw/data.tif",  # Update with actual path
            "written": baselines_dir / "imwrite_tiff.tif",
            "config": {
                "num_planes": 14,
                "expected_shape": (100, 512, 512),  # Adjust
                "expected_dtype": np.int16,
                "plane_idx": 0,
                "num_test_frames": 50,
            }
        },
        {
            "name": "imwrite_zarr",
            "source": "path/to/raw/data.tif",  # Update with actual path
            "written": baselines_dir / "imwrite_zarr.zarr",
            "config": {
                "num_planes": 14,
                "expected_shape": (100, 14, 512, 512),  # Adjust
                "expected_dtype": np.int16,
                "plane_idx": 0,
                "num_test_frames": 50,
            }
        },
        # Add more baselines as needed
    ]

    # Run validation
    all_results = []

    for baseline in baselines:
        if not Path(baseline["written"]).exists():
            print(f"\nSkipping {baseline['name']} - file does not exist")
            continue

        try:
            results = validate_baseline(
                baseline["name"],
                baseline["source"],
                baseline["written"],
                baseline["config"]
            )
            all_results.append(results)
        except Exception as e:
            print(f"\nFAIL Validation failed with error: {e}")
            all_results.append({
                "baseline": baseline["name"],
                "error": str(e)
            })

    # Summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for result in all_results:
        print(f"\n{result['baseline']}:")

        if "error" in result:
            print(f"  FAIL ERROR: {result['error']}")
            continue

        # Shape validation
        if result.get("shape_validation") == "passed":
            print(f"  PASS Shape validation passed")
        else:
            print(f"  FAIL Shape validation: {result.get('shape_validation')}")

        # Frame validation
        frame_val = result.get("frame_validation", {})
        if isinstance(frame_val, dict):
            status = frame_val.get("status", "unknown")
            corr_min = frame_val.get("correlation_min", 0)
            print(f"  Frame order: {status} (min correlation: {corr_min:.6f})")
        else:
            print(f"  FAIL Frame validation: {frame_val}")

        # imread validation
        imread_val = result.get("imread_validation", {})
        if isinstance(imread_val, dict):
            if imread_val.get("shape_match") and imread_val.get("dtype_match"):
                print(f"  PASS imread validation passed")
            else:
                print(f"  FAIL imread validation failed")
        else:
            print(f"  FAIL imread validation: {imread_val}")

    # Save results to JSON
    results_file = Path(__file__).parent / "baseline_validation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
