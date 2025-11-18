"""
Pytest configuration for mbo_utilities tests.

Automatically validates baselines when they are generated or updated.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import mbo_utilities as mbo


# Track generated baselines during test run
generated_baselines = []


def validate_baseline_integrity(baseline_path, source_data, written_data):
    """
    Validate that baseline data was correctly written.

    Returns dict with validation results.
    """
    results = {
        "baseline": str(baseline_path),
        "checks": {}
    }

    # Shape check
    if source_data.shape == written_data.shape:
        results["checks"]["shape"] = "passed"
    else:
        results["checks"]["shape"] = {
            "status": "failed",
            "source_shape": source_data.shape,
            "written_shape": written_data.shape
        }

    # Dtype check
    if source_data.dtype == written_data.dtype:
        results["checks"]["dtype"] = "passed"
    else:
        results["checks"]["dtype"] = {
            "status": "failed",
            "source_dtype": str(source_data.dtype),
            "written_dtype": str(written_data.dtype)
        }

    # Frame correlation check
    try:
        correlations = []

        if len(source_data.shape) == 3:
            # 3D: (T, Y, X)
            num_test = min(50, source_data.shape[0])
            test_indices = np.random.choice(source_data.shape[0], size=num_test, replace=False)

            for idx in test_indices:
                src = source_data[idx].astype(np.float32).flatten()
                dst = written_data[idx].astype(np.float32).flatten()
                corr, _ = pearsonr(src, dst)
                correlations.append(corr)

        elif len(source_data.shape) == 4:
            # 4D: (T, Z, Y, X)
            # Read FULL planes to match write behavior - this ensures phase correction
            # is applied to the same data range as during the write
            for z_idx in range(source_data.shape[1]):
                # Read entire plane at once (same as write does)
                src_plane = source_data[:, z_idx].astype(np.float32)
                dst_plane = written_data[:, z_idx].astype(np.float32)

                # Test random subset of frames from the full plane
                num_test = min(50, src_plane.shape[0])
                test_indices = np.random.choice(src_plane.shape[0], size=num_test, replace=False)

                for idx in test_indices:
                    corr, _ = pearsonr(src_plane[idx].flatten(), dst_plane[idx].flatten())
                    correlations.append(corr)

        correlations = np.array(correlations)

        if correlations.min() > 0.99:
            results["checks"]["correlation"] = {
                "status": "passed",
                "mean": float(correlations.mean()),
                "min": float(correlations.min())
            }
        else:
            results["checks"]["correlation"] = {
                "status": "failed",
                "mean": float(correlations.mean()),
                "min": float(correlations.min()),
                "warning": "Low correlation suggests frame shuffling or data corruption!"
            }

    except Exception as e:
        results["checks"]["correlation"] = {
            "status": "error",
            "error": str(e)
        }

    # Overall status
    all_passed = all(
        v == "passed" or (isinstance(v, dict) and v.get("status") == "passed")
        for v in results["checks"].values()
    )

    results["status"] = "passed" if all_passed else "failed"

    return results


@pytest.fixture
def validate_on_write(request):
    """
    Fixture that automatically validates data when a baseline is written.

    Usage in test:
        def test_something(validate_on_write):
            source_data = imread(source_path)
            imwrite(baseline_path, source_data)

            # Validation happens automatically
            validate_on_write(source_data, baseline_path)
    """
    def _validate(source_data, baseline_path, metadata=None):
        """
        Validate that baseline was correctly written.

        Args:
            source_data: Original source data array (or path)
            baseline_path: Path to written baseline
            metadata: Optional metadata dict to save alongside
        """
        baseline_path = Path(baseline_path)

        # Load source if path provided
        if isinstance(source_data, (str, Path)):
            source_data = mbo.imread(source_path)

        # Ensure we compare the SAME data that was written
        # If source has phase correction enabled, it will be applied during write
        # So we need to compare with phase correction enabled
        # But we need to ensure consistent chunk-based reading
        source_for_validation = source_data

        # Load written data and materialize to ensure file handle is closed
        try:
            written_data_lazy = mbo.imread(baseline_path)
            written_data = np.asarray(written_data_lazy)

            # Close file handle if it exists
            if hasattr(written_data_lazy, 'close'):
                written_data_lazy.close()
            elif hasattr(written_data_lazy, '_fh') and hasattr(written_data_lazy._fh, 'close'):
                written_data_lazy._fh.close()

            # Delete reference and force garbage collection
            del written_data_lazy
        except (KeyError, AttributeError):
            # Fallback for simple TIFF files without metadata
            from tifffile import imread as tiff_imread
            written_data = tiff_imread(baseline_path)

        # Force garbage collection to release file handles (Windows issue)
        import gc
        gc.collect()

        # Validate
        results = validate_baseline_integrity(baseline_path, source_data, written_data)

        # Track this baseline
        generated_baselines.append(results)

        # Save metadata JSON alongside baseline
        if metadata or results["status"] != "passed":
            json_path = baseline_path.parent / f"{baseline_path.stem}.json"

            metadata_to_save = metadata or {}
            metadata_to_save.update({
                "shape": list(written_data.shape),
                "dtype": str(written_data.dtype),
                "validation": results
            })

            with open(json_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)

        # Fail test if validation failed
        if results["status"] != "passed":
            failure_msg = f"Baseline validation FAILED for {baseline_path.name}:\n"
            for check, result in results["checks"].items():
                if isinstance(result, dict) and result.get("status") == "failed":
                    failure_msg += f"  - {check}: {result}\n"
            pytest.fail(failure_msg)

        return results

    return _validate


@pytest.fixture
def baseline_validator(request):
    """
    Fixture that provides a context manager for automatic baseline validation.

    Usage:
        def test_something(baseline_validator):
            with baseline_validator(source_path, baseline_path) as validator:
                # Write your baseline
                imwrite(baseline_path, data)
            # Validation happens automatically on exit
    """
    class BaselineValidator:
        def __init__(self, source_path, baseline_path, metadata=None):
            self.source_path = Path(source_path)
            self.baseline_path = Path(baseline_path)
            self.metadata = metadata

        def __enter__(self):
            # Load source data
            self.source_data = mbo.imread(self.source_path)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                # Test failed, don't validate
                return False

            # Validate the written baseline
            written_data = mbo.imread(self.baseline_path)
            results = validate_baseline_integrity(
                self.baseline_path,
                self.source_data,
                written_data
            )

            # Track this baseline
            generated_baselines.append(results)

            # Save metadata
            json_path = self.baseline_path.parent / f"{self.baseline_path.stem}.json"
            metadata_to_save = self.metadata or {}
            metadata_to_save.update({
                "shape": list(written_data.shape),
                "dtype": str(written_data.dtype),
                "source_path": str(self.source_path),
                "validation": results
            })

            with open(json_path, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)

            # Fail if validation failed
            if results["status"] != "passed":
                failure_msg = f"Baseline validation FAILED for {self.baseline_path.name}:\n"
                for check, result in results["checks"].items():
                    if isinstance(result, dict) and result.get("status") == "failed":
                        failure_msg += f"  - {check}: {result}\n"
                pytest.fail(failure_msg)

            return False

    return BaselineValidator


def pytest_sessionfinish(session, exitstatus):
    """
    Hook called after all tests finish.

    Generates a summary report of all baseline validations.
    """
    if not generated_baselines:
        return

    print("\n" + "="*80)
    print("BASELINE VALIDATION SUMMARY")
    print("="*80)

    passed = sum(1 for r in generated_baselines if r["status"] == "passed")
    failed = sum(1 for r in generated_baselines if r["status"] == "failed")

    print(f"\nBaselines generated: {len(generated_baselines)}")
    print(f"  PASS Passed: {passed}")
    print(f"  FAIL Failed: {failed}")

    if failed > 0:
        print(f"\nWARNING WARNING: {failed} baseline(s) failed validation!")
        print("Failed baselines:")
        for result in generated_baselines:
            if result["status"] == "failed":
                print(f"  - {Path(result['baseline']).name}")
                for check, check_result in result["checks"].items():
                    if isinstance(check_result, dict) and check_result.get("status") == "failed":
                        print(f"      {check}: {check_result.get('warning', 'failed')}")

    # Save summary
    summary_path = Path(__file__).parent / "baseline_validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(generated_baselines, f, indent=2, default=str)

    print(f"\nFull validation report saved to: {summary_path}")
    print("="*80 + "\n")


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory."""
    # Adjust this path to your test data location
    data_dir = Path(__file__).parent / "test_data"
    if not data_dir.exists():
        pytest.skip(f"Test data directory not found: {data_dir}")
    return data_dir


@pytest.fixture(scope="session")
def baselines_dir():
    """Fixture providing path to baselines directory."""
    baseline_dir = Path(__file__).parent / "lbm" / "mbo_utilities" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir
