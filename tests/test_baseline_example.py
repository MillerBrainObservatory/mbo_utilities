"""
Example test showing how to use automatic baseline validation.

This shows two approaches:
1. Using validate_on_write fixture (manual validation call)
2. Using baseline_validator context manager (automatic validation)
"""

import pytest
import numpy as np
from pathlib import Path
import mbo_utilities as mbo


# Example 1: Using validate_on_write fixture
def test_imwrite_with_validation(validate_on_write, test_data_dir, baselines_dir):
    """
    Example test that validates baseline after writing.

    The validate_on_write fixture will:
    1. Compare source data with written baseline
    2. Check shape, dtype, and frame correlation
    3. Save metadata JSON
    4. Fail the test if validation fails
    """
    # Assume you have test data
    source_path = test_data_dir / "example_source.tif"

    # Skip if source doesn't exist (replace with your actual test data)
    if not source_path.exists():
        pytest.skip("Test data not available")

    # Load source
    source_data = mbo.imread(source_path)

    # Write baseline
    baseline_path = baselines_dir / "example_baseline.tif"
    mbo.imwrite(baseline_path, source_data, overwrite=True)

    # Validate (automatically checks correlation, shape, dtype)
    # Will fail test if validation fails
    validate_on_write(source_data, baseline_path, metadata={
        "test_name": "test_imwrite_with_validation",
        "source_path": str(source_path),
        "description": "Example baseline for documentation"
    })


# Example 2: Using baseline_validator context manager (recommended)
def test_imwrite_with_context_manager(baseline_validator, test_data_dir, baselines_dir):
    """
    Example test using context manager for automatic validation.

    This is cleaner - validation happens automatically when exiting the context.
    """
    source_path = test_data_dir / "example_source.tif"
    baseline_path = baselines_dir / "example_baseline_v2.tif"

    if not source_path.exists():
        pytest.skip("Test data not available")

    # Use context manager - validation happens on exit
    with baseline_validator(source_path, baseline_path, metadata={
        "test_name": "test_imwrite_with_context_manager",
        "description": "Example using context manager"
    }) as validator:
        # Access source data from validator
        source_data = validator.source_data

        # Write baseline
        mbo.imwrite(baseline_path, source_data, overwrite=True)

        # Validation happens automatically when exiting context
        # Test will fail if validation fails


# Example 3: Testing multi-plane data
def test_multiplane_baseline(baseline_validator, test_data_dir, baselines_dir):
    """
    Example test for multi-plane (4D) data.

    Validation automatically handles both 3D and 4D data.
    """
    source_path = test_data_dir / "multiplane_source.tif"
    baseline_path = baselines_dir / "multiplane_baseline.zarr"

    if not source_path.exists():
        pytest.skip("Test data not available")

    with baseline_validator(source_path, baseline_path) as validator:
        source_data = validator.source_data

        # Write as zarr with all planes
        mbo.imwrite(baseline_path, source_data, overwrite=True)

        # Validation will check correlation for all z-planes


# Example 4: Testing specific write options
def test_imwrite_options(baseline_validator, test_data_dir, baselines_dir):
    """
    Example testing specific write options like compression, chunking, etc.
    """
    source_path = test_data_dir / "example_source.tif"
    baseline_path = baselines_dir / "compressed_baseline.tif"

    if not source_path.exists():
        pytest.skip("Test data not available")

    with baseline_validator(source_path, baseline_path, metadata={
        "compression": "lzw",
        "description": "Testing LZW compression"
    }) as validator:
        source_data = validator.source_data

        # Write with specific options
        mbo.imwrite(
            baseline_path,
            source_data,
            compression='lzw',
            overwrite=True
        )
