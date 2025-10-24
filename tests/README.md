# MBO Utilities Test Suite

This directory contains the test suite for mbo_utilities, split between CI tests (using synthetic data) and local tests (using real microscopy data).

## Test Organization

Tests are organized using pytest markers:

- **`@pytest.mark.ci`**: Tests that run in CI with synthetic/small data
- **`@pytest.mark.local`**: Tests that require real data files (too large for CI)
- **`@pytest.mark.slow`**: Tests that take significant time to run
- **`@pytest.mark.io`**: Tests involving file I/O operations
- **`@pytest.mark.formats`**: Tests for different file format conversions

## Running Tests

### CI Tests (Fast, No Data Required)

These tests run automatically in GitHub Actions and use synthetic data:

```bash
# Run all CI tests
pytest -m ci

# Run specific CI test file
pytest -m ci tests/test_io_comprehensive.py
```

### Local Tests (Require Real Data)

These tests require real microscopy data files and should be run before publishing:

```bash
# Set path to your test data
export MBO_TEST_DATA=/path/to/your/test/data

# Run all local tests
pytest -m local

# Run local tests excluding slow ones
pytest -m "local and not slow"

# Run specific test categories
pytest -m "local and io"
pytest -m "local and formats"
```

### Run All Tests

```bash
# Run everything (requires real data)
pytest tests/

# Run with verbose output and timing
pytest -v -m local --durations=10
```

## Test Data Setup

### For CI Tests
No setup required - synthetic data is generated automatically.

### For Local Tests
You need real ScanImage TIFF files. Set the `MBO_TEST_DATA` environment variable:

```bash
# Linux/Mac
export MBO_TEST_DATA=/path/to/your/scanimage/tiffs

# Windows (PowerShell)
$env:MBO_TEST_DATA="C:\path\to\your\scanimage\tiffs"
```

The test suite will look for `.tif` or `.tiff` files in this directory.

## Test Files

### `test_io_comprehensive.py`
Comprehensive tests for `imread` and `imwrite` functionality:

- **Format tests**: `.tiff`, `.zarr`, `.bin`, `.h5`
- **Roundtrip tests**: Read → Write → Read validation
- **ROI tests**: Stitched (roi=None) and individual ROIs (roi=0, roi=1)
- **Performance benchmarks**: Timing and throughput measurements
- **Metadata preservation**: Ensures metadata flows through pipelines

Key test functions:
- `test_imwrite_formats_synthetic()`: Test writing to different formats (CI)
- `test_imwrite_formats_real_data()`: Test with real data (local)
- `test_roi_stitched()`: Test stitched multi-ROI output
- `test_roi_individual()`: Test individual ROI extraction
- `test_roi_comparison()`: Compare stitched vs individual ROIs
- `test_format_comparison_benchmark()`: Performance benchmark across formats

### `test_file_io.py`
Tests for file discovery, metadata extraction, and basic I/O:

- File discovery utilities (`get_files`, `expand_paths`)
- Metadata parsing from ScanImage TIFFs
- Overwrite behavior
- Multi-ROI handling

### `test_graphics.py`
Tests for GUI components (imgui context creation).

## Pre-Release Testing

Before publishing a new version, run the full local test suite:

```bash
# Set your test data path
export MBO_TEST_DATA=/path/to/test/data

# Run all local tests with timing
pytest -m local -v --durations=20

# Run comprehensive I/O tests
pytest -m "local and io" -v

# Run format comparison benchmark
pytest -k "test_format_comparison_benchmark" -v -s
```

Expected output includes:
- ✅ All tests pass
- ⏱️ Timing information for each operation
- 📊 Data shapes and sizes
- 📈 Throughput measurements (GB/s)

## Adding New Tests

When adding new tests:

1. **Add appropriate markers**:
   ```python
   @pytest.mark.ci  # or @pytest.mark.local
   @pytest.mark.io  # if it involves file I/O
   def test_my_new_feature():
       ...
   ```

2. **Use fixtures** for common setup:
   - `synthetic_data`: Small generated array
   - `synthetic_tiff`: Pre-created TIFF file
   - `real_data_files`: Real data (if available)

3. **Include timing** for performance-sensitive operations:
   ```python
   with Timer("My operation"):
       result = my_function()
   ```

4. **Add docstrings** explaining what the test validates

## Continuous Integration

The GitHub Actions workflow (`.github/workflows/test_python.yml`) runs:

```bash
pytest -m ci tests/
```

This ensures:
- Package installs correctly
- Basic functionality works
- No import errors
- Synthetic data tests pass

Local tests are **not** run in CI due to data size constraints.

## Troubleshooting

### Tests fail with "No data files found"
- Set `MBO_TEST_DATA` environment variable
- Ensure path contains `.tif` or `.tiff` files
- Check that files are readable

### Import errors
```bash
# Reinstall package in development mode
uv pip install -e .
```

### Slow tests taking too long
```bash
# Skip slow tests
pytest -m "local and not slow"
```

### Want to see detailed output
```bash
# Use -s flag to show print statements
pytest -v -s -m local
```

## Performance Expectations

Typical timing for local tests with ~1GB data:

| Operation | Expected Time | Throughput |
|-----------|--------------|------------|
| Read TIFF | 0.5-2s | 0.5-2 GB/s |
| Write TIFF | 1-3s | 0.3-1 GB/s |
| Write Zarr | 2-5s | 0.2-0.5 GB/s |
| Write Binary | 1-3s | 0.3-1 GB/s |
| Write HDF5 | 2-4s | 0.25-0.5 GB/s |

Times vary based on:
- Data size and shape
- Disk speed (SSD vs HDD)
- Compression settings
- System load
