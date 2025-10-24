# Test Suite Summary

## Overview

The test suite has been reorganized and updated to:

1. **Separate CI and local tests** using pytest markers
2. **Add timing information** for all I/O operations
3. **Test multiple formats** (.tiff, .zarr, .bin, .h5)
4. **Test ROI functionality** (stitched and individual ROIs)

## Test Status

### ✅ Working Tests (CI)

These tests pass in CI with synthetic data:

- `test_expand_paths`: File path expansion utility
- `test_files_to_dask`: Dask array creation from files
- `test_jupyter_check`: Jupyter environment detection
- `test_imgui_check`: ImGui installation detection
- `test_imgui_context_creation`: ImGui context management
- `test_imread_synthetic_tiff`: Basic imread functionality

### ⚠️  Known Issues

Several comprehensive tests currently fail due to library limitations:

**Issue 1: Metadata Property Read-Only**
- **Problem**: `MBOTiffArray.metadata` property has no setter
- **Impact**: Cannot modify metadata after reading TIFF files
- **Fix Applied**: Updated `imwrite()` to handle read-only metadata properties gracefully
- **Location**: `mbo_utilities/lazy_array.py:220-229`

**Issue 2: Plane Number Requirement**
- **Problem**: `MBOTiffArray._imwrite()` requires plane number in metadata or filename
- **Impact**: Cannot write files to arbitrary output directories
- **Workaround Needed**: Either:
  - Include "plane" in metadata that persists through read/write
  - Use filenames with plane patterns (e.g., `data_plane0.tif`)
  - Explicitly pass `planes=[0]` parameter
- **Location**: `mbo_utilities/array_types.py:516`

**Issue 3: Metadata Persistence**
- **Problem**: Custom metadata written to TIFF tags doesn't always persist when read back
- **Impact**: Metadata like "plane" number may be lost in roundtrip
- **Needs**: Investigation of TIFF tag reading/writing

## Test Organization

### File Structure

```
tests/
├── README.md                    # User guide for running tests
├── TEST_SUMMARY.md             # This file - status and issues
├── pytest.ini                  # Pytest configuration (in repo root)
├── test_file_io.py            # Basic file I/O tests (updated with markers)
├── test_graphics.py           # GUI tests
└── test_io_comprehensive.py    # Comprehensive I/O tests (new)
```

###  Pytest Markers

Defined in `pytest.ini`:

- `@pytest.mark.ci`: Tests that run in CI (no large data required)
- `@pytest.mark.local`: Tests requiring local data files
- `@pytest.mark.slow`: Time-consuming tests
- `@pytest.mark.io`: File I/O operations
- `@pytest.mark.formats`: Format conversion tests

### Running Tests

```bash
# CI tests only (fast, no data required)
pytest -m ci

# Local tests (requires MBO_TEST_DATA env var)
export MBO_TEST_DATA=/path/to/test/data
pytest -m local

# Specific categories
pytest -m "local and io"
pytest -m "local and formats and not slow"
```

## Recommendations for Fixing Tests

### Short Term
1. **Document workarounds** in test docstrings
2. **Skip problematic tests** until library issues are resolved
3. **Focus on tests that work** with real multi-ROI ScanImage data

### Medium Term
1. **Fix metadata setter**: Add setter to `MBOTiffArray.metadata` property
2. **Improve plane handling**: Make plane number optional or auto-detect from context
3. **Enhance metadata persistence**: Ensure custom metadata survives read/write cycles

### Long Term
1. **Refactor array types**: Unify metadata handling across all array types
2. **Add metadata tests**: Specific tests for metadata read/write
3. **Document metadata schema**: Clear specification of what metadata is supported

## Test Coverage Goals

### Current Coverage (estimate)
- Basic I/O: ~60%
- Format conversion: ~30% (blocked by issues)
- ROI handling: ~40% (needs real data)
- Metadata: ~20% (blocked by issues)

### Target Coverage
- Basic I/O: 90%+
- Format conversion: 80%+
- ROI handling: 80%+
- Metadata: 70%+

## Next Steps

1. **Create fixtures for real ScanImage data** with known properties
2. **Add skipping decorators** to tests that expose known bugs
3. **File issues** for the identified problems
4. **Create minimal reproducible examples** for each issue
5. **Add integration tests** that test full workflows end-to-end

## CI Configuration

The GitHub Actions workflow (`.github/workflows/test_python.yml`) has been updated to:

```yaml
- name: Install pytest
  run: uv pip install pytest

- name: Run CI tests
  run: GIT_LFS_SKIP_SMUDGE=1 uv run pytest -m ci tests/
```

This ensures only CI-appropriate tests run in automated testing.

## Local Testing Before Release

Before publishing a new version, run:

```bash
# Set test data location
export MBO_TEST_DATA=/path/to/your/scanimage/data

# Run all local tests
pytest -m local -v

# Run tests by category
pytest -m "local and io" -v
pytest -m "local and formats" -v --durations=10

# Check specific functionality
pytest -k "roi" -v
pytest -k "roundtrip" -v
```

## Contact

For questions about the test suite, see:
- `tests/README.md` - User guide
- Test docstrings - Specific test documentation
- GitHub Issues - Report problems
