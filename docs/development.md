# Development

## Setup

```bash
git clone https://github.com/MillerBrainObservatory/mbo_utilities.git
cd mbo_utilities
uv sync --all-extras
```

## Testing

Tests use synthetic data by default.

There are additional tests for filetype-specific properties that use tiffs e.g. raw scanimage tiffs.

```bash
# run all tests
uv run pytest tests/ -v

# run specific test file
uv run pytest tests/test_arrays.py -v

# for CI: run only synthetic data tests
uv run pytest tests/test_arrays.py tests/test_to_video.py -v -k "synthetic or Synthetic"

# keep output files
KEEP_TEST_OUTPUT=1 uv run pytest tests/
```

Test structure:
- `tests/conftest.py` - fixtures for synthetic 3D/4D data, temp files, comparison helpers
- `tests/test_arrays.py` - array class tests (indexing, protocols, volume detection)
- `tests/test_roundtrip.py` - format conversion tests
- `tests/test_to_video.py` - video export tests

## Code Formatting

[ruff](https://github.com/astral-sh/ruff) for formatting and linting.

```bash
# format code
uv tool run ruff format .

# check linting
uv tool run ruff check .

# fix auto-fixable issues
uv tool run ruff check --fix .

## Building Docs

```bash
uv pip install "mbo_utilities[docs]"
cd docs
uv run make clean
uv run make html
```
