import numpy as np
import pytest
from pathlib import Path
import mbo_utilities as mbo

DATA_ROOT = Path(r"D:\tests\data")

skip_if_missing_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason=f"Test data directory not found: {DATA_ROOT}"
)

@skip_if_missing_data
def test_metadata():
    """Test that metadata can be read from a file."""
    files = mbo.get_files(DATA_ROOT, "tif")
    assert len(files) > 0
    metadata = mbo.get_metadata(files[0])
    assert isinstance(metadata, dict)
    assert "frame_rate" in metadata.keys()

@skip_if_missing_data
def test_get_files_returns_valid_tiffs():
    files = mbo.get_files(DATA_ROOT, "tif")
    assert isinstance(files, list)
    assert len(files) == 2
    for f in files:
        assert Path(f).suffix in (".tif", ".tiff")
        assert Path(f).exists()


@skip_if_missing_data
def test_read_metadata():
    files = mbo.get_files(DATA_ROOT, "tif")
    metadata = mbo.get_metadata(files[0])
    assert isinstance(metadata, dict)
    assert "frame_rate" in metadata.keys()


def test_expand_paths(tmp_path):
    """Test expand_paths returns sorted file paths."""
    (tmp_path / "a.txt").write_text("dummy")
    (tmp_path / "b.txt").write_text("dummy")
    (tmp_path / "c.md").write_text("dummy")
    results = mbo.expand_paths(tmp_path)
    names = sorted([Path(p).name for p in results])
    expected = sorted(["a.txt", "b.txt", "c.md"])
    assert names == expected


def test_npy_to_dask(tmp_path):
    """Test npy_to_dask creates a dask array of the expected shape."""
    shape = (10, 20, 30, 40)
    files = []
    for i in range(3):
        arr = np.full(shape, i, dtype=np.float32)
        file_path = tmp_path / f"dummy_{i}.npy"
        np.save(file_path, arr)
        files.append(str(file_path))
    darr = mbo.npy_to_dask(files, name="test", axis=1, astype=np.float32)
    expected_shape = (10, 60, 30, 40)
    assert darr.shape == expected_shape


def test_jupyter_check():
    assert isinstance(mbo.is_running_jupyter(), bool)


def test_imgui_check():
    result = mbo.is_imgui_installed()
    assert isinstance(result, bool)


@skip_if_missing_data
def test_get_files():
    """Test get_files returns a list of files with the specified extension."""
    test_path = Path(r"D:\tests\data")
    if test_path.is_dir():
        files = mbo.get_files(test_path, "tif")
        assert isinstance(files, list)
        assert len(files) > 0
        for file in files:
            assert Path(file).suffix == ".tif"


@skip_if_missing_data
def test_demo_files():
    test_path = Path(r"D:\tests\data")
    files = mbo.get_files(test_path, "tif")
    scan = mbo.read_scan(files)
    assert hasattr(scan, "shape")
