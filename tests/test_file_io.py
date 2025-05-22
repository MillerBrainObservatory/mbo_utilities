import numpy as np
import tifffile
import dask.array as da
from pathlib import Path
import mbo_utilities as mbo


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

def test_demo_files():
    test_path = Path("D://demo//raw_data")
    if test_path.is_dir():
        files = mbo.get_files(test_path, "tif")
        scan = mbo.read_scan(files)
        assert scan.min == -169
        assert scan.max == 4381
        assert scan.shape == (1437, 14, 448, 448)
    else:
        print("Skipping demo tests.")


# def test_scan_shape():
#     test_dir = Path(__file__).parent
#     files = mbo.get_files(test_dir, "tif")
#     scan = mbo.read_scan(files)
#     assert scan.shape is not None
#     assert len(scan.shape) == 4
