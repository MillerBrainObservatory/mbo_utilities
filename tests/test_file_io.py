import pytest
from pathlib import Path
from itertools import product

import numpy as np
from icecream import ic

import mbo_utilities as mbo

ic.enable()

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
    assert "pixel_resolution" in metadata.keys()
    assert "objective_resolution" in metadata.keys()
    assert "dtype" in metadata.keys()
    assert "frame_rate" in metadata.keys()

@skip_if_missing_data
def test_get_files_returns_valid_tiffs():
    files = mbo.get_files(DATA_ROOT, "tif")
    assert isinstance(files, list)
    assert len(files) == 2
    for f in files:
        assert Path(f).suffix in (".tif", ".tiff")
        assert Path(f).exists()


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
def test_demo_files(tmp_path: Path):
    test_path = Path(r"D:\tests\data")

    assembled_path = test_path / "assembled"
    assembled_path.mkdir(exist_ok=True)

    test_files = mbo.get_files(test_path, "tif")
    for roi in [0, 1, None]:
        if roi is None:
            savedir = assembled_path / "full"
        else:
            savedir = assembled_path
        test_scan = mbo.read_scan(test_files, roi=roi)
        mbo.save_as(
            test_scan,
            savedir.expanduser().resolve(),
            ext=".tiff",
            overwrite=True,
            fix_phase=True,
            planes=[1, 7, 14],
            debug=True,
        )
    outputs = mbo.get_files(assembled_path, "tif", max_depth=3)
    print(outputs)
