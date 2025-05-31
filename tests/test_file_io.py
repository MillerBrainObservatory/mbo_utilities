import pytest
from pathlib import Path

import numpy as np
from icecream import ic
from tifffile import imread

import mbo_utilities as mbo

ic.enable()

BASE = Path(r"D:\tests\data")
ASSEMBLED = BASE / "assembled"
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


@pytest.mark.parametrize(
    "roi,subdir",
    [
        (0, ""),  # individual ROIs in ASSEMBLED/roi1, roi2…
        (1, ""),  # same, just roi=1
        (None, "full"),  # full‐stack in ASSEMBLED/full
    ],
)
def test_demo_files(tmp_path, roi, subdir):
    ASSEMBLED.mkdir(exist_ok=True)
    files = mbo.get_files(BASE, "tif")

    save_dir = ASSEMBLED / subdir if subdir else ASSEMBLED
    scan = mbo.read_scan(files, roi=roi)
    mbo.save_as(
        scan,
        save_dir,
        ext=".tiff",
        overwrite=True,
        fix_phase=False,
        planes=[1, 7, 14],
    )

    out = mbo.get_files(ASSEMBLED, "plane", max_depth=2)
    assert out, "No plane files written"


@pytest.fixture
def plane_paths():
    return mbo.get_files(ASSEMBLED, "plane_01.tif", max_depth=3)


def test_full_contains_rois_side_by_side(plane_paths):
    # map parent‐dir → path, e.g. "full", "roi1", "roi2"
    by_dir = {Path(p).parent.name: Path(p) for p in plane_paths}
    full = imread(by_dir["full"])
    roi1 = imread(by_dir["roi1"])
    roi2 = imread(by_dir["roi2"])

    T, H, W = full.shape
    assert roi1.shape == (T, H, W // 2)
    assert roi2.shape == (T, H, W - W // 2)

    left, right = full[:, :, : W // 2], full[:, :, W // 2 :]
    np.testing.assert_array_equal(left, roi1)
    np.testing.assert_array_equal(right, roi2)


def test_overwrite_false_skips_existing(tmp_path, capsys):
    # First write with overwrite=True
    files = mbo.get_files(BASE, "tif")
    scan = mbo.read_scan(files, roi=None)
    mbo.save_as(
        scan, ASSEMBLED, ext=".tiff", overwrite=True, fix_phase=False, planes=[1]
    )

    # Capture output of second call with overwrite=False
    mbo.save_as(
        scan, ASSEMBLED, ext=".tiff", overwrite=False, fix_phase=False, planes=[1]
    )
    captured = capsys.readouterr().out

    assert "All output files exist; skipping save." in captured


def test_overwrite_true_rewrites(tmp_path, capsys):
    # first write with overwrite=True
    files = mbo.get_files(BASE, "tif")
    scan = mbo.read_scan(files, roi=None)
    mbo.save_as(
        scan, ASSEMBLED, ext=".tiff", overwrite=True, fix_phase=False, planes=[1]
    )

    # second write with overwrite=True
    mbo.save_as(
        scan, ASSEMBLED, ext=".tiff", overwrite=True, fix_phase=False, planes=[1]
    )
    captured = capsys.readouterr().out

    # Should not skip entirely
    assert "All output files exist; skipping save." not in captured

    # And it should print the elapsed‐time message twice (once per call)
    assert captured.count("Time elapsed:") >= 2
