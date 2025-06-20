import os
import functools

import pytest
from pathlib import Path

import numpy as np
from icecream import ic
from tifffile import imread as tfile_imread

import mbo_utilities as mbo
from mbo_utilities import get_mbo_dirs
from mbo_utilities._benchmark import _benchmark_indexing
from mbo_utilities.lazy_array import imread as mbo_imread, imwrite as mbo_imwrite

try:
    import dask.array as da
    import zarr
except ImportError:
    raise ImportError(
        "Dask and Zarr are required for this test. "
        "Please install them with `pip install dask zarr`."
    )

ic.enable()

DEFAULT_DATA_ROOT = get_mbo_dirs()["tests"]
DATA_ROOT = Path(os.getenv("MBO_TEST_DATA", DEFAULT_DATA_ROOT))
BASE = DATA_ROOT
ASSEMBLED = BASE / "assembled"


def skip_if_missing_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DATA_ROOT.exists() or len(list(DATA_ROOT.glob("*.tif"))) == 0:
            pytest.skip("Required TIFF files not found.")
        return func(*args, **kwargs)
    return wrapper

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
@pytest.mark.parametrize(
    "roi,subdir",
    [
        (0, ""),  # individual ROIs in ASSEMBLED/roi1, roi2…
        (1, ""),  # same, just roi=1
        (None, "full"),  # full‐stack in ASSEMBLED/full
    ],
)
def test_demo_files(roi, subdir):
    ASSEMBLED.mkdir(exist_ok=True)
    files = mbo.get_files(BASE, "tif")
    lazy_array = mbo_imread(files)
    save_dir = ASSEMBLED / subdir if subdir else ASSEMBLED
    lazy_array.roi = roi
    mbo_imwrite(
        lazy_array,
        save_dir,
        ext=".tiff",
        overwrite=True,
        planes=[7, 10],
    )

    out = mbo.get_files(save_dir, "plane", max_depth=2)
    print('------------')
    print(out)


@pytest.fixture
def plane_paths():
    ASSEMBLED.mkdir(exist_ok=True)
    return mbo.get_files(ASSEMBLED, "plane", max_depth=3)


@skip_if_missing_data
def test_full_contains_rois_side_by_side(plane_paths):
    # map parent‐dir → path, e.g. "full", "roi1", "roi2"
    if ASSEMBLED:
        by_dir = {Path(p).parent.name: Path(p) for p in plane_paths}
        full = tfile_imread(by_dir["full"])
        roi1 = tfile_imread(by_dir["roi1"])
        roi2 = tfile_imread(by_dir["roi2"])

        T, H, W = full.shape
        assert roi1.shape == (T, H, W // 2)
        assert roi2.shape == (T, H, W - W // 2)

        left, right = full[:, :, : W // 2], full[:, :, W // 2 :]
        np.testing.assert_array_equal(left, roi1)
        np.testing.assert_array_equal(right, roi2)


@skip_if_missing_data
def test_overwrite_false_skips_existing():
    # First write with overwrite=True
    files = mbo.get_files(BASE, "tif")
    data = mbo_imread(files)
    data.fix_phase = False
    mbo_imwrite(data, ASSEMBLED, ext=".tiff", overwrite=True, planes=[1])
    # Capture output of second call with overwrite=False
    mbo_imwrite(data, ASSEMBLED, ext=".tiff", overwrite=False, planes=[1])
    # Ensure the file exists
    import tifffile
    data = tifffile.imread(ASSEMBLED / "plane1.tif")
    assert data.shape is not None
    print('------------')
    print(data)
    # captured = capsys.readouterr().out
    # assert "All output files exist; skipping save." in captured


@skip_if_missing_data
def test_overwrite_true_rewrites():
    # First write with overwrite=True
    files = mbo.get_files(BASE, "tif")
    data = mbo_imread(files)
    data.fix_phase = False
    mbo_imwrite(data, ASSEMBLED, ext=".tiff", overwrite=True, planes=[1])
    # Capture output of second call with overwrite=True
    mbo_imwrite(data, ASSEMBLED, ext=".tiff", overwrite=True, planes=[1])
    import tifffile
    data = tifffile.imread(ASSEMBLED / "plane1.tif")  # Ensure file exists
    assert data.shape is not None
    print(data)
    # captured = capsys.readouterr().out
    # # And it should print the elapsed‐time message twice (once per call)
    # assert captured.count("Time elapsed:") >= 2

@skip_if_missing_data
def test_benchmark_indexing_test(tmp_path):
    """Benchmark indexing performance for different array types."""
    files = mbo.get_files(BASE, "tif")
    data = mbo_imread(files)

    # Convert to dask and zarr
    dask_array = data.as_dask()
    zarr_array = data.as_zarr()

    arrays = {
        "numpy": data,
        "dask": dask_array,
        "zarr": zarr_array,
    }

    save_path = tmp_path / "benchmark_results.json"
    results = _benchmark_indexing(
        arrays=arrays,
        save_path=save_path,
        num_repeats=3,
        label="Indexing Benchmark",
    )

    assert isinstance(results, dict)
    assert len(results) == 3
    print(results)
