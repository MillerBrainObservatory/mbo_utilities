"""
full pipeline test: save to all formats, read back, run suite2p on plane 7.

usage:
    uv run pytest tests/local/test_full_pipeline.py -v -s
    KEEP_TEST_OUTPUT=1 uv run pytest tests/local/test_full_pipeline.py -v -s

test input: E:/tests/lbm/mbo_utilities/test_input.tif (14 z-planes)
"""

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import mbo_utilities as mbo

TEST_INPUT = Path("E:/tests/lbm/mbo_utilities/test_input.tif")
OUTPUT_ROOT = Path("E:/tests/lbm/mbo_utilities/test_outputs/full_pipeline")

# formats to test
WRITE_FORMATS = [".tiff", ".zarr", ".h5", ".bin"]
SUITE2P_PLANE = 7


@dataclass
class TimingResult:
    """timing result for a single operation."""
    operation: str
    format: str
    elapsed_ms: float
    details: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.operation:20s} [{self.format:6s}]: {self.elapsed_ms:8.1f} ms"


@dataclass
class PipelineResults:
    """aggregated results from the full pipeline test."""
    timings: list[TimingResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    write_paths: dict = field(default_factory=dict)
    read_shapes: dict = field(default_factory=dict)
    suite2p_results: dict = field(default_factory=dict)

    def add_timing(self, operation: str, fmt: str, elapsed_ms: float, **details):
        self.timings.append(TimingResult(operation, fmt, elapsed_ms, details))

    def print_summary(self):
        print("\n" + "=" * 70)
        print("PIPELINE TIMING SUMMARY")
        print("=" * 70)

        # group by operation
        ops = {}
        for t in self.timings:
            if t.operation not in ops:
                ops[t.operation] = []
            ops[t.operation].append(t)

        for op, timings in ops.items():
            print(f"\n{op}:")
            for t in timings:
                print(f"  {t.format:6s}: {t.elapsed_ms:8.1f} ms")

        # totals per format
        print("\n" + "-" * 40)
        print("TOTAL TIME PER FORMAT:")
        fmt_totals = {}
        for t in self.timings:
            if t.format not in fmt_totals:
                fmt_totals[t.format] = 0
            fmt_totals[t.format] += t.elapsed_ms

        for fmt, total in sorted(fmt_totals.items(), key=lambda x: x[1]):
            print(f"  {fmt:6s}: {total:8.1f} ms ({total/1000:.2f} s)")

        if self.errors:
            print("\n" + "-" * 40)
            print("ERRORS:")
            for err in self.errors:
                print(f"  - {err}")

        print("=" * 70)


def time_operation(func, *args, **kwargs):
    """time a function call, return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


@pytest.fixture(scope="module")
def source_array():
    """load source array once per module."""
    if not TEST_INPUT.exists():
        pytest.skip(f"Test input not found: {TEST_INPUT}")
    return mbo.imread(TEST_INPUT)


@pytest.fixture(scope="module")
def output_root():
    """create clean output directory."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


@pytest.fixture(scope="module")
def pipeline_results():
    """shared results collector."""
    return PipelineResults()


class TestFullPipeline:
    """test full pipeline: write all formats, read back, run suite2p."""

    def test_01_source_info(self, source_array, pipeline_results):
        """verify source array properties."""
        arr = source_array
        print(f"\nSource array: {arr.shape}, dtype={arr.dtype}")
        print(f"  Type: {type(arr).__name__}")

        assert arr.ndim == 4, f"Expected 4D array, got {arr.ndim}D"
        assert arr.shape[1] == 14, f"Expected 14 z-planes, got {arr.shape[1]}"

        pipeline_results.read_shapes["source"] = arr.shape

    @pytest.mark.parametrize("ext", WRITE_FORMATS)
    def test_02_write_format(self, source_array, output_root, pipeline_results, ext):
        """write source to each format."""
        out_dir = output_root / f"write_{ext.lstrip('.')}"
        out_dir.mkdir(exist_ok=True)

        print(f"\nWriting to {ext}...")

        _, elapsed_ms = time_operation(
            mbo.imwrite,
            source_array,
            out_dir,
            ext=ext,
            overwrite=True,
        )

        pipeline_results.add_timing("write", ext, elapsed_ms)
        pipeline_results.write_paths[ext] = out_dir

        print(f"  Wrote to {out_dir} in {elapsed_ms:.1f} ms")

        # verify output exists
        assert out_dir.exists()
        files = list(out_dir.rglob("*"))
        assert len(files) > 0, f"No files written to {out_dir}"

    @pytest.mark.parametrize("ext", WRITE_FORMATS)
    def test_03_read_format(self, output_root, pipeline_results, ext):
        """read back each format and verify shape."""
        out_dir = output_root / f"write_{ext.lstrip('.')}"

        if not out_dir.exists():
            pytest.skip(f"Write output not found: {out_dir}")

        print(f"\nReading {ext}...")

        # find the output file/directory
        read_path = self._find_readable_path(out_dir, ext)
        if read_path is None:
            pytest.fail(f"Could not find readable output in {out_dir}")

        arr, elapsed_ms = time_operation(mbo.imread, read_path)

        pipeline_results.add_timing("read", ext, elapsed_ms)
        pipeline_results.read_shapes[ext] = arr.shape

        print(f"  Read {arr.shape} in {elapsed_ms:.1f} ms")

        # verify we can index into it
        sample, sample_ms = time_operation(lambda: np.asarray(arr[0]))
        pipeline_results.add_timing("index_frame", ext, sample_ms)

        print(f"  First frame shape: {sample.shape}, indexed in {sample_ms:.1f} ms")

    @pytest.mark.parametrize("ext", WRITE_FORMATS)
    def test_04_suite2p_plane7(self, output_root, pipeline_results, ext):
        """run suite2p on plane 7 from each format."""
        pytest.importorskip("lbm_suite2p_python")
        from lbm_suite2p_python import run_plane

        out_dir = output_root / f"write_{ext.lstrip('.')}"
        s2p_out = output_root / f"suite2p_{ext.lstrip('.')}"
        s2p_out.mkdir(exist_ok=True)

        if not out_dir.exists():
            pytest.skip(f"Write output not found: {out_dir}")

        print(f"\nRunning suite2p on plane {SUITE2P_PLANE} from {ext}...")

        # find readable path
        read_path = self._find_readable_path(out_dir, ext)
        if read_path is None:
            pytest.fail(f"Could not find readable output in {out_dir}")

        # read array
        arr = mbo.imread(read_path)

        # extract plane 7
        if arr.ndim == 4:
            plane_data = arr[:, SUITE2P_PLANE, :, :]
        else:
            plane_data = arr

        print(f"  Plane data shape: {plane_data.shape}")

        # write to bin format for suite2p
        bin_dir = s2p_out / "plane07"
        bin_dir.mkdir(exist_ok=True)

        # prepare metadata
        n_frames = plane_data.shape[0]
        Ly, Lx = plane_data.shape[-2:]

        ops = {
            "nframes": n_frames,
            "Ly": Ly,
            "Lx": Lx,
            "fs": getattr(arr, "frame_rate", 10.0) or 10.0,
            "plane": SUITE2P_PLANE,
        }

        # write bin file
        _, write_ms = time_operation(
            mbo.imwrite,
            plane_data,
            bin_dir,
            ext=".bin",
            overwrite=True,
            metadata=ops,
        )
        pipeline_results.add_timing("extract_plane", ext, write_ms)

        print(f"  Extracted plane in {write_ms:.1f} ms")

        # run suite2p
        bin_file = bin_dir / "data_raw.bin"
        if not bin_file.exists():
            # check for data.bin
            bin_file = bin_dir / "data.bin"

        if not bin_file.exists():
            pytest.fail(f"No bin file found in {bin_dir}")

        print(f"  Running suite2p on {bin_file}...")

        _, s2p_ms = time_operation(
            run_plane,
            str(bin_file),
            save_path=str(bin_dir),
            ops=ops,
            keep_raw=False,
            keep_reg=True,
        )

        pipeline_results.add_timing("suite2p", ext, s2p_ms)
        print(f"  Suite2p completed in {s2p_ms:.1f} ms ({s2p_ms/1000:.1f} s)")

        # verify suite2p outputs
        stat_file = bin_dir / "stat.npy"
        if stat_file.exists():
            stat = np.load(stat_file, allow_pickle=True)
            n_cells = len(stat)
            pipeline_results.suite2p_results[ext] = {"n_cells": n_cells}
            print(f"  Found {n_cells} cells")
        else:
            pipeline_results.suite2p_results[ext] = {"n_cells": 0, "error": "no stat.npy"}
            print("  Warning: no stat.npy found")

    def test_99_summary(self, pipeline_results):
        """print timing summary."""
        pipeline_results.print_summary()

        # print suite2p results
        if pipeline_results.suite2p_results:
            print("\nSUITE2P RESULTS:")
            for fmt, res in pipeline_results.suite2p_results.items():
                print(f"  {fmt}: {res.get('n_cells', 0)} cells")

        # basic assertions
        assert len(pipeline_results.timings) > 0, "No timings recorded"

    def _find_readable_path(self, out_dir: Path, ext: str) -> Path | None:
        """find the path to read from a write output directory."""
        ext_clean = ext.lstrip(".").lower()

        if ext_clean in ("tif", "tiff"):
            # look for tiff files or plane directories
            tiffs = list(out_dir.rglob("*.tiff")) + list(out_dir.rglob("*.tif"))
            if tiffs:
                # if multiple plane files, return parent dir
                if len(tiffs) > 1 and "plane" in str(tiffs[0]):
                    return out_dir
                return tiffs[0]
            return out_dir

        elif ext_clean == "zarr":
            zarrs = list(out_dir.rglob("*.zarr"))
            if zarrs:
                return zarrs[0]
            # maybe out_dir itself is the zarr
            if (out_dir / "zarr.json").exists() or (out_dir / ".zarray").exists():
                return out_dir
            return None

        elif ext_clean == "bin":
            # look for ops.npy (suite2p format)
            ops_files = list(out_dir.rglob("ops.npy"))
            if ops_files:
                return ops_files[0].parent
            return None

        elif ext_clean in ("h5", "hdf5"):
            h5s = list(out_dir.rglob("*.h5")) + list(out_dir.rglob("*.hdf5"))
            if h5s:
                return h5s[0]
            return None

        return None
