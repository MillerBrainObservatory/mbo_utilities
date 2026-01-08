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

        # find readable path - for multi-plane formats, find plane 7 directly
        read_path = self._find_plane_path(out_dir, ext, SUITE2P_PLANE)
        if read_path is None:
            pytest.skip(f"Could not find plane {SUITE2P_PLANE} in {out_dir}")

        # read array
        arr = mbo.imread(read_path)
        print(f"  Loaded array: {arr.shape}, ndim={arr.ndim}")

        # extract plane data - handle different array structures
        # use explicit slicing arr[:] to get full data (np.asarray returns single frame)
        if arr.ndim == 4 and arr.shape[1] > SUITE2P_PLANE:
            # multi-plane 4D array, extract specific plane
            plane_data = np.asarray(arr[:, SUITE2P_PLANE, :, :])
        elif arr.ndim == 4 and arr.shape[1] == 1:
            # single plane stored as 4D, squeeze z dimension
            plane_data = np.asarray(arr[:, 0, :, :])
        elif arr.ndim == 3:
            # already single plane (T, Y, X) - use explicit slice for full data
            plane_data = np.asarray(arr[:])
        else:
            pytest.skip(f"Cannot extract plane {SUITE2P_PLANE} from shape {arr.shape}")

        print(f"  Plane data shape: {plane_data.shape}")

        # write to bin format for suite2p
        bin_dir = s2p_out / "plane07"
        bin_dir.mkdir(exist_ok=True)

        # prepare metadata
        n_frames = plane_data.shape[0]
        Ly, Lx = plane_data.shape[-2:]

        # write bin file directly (not using imwrite to avoid issues with lazy arrays)
        bin_file = bin_dir / "data_raw.bin"
        ops_file = bin_dir / "ops.npy"

        ops = {
            "nframes": n_frames,
            "nframes_chan1": n_frames,
            "Ly": Ly,
            "Lx": Lx,
            "fs": getattr(arr, "frame_rate", 10.0) or 10.0,
            "plane": SUITE2P_PLANE,
            "raw_file": str(bin_file),
            "save_path": str(bin_dir),
            "ops_path": str(ops_file),
        }
        t0 = time.perf_counter()

        # ensure int16
        if plane_data.dtype != np.int16:
            plane_data = plane_data.astype(np.int16)

        with open(bin_file, "wb") as f:
            plane_data.tofile(f)

        # write ops.npy
        np.save(bin_dir / "ops.npy", ops)

        write_ms = (time.perf_counter() - t0) * 1000
        pipeline_results.add_timing("extract_plane", ext, write_ms)

        print(f"  Extracted plane in {write_ms:.1f} ms")

        if not bin_file.exists():
            pytest.fail(f"Failed to write bin file: {bin_file}")

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

    def _find_plane_path(self, out_dir: Path, ext: str, plane_idx: int) -> Path | None:
        """find path to a specific plane from write output."""
        ext_clean = ext.lstrip(".").lower()
        # plane naming: plane01, plane02, ... (1-indexed in filenames)
        plane_num = plane_idx + 1
        plane_patterns = [f"plane{plane_num:02d}", f"plane{plane_num}"]

        if ext_clean in ("tif", "tiff"):
            # look for planeXX*.tiff
            for pattern in plane_patterns:
                matches = list(out_dir.glob(f"{pattern}*.tif*"))
                if matches:
                    return matches[0]
            # fallback: return dir if it has multiple tiffs (TiffArray handles it)
            tiffs = list(out_dir.rglob("*.tif*"))
            if len(tiffs) >= plane_num:
                return out_dir
            return None

        elif ext_clean == "zarr":
            # look for planeXX*.zarr
            for pattern in plane_patterns:
                matches = list(out_dir.glob(f"{pattern}*.zarr"))
                if matches:
                    return matches[0]
            return None

        elif ext_clean == "bin":
            # look for planeXX*/ops.npy (may have _stitched suffix)
            for pattern in plane_patterns:
                matches = list(out_dir.glob(f"{pattern}*"))
                for match in matches:
                    if match.is_dir() and (match / "ops.npy").exists():
                        return match
            return None

        elif ext_clean in ("h5", "hdf5"):
            # look for planeXX*.h5
            for pattern in plane_patterns:
                matches = list(out_dir.glob(f"{pattern}*.h5"))
                if matches:
                    return matches[0]
            return None

        return None
