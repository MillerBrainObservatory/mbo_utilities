"""
benchmarking utilities for mbo_utilities.

provides reproducible performance benchmarks for:
- MboRawArray initialization and metadata extraction
- frame indexing (single, batch, z-plane selection)
- phase correction variants (off, correlation, fft)
- writing to supported formats (.zarr, .tiff, .h5, .bin, .npy)
"""
from __future__ import annotations

import json
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from mbo_utilities import log

logger = log.get("benchmarks")


@dataclass
class BenchmarkConfig:
    """
    configuration for benchmark runs.

    use presets for common scenarios:
        config = BenchmarkConfig.quick()   # fast sanity check
        config = BenchmarkConfig.full()    # comprehensive suite
        config = BenchmarkConfig.read_only()  # skip writes
    """

    # frame counts to test for indexing
    frame_counts: tuple[int, ...] = (1, 10, 200, 1000)

    # phase correction variants
    test_no_phase: bool = True
    test_phase_corr: bool = True
    test_phase_fft: bool = True

    # z-plane indexing tests
    test_zplane_indexing: bool = True

    # access pattern tests
    test_random_access: bool = True  # contiguous vs strided vs random
    test_parallel_planes: bool = True  # first/middle/last z-plane comparison
    test_cold_warm: bool = True  # cold start vs warm cache reads

    # performance analysis
    test_throughput: bool = True  # MB/s calculation
    test_scaling: bool = True  # time vs frame count analysis

    # write tests
    write_formats: tuple[str, ...] = (".zarr", ".tiff", ".h5", ".bin")
    write_num_frames: int = 100
    write_full_dataset: bool = False  # write entire dataset (no num_frames limit)
    keep_written_files: bool = False  # keep output files after benchmark
    zarr_levels: tuple[int, ...] = (0, 1, 5, 9)  # gzip compression levels (0=none)
    zarr_sharding: tuple[bool, ...] = (True, False)  # test sharded vs non-sharded

    # advanced write tests
    test_chunk_sizes: bool = True  # vary target_chunk_mb
    chunk_sizes_mb: tuple[int, ...] = (10, 25, 50, 100, 200)
    test_readback: bool = True  # write then read back
    test_file_sizes: bool = True  # measure compression ratios

    # timing settings
    repeats: int = 3
    warmup: bool = True

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """quick test: 10, 200 frames, no FFT, fewer formats."""
        return cls(
            frame_counts=(10, 200),
            test_phase_fft=False,
            test_zplane_indexing=False,
            test_random_access=False,
            test_parallel_planes=False,
            test_cold_warm=False,
            test_throughput=False,
            test_scaling=False,
            write_formats=(".zarr", ".tiff"),
            zarr_levels=(1,),
            zarr_sharding=(True,),
            test_chunk_sizes=False,
            test_readback=False,
            test_file_sizes=False,
            repeats=2,
        )

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        """full benchmark suite with all tests."""
        return cls()

    @classmethod
    def read_only(cls) -> "BenchmarkConfig":
        """skip write benchmarks entirely."""
        return cls(
            write_formats=(),
            test_chunk_sizes=False,
            test_readback=False,
            test_file_sizes=False,
        )

    @classmethod
    def write_only(cls) -> "BenchmarkConfig":
        """only test write operations."""
        return cls(
            frame_counts=(),
            test_no_phase=False,
            test_phase_corr=False,
            test_phase_fft=False,
            test_zplane_indexing=False,
            test_random_access=False,
            test_parallel_planes=False,
            test_cold_warm=False,
            test_throughput=False,
            test_scaling=False,
        )

    @classmethod
    def analysis(cls) -> "BenchmarkConfig":
        """performance analysis focus: throughput, scaling, access patterns."""
        return cls(
            frame_counts=(10, 50, 100, 200),
            test_no_phase=False,
            test_phase_corr=False,
            test_phase_fft=False,
            test_random_access=True,
            test_parallel_planes=True,
            test_cold_warm=True,
            test_throughput=True,
            test_scaling=True,
            write_formats=(),
            test_chunk_sizes=False,
            test_readback=False,
            test_file_sizes=False,
        )


@dataclass
class TimingStats:
    """statistics for a set of timing measurements."""

    times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_times(cls, times_ms: list[float]) -> "TimingStats":
        """compute stats from raw timing list."""
        arr = np.array(times_ms)
        return cls(
            times_ms=times_ms,
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )


@dataclass
class BenchmarkResult:
    """complete benchmark results with metadata."""

    timestamp: str = ""
    git_commit: str = ""
    label: str = ""
    system_info: dict = field(default_factory=dict)
    data_info: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """convert to serializable dict."""
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """save results to json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"saved benchmark results to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResult":
        """load results from json file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def get_system_info() -> dict:
    """collect system information for benchmark context."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # try to get cpu count
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # try to get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_gb"] = round(mem.total / (1024**3), 1)
    except ImportError:
        pass

    # try to get gpu info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
    except Exception:
        pass

    return info


def get_git_commit() -> str:
    """get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _time_func(func, *args, **kwargs) -> tuple[Any, float]:
    """time a function call, return (result, time_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0) * 1000


def _suppress_tifffile_warnings():
    """suppress tifffile stderr output during benchmark reads."""
    import sys
    import os
    import logging

    # suppress tifffile logger
    logging.getLogger("tifffile").setLevel(logging.ERROR)

    # context manager to suppress stderr
    class SuppressStderr:
        def __enter__(self):
            self._stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            return self

        def __exit__(self, *args):
            sys.stderr.close()
            sys.stderr = self._stderr

    return SuppressStderr()


def benchmark_init(
    files: str | Path | list,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, TimingStats]:
    """
    benchmark MboRawArray initialization.

    measures:
    - total imread() time
    - internal timing breakdown (if available)

    parameters
    ----------
    files : path or list of paths
        scanimage tiff files to load
    repeats : int
        number of timing iterations
    warmup : bool
        run one warmup iteration first

    returns
    -------
    dict
        timing stats for each metric
    """
    from mbo_utilities import imread

    times_total = []

    # warmup run
    if warmup:
        _ = imread(files)

    for _ in range(repeats):
        _, elapsed = _time_func(imread, files)
        times_total.append(elapsed)

    return {
        "init_total": TimingStats.from_times(times_total),
    }


def benchmark_indexing(
    arr,
    frame_counts: tuple[int, ...] = (1, 10, 200, 1000),
    repeats: int = 3,
    warmup: bool = True,
    test_zplane: bool = True,
) -> dict[str, TimingStats]:
    """
    benchmark array indexing operations.

    tests frame batches and z-plane selection patterns.

    parameters
    ----------
    arr : MboRawArray
        array to benchmark
    frame_counts : tuple of int
        number of frames to read in each test
    repeats : int
        timing iterations per test
    warmup : bool
        run warmup iteration
    test_zplane : bool
        include z-plane indexing tests

    returns
    -------
    dict
        timing stats for each indexing pattern
    """
    results = {}
    max_frames = arr.shape[0]
    num_zplanes = arr.shape[1] if arr.ndim >= 4 else 1

    # warmup
    if warmup:
        _ = arr[0]

    # full volume reads: arr[0:N] reads N timepoints with all z-planes
    for n in frame_counts:
        if n > max_frames:
            logger.warning(f"skipping {n} timepoints test (only {max_frames} available)")
            continue

        times = []
        for _ in range(repeats):
            if n == 1:
                _, elapsed = _time_func(lambda: arr[0])
            else:
                _, elapsed = _time_func(lambda: arr[0:n])
            times.append(elapsed)

        results[f"arr[:{n}] ({n}t x {num_zplanes}z)"] = TimingStats.from_times(times)

    # single z-plane reads: arr[0:N, z] reads N timepoints but only one z-plane
    if test_zplane and num_zplanes > 1:
        # single timepoint, single z-plane
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0, 0])
            times.append(elapsed)
        results["arr[0, 0] (1 frame)"] = TimingStats.from_times(times)

        # 10 timepoints, single z-plane
        if max_frames >= 10:
            times = []
            for _ in range(repeats):
                _, elapsed = _time_func(lambda: arr[0:10, 0])
                times.append(elapsed)
            results["arr[:10, 0] (10 frames)"] = TimingStats.from_times(times)

        # single timepoint, single z-plane, spatial crop
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0, 0, 50:150, 50:150])
            times.append(elapsed)
        results["arr[0, 0, 50:150, 50:150]"] = TimingStats.from_times(times)

    return results


def benchmark_random_access(
    arr,
    num_samples: int = 20,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, TimingStats]:
    """
    benchmark random vs sequential access patterns.

    compares contiguous reads against strided/random reads to measure
    seek overhead and cache effects. useful for understanding whether
    your access pattern matters for performance.

    tests:
    - contiguous: arr[0:N] - sequential read
    - strided: arr[::10] - every 10th frame (forces seeks)
    - random: arr[random_indices] - worst case random access

    parameters
    ----------
    arr : MboRawArray
        array to benchmark
    num_samples : int
        number of frames to read in each test
    repeats : int
        timing iterations per test
    warmup : bool
        run warmup iteration

    returns
    -------
    dict
        timing stats for each access pattern
    """
    results = {}
    max_frames = arr.shape[0]
    num_samples = min(num_samples, max_frames // 10)  # ensure we have room for stride

    if num_samples < 5:
        logger.warning("not enough frames for random access benchmark")
        return results

    # warmup
    if warmup:
        _ = arr[0]

    # contiguous read: arr[0:N]
    times = []
    for _ in range(repeats):
        _, elapsed = _time_func(lambda: arr[0:num_samples])
        times.append(elapsed)
    results[f"contiguous arr[:{num_samples}]"] = TimingStats.from_times(times)

    # strided read: arr[::stride] where stride gives ~same number of frames
    stride = max(2, max_frames // num_samples)
    actual_strided = len(range(0, max_frames, stride))
    times = []
    for _ in range(repeats):
        _, elapsed = _time_func(lambda: arr[::stride])
        times.append(elapsed)
    results[f"strided arr[::{ stride}] ({actual_strided} frames)"] = TimingStats.from_times(times)

    # random access: random indices
    rng = np.random.default_rng(42)
    random_indices = sorted(rng.choice(max_frames, size=num_samples, replace=False))
    times = []
    for _ in range(repeats):
        _, elapsed = _time_func(lambda: arr[random_indices])
        times.append(elapsed)
    results[f"random ({num_samples} indices)"] = TimingStats.from_times(times)

    return results


def benchmark_parallel_planes(
    arr,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, TimingStats]:
    """
    benchmark reading different z-planes.

    checks if plane position affects read speed (e.g., first vs middle vs last).
    useful for detecting whether certain planes have higher I/O cost.

    parameters
    ----------
    arr : MboRawArray
        array to benchmark (must be 4D with Z dimension)
    repeats : int
        timing iterations per test
    warmup : bool
        run warmup iteration

    returns
    -------
    dict
        timing stats for each plane position
    """
    results = {}
    if arr.ndim < 4 or arr.shape[1] < 3:
        logger.warning("need 4D array with at least 3 z-planes for parallel plane benchmark")
        return results

    num_zplanes = arr.shape[1]
    max_frames = min(50, arr.shape[0])

    # warmup
    if warmup:
        _ = arr[0, 0]

    # test first, middle, and last z-planes
    planes_to_test = [
        (0, "first"),
        (num_zplanes // 2, "middle"),
        (num_zplanes - 1, "last"),
    ]

    for plane_idx, label in planes_to_test:
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda p=plane_idx: arr[:max_frames, p])
            times.append(elapsed)
        results[f"plane {plane_idx} ({label})"] = TimingStats.from_times(times)

    return results


def benchmark_cold_warm_reads(
    files: str | Path | list,
    num_frames: int = 50,
    repeats: int = 3,
) -> dict[str, TimingStats]:
    """
    benchmark cold start vs warm cache reads.

    measures the difference between first read after opening a file (cold)
    and subsequent reads (warm). large differences indicate filesystem/OS
    caching is significant for your workload.

    parameters
    ----------
    files : path or list
        scanimage tiff files
    num_frames : int
        frames to read in each test
    repeats : int
        number of cold/warm cycles

    returns
    -------
    dict
        timing stats for cold and warm reads
    """
    from mbo_utilities import imread
    import gc

    cold_times = []
    warm_times = []

    for _ in range(repeats):
        # force garbage collection and clear any cached state
        gc.collect()

        # cold read: first read after fresh imread
        arr = imread(files)
        num_frames_actual = min(num_frames, arr.shape[0])
        _, cold_elapsed = _time_func(lambda: arr[0:num_frames_actual])
        cold_times.append(cold_elapsed)

        # warm read: subsequent read on same array
        _, warm_elapsed = _time_func(lambda: arr[0:num_frames_actual])
        warm_times.append(warm_elapsed)

        # cleanup
        del arr
        gc.collect()

    return {
        f"cold read ({num_frames} frames)": TimingStats.from_times(cold_times),
        f"warm read ({num_frames} frames)": TimingStats.from_times(warm_times),
    }


def benchmark_throughput(
    arr,
    frame_counts: tuple[int, ...] = (10, 50, 100, 200),
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, dict]:
    """
    measure memory throughput in MB/s.

    calculates actual read speed to compare against theoretical disk limits.
    helps identify whether you're I/O bound or compute bound.

    parameters
    ----------
    arr : MboRawArray
        array to benchmark
    frame_counts : tuple of int
        frame counts to test
    repeats : int
        timing iterations
    warmup : bool
        run warmup iteration

    returns
    -------
    dict
        throughput stats (MB/s) for each frame count
    """
    results = {}
    max_frames = arr.shape[0]
    dtype_size = np.dtype(arr.dtype).itemsize

    # warmup
    if warmup:
        _ = arr[0]

    for n in frame_counts:
        if n > max_frames:
            continue

        # calculate bytes for this read
        if arr.ndim == 4:
            bytes_read = n * arr.shape[1] * arr.shape[2] * arr.shape[3] * dtype_size
        else:
            bytes_read = n * arr.shape[1] * arr.shape[2] * dtype_size

        mb_read = bytes_read / (1024 * 1024)

        times = []
        throughputs = []
        for _ in range(repeats):
            _, elapsed_ms = _time_func(lambda: arr[0:n])
            times.append(elapsed_ms)
            throughputs.append(mb_read / (elapsed_ms / 1000))  # MB/s

        results[f"{n} frames ({mb_read:.1f} MB)"] = {
            "time_ms": TimingStats.from_times(times),
            "throughput_mbs": {
                "mean": float(np.mean(throughputs)),
                "std": float(np.std(throughputs)),
                "min": float(np.min(throughputs)),
                "max": float(np.max(throughputs)),
            },
        }

    return results


def benchmark_scaling(
    arr,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, dict]:
    """
    analyze read time scaling with frame count.

    tests whether read time scales linearly with data size or has
    fixed overhead. plots time vs N to identify overhead costs.

    returns both raw timings and computed overhead/per-frame costs.

    parameters
    ----------
    arr : MboRawArray
        array to benchmark
    repeats : int
        timing iterations
    warmup : bool
        run warmup iteration

    returns
    -------
    dict
        scaling analysis with overhead estimate
    """
    results = {}
    max_frames = arr.shape[0]

    # test exponentially increasing frame counts
    frame_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    frame_counts = [n for n in frame_counts if n <= max_frames]

    if len(frame_counts) < 3:
        logger.warning("not enough frames for scaling analysis")
        return results

    # warmup
    if warmup:
        _ = arr[0]

    timings = []
    for n in frame_counts:
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0:n])
            times.append(elapsed)
        mean_time = np.mean(times)
        timings.append((n, mean_time))
        results[f"{n} frames"] = {
            "mean_ms": float(mean_time),
            "std_ms": float(np.std(times)),
        }

    # linear regression to estimate overhead and per-frame cost
    if len(timings) >= 3:
        x = np.array([t[0] for t in timings])
        y = np.array([t[1] for t in timings])
        # fit: time = overhead + per_frame * N
        slope, intercept = np.polyfit(x, y, 1)
        results["_scaling_analysis"] = {
            "overhead_ms": float(max(0, intercept)),
            "per_frame_ms": float(slope),
            "r_squared": float(1 - np.var(y - (slope * x + intercept)) / np.var(y)),
        }

    return results


def benchmark_chunk_sizes(
    arr,
    chunk_sizes_mb: tuple[int, ...] = (10, 25, 50, 100, 200),
    num_frames: int = 100,
    output_dir: Path | None = None,
    repeats: int = 2,
    keep_files: bool = False,
) -> dict[str, TimingStats]:
    """
    benchmark impact of chunk size on write performance.

    varies target_chunk_mb parameter to find optimal chunk size
    for your disk and data characteristics.

    parameters
    ----------
    arr : lazy array
        source array
    chunk_sizes_mb : tuple of int
        chunk sizes to test
    num_frames : int
        frames to write
    output_dir : Path, optional
        output directory
    repeats : int
        timing iterations
    keep_files : bool
        keep output files

    returns
    -------
    dict
        timing stats for each chunk size
    """
    from mbo_utilities import imwrite
    import shutil

    results = {}
    write_frames = min(num_frames, arr.shape[0])

    cleanup_temp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="mbo_chunk_bench_"))
        cleanup_temp = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_mb in chunk_sizes_mb:
        times = []
        for i in range(repeats):
            out_path = output_dir / f"chunk_{chunk_mb}mb_{i}"
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)

            _, elapsed = _time_func(
                imwrite,
                arr,
                out_path,
                ext=".zarr",
                num_frames=write_frames,
                overwrite=True,
                target_chunk_mb=chunk_mb,
            )
            times.append(elapsed)

            if not keep_files and i < repeats - 1:
                shutil.rmtree(out_path, ignore_errors=True)

        results[f"chunk={chunk_mb}MB"] = TimingStats.from_times(times)

    if cleanup_temp and not keep_files:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    return results


def benchmark_readback(
    arr,
    formats: tuple[str, ...] = (".zarr", ".tiff", ".h5"),
    num_frames: int = 50,
    output_dir: Path | None = None,
    repeats: int = 3,
) -> dict[str, dict]:
    """
    benchmark write then read-back performance.

    writes data to each format, then reads it back to measure
    round-trip performance. helps choose format for read-heavy workloads.

    note: .bin format is excluded from readback tests because Suite2pArray
    requires matching ops.npy metadata which doesn't work with partial writes.

    parameters
    ----------
    arr : lazy array
        source array
    formats : tuple of str
        formats to test (default excludes .bin)
    num_frames : int
        frames to write/read
    output_dir : Path, optional
        output directory
    repeats : int
        timing iterations

    returns
    -------
    dict
        write and read timing for each format
    """
    from mbo_utilities import imread, imwrite
    import shutil

    results = {}
    write_frames = min(num_frames, arr.shape[0])

    # exclude .bin from readback - ops.npy metadata mismatch causes errors
    formats = tuple(f for f in formats if f != ".bin")

    cleanup_temp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="mbo_readback_"))
        cleanup_temp = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        write_times = []
        read_times = []

        for i in range(repeats):
            out_path = output_dir / f"readback_{ext.lstrip('.')}_{i}"
            if out_path.exists():
                if out_path.is_dir():
                    shutil.rmtree(out_path, ignore_errors=True)
                else:
                    out_path.unlink(missing_ok=True)

            # write
            _, write_elapsed = _time_func(
                imwrite, arr, out_path, ext=ext, num_frames=write_frames, overwrite=True
            )
            write_times.append(write_elapsed)

            # find the actual written file/dir
            if ext == ".zarr":
                read_path = out_path
            else:
                # find written file (tiff, h5, etc)
                files = list(out_path.glob(f"*{ext}")) + list(out_path.glob("*.tif"))
                read_path = files[0] if files else out_path

            # read back with warnings suppressed
            if read_path.exists():
                def do_read(p=read_path):
                    with _suppress_tifffile_warnings():
                        data = imread(p)
                        # just read the array, don't slice (may fail for some formats)
                        return np.asarray(data[:write_frames] if hasattr(data, '__getitem__') else data)
                try:
                    _, read_elapsed = _time_func(do_read)
                    read_times.append(read_elapsed)
                except Exception as e:
                    logger.warning(f"readback failed for {ext}: {e}")

            # cleanup between iterations
            if out_path.exists():
                if out_path.is_dir():
                    shutil.rmtree(out_path, ignore_errors=True)
                else:
                    out_path.unlink(missing_ok=True)

        results[ext] = {
            "write": asdict(TimingStats.from_times(write_times)),
            "read": asdict(TimingStats.from_times(read_times)) if read_times else None,
        }

    if cleanup_temp:
        shutil.rmtree(output_dir, ignore_errors=True)

    return results


def benchmark_file_sizes(
    arr,
    formats: tuple[str, ...] = (".zarr", ".tiff", ".h5", ".bin"),
    num_frames: int = 100,
    output_dir: Path | None = None,
    zarr_levels: tuple[int, ...] = (1, 5, 9),
) -> dict[str, dict]:
    """
    measure output file sizes and compression ratios.

    writes data to each format and measures resulting file size.
    helps choose format based on storage constraints.

    parameters
    ----------
    arr : lazy array
        source array
    formats : tuple of str
        formats to test
    num_frames : int
        frames to write
    output_dir : Path, optional
        output directory
    zarr_levels : tuple of int
        compression levels for zarr

    returns
    -------
    dict
        file sizes and compression ratios
    """
    from mbo_utilities import imwrite
    import shutil

    results = {}
    write_frames = min(num_frames, arr.shape[0])

    # calculate theoretical uncompressed size
    dtype_size = np.dtype(arr.dtype).itemsize
    if arr.ndim == 4:
        raw_bytes = write_frames * arr.shape[1] * arr.shape[2] * arr.shape[3] * dtype_size
    else:
        raw_bytes = write_frames * arr.shape[1] * arr.shape[2] * dtype_size
    raw_mb = raw_bytes / (1024 * 1024)

    cleanup_temp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="mbo_filesize_"))
        cleanup_temp = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def get_dir_size(path: Path) -> int:
        """get total size of directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    for ext in formats:
        if ext == ".zarr":
            # test multiple compression levels
            for level in zarr_levels:
                out_path = output_dir / f"size_zarr_L{level}"
                if out_path.exists():
                    shutil.rmtree(out_path, ignore_errors=True)

                imwrite(arr, out_path, ext=ext, num_frames=write_frames, overwrite=True, level=level)

                size_bytes = get_dir_size(out_path)
                size_mb = size_bytes / (1024 * 1024)
                ratio = raw_bytes / size_bytes if size_bytes > 0 else 0

                results[f"zarr L{level}"] = {
                    "size_mb": round(size_mb, 2),
                    "raw_mb": round(raw_mb, 2),
                    "compression_ratio": round(ratio, 2),
                }

                shutil.rmtree(out_path, ignore_errors=True)
        else:
            out_path = output_dir / f"size_{ext.lstrip('.')}"
            if out_path.exists():
                if out_path.is_dir():
                    shutil.rmtree(out_path, ignore_errors=True)
                else:
                    out_path.unlink(missing_ok=True)

            imwrite(arr, out_path, ext=ext, num_frames=write_frames, overwrite=True)

            size_bytes = get_dir_size(out_path)
            size_mb = size_bytes / (1024 * 1024)
            ratio = raw_bytes / size_bytes if size_bytes > 0 else 0

            results[ext] = {
                "size_mb": round(size_mb, 2),
                "raw_mb": round(raw_mb, 2),
                "compression_ratio": round(ratio, 2),
            }

            if out_path.exists():
                if out_path.is_dir():
                    shutil.rmtree(out_path, ignore_errors=True)
                else:
                    out_path.unlink(missing_ok=True)

    if cleanup_temp:
        shutil.rmtree(output_dir, ignore_errors=True)

    return results


def benchmark_phase_variants(
    files: str | Path | list,
    frame_count: int = 100,
    repeats: int = 3,
    test_no_phase: bool = True,
    test_phase_corr: bool = True,
    test_phase_fft: bool = True,
) -> dict[str, TimingStats]:
    """
    benchmark phase correction variants.

    compares reading frames with different phase correction settings.

    parameters
    ----------
    files : path or list
        scanimage tiff files
    frame_count : int
        frames to read in each test
    repeats : int
        timing iterations
    test_no_phase : bool
        test with fix_phase=False
    test_phase_corr : bool
        test with fix_phase=True, use_fft=False
    test_phase_fft : bool
        test with fix_phase=True, use_fft=True

    returns
    -------
    dict
        timing stats for each variant
    """
    from mbo_utilities import imread

    results = {}

    variants = []
    if test_no_phase:
        variants.append(("no_phase", {"fix_phase": False}))
    if test_phase_corr:
        variants.append(("phase_corr", {"fix_phase": True, "use_fft": False}))
    if test_phase_fft:
        variants.append(("phase_fft", {"fix_phase": True, "use_fft": True}))

    for name, kwargs in variants:
        arr = imread(files, **kwargs)
        max_frames = min(frame_count, arr.shape[0])

        # warmup
        _ = arr[0]

        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0:max_frames])
            times.append(elapsed)

        results[name] = TimingStats.from_times(times)

    return results


def benchmark_writes(
    arr,
    formats: tuple[str, ...] = (".zarr", ".tiff", ".h5", ".bin"),
    num_frames: int | None = 100,
    output_dir: Path | None = None,
    repeats: int = 3,
    keep_files: bool = False,
    zarr_levels: tuple[int, ...] = (1, 3, 5, 9),
    zarr_sharding: tuple[bool, ...] = (True, False),
) -> dict[str, TimingStats]:
    """
    benchmark writing to different file formats.

    parameters
    ----------
    arr : lazy array
        source array to write from
    formats : tuple of str
        file extensions to test
    num_frames : int or None
        frames to write in each test. None = full dataset
    output_dir : Path, optional
        directory for output files (temp dir if None)
    repeats : int
        timing iterations per format
    keep_files : bool
        keep output files after benchmark (default: cleanup)
    zarr_levels : tuple of int
        compression levels to test for zarr (gzip 1-9)
    zarr_sharding : tuple of bool
        sharding modes to test (True=sharded, False=non-sharded)

    returns
    -------
    dict
        timing stats for each format
    """
    from mbo_utilities import imwrite

    results = {}
    total_frames = arr.shape[0]
    write_frames = total_frames if num_frames is None else min(num_frames, total_frames)
    is_full = num_frames is None or write_frames == total_frames

    logger.info(f"writing {write_frames} frames ({'full dataset' if is_full else 'subset'})")

    # use temp dir if not specified
    cleanup_temp = False
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="mbo_bench_"))
        cleanup_temp = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in formats:
        # for zarr, test multiple compression levels and sharding modes
        if ext == ".zarr" and zarr_levels:
            for sharded in zarr_sharding:
                shard_suffix = "" if sharded else " no-shard"
                file_label = "sharded" if sharded else "noshard"
                for level in zarr_levels:
                    times = []
                    for i in range(repeats):
                        out_path = output_dir / f"bench_zarr_{file_label}_L{level}_{i}"
                        if out_path.exists():
                            import shutil
                            shutil.rmtree(out_path, ignore_errors=True)

                        _, elapsed = _time_func(
                            imwrite,
                            arr,
                            out_path,
                            ext=ext,
                            num_frames=write_frames,
                            overwrite=True,
                            level=level,
                            sharded=sharded,
                        )
                        times.append(elapsed)

                        if not keep_files and i < repeats - 1:
                            import shutil
                            if out_path.exists():
                                shutil.rmtree(out_path, ignore_errors=True)

                    key = f"zarr L{level}{shard_suffix}"
                    results[key] = TimingStats.from_times(times)
                    logger.info(f"  {key}: {results[key].mean_ms:.1f} ± {results[key].std_ms:.1f} ms")
        else:
            times = []
            for i in range(repeats):
                out_path = output_dir / f"bench_{ext.lstrip('.')}_{i}"
                if out_path.exists():
                    import shutil
                    shutil.rmtree(out_path, ignore_errors=True)

                _, elapsed = _time_func(
                    imwrite,
                    arr,
                    out_path,
                    ext=ext,
                    num_frames=write_frames,
                    overwrite=True,
                )
                times.append(elapsed)

                if not keep_files and i < repeats - 1:
                    import shutil
                    if out_path.exists():
                        if out_path.is_dir():
                            shutil.rmtree(out_path, ignore_errors=True)
                        else:
                            out_path.unlink(missing_ok=True)

            results[ext] = TimingStats.from_times(times)
            logger.info(f"  {ext}: {results[ext].mean_ms:.1f} ± {results[ext].std_ms:.1f} ms")

    # cleanup temp dir (unless keeping files)
    if cleanup_temp and not keep_files:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
    elif keep_files:
        logger.info(f"output files saved to: {output_dir}")

    return results


def get_memory_mb() -> float:
    """get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_zarr_chunking(
    arr,
    output_dir: Path | None = None,
    num_timepoints: int | None = None,
    level: int = 0,
    repeats: int = 2,
    keep_files: bool = False,
) -> dict[str, dict]:
    """
    benchmark zarr write/read with various chunk configurations.

    tests different combinations of shard size (outer) and chunk size (inner)
    to find optimal configuration for specific access patterns.

    parameters
    ----------
    arr : array
        source array to write (must support imwrite)
    output_dir : Path, optional
        output directory for test files
    num_timepoints : int, optional
        timepoints to write/read in each test (default: all)
    level : int
        compression level (0 = no compression)
    repeats : int
        timing iterations
    keep_files : bool
        keep output files after benchmark

    returns
    -------
    dict
        results with write_ms, read_ms, fps, ram_mb, file_size_mb for each config
    """
    import shutil
    import gc
    from mbo_utilities import imread
    from mbo_utilities._writers import _write_zarr

    cleanup_temp = output_dir is None
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="bench_zarr_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = arr.shape[-2:]
    total_available = arr.shape[0]
    write_timepoints = min(num_timepoints, total_available) if num_timepoints else total_available

    # for multi-plane arrays, benchmark uses plane 0 only (single 3D slice)
    # this avoids multi-file output and tests pure zarr chunking performance
    is_multiplane = arr.ndim == 4 and arr.shape[1] > 1
    if is_multiplane:
        logger.info(f"using plane 0 of {arr.shape[1]} planes for zarr benchmark")

    frame_bytes = h * w * 2  # assuming uint16
    total_bytes = write_timepoints * frame_bytes

    results = {}

    # build configurations dynamically based on num_timepoints
    # constraint: shard_size must be divisible by chunk_t (zarr requirement)
    # (shard_frames, chunk_shape, label)
    configs = []

    # candidate shard sizes (filter to those <= num_timepoints)
    candidate_shards = [s for s in [1, 10, 50, 100, 200, 500] if s <= write_timepoints]

    # 1-frame inner chunks with varying shard sizes (always valid since any N % 1 == 0)
    for shard in candidate_shards:
        configs.append((shard, (1, h, w), f"shard={shard} chunk=1t"))

    # sub-frame chunks (spatial tiling) - only with shard=1
    if 1 in candidate_shards:
        configs.append((1, (1, h // 2, w // 2), "shard=1 chunk=1t/4tile"))
        configs.append((1, (1, h // 4, w // 4), "shard=1 chunk=1t/16tile"))

    # multi-frame inner chunks - shard must be divisible by chunk_t
    # test with shard sizes that are multiples of common chunk sizes
    chunk_t_candidates = [5, 10, 25, 50, 100]
    for chunk_t in chunk_t_candidates:
        if chunk_t > write_timepoints:
            continue
        # find a shard size that's divisible by chunk_t and <= write_timepoints
        # prefer 100, then 200, then 500, then exact multiple
        for shard in [100, 200, 500]:
            if shard <= write_timepoints and shard % chunk_t == 0:
                configs.append((shard, (chunk_t, h, w), f"shard={shard} chunk={chunk_t}t"))
                break
        else:
            # use chunk_t as shard (shard == chunk, single chunk per shard)
            if chunk_t <= write_timepoints:
                configs.append((chunk_t, (chunk_t, h, w), f"shard={chunk_t} chunk={chunk_t}t"))

    # filter valid configs
    valid_configs = [
        (sf, cs, lbl) for sf, cs, lbl in configs if cs[0] <= sf
    ]
    total_configs = len(valid_configs)

    try:
        from tqdm import tqdm
        config_iter = tqdm(valid_configs, desc="zarr configs", unit="cfg")
    except ImportError:
        config_iter = valid_configs
        print(f"testing {total_configs} zarr configurations...")

    for idx, (shard_frames, chunk_shape, label) in enumerate(config_iter):
        shard_frames_actual = shard_frames

        write_times = []
        read_times = []
        ram_samples = []
        file_size = 0

        for i in range(repeats):
            out_path = output_dir / f"zarr_bench_{idx}.zarr"
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)

            # measure write
            gc.collect()
            ram_before = get_memory_mb()

            # extract data to write (plane 0 for 4D, full array for 3D)
            if is_multiplane:
                write_data = arr[:write_timepoints, 0]
            else:
                write_data = arr[:write_timepoints]

            # force load into memory for consistent timing
            if hasattr(write_data, 'compute'):
                write_data = write_data.compute()
            elif not isinstance(write_data, np.ndarray):
                write_data = np.asarray(write_data)

            # write directly to zarr, bypassing multi-plane logic
            _, write_elapsed = _time_func(
                _write_zarr,
                out_path,
                write_data,
                overwrite=True,
                metadata={"num_frames": write_timepoints},
                level=level,
                sharded=True,
                shard_frames=shard_frames_actual,
                chunk_shape=chunk_shape,
            )
            write_times.append(write_elapsed)

            # clear zarr writer cache to flush and close the store
            if hasattr(_write_zarr, "_arrays"):
                _write_zarr._arrays.pop(out_path, None)
                _write_zarr._offsets.pop(out_path, None)
                _write_zarr._groups.pop(out_path, None)

            ram_after = get_memory_mb()
            ram_samples.append(ram_after - ram_before)

            # measure file size (first iteration only)
            if i == 0:
                file_size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())

            # measure read
            with _suppress_tifffile_warnings():
                z = imread(out_path)
                _, read_elapsed = _time_func(lambda: np.asarray(z[:write_timepoints]))
                read_times.append(read_elapsed)
                del z

            # always cleanup after each repeat to save space
            if not keep_files:
                shutil.rmtree(out_path, ignore_errors=True)

        # compute stats
        write_mean = np.mean(write_times)
        read_mean = np.mean(read_times)
        fps_write = (write_timepoints / write_mean) * 1000
        fps_read = (write_timepoints / read_mean) * 1000
        ram_mean = np.mean(ram_samples) if ram_samples else 0

        results[label] = {
            "write_ms": round(write_mean, 1),
            "write_std": round(np.std(write_times), 1),
            "read_ms": round(read_mean, 1),
            "read_std": round(np.std(read_times), 1),
            "fps_write": round(fps_write, 1),
            "fps_read": round(fps_read, 1),
            "ram_delta_mb": round(ram_mean, 1),
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "compression_ratio": round(total_bytes / file_size, 2) if file_size > 0 else 0,
            "shard_frames": shard_frames_actual,
            "chunk_shape": chunk_shape,
        }

        # update progress bar description with current result
        try:
            config_iter.set_postfix_str(
                f"w:{fps_write:.0f} r:{fps_read:.0f} fps"
            )
        except AttributeError:
            # no tqdm, print result
            print(
                f"  [{idx + 1}/{total_configs}] {label:30s}: "
                f"write {write_mean:7.1f} ms ({fps_write:5.0f} fps), "
                f"read {read_mean:7.1f} ms ({fps_read:5.0f} fps)"
            )

    # cleanup temp directory
    if cleanup_temp and not keep_files:
        shutil.rmtree(output_dir, ignore_errors=True)

    return results


def print_zarr_benchmark(results: dict):
    """print zarr chunking benchmark results."""
    print("\nZARR CHUNKING BENCHMARK")
    print("-" * 90)
    print(f"{'config':32s} {'write':>10s} {'read':>10s} {'w-fps':>8s} {'r-fps':>8s} {'size':>8s} {'ratio':>6s}")
    print("-" * 90)

    for label, data in results.items():
        print(
            f"{label:32s} "
            f"{data['write_ms']:>7.0f} ms "
            f"{data['read_ms']:>7.0f} ms "
            f"{data['fps_write']:>7.0f} "
            f"{data['fps_read']:>7.0f} "
            f"{data['file_size_mb']:>6.1f} MB "
            f"{data['compression_ratio']:>5.2f}x"
        )


def plot_zarr_benchmark(
    results: dict,
    output_path: Path | str | None = None,
    show: bool = True,
    title: str = "Zarr Chunking Benchmark",
) -> "Figure | None":
    """
    generate dark-mode visualization of zarr chunking benchmark.

    creates a 2-panel figure showing:
    - write/read FPS by configuration (grouped bar chart)
    - file size and compression ratio

    parameters
    ----------
    results : dict
        benchmark results from benchmark_zarr_chunking()
    output_path : Path or str, optional
        save figure to this path
    show : bool
        display the figure
    title : str
        figure title

    returns
    -------
    Figure or None
        matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return None

    # lazy import for theme
    from mbo_utilities.benchmarks import MBO_DARK_THEME, _apply_mbo_style
    colors = MBO_DARK_THEME

    # separate configs into categories for clearer visualization
    shard_configs = {}  # shard=X chunk=1t
    tile_configs = {}   # spatial tiling
    multi_configs = {}  # multi-frame chunks

    for label, data in results.items():
        if "tile" in label:
            tile_configs[label] = data
        elif "chunk=1t" in label and "tile" not in label:
            shard_configs[label] = data
        else:
            multi_configs[label] = data

    # create figure with 2 rows: FPS comparison, file size
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor(colors["background"])
    fig.suptitle(title, color=colors["text"], fontsize=14, fontweight="bold")

    # panel 1: FPS comparison (grouped bar chart)
    ax1 = axes[0]
    _apply_mbo_style(ax1, fig)

    all_labels = list(results.keys())
    x = np.arange(len(all_labels))
    width = 0.35

    write_fps = [results[l]["fps_write"] for l in all_labels]
    read_fps = [results[l]["fps_read"] for l in all_labels]

    bars1 = ax1.bar(x - width/2, write_fps, width, label="Write FPS",
                    color=colors["primary"], edgecolor=colors["border"])
    bars2 = ax1.bar(x + width/2, read_fps, width, label="Read FPS",
                    color=colors["success"], edgecolor=colors["border"])

    ax1.set_ylabel("Frames per Second")
    ax1.set_title("Write vs Read Performance", color=colors["text"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(facecolor=colors["surface"], edgecolor=colors["border"],
               labelcolor=colors["text"])
    ax1.grid(axis="y", alpha=0.3, color=colors["border"])

    # add value labels on bars
    for bar, val in zip(bars1, write_fps):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=7, color=colors["text_muted"])
    for bar, val in zip(bars2, read_fps):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=7, color=colors["text_muted"])

    # panel 2: time comparison (stacked bar for write+read)
    ax2 = axes[1]
    _apply_mbo_style(ax2, fig)

    write_ms = [results[l]["write_ms"] for l in all_labels]
    read_ms = [results[l]["read_ms"] for l in all_labels]

    bars1 = ax2.bar(x, write_ms, width * 2, label="Write Time",
                    color=colors["warning"], edgecolor=colors["border"])
    bars2 = ax2.bar(x, read_ms, width * 2, bottom=write_ms, label="Read Time",
                    color=colors["accent"], edgecolor=colors["border"])

    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Write + Read Time (lower is better)", color=colors["text"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(facecolor=colors["surface"], edgecolor=colors["border"],
               labelcolor=colors["text"])
    ax2.grid(axis="y", alpha=0.3, color=colors["border"])

    # add total time labels
    for i, (w, r) in enumerate(zip(write_ms, read_ms)):
        total = w + r
        ax2.text(i, total + 20, f"{total:.0f}",
                ha="center", va="bottom", fontsize=7, color=colors["text_muted"])

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # make room for suptitle

    if output_path:
        output_path = Path(output_path)
        fig.savefig(
            output_path,
            facecolor=colors["background"],
            edgecolor="none",
            dpi=150,
            bbox_inches="tight",
        )
        logger.info(f"saved zarr benchmark plot to {output_path}")

    if show:
        plt.show()

    return fig


def benchmark_mboraw(
    data_path: str | Path,
    config: BenchmarkConfig | None = None,
    output_dir: Path | None = None,
    label: str = "",
) -> BenchmarkResult:
    """
    run full benchmark suite on MboRawArray.

    parameters
    ----------
    data_path : path
        path to scanimage tiff files (file or directory)
    config : BenchmarkConfig, optional
        benchmark configuration (defaults to full suite)
    output_dir : Path, optional
        directory for write tests (temp dir if None)
    label : str
        label for this benchmark run

    returns
    -------
    BenchmarkResult
        complete benchmark results

    examples
    --------
    >>> result = benchmark_mboraw("/path/to/raw", config=BenchmarkConfig.quick())
    >>> result.save("benchmarks/results/run_001.json")
    """
    from mbo_utilities import imread

    if config is None:
        config = BenchmarkConfig.full()

    logger.info(f"starting benchmark with config: {type(config).__name__}")

    # load array once to get data info
    arr = imread(data_path)
    data_info = {
        "path": str(data_path),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "num_files": len(arr.filenames) if hasattr(arr, "filenames") else 1,
    }
    logger.info(f"data: shape={arr.shape}, dtype={arr.dtype}")

    results = {}

    # initialization benchmark
    logger.info("benchmarking initialization...")
    results["init"] = {
        k: asdict(v) for k, v in
        benchmark_init(data_path, repeats=config.repeats, warmup=config.warmup).items()
    }

    # indexing benchmark
    if config.frame_counts:
        logger.info("benchmarking indexing...")
        results["indexing"] = {
            k: asdict(v) for k, v in
            benchmark_indexing(
                arr,
                frame_counts=config.frame_counts,
                repeats=config.repeats,
                warmup=config.warmup,
                test_zplane=config.test_zplane_indexing,
            ).items()
        }

    # random access patterns (contiguous vs strided vs random)
    if config.test_random_access:
        logger.info("benchmarking random access patterns...")
        results["access_patterns"] = {
            k: asdict(v) for k, v in
            benchmark_random_access(
                arr,
                repeats=config.repeats,
                warmup=config.warmup,
            ).items()
        }

    # parallel plane reads (first/middle/last z-plane)
    if config.test_parallel_planes:
        logger.info("benchmarking parallel plane reads...")
        results["plane_position"] = {
            k: asdict(v) for k, v in
            benchmark_parallel_planes(
                arr,
                repeats=config.repeats,
                warmup=config.warmup,
            ).items()
        }

    # cold vs warm cache reads
    if config.test_cold_warm:
        logger.info("benchmarking cold vs warm reads...")
        results["cache_effects"] = {
            k: asdict(v) for k, v in
            benchmark_cold_warm_reads(
                data_path,
                repeats=config.repeats,
            ).items()
        }

    # throughput measurement (MB/s)
    if config.test_throughput:
        logger.info("benchmarking throughput...")
        throughput_results = benchmark_throughput(
            arr,
            repeats=config.repeats,
            warmup=config.warmup,
        )
        # flatten nested structure for JSON
        results["throughput"] = {}
        for name, data in throughput_results.items():
            results["throughput"][name] = {
                "time_ms": asdict(data["time_ms"]),
                "throughput_mbs": data["throughput_mbs"],
            }

    # scaling analysis (time vs frame count)
    if config.test_scaling:
        logger.info("benchmarking scaling...")
        results["scaling"] = benchmark_scaling(
            arr,
            repeats=config.repeats,
            warmup=config.warmup,
        )

    # phase correction variants
    phase_tests = any([config.test_no_phase, config.test_phase_corr, config.test_phase_fft])
    if phase_tests:
        logger.info("benchmarking phase correction variants...")
        # use medium frame count for phase tests
        phase_frames = 100
        if config.frame_counts:
            phase_frames = min(200, max(config.frame_counts))

        results["phase_variants"] = {
            k: asdict(v) for k, v in
            benchmark_phase_variants(
                data_path,
                frame_count=phase_frames,
                repeats=config.repeats,
                test_no_phase=config.test_no_phase,
                test_phase_corr=config.test_phase_corr,
                test_phase_fft=config.test_phase_fft,
            ).items()
        }

    # write benchmarks
    if config.write_formats:
        logger.info("benchmarking writes...")
        write_frames = None if config.write_full_dataset else config.write_num_frames
        results["writes"] = {
            k: asdict(v) for k, v in
            benchmark_writes(
                arr,
                formats=config.write_formats,
                num_frames=write_frames,
                output_dir=output_dir,
                repeats=config.repeats,
                keep_files=config.keep_written_files,
                zarr_levels=config.zarr_levels,
                zarr_sharding=config.zarr_sharding,
            ).items()
        }

    # chunk size impact on writes
    if config.test_chunk_sizes:
        logger.info("benchmarking chunk sizes...")
        results["chunk_sizes"] = {
            k: asdict(v) for k, v in
            benchmark_chunk_sizes(
                arr,
                chunk_sizes_mb=config.chunk_sizes_mb,
                num_frames=config.write_num_frames,
                output_dir=output_dir,
                repeats=max(2, config.repeats - 1),  # slightly fewer repeats
                keep_files=config.keep_written_files,
            ).items()
        }

    # readback performance (write then read)
    if config.test_readback:
        logger.info("benchmarking readback...")
        results["readback"] = benchmark_readback(
            arr,
            formats=config.write_formats if config.write_formats else (".zarr", ".tiff"),
            num_frames=min(50, config.write_num_frames),
            output_dir=output_dir,
            repeats=config.repeats,
        )

    # file size comparison
    if config.test_file_sizes:
        logger.info("benchmarking file sizes...")
        results["file_sizes"] = benchmark_file_sizes(
            arr,
            formats=config.write_formats if config.write_formats else (".zarr", ".tiff", ".h5", ".bin"),
            num_frames=config.write_num_frames,
            output_dir=output_dir,
            zarr_levels=config.zarr_levels,
        )

    logger.info("benchmark complete")

    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        label=label,
        system_info=get_system_info(),
        data_info=data_info,
        config=asdict(config),
        results=results,
    )


def print_summary(result: BenchmarkResult) -> None:
    """print a formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"timestamp: {result.timestamp}")
    print(f"git commit: {result.git_commit}")
    print(f"label: {result.label}")
    print(f"\ndata: {result.data_info.get('path', 'unknown')}")
    print(f"shape: {result.data_info.get('shape')}")
    print()

    for category, tests in result.results.items():
        print(f"\n{category.upper()}")
        print("-" * 50)

        # handle special categories with nested/custom structure
        if category == "throughput":
            for name, data in tests.items():
                time_stats = data.get("time_ms", {})
                tp_stats = data.get("throughput_mbs", {})
                mean_ms = time_stats.get("mean_ms", 0)
                mean_mbs = tp_stats.get("mean", 0)
                print(f"  {name:30s}: {mean_ms:7.1f} ms  ({mean_mbs:6.1f} MB/s)")

        elif category == "scaling":
            # print frame count timings
            for name, stats in tests.items():
                if name.startswith("_"):
                    continue  # skip analysis metadata
                mean = stats.get("mean_ms", 0)
                std = stats.get("std_ms", 0)
                print(f"  {name:25s}: {mean:8.1f} ± {std:6.1f} ms")
            # print scaling analysis if present
            if "_scaling_analysis" in tests:
                analysis = tests["_scaling_analysis"]
                print(f"\n  scaling model: time = {analysis['overhead_ms']:.1f}ms + {analysis['per_frame_ms']:.2f}ms/frame")
                print(f"  r² = {analysis['r_squared']:.3f}")

        elif category == "readback":
            for fmt, data in tests.items():
                write_stats = data.get("write", {})
                read_stats = data.get("read", {})
                w_mean = write_stats.get("mean_ms", 0) if write_stats else 0
                r_mean = read_stats.get("mean_ms", 0) if read_stats else 0
                print(f"  {fmt:15s}: write {w_mean:7.1f} ms, read {r_mean:7.1f} ms")

        elif category == "file_sizes":
            for fmt, data in tests.items():
                size_mb = data.get("size_mb", 0)
                raw_mb = data.get("raw_mb", 0)
                ratio = data.get("compression_ratio", 0)
                print(f"  {fmt:20s}: {size_mb:6.1f} MB  (ratio: {ratio:.2f}x)")
            # print raw size once
            if tests:
                first = next(iter(tests.values()))
                print(f"\n  raw uncompressed: {first.get('raw_mb', 0):.1f} MB")

        else:
            # standard timing format
            for name, stats in tests.items():
                if isinstance(stats, dict) and "mean_ms" in stats:
                    mean = stats.get("mean_ms", 0)
                    std = stats.get("std_ms", 0)
                    print(f"  {name:30s}: {mean:8.1f} ± {std:6.1f} ms")
                elif isinstance(stats, dict):
                    # nested dict without mean_ms (skip or show raw)
                    pass

    print("\n" + "=" * 70)


# MBO dark theme colors (from docs/_static/custom.css)
MBO_DARK_THEME = {
    "background": "#121212",
    "surface": "#1e1e1e",
    "text": "#e0e0e0",
    "text_muted": "#9e9e9e",
    "border": "#333333",
    "primary": "#82aaff",  # blue
    "secondary": "#c792ea",  # purple
    "success": "#c3e88d",  # green
    "warning": "#ffcb6b",  # yellow
    "error": "#f07178",  # red
    "accent": "#89ddff",  # cyan
    "orange": "#f78c6c",
}


def _apply_mbo_style(ax, fig=None):
    """apply MBO dark theme to matplotlib axes."""
    colors = MBO_DARK_THEME

    ax.set_facecolor(colors["surface"])
    if fig:
        fig.patch.set_facecolor(colors["background"])

    ax.spines["bottom"].set_color(colors["border"])
    ax.spines["top"].set_color(colors["border"])
    ax.spines["left"].set_color(colors["border"])
    ax.spines["right"].set_color(colors["border"])

    ax.tick_params(colors=colors["text_muted"], which="both")
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])

    ax.grid(True, alpha=0.2, color=colors["text_muted"], linestyle="-", linewidth=0.5)


def plot_benchmark_results(
    result: BenchmarkResult,
    output_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    generate dark-mode visualization of benchmark results.

    creates a multi-panel figure showing:
    - scaling analysis (time vs frame count)
    - throughput (MB/s)
    - write format comparison
    - file size comparison

    parameters
    ----------
    result : BenchmarkResult
        benchmark results from benchmark_mboraw()
    output_path : Path or str, optional
        save figure to this path (png, pdf, svg)
    show : bool
        display the figure

    returns
    -------
    Figure or None
        matplotlib figure if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return None

    colors = MBO_DARK_THEME
    bar_colors = [
        colors["primary"],
        colors["success"],
        colors["warning"],
        colors["accent"],
        colors["secondary"],
        colors["orange"],
        colors["error"],
    ]

    # count available plot panels
    panels = []
    if "scaling" in result.results:
        panels.append("scaling")
    if "throughput" in result.results:
        panels.append("throughput")
    if "writes" in result.results:
        panels.append("writes")
    if "file_sizes" in result.results:
        panels.append("file_sizes")

    if not panels:
        logger.warning("no plottable results found")
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor(colors["background"])

    for ax, panel in zip(axes, panels):
        _apply_mbo_style(ax, fig)

        if panel == "scaling":
            _plot_scaling(ax, result.results["scaling"], colors)
        elif panel == "throughput":
            _plot_throughput(ax, result.results["throughput"], colors)
        elif panel == "writes":
            _plot_writes(ax, result.results["writes"], colors, bar_colors)
        elif panel == "file_sizes":
            _plot_file_sizes(ax, result.results["file_sizes"], colors, bar_colors)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        fig.savefig(
            output_path,
            facecolor=colors["background"],
            edgecolor="none",
            dpi=150,
            bbox_inches="tight",
        )
        logger.info(f"saved benchmark plot to {output_path}")

    if show:
        plt.show()

    return fig


def _plot_scaling(ax, data: dict, colors: dict):
    """plot scaling analysis: time vs frame count."""
    frame_counts = []
    times = []

    for name, stats in data.items():
        if name.startswith("_"):
            continue
        # parse "N frames" format
        try:
            n = int(name.split()[0])
            frame_counts.append(n)
            times.append(stats.get("mean_ms", 0))
        except (ValueError, IndexError):
            continue

    if not frame_counts:
        return

    ax.plot(
        frame_counts, times,
        "o-",
        color=colors["primary"],
        linewidth=2,
        markersize=6,
        markerfacecolor=colors["primary"],
        markeredgecolor=colors["text"],
        markeredgewidth=0.5,
    )

    # add regression line if available
    if "_scaling_analysis" in data:
        analysis = data["_scaling_analysis"]
        overhead = analysis["overhead_ms"]
        per_frame = analysis["per_frame_ms"]
        r_squared = analysis["r_squared"]

        x_fit = np.array([0, max(frame_counts)])
        y_fit = overhead + per_frame * x_fit
        ax.plot(
            x_fit, y_fit,
            "--",
            color=colors["warning"],
            linewidth=1.5,
            alpha=0.8,
            label=f"fit: {overhead:.0f} + {per_frame:.1f}×N (r²={r_squared:.3f})",
        )
        ax.legend(
            loc="upper left",
            facecolor=colors["surface"],
            edgecolor=colors["border"],
            labelcolor=colors["text_muted"],
            fontsize=8,
        )

    ax.set_xlabel("frames", fontsize=10)
    ax.set_ylabel("time (ms)", fontsize=10)
    ax.set_title("Read Scaling", fontsize=11, fontweight="bold")


def _plot_throughput(ax, data: dict, colors: dict):
    """plot throughput: MB/s for different frame counts."""
    names = []
    throughputs = []

    for name, stats in data.items():
        tp = stats.get("throughput_mbs", {}).get("mean", 0)
        if tp > 0:
            # extract just the frame count for shorter labels
            short_name = name.split()[0] + "f"
            names.append(short_name)
            throughputs.append(tp)

    if not names:
        return

    x = np.arange(len(names))
    bars = ax.bar(
        x, throughputs,
        color=colors["success"],
        edgecolor=colors["border"],
        alpha=0.85,
    )

    # add value labels
    for bar, tp in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.02,
            f"{tp:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors["text_muted"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_xlabel("frames", fontsize=10)
    ax.set_ylabel("throughput (MB/s)", fontsize=10)
    ax.set_title("Read Throughput", fontsize=11, fontweight="bold")


def _plot_writes(ax, data: dict, colors: dict, bar_colors: list):
    """plot write benchmark: time for each format."""
    names = []
    times = []
    stds = []

    for name, stats in data.items():
        mean = stats.get("mean_ms", 0)
        std = stats.get("std_ms", 0)
        if mean > 0:
            # shorten format extensions
            short_name = name.replace(".tiff", "tiff").replace(".h5", "h5").replace(".bin", "bin")
            names.append(short_name)
            times.append(mean)
            stds.append(std)

    if not names:
        return

    x = np.arange(len(names))
    bar_cols = [bar_colors[i % len(bar_colors)] for i in range(len(names))]
    bars = ax.bar(
        x, times,
        yerr=stds,
        color=bar_cols,
        edgecolor=colors["border"],
        alpha=0.85,
        capsize=3,
        error_kw={"ecolor": colors["text_muted"], "capthick": 1},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("time (ms)", fontsize=10)
    ax.set_title("Write Performance", fontsize=11, fontweight="bold")


def _plot_file_sizes(ax, data: dict, colors: dict, bar_colors: list):
    """plot file size comparison with compression ratios."""
    names = []
    sizes = []
    ratios = []

    for name, stats in data.items():
        size = stats.get("size_mb", 0)
        ratio = stats.get("compression_ratio", 0)
        if size > 0:
            short_name = name.replace(".zarr", "zarr").replace(".tiff", "tiff")
            short_name = short_name.replace(".h5", "h5").replace(".bin", "bin")
            names.append(short_name)
            sizes.append(size)
            ratios.append(ratio)

    if not names:
        return

    x = np.arange(len(names))
    bar_cols = [bar_colors[i % len(bar_colors)] for i in range(len(names))]
    bars = ax.bar(
        x, sizes,
        color=bar_cols,
        edgecolor=colors["border"],
        alpha=0.85,
    )

    # add ratio labels
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(sizes) * 0.02,
            f"{ratio:.1f}×",
            ha="center",
            va="bottom",
            fontsize=8,
            color=colors["text_muted"],
        )

    # add raw size reference line
    if data:
        raw_mb = next(iter(data.values())).get("raw_mb", 0)
        if raw_mb > 0:
            ax.axhline(
                raw_mb,
                color=colors["error"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"raw: {raw_mb:.0f} MB",
            )
            ax.legend(
                loc="upper right",
                facecolor=colors["surface"],
                edgecolor=colors["border"],
                labelcolor=colors["text_muted"],
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("size (MB)", fontsize=10)
    ax.set_title("File Sizes", fontsize=11, fontweight="bold")


def plot_comparison(
    results: list[BenchmarkResult],
    metric: str = "scaling",
    output_path: Path | str | None = None,
    show: bool = True,
) -> "Figure | None":
    """
    compare multiple benchmark runs on a single plot.

    parameters
    ----------
    results : list of BenchmarkResult
        benchmark results to compare
    metric : str
        which metric to compare: "scaling", "throughput", "writes"
    output_path : Path or str, optional
        save figure to this path
    show : bool
        display the figure

    returns
    -------
    Figure or None
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return None

    colors = MBO_DARK_THEME
    line_colors = [
        colors["primary"],
        colors["success"],
        colors["warning"],
        colors["accent"],
        colors["secondary"],
        colors["orange"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(colors["background"])
    _apply_mbo_style(ax, fig)

    for i, result in enumerate(results):
        color = line_colors[i % len(line_colors)]
        label = result.label or f"run {i+1}"

        if metric == "scaling" and "scaling" in result.results:
            data = result.results["scaling"]
            frame_counts = []
            times = []
            for name, stats in data.items():
                if name.startswith("_"):
                    continue
                try:
                    n = int(name.split()[0])
                    frame_counts.append(n)
                    times.append(stats.get("mean_ms", 0))
                except (ValueError, IndexError):
                    continue
            if frame_counts:
                ax.plot(
                    frame_counts, times,
                    "o-",
                    color=color,
                    linewidth=2,
                    markersize=5,
                    label=label,
                )
                ax.set_xlabel("frames", fontsize=10)
                ax.set_ylabel("time (ms)", fontsize=10)
                ax.set_title("Scaling Comparison", fontsize=11, fontweight="bold")

        elif metric == "throughput" and "throughput" in result.results:
            data = result.results["throughput"]
            frame_counts = []
            throughputs = []
            for name, stats in data.items():
                tp = stats.get("throughput_mbs", {}).get("mean", 0)
                if tp > 0:
                    try:
                        n = int(name.split()[0])
                        frame_counts.append(n)
                        throughputs.append(tp)
                    except (ValueError, IndexError):
                        continue
            if frame_counts:
                ax.plot(
                    frame_counts, throughputs,
                    "o-",
                    color=color,
                    linewidth=2,
                    markersize=5,
                    label=label,
                )
                ax.set_xlabel("frames", fontsize=10)
                ax.set_ylabel("throughput (MB/s)", fontsize=10)
                ax.set_title("Throughput Comparison", fontsize=11, fontweight="bold")

    ax.legend(
        facecolor=colors["surface"],
        edgecolor=colors["border"],
        labelcolor=colors["text"],
        fontsize=9,
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        fig.savefig(
            output_path,
            facecolor=colors["background"],
            edgecolor="none",
            dpi=150,
            bbox_inches="tight",
        )
        logger.info(f"saved comparison plot to {output_path}")

    if show:
        plt.show()

    return fig
