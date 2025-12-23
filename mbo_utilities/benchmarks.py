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
from typing import Any

import numpy as np

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

    # write formats to test
    write_formats: tuple[str, ...] = (".zarr", ".tiff", ".h5", ".bin")
    write_num_frames: int = 100
    write_full_dataset: bool = False  # write entire dataset (no num_frames limit)
    keep_written_files: bool = False  # keep output files after benchmark

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
            write_formats=(".zarr", ".tiff"),
            repeats=2,
        )

    @classmethod
    def full(cls) -> "BenchmarkConfig":
        """full benchmark suite with all tests."""
        return cls()

    @classmethod
    def read_only(cls) -> "BenchmarkConfig":
        """skip write benchmarks entirely."""
        return cls(write_formats=())

    @classmethod
    def write_only(cls) -> "BenchmarkConfig":
        """only test write operations."""
        return cls(
            frame_counts=(),
            test_no_phase=False,
            test_phase_corr=False,
            test_phase_fft=False,
            test_zplane_indexing=False,
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

    # volume batch tests (all z-planes per timepoint)
    for n in frame_counts:
        if n > max_frames:
            logger.warning(f"skipping {n} volumes test (only {max_frames} available)")
            continue

        times = []
        for _ in range(repeats):
            if n == 1:
                _, elapsed = _time_func(lambda: arr[0])
            else:
                _, elapsed = _time_func(lambda: arr[0:n])
            times.append(elapsed)

        results[f"{n}_volumes"] = TimingStats.from_times(times)

    # single z-plane tests (subset of volume)
    if test_zplane and num_zplanes > 1:
        # single timepoint, single z-plane
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0, 0])
            times.append(elapsed)
        results["1_frame"] = TimingStats.from_times(times)

        # 10 timepoints, single z-plane
        if max_frames >= 10:
            times = []
            for _ in range(repeats):
                _, elapsed = _time_func(lambda: arr[0:10, 0])
                times.append(elapsed)
            results["10_frames"] = TimingStats.from_times(times)

        # single frame, spatial crop
        times = []
        for _ in range(repeats):
            _, elapsed = _time_func(lambda: arr[0, 0, 50:150, 50:150])
            times.append(elapsed)
        results["1_frame_crop"] = TimingStats.from_times(times)

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
        times = []
        for i in range(repeats):
            # create fresh output path each iteration
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

            # cleanup intermediate iterations if not keeping files
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
            ).items()
        }

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
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"timestamp: {result.timestamp}")
    print(f"git commit: {result.git_commit}")
    print(f"label: {result.label}")
    print(f"\ndata: {result.data_info.get('path', 'unknown')}")
    print(f"shape: {result.data_info.get('shape')}")
    print()

    for category, tests in result.results.items():
        print(f"\n{category.upper()}")
        print("-" * 40)
        for name, stats in tests.items():
            mean = stats.get("mean_ms", 0)
            std = stats.get("std_ms", 0)
            print(f"  {name:25s}: {mean:8.1f} ± {std:6.1f} ms")

    print("\n" + "=" * 60)
