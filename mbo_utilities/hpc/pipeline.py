"""Scheduler-agnostic compute core for the HPC Suite2p runner.

These functions run the same whether invoked locally, in a single SLURM job, or
in one array task; the submitit layer in ``submit.py`` only decides which role
each process plays. ``run_job`` is the module-level entry point submitit pickles
and executes on the compute node.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .config import HpcConfig

# Per-plane timing columns (see _read_plane_timings).
_COLS = ["io", "reg", "regmetrics", "detect", "extract", "plots", "total"]


def cpu_quota() -> int:
    """CPUs this process may use — respects SLURM cgroup, unlike os.cpu_count()."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def num_planes(arr) -> int:
    try:
        from lbm_suite2p_python.utils import _get_num_planes
        return int(_get_num_planes(arr))
    except Exception:
        pass
    n = getattr(arr, "nz", None)
    if n:
        return int(n)
    shp = getattr(arr, "shape", None)
    if shp is not None and len(shp) >= 3:
        return int(shp[-3])
    return 1


def resolve_workers(n_shard: int, pack: int, threads_override: int = 0) -> tuple[int, int]:
    workers = max(1, min(n_shard, pack, cpu_quota()))
    threads = threads_override if threads_override else max(1, cpu_quota() // workers)
    return workers, threads


def shard_for_task(plane_indices, pack, task_id):
    start = task_id * pack
    return plane_indices[start:start + pack]


def num_tasks(n_planes: int, pack: int) -> int:
    return math.ceil(n_planes / pack)


def _run(arr, output_dir, ops, planes, workers, threads, skip_volumetric, force,
         replot=True, passthrough=None):
    from lbm_suite2p_python import pipeline

    kw = dict(passthrough or {})
    keep_reg = kw.pop("keep_reg", True)
    keep_raw = kw.pop("keep_raw", False)
    writer_kwargs = kw.pop("writer_kwargs", {"fix_phase": True, "use_fft": True})

    t0 = time.perf_counter()
    pipeline(
        arr,
        save_path=str(output_dir),
        ops=ops,
        planes=planes,
        keep_reg=keep_reg,
        keep_raw=keep_raw,
        force_reg=force,
        force_detect=force,
        replot=replot,
        skip_volumetric=skip_volumetric,
        workers=workers,
        threads_per_worker=threads,
        writer_kwargs=writer_kwargs,
        **kw,
    )
    return time.perf_counter() - t0


def _aggregate(input_dir, output_dir, ops, passthrough=None):
    from mbo_utilities import imread

    arr = imread(input_dir)
    # replot=False: per-plane figures already exist from the shard runs; the
    # aggregate only needs the volumetric merge + volume plots.
    _run(arr, output_dir, ops, planes=None, workers=1, threads=cpu_quota(),
         skip_volumetric=False, force=False, replot=False, passthrough=passthrough)


def _read_plane_timings(output_dir):
    """Per-plane timing (s): IO + suite2p stage split + plotting."""
    try:
        import numpy as np
    except Exception:
        return []

    def _g(d, k):
        return float(d.get(k, 0) or 0)

    rows = []
    for p in sorted(Path(output_dir).glob("**/ops.npy")):
        try:
            ops = np.load(p, allow_pickle=True).item()
        except Exception:
            continue
        if not isinstance(ops, dict):
            continue
        pt = ops.get("plane_times") or {}
        io = plots = 0.0
        seen = set()
        for step in (ops.get("processing_history") or []):
            name, dur = step.get("step"), step.get("duration_seconds")
            if dur is None:
                continue
            if name == "binary_write" and "io" not in seen:
                io = float(dur); seen.add("io")
            elif name == "plots" and "plots" not in seen:
                plots = float(dur); seen.add("plots")
        if not pt and not seen:
            continue
        row = {
            "plane": p.parent.name,
            "io": io,
            "reg": _g(pt, "registration"),
            "regmetrics": _g(pt, "registration_metrics"),
            "detect": _g(pt, "detection"),
            "extract": _g(pt, "extraction") + _g(pt, "classification") + _g(pt, "deconvolution"),
            "plots": plots,
        }
        row["total"] = (row["io"] + row["reg"] + row["regmetrics"]
                        + row["detect"] + row["extract"] + row["plots"])
        rows.append(row)
    return rows


def write_timing_report(output_dir, wall=None, n_workers=None):
    """Write timings.json into output_dir and print a per-plane breakdown."""
    rows = _read_plane_timings(output_dir)
    report = {"wall": wall or {}, "n_workers": n_workers, "planes": rows}
    if rows:
        report["totals"] = {
            c: {"sum": sum(r[c] for r in rows),
                "mean": sum(r[c] for r in rows) / len(rows),
                "max": max(r[c] for r in rows)}
            for c in _COLS
        }
    try:
        (Path(output_dir) / "timings.json").write_text(json.dumps(report, indent=2))
    except Exception:
        pass

    if not rows:
        print(f"timing: no ops.npy timing under {output_dir}", flush=True)
    else:
        print("\n--- per-plane timing (s): io=tiff->bin  reg=motion  regmetrics=reg-quality  "
              "detect=cellpose  extract=+class+decon  plots=figures ---", flush=True)
        print(f"{'plane':<24}" + "".join(f"{c:>11}" for c in _COLS), flush=True)
        for r in rows:
            print(f"{r['plane'][:24]:<24}" + "".join(f"{r[c]:>11.1f}" for c in _COLS), flush=True)
        tot = report["totals"]
        print(f"{'sum':<24}" + "".join(f"{tot[c]['sum']:>11.1f}" for c in _COLS), flush=True)
        print(f"{'mean':<24}" + "".join(f"{tot[c]['mean']:>11.1f}" for c in _COLS), flush=True)
    for k, v in (wall or {}).items():
        print(f"wall.{k}: {v:.1f}s", flush=True)
    return report


def _apply_thread_env(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))


def disk_free_gb(path) -> float:
    """Free space (GB) on the filesystem holding `path`'s nearest existing parent."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return shutil.disk_usage(p).free / 1e9
    except OSError:
        return -1.0


def probe_writable(path) -> tuple[bool, str]:
    """Write+delete a probe file under `path`'s nearest existing dir.

    A real write catches read-only mounts and permission denials that
    os.access misses. The reported free space is filesystem-level and does
    NOT reflect a per-user quota — a quota only surfaces on the real write.
    """
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    free = disk_free_gb(p)
    try:
        probe = p / f".mbo_write_test_{os.getpid()}"
        probe.write_bytes(b"mbo")
        probe.unlink()
    except OSError as e:
        return False, f"{p}: {e} (free {free:.1f} GB)"
    return True, f"writable, {free:.1f} GB free"


def assert_output_writable(path) -> None:
    """Raise with actionable text if `path` can't be written to."""
    ok, detail = probe_writable(path)
    if not ok:
        raise RuntimeError(
            f"output not writable ({detail}). Point [io] output at a writable "
            f"location with room (home dirs often have small quotas; prefer scratch)."
        )


def _bin_gb_per_plane(arr) -> float:
    """int16 data.bin size for one plane, or 0 if shape is unreadable."""
    try:
        nt, ly, lx = int(arr.shape[0]), int(arr.shape[-2]), int(arr.shape[-1])
    except Exception:
        return 0.0
    return nt * ly * lx * 2 / 1e9


def _log_output_estimate(arr, n_planes: int, output_dir) -> None:
    """Print the expected registered-binary footprint vs. free space."""
    per = _bin_gb_per_plane(arr)
    if not per:
        return
    est_gb = per * n_planes  # int16 data.bin per plane
    free_gb = disk_free_gb(output_dir)
    print(f"output estimate: ~{est_gb:.0f} GB for {n_planes} plane(s); "
          f"{free_gb:.0f} GB free on output FS (quota not included)", flush=True)
    if 0 <= free_gb < est_gb:
        print("WARNING: estimated output exceeds free space — likely to fail at "
              "write/copy. Use scratch, or disable keep_reg.", flush=True)


def assert_stage_fits(arr, work_dir, n_planes: int, keep_raw: bool = False) -> None:
    """Fail fast if node-local /tmp can't hold this shard's staged binaries.

    Staging writes the full data.bin per shard plane to ``work_dir`` (and, while
    keep_raw, data_raw.bin alongside it). Checked on the node where /tmp free is
    actually known, BEFORE the long compute, so an undersized /tmp costs seconds.
    """
    per = _bin_gb_per_plane(arr)
    if not per:
        return
    est_gb = per * n_planes * (2.0 if keep_raw else 1.0)
    free_gb = disk_free_gb(work_dir)
    print(f"node-local staging: ~{est_gb:.0f} GB for {n_planes} plane(s); "
          f"{free_gb:.0f} GB free on {work_dir}", flush=True)
    if est_gb and 0 <= free_gb < est_gb * 1.1:  # 10% headroom for fs overhead
        raise RuntimeError(
            f"node-local /tmp ({work_dir}) has {free_gb:.0f} GB free but staging "
            f"this shard needs ~{est_gb:.0f} GB. Lower [pipeline] planes_per_gpu, "
            f"or set [pipeline] node_local = false to write to the output FS directly."
        )


def write_failure_report(output_dir, role, task_id, exc) -> str | None:
    """Write a failure report to the first writable of: <output>/logs,
    ~/.mbo/hpc_logs, the temp dir. Returns the path written, or None.

    Guarantees a debug breadcrumb survives even when the output dir is the
    thing that failed (e.g. a full quota that broke the result copy).
    """
    import traceback

    lines = [
        "mbo hpc job failure",
        f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"role={role} task_id={task_id} pid={os.getpid()}",
        f"node={os.environ.get('SLURMD_NODENAME', '?')} "
        f"job={os.environ.get('SLURM_JOB_ID', '?')}",
        f"output={output_dir} (free {disk_free_gb(output_dir):.1f} GB)",
        f"tmp={os.environ.get('TMPDIR') or tempfile.gettempdir()} "
        f"(free {disk_free_gb(os.environ.get('TMPDIR') or tempfile.gettempdir()):.1f} GB)",
        "",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ]
    text = "\n".join(lines)
    name = f"FAILURE_{role}_{task_id}_{os.getpid()}.log"
    for base in (Path(output_dir) / "logs",
                 Path.home() / ".mbo" / "hpc_logs",
                 Path(tempfile.gettempdir())):
        try:
            base.mkdir(parents=True, exist_ok=True)
            dest = base / name
            dest.write_text(text, encoding="utf-8")
            return str(dest)
        except OSError:
            continue
    return None


def run_job(cfg: HpcConfig | dict, output_dir, role: str = "single",
            task_id: int | None = None, shard=None):
    """Entry point executed on the compute node (submitit pickles this).

    role:
      single    - all planes on one GPU, volumetric merge, timing report.
      array     - this task's plane shard (skip_volumetric); merged later.
      aggregate - volumetric merge over already-processed planes (no node-local).
    """
    if isinstance(cfg, dict):
        cfg = HpcConfig.from_dict(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ops = cfg.ops()

    try:
        # Fail fast: confirm the destination is writable before the long
        # compute, so a bad output costs seconds instead of hours.
        assert_output_writable(output_dir)

        if role == "aggregate":
            t0 = time.perf_counter()
            _aggregate(cfg.io.input, output_dir, ops, passthrough=cfg.pipeline_kwargs())
            write_timing_report(output_dir, {"aggregate": time.perf_counter() - t0})
            return str(output_dir)

        from mbo_utilities import imread

        # Node-local NVMe staging is a cluster optimization; off-cluster it would
        # only copy results through /tmp, so it applies only when running under SLURM.
        jid = os.environ.get("SLURM_JOB_ID")
        node_local = cfg.pipeline.node_local and bool(jid)
        work_dir = output_dir
        if node_local:
            base = os.environ.get("TMPDIR") or tempfile.gettempdir()
            work_dir = Path(base) / f"{cfg.io.name}_{jid}_{task_id if task_id is not None else 0}"
            work_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.perf_counter()
            arr = imread(cfg.io.input)
            t_imread = time.perf_counter() - t0
            print(f"shape={arr.shape} dims={getattr(arr, 'dims', None)}", flush=True)
            plane_indices = list(range(1, num_planes(arr) + 1))  # lbm planes are 1-based
            pack = cfg.pipeline.planes_per_gpu
            thr = cfg.pipeline.threads_per_worker
            keep_raw = cfg.pipeline_kwargs().get("keep_raw", False)

            if role == "array":
                this_shard = shard if shard is not None else shard_for_task(
                    plane_indices, pack, int(task_id or 0))
                if not this_shard:
                    print(f"array task {task_id}: no planes, exiting", flush=True)
                    return str(output_dir)
                workers, threads = resolve_workers(len(this_shard), pack, thr)
                _apply_thread_env(threads)
                _log_output_estimate(arr, len(this_shard), output_dir)
                if node_local:
                    assert_stage_fits(arr, work_dir, len(this_shard), keep_raw)
                print(f"array task {task_id}: planes {this_shard} "
                      f"workers={workers} threads={threads}", flush=True)
                _run(arr, work_dir, ops, this_shard, workers, threads,
                     skip_volumetric=True, force=False,
                     passthrough=cfg.pipeline_kwargs())
            else:  # single
                workers, threads = resolve_workers(len(plane_indices), pack, thr)
                _apply_thread_env(threads)
                _log_output_estimate(arr, len(plane_indices), output_dir)
                if node_local:
                    assert_stage_fits(arr, work_dir, len(plane_indices), keep_raw)
                print(f"single job: {len(plane_indices)} planes "
                      f"workers={workers} threads={threads}", flush=True)
                wall = _run(arr, work_dir, ops, plane_indices, workers, threads,
                            skip_volumetric=False, force=False,
                            passthrough=cfg.pipeline_kwargs())
                write_timing_report(work_dir, {"imread": t_imread, "pipeline": wall},
                                    n_workers=workers)
        finally:
            if node_local and Path(work_dir) != output_dir:
                pending = sys.exc_info()[1]  # compute error already unwinding, if any
                try:
                    t0 = time.perf_counter()
                    shutil.copytree(work_dir, output_dir, dirs_exist_ok=True)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    print(f"timing: transfer={time.perf_counter() - t0:.1f}s -> {output_dir}",
                          flush=True)
                except Exception as copy_err:
                    detail = copy_err
                    if isinstance(copy_err, shutil.Error) and copy_err.args:
                        detail = "; ".join(str(x) for x in copy_err.args[0][:15])
                    msg = (f"copy-back failed: node-local results in {work_dir} were not "
                           f"copied to {output_dir} (check destination disk quota / free "
                           f"space): {detail}")
                    # Never let a copy-back failure mask the real compute error.
                    if pending is not None:
                        print(f"WARNING: {msg}", flush=True)
                    else:
                        raise RuntimeError(msg) from copy_err
        return str(output_dir)
    except BaseException as e:
        report = write_failure_report(output_dir, role, task_id, e)
        if report:
            print(f"failure report written: {report}", flush=True)
        raise
