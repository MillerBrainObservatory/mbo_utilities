#!/usr/bin/env python
"""Unified Suite2p runner for local, single-job, and SLURM-array execution.

The same script runs three ways, selected from the environment:

  - local / single cluster job : process all planes, GPU time-shares F at a time
  - SLURM array task           : process this task's plane shard (skip_volumetric)
  - aggregate                  : volumetric merge/stats over already-run planes

The only tuning knob is the pack factor F (planes per GPU), measured with
`nvidia-smi dmon` — the largest worker count that fills one GPU without
cellpose OOM. Shard size and per-job worker count both derive from it.

Config comes from env vars (overridable) with the values below as defaults:
  MBO_INPUT, MBO_OUTPUT, MBO_PLANES_PER_GPU, MBO_THREADS_PER_WORKER, MBO_OPS_JSON
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# suite2p records these per-plane stage times in ops.npy['timing'] (seconds).
_STAGES = ["registration", "detection", "extraction", "classification", "deconvolution"]

DEFAULT_INPUT = "/lustre/fs8/mbo/scratch/mbo_data/lbm/2025-07-27_mk355/raw"
DEFAULT_OUTPUT = "/lustre/fs8/mbo/scratch/mbo_data/lbm/2025-07-27_mk355/results"
DEFAULT_PLANES_PER_GPU = 4

DEFAULT_OPS = {
    "anatomical_only": 4,
    "diameter": 2,
    "cellprob_threshold": -6,
    "spatial_hp_cp": 3,
    "niter": 200,
    "do_registration": 1,
    "two_step_registration": 1,
    "lam_percentile": 0,
    "min_neuropil_pixels": 0,
    "max_overlap": 0.99,
}


def _cpu_quota() -> int:
    """CPUs this process may use — respects SLURM cgroup, unlike os.cpu_count()."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _config():
    input_dir = Path(os.environ.get("MBO_INPUT", DEFAULT_INPUT))
    output_dir = Path(os.environ.get("MBO_OUTPUT", DEFAULT_OUTPUT))
    pack = int(os.environ.get("MBO_PLANES_PER_GPU", DEFAULT_PLANES_PER_GPU))
    ops = dict(DEFAULT_OPS)
    ops_json = os.environ.get("MBO_OPS_JSON")
    if ops_json:
        ops.update(json.loads(Path(ops_json).read_text()))
    ops_inline = os.environ.get("MBO_OPS")  # inline JSON, overrides MBO_OPS_JSON
    if ops_inline:
        ops.update(json.loads(ops_inline))
    return input_dir, output_dir, pack, ops


def _num_planes(arr) -> int:
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


def _run(arr, output_dir, ops, planes, workers, threads, skip_volumetric, force):
    from lbm_suite2p_python import pipeline

    t0 = time.perf_counter()
    pipeline(
        arr,
        save_path=str(output_dir),
        ops=ops,
        planes=planes,
        keep_reg=True,
        keep_raw=False,
        force_reg=force,
        force_detect=force,
        skip_volumetric=skip_volumetric,
        workers=workers,
        threads_per_worker=threads,
        writer_kwargs={"fix_phase": True, "use_fft": True},
    )
    return time.perf_counter() - t0


def _read_plane_timings(output_dir):
    """Per-plane suite2p stage times (s) from each ops.npy['timing']."""
    try:
        import numpy as np
    except Exception:
        return []
    rows = []
    for p in sorted(Path(output_dir).glob("**/ops.npy")):
        try:
            ops = np.load(p, allow_pickle=True).item()
        except Exception:
            continue
        t = ops.get("timing") if isinstance(ops, dict) else None
        if not t:
            continue
        row = {"plane": p.parent.name}
        row.update({s: float(t.get(s, 0) or 0) for s in _STAGES})
        row["total_plane"] = float(t.get("total_plane_runtime", 0) or 0)
        rows.append(row)
    return rows


def _write_timing_report(output_dir, wall=None, n_workers=None):
    """Write timings.json into output_dir and print a per-plane breakdown.

    Per-plane stage times come from ops.npy (suite2p); wall holds runner-level
    seconds (imread, pipeline). Per plane == per worker (one plane per worker).
    """
    rows = _read_plane_timings(output_dir)
    report = {"wall": wall or {}, "n_workers": n_workers, "planes": rows}
    cols = _STAGES + ["total_plane"]
    if rows:
        report["totals"] = {
            c: {"sum": sum(r[c] for r in rows),
                "mean": sum(r[c] for r in rows) / len(rows),
                "max": max(r[c] for r in rows)}
            for c in cols
        }
    try:
        (Path(output_dir) / "timings.json").write_text(json.dumps(report, indent=2))
    except Exception:
        pass

    if not rows:
        print("timing: no suite2p ops.npy['timing'] found under "
              f"{output_dir}", flush=True)
    else:
        print("\n--- per-plane timing (s) ---", flush=True)
        print(f"{'plane':<26}" + "".join(f"{s[:5]:>9}" for s in _STAGES)
              + f"{'total':>9}", flush=True)
        for r in rows:
            print(f"{r['plane'][:26]:<26}"
                  + "".join(f"{r[s]:>9.1f}" for s in _STAGES)
                  + f"{r['total_plane']:>9.1f}", flush=True)
        tot = report["totals"]
        print(f"{'sum':<26}" + "".join(f"{tot[s]['sum']:>9.1f}" for s in _STAGES)
              + f"{tot['total_plane']['sum']:>9.1f}", flush=True)
        print(f"{'max (slowest plane)':<26}"
              + "".join(f"{tot[s]['max']:>9.1f}" for s in _STAGES)
              + f"{tot['total_plane']['max']:>9.1f}", flush=True)
    for k, v in (wall or {}).items():
        print(f"wall.{k}: {v:.1f}s", flush=True)
    return report


def _resolve_workers(n_shard: int, pack: int) -> tuple[int, int]:
    workers = max(1, min(n_shard, pack, _cpu_quota()))
    env_threads = os.environ.get("MBO_THREADS_PER_WORKER")
    threads = int(env_threads) if env_threads else max(1, _cpu_quota() // workers)
    return workers, threads


def _shard_for_task(plane_indices, pack, task_id):
    start = task_id * pack
    return plane_indices[start:start + pack]


def _local_multi_gpu(gpus, plane_indices, input_dir, output_dir):
    """Fork one pinned subprocess per GPU, each processing a contiguous shard."""
    n = len(gpus)
    per = math.ceil(len(plane_indices) / n)
    procs = []
    for i, gpu in enumerate(gpus):
        shard = plane_indices[i * per:(i + 1) * per]
        if not shard:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MBO_HPC_PLANES"] = ",".join(map(str, shard))
        env["MBO_HPC_SKIP_VOLUMETRIC"] = "1"
        env.pop("MBO_GPUS", None)
        print(f"GPU {gpu}: planes {shard}", flush=True)
        procs.append(subprocess.Popen([sys.executable, __file__], env=env))
    failed = [p.wait() for p in procs]
    if any(rc != 0 for rc in failed):
        raise SystemExit(f"local multi-GPU shard(s) failed: returncodes={failed}")
    print("All GPU shards done; running volumetric aggregate...", flush=True)
    _aggregate(input_dir, output_dir)


def _aggregate(input_dir, output_dir):
    from mbo_utilities import imread

    _, _, _, ops = _config()
    arr = imread(input_dir)
    _run(arr, output_dir, ops, planes=None, workers=1, threads=_cpu_quota(),
         skip_volumetric=False, force=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    input_dir, output_dir, pack, ops = _config()

    if "--print-num-tasks" in argv:
        from mbo_utilities import imread
        p = _num_planes(imread(input_dir))
        print(math.ceil(p / pack))
        return

    if "--report-timings" in argv:
        i = argv.index("--report-timings")
        d = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else str(output_dir)
        _write_timing_report(d)
        return

    gpus = None
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--gpus":
            gpus = [g for g in argv[i + 1].split(",") if g != ""]
            i += 2
            continue
        if not a.startswith("--"):
            pos.append(a)
        i += 1
    if gpus is None and os.environ.get("MBO_GPUS"):
        gpus = [g for g in os.environ["MBO_GPUS"].split(",") if g != ""]

    if len(pos) >= 1:
        input_dir = Path(pos[0])
    if len(pos) >= 2:
        output_dir = Path(pos[1])

    output_dir.mkdir(parents=True, exist_ok=True)

    from mbo_utilities import imread

    mode = os.environ.get("MBO_HPC_MODE")
    explicit = os.environ.get("MBO_HPC_PLANES")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if mode == "aggregate":
        _aggregate(input_dir, output_dir)
        return

    t0 = time.perf_counter()
    arr = imread(input_dir)
    t_imread = time.perf_counter() - t0
    print(f"shape={arr.shape} dims={getattr(arr, 'dims', None)}", flush=True)
    plane_indices = list(range(_num_planes(arr)))

    if explicit is not None:
        shard = [int(p) for p in explicit.split(",") if p != ""]
        skip = os.environ.get("MBO_HPC_SKIP_VOLUMETRIC") == "1"
        workers, threads = _resolve_workers(len(shard), pack)
        print(f"explicit shard {shard} workers={workers} threads={threads}", flush=True)
        _run(arr, output_dir, ops, shard, workers, threads, skip, force=False)
        return

    if gpus:
        _local_multi_gpu(gpus, plane_indices, input_dir, output_dir)
        _write_timing_report(output_dir, {"imread": t_imread})
        return

    if array_id is not None:
        shard = _shard_for_task(plane_indices, pack, int(array_id))
        if not shard:
            print(f"array task {array_id}: no planes, exiting", flush=True)
            return
        workers, threads = _resolve_workers(len(shard), pack)
        print(f"array task {array_id}: planes {shard} workers={workers} threads={threads}", flush=True)
        _run(arr, output_dir, ops, shard, workers, threads, skip_volumetric=True, force=False)
        return

    workers, threads = _resolve_workers(len(plane_indices), pack)
    print(f"single job: {len(plane_indices)} planes workers={workers} threads={threads}", flush=True)
    wall = _run(arr, output_dir, ops, plane_indices, workers, threads, skip_volumetric=False, force=False)
    _write_timing_report(output_dir, {"imread": t_imread, "pipeline": wall}, n_workers=workers)


if __name__ == "__main__":
    main()
