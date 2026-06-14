"""submitit wiring: turn an HpcConfig into SLURM jobs (or a local run).

Modes:
  single  - one GPU job over all planes, volumetric merge inline.
  array   - one array task per plane shard, then a dependent aggregate job.
  local   - run the compute inline in this process (no SLURM, no submitit).

``dry_run`` prints the resolved scheduler parameters and job layout without
importing submitit, so it works on a machine that has neither submitit nor SLURM.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from .config import HpcConfig
from . import pipeline as _pipe


def resolve_output_dir(cfg: HpcConfig) -> Path:
    """<output>/<date>_<name>, with a _2/_3... suffix if it already exists."""
    root = Path(cfg.io.output or ".")
    if not cfg.io.dated_subfolder:
        return root
    date = datetime.date.today().strftime("%Y_%m_%d")
    base = root / f"{date}_{cfg.io.name}"
    final, n = base, 2
    while final.exists():
        final = base.with_name(f"{base.name}_{n}")
        n += 1
    return final


def _executor_params(cfg: HpcConfig, array: bool = False) -> dict:
    p = {
        "name": cfg.io.name,
        "nodes": 1,
        "tasks_per_node": 1,
        "cpus_per_task": cfg.slurm.cpus_per_task,
        "mem_gb": cfg.slurm.mem_gb,
        "timeout_min": cfg.timeout_min(),
        "slurm_partition": cfg.slurm.partition,
        "slurm_gres": cfg.slurm.gres,
    }
    if cfg.slurm.account:
        p["slurm_account"] = cfg.slurm.account
    if cfg.slurm.qos:
        p["slurm_qos"] = cfg.slurm.qos
    if cfg.slurm.exclusive:
        p["slurm_exclusive"] = True
    if array and cfg.slurm.array_parallelism:
        p["slurm_array_parallelism"] = cfg.slurm.array_parallelism
    return p


def _make_executor(folder, params: dict):
    try:
        import submitit
    except ImportError as e:
        raise ImportError(
            'submitit not installed. Install with: pip install "mbo_utilities[hpc]"'
        ) from e
    ex = submitit.AutoExecutor(folder=str(folder))
    ex.update_parameters(**params)
    return ex


def plan(cfg: HpcConfig):
    """Login-side metadata read: (n_planes, n_tasks, shards)."""
    from mbo_utilities import imread

    arr = imread(cfg.io.input)
    n = _pipe.num_planes(arr)
    pack = cfg.pipeline.planes_per_gpu
    ntasks = _pipe.num_tasks(n, pack)
    planes = list(range(1, n + 1))
    shards = [_pipe.shard_for_task(planes, pack, t) for t in range(ntasks)]
    return n, ntasks, shards


def _print_plan(cfg: HpcConfig, mode: str, output_dir: Path) -> None:
    print(f"mode:    {mode}")
    print(f"input:   {cfg.io.input}")
    print(f"output:  {output_dir}")
    ok, detail = _pipe.probe_writable(output_dir)
    print(f"writable: {detail}" if ok else
          f"WARNING: output not writable ({detail}); set [io] output to a writable "
          f"location with room (prefer scratch)")
    print("slurm parameters:")
    for k, v in _executor_params(cfg, array=(mode == "array")).items():
        print(f"  {k} = {v}")
    if mode == "array":
        if not cfg.pipeline_kwargs().get("keep_reg", True):
            print("WARNING: keep_reg=false + array makes the aggregate re-process "
                  "every plane; use single/local mode for small outputs.")
        try:
            n, ntasks, shards = plan(cfg)
            print(f"planes:  {n}  ->  {ntasks} array task(s), F={cfg.pipeline.planes_per_gpu}")
            for t, s in enumerate(shards):
                print(f"  task {t}: planes {s}")
            print("  + 1 dependent aggregate job (afterok)")
        except Exception as e:
            print(f"planes:  unreadable here ({type(e).__name__}: {e})")


def submit(cfg: HpcConfig, mode: str = "single", dry_run: bool = False):
    """Submit (or simulate) the run. Returns a dict describing what was launched."""
    output_dir = resolve_output_dir(cfg)

    if dry_run:
        _print_plan(cfg, mode, output_dir)
        return {"mode": mode, "output_dir": str(output_dir), "dry_run": True}

    output_dir.mkdir(parents=True, exist_ok=True)
    _pipe.assert_output_writable(output_dir)
    cfg_dict = cfg.to_dict()

    if mode == "local":
        result = _pipe.run_job(cfg, output_dir, role="single")
        return {"mode": "local", "output_dir": result}

    log_folder = output_dir / "logs"

    if mode == "single":
        ex = _make_executor(log_folder, _executor_params(cfg))
        job = ex.submit(_pipe.run_job, cfg_dict, str(output_dir), "single")
        print(f"submitted single job {job.job_id} -> {output_dir}")
        print(f"logs:   {log_folder}")
        return {"mode": "single", "job_id": job.job_id, "output_dir": str(output_dir)}

    if mode == "array":
        if not cfg.pipeline_kwargs().get("keep_reg", True):
            print("WARNING: keep_reg=false with --mode array makes the aggregate "
                  "re-process every plane (its binaries are gone). Use single/local "
                  "mode for small outputs, or keep keep_reg=true for array.")
        n, ntasks, shards = plan(cfg)
        ex = _make_executor(log_folder, _executor_params(cfg, array=True))
        jobs = []
        with ex.batch():
            for t in range(ntasks):
                jobs.append(
                    ex.submit(_pipe.run_job, cfg_dict, str(output_dir), "array", t, shards[t])
                )
        array_id = jobs[0].job_id.split("_")[0]
        print(f"submitted array job {array_id} ({ntasks} task(s)) -> {output_dir}")

        agg_params = _executor_params(cfg)
        agg_params["slurm_dependency"] = f"afterok:{array_id}"
        agg_ex = _make_executor(log_folder, agg_params)
        agg_job = agg_ex.submit(_pipe.run_job, cfg_dict, str(output_dir), "aggregate")
        print(f"submitted aggregate job {agg_job.job_id} (after {array_id} succeeds)")
        print(f"logs:   {log_folder}")
        return {
            "mode": "array",
            "array_id": array_id,
            "aggregate_id": agg_job.job_id,
            "n_tasks": ntasks,
            "output_dir": str(output_dir),
        }

    raise ValueError(f"unknown mode {mode!r}; use single, array, or local")
