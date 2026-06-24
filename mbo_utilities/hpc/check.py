"""`mbo hpc check`: report a config's requested resources against the data and
the partition, and suggest concrete TOML deltas for structural problems.

Per the chosen design: the memory math is REPORTED (no hard pass/fail — the peak
estimate is rough), while the deterministic issues (array on a single-node
partition, CPU oversubscription, GPU over-request) WARN with suggested fixes.
"""

from __future__ import annotations

import math
import re

from . import cluster

# Peak host RAM ~ FACTOR * planes_per_gpu * per-plane binary size. Calibrated to
# the mk355 run (F=4, ~22.8 GB/plane -> OOM at the 128 GB cap). A rough guide.
_RAM_FACTOR = 1.5


def _gres_count(gres: str) -> int:
    m = re.search(r":(\d+)\s*$", cluster._clean_gres(gres))
    return int(m.group(1)) if m else (1 if cluster._clean_gres(gres) else 0)


def analyze(cfg, n_planes, nt, ly, lx, partition, mode):
    """Pure analysis. Returns (report_lines, suggestions) where each suggestion
    is (toml_key, value_or_None, reason)."""
    report, suggestions = [], []
    F = cfg.pipeline.planes_per_gpu
    bin_gb = nt * ly * lx * 2 / 1e9
    est_peak = _RAM_FACTOR * F * bin_gb
    mem = cfg.slurm.mem_gb

    report.append(f"{n_planes} planes, {nt} frames, {ly}x{lx}")
    report.append(f"per-plane binary : {bin_gb:.1f} GB")
    report.append(f"planes_per_gpu F : {F}")
    report.append(f"est. peak RAM    : ~{est_peak:.0f} GB   (~{_RAM_FACTOR} x F x binary, rough)")
    report.append(f"mem_gb requested : {mem}")
    if est_peak > mem:
        report.append(f"-> est. peak (~{est_peak:.0f} GB) exceeds mem_gb ({mem}); OOM risk")

    fit_F = max(1, int(mem / (_RAM_FACTOR * bin_gb))) if bin_gb else F

    if partition is None:
        report.append("partition not found via sinfo (off-cluster?); node checks skipped")
        if est_peak > mem:
            suggestions.append(("[slurm] mem_gb", math.ceil(est_peak * 1.1),
                                f"est. peak ~{est_peak:.0f} GB > {mem}"))
            if fit_F < F:
                suggestions.append(("[pipeline] planes_per_gpu", fit_F,
                                    f"or drop F to {fit_F} to fit {mem} GB"))
        return report, suggestions

    node_mem = partition.mem_mb / 1024
    free = partition.free_mb_max / 1024
    report.append(f"partition {partition.name}: {partition.nodes} node(s), "
                  f"{partition.cpus_per_node} CPUs/node, {node_mem:.0f} GB/node "
                  f"({free:.0f} GB free), {partition.gres or '-'}")

    # memory: advisory (no hard fail)
    if est_peak > mem and node_mem > mem:
        rec = int(min(node_mem, math.ceil(est_peak * 1.1)))
        suggestions.append(("[slurm] mem_gb", rec,
            f"node has {node_mem:.0f} GB; {mem} caps below est. peak ~{est_peak:.0f}"))
    if est_peak > mem and fit_F < F:
        suggestions.append(("[pipeline] planes_per_gpu", fit_F,
                            f"or drop F to {fit_F} to fit {mem} GB"))

    n_tasks = math.ceil(n_planes / F) if F else n_planes

    # node-local /tmp staging capacity (only matters when node_local is on).
    # The full per-shard binaries are written to the node's /tmp before copy-back;
    # on a single-node partition all co-resident tasks share that one /tmp.
    if cfg.pipeline.node_local and partition.tmp_mb:
        tmp_gb = partition.tmp_mb / 1024
        keep_raw = cfg.pipeline_kwargs().get("keep_raw", False)
        per_task = F * bin_gb * (2 if keep_raw else 1)
        concurrent = 1
        if mode == "array" and partition.nodes <= 1:
            fit_cpu = partition.cpus_per_node // max(1, cfg.slurm.cpus_per_task)
            fit_gpu = partition.gpus_per_node or n_tasks
            concurrent = max(1, min(n_tasks, fit_cpu, fit_gpu))
        need = per_task * concurrent
        line = f"node-local /tmp  : {tmp_gb:.0f} GB/node; staging ~{per_task:.0f} GB/task"
        if concurrent > 1:
            line += f" x {concurrent} co-resident = ~{need:.0f} GB"
        report.append(line)
        if need > tmp_gb:
            suggestions.append(("[pipeline] node_local", "false",
                f"staging ~{need:.0f} GB > {tmp_gb:.0f} GB /tmp on {partition.name}; "
                f"or lower planes_per_gpu"))

    # array buys nothing on a single-node partition
    if mode == "array" and partition.nodes <= 1:
        suggestions.append(("--mode", "single",
            f"{partition.name} is {partition.nodes} node; array can't spread, only packs onto it"))

    # CPU oversubscription when array packs n_tasks onto one node
    if mode == "array" and partition.nodes <= 1:
        cpt = cfg.slurm.cpus_per_task
        fit = partition.cpus_per_node // max(1, cpt)
        if n_tasks > fit:
            suggestions.append(("[slurm] cpus_per_task", None,
                f"{cpt} x {n_tasks} tasks = {cpt * n_tasks} CPUs, node has "
                f"{partition.cpus_per_node}; only {fit} run at once"))

    # GPUs per job vs node
    gj = _gres_count(cfg.slurm.gres)
    if partition.gpus_per_node and gj > partition.gpus_per_node:
        suggestions.append(("[slurm] gres", None,
            f"requests {gj} GPU(s)/job but node has {partition.gpus_per_node}"))

    return report, suggestions


def run_check(cfg, mode="single"):
    import click
    from mbo_utilities import imread
    from .pipeline import num_planes

    arr = imread(cfg.io.input)
    nt, ly, lx = arr.nt, arr.ny, arr.nx
    n_planes = num_planes(arr)

    partition = None
    if cluster.sinfo_available():
        match = [p for p in cluster.query_partitions(cfg.slurm.partition)
                 if p.name == cfg.slurm.partition]
        partition = match[0] if match else None

    report, suggestions = analyze(cfg, n_planes, nt, ly, lx, partition, mode)

    click.echo(f"input: {cfg.io.input}")
    for line in report:
        click.echo(f"  {line}")
    if suggestions:
        click.secho("\nsuggested changes:", fg="cyan")
        for key, val, why in suggestions:
            kv = f"{key} = {val}" if val is not None else key
            click.echo(f"  {kv}   # {why}")
    else:
        click.secho("\nrequest looks consistent with the data and partition.", fg="green")
