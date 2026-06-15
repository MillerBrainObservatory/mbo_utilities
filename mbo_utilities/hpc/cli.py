"""`mbo hpc` command group: init a config, submit it, check status."""

from __future__ import annotations

from pathlib import Path

import click


@click.group("hpc")
def hpc():
    """Run the Suite2p pipeline on SLURM (or locally) from a TOML config.

    \b
    Typical flow:
      mbo hpc info                     show partitions: nodes/CPUs/GPUs/free mem
      mbo hpc init /data/raw           write hpc.toml (edit input/output/params)
      mbo hpc check hpc.toml           does the request fit the data + partition?
      mbo hpc run hpc.toml --mode array   submit array + dependent aggregate

    \b
    Resources (size from `mbo hpc info`, verify with `mbo hpc check`):
      One job = one GPU (gres). planes_per_gpu (F) planes share it, each holding a
      movie in RAM, so peak RAM ~= F x per-plane size; set mem_gb to the node's
      capacity, not a small default (too low OOMs at the cgroup cap).
      --mode array only helps across MULTIPLE nodes. On a single-node partition all
      tasks pile onto one node, and cpus_per_task x tasks must fit the node's CPUs.
    """


@hpc.command("init")
@click.argument("data_path", required=False, type=click.Path())
@click.option("-o", "--config", "config_path", type=click.Path(), default=None,
              help="Config file to write (default: <data_path>/hpc.toml, else ./hpc.toml).")
@click.option("-O", "--output", "output_root", type=click.Path(), default=None,
              help="Results root (default: <config dir>/results).")
@click.option("--overwrite/--no-overwrite", default=False)
def hpc_init(data_path, config_path, output_root, overwrite):
    """
    Write a commented TOML config next to your data.

    \b
    Examples:
      mbo hpc init                       # ./hpc.toml in the current directory
      mbo hpc init /data/raw             # /data/raw/hpc.toml, fills input + results/
      mbo hpc init /data/raw -o run.toml # explicit config path
    """
    from mbo_utilities.hpc.config import render_template

    # Config lands in the given path; CWD only when no path is passed.
    if config_path:
        cfg_file = Path(config_path).expanduser()
    elif data_path:
        cfg_file = Path(data_path).expanduser() / "hpc.toml"
    else:
        cfg_file = Path("hpc.toml")

    if cfg_file.exists() and not overwrite:
        click.secho(f"Exists: {cfg_file}  (--overwrite to replace)", fg="yellow")
        return

    # Absolute paths so the config works from any directory. Results default next
    # to the config.
    inp = str(Path(data_path).expanduser().resolve()) if data_path else ""
    out = output_root or str(cfg_file.resolve().parent / "results")

    try:
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(
            render_template(input_path=inp, output_path=out),
            encoding="utf-8",
        )
    except OSError as e:
        raise click.ClickException(
            f"cannot write {cfg_file}: {e}\n"
            f"{cfg_file.parent} may be read-only (data directories often are). "
            "Pass -o <writable-dir>/hpc.toml, or run from a writable directory."
        )
    click.secho(f"Created: {cfg_file.resolve()}", fg="green")
    click.echo(f"Edit it, then: mbo hpc run {cfg_file}")


@hpc.command("run")
@click.argument("config_path", type=click.Path(exists=True), default="hpc.toml")
@click.option("--mode", type=click.Choice(["single", "array", "local"]), default="single",
              help="single GPU job, SLURM array+aggregate, or inline local run.")
@click.option("--dry-run", is_flag=True, help="Print the job layout; submit nothing.")
@click.option("--local", "force_local", is_flag=True, help="Shortcut for --mode local.")
@click.option("--input", "input_", default=None, help="Override [io] input.")
@click.option("--output", default=None, help="Override [io] output.")
@click.option("--name", default=None, help="Override [io] name.")
@click.option("--partition", default=None, help="Override [slurm] partition.")
@click.option("--gres", default=None, help="Override [slurm] gres.")
@click.option("--time", "time_", default=None, help="Override [slurm] time.")
@click.option("--planes-per-gpu", type=int, default=None, help="Override pack factor F.")
def hpc_run(config_path, mode, dry_run, force_local, input_, output, name,
            partition, gres, time_, planes_per_gpu):
    """
    Submit the pipeline described by CONFIG_PATH.

    \b
    Examples:
      mbo hpc run hpc.toml --dry-run
      mbo hpc run hpc.toml                       # single GPU job
      mbo hpc run hpc.toml --mode array          # array + dependent aggregate
      mbo hpc run hpc.toml --local               # run here, no SLURM
      mbo hpc run hpc.toml --partition hpc_l40s --gres gpu:l40s:1
    """
    from mbo_utilities.hpc.config import HpcConfig
    from mbo_utilities.hpc.submit import submit

    try:
        cfg = HpcConfig.from_toml(config_path)
        if input_:
            cfg.io.input = input_
        if output:
            cfg.io.output = output
        if name:
            cfg.io.name = name
        if partition:
            cfg.slurm.partition = partition
        if gres:
            cfg.slurm.gres = gres
        if time_:
            cfg.slurm.time = time_
        if planes_per_gpu:
            cfg.pipeline.planes_per_gpu = planes_per_gpu
        cfg.validate()
    except ValueError as e:  # bad TOML or failed validation
        raise click.ClickException(str(e))

    if force_local:
        mode = "local"

    try:
        submit(cfg, mode=mode, dry_run=dry_run)
    except (ValueError, OSError, ImportError, RuntimeError) as e:
        raise click.ClickException(str(e))


@hpc.command("status")
@click.argument("target", required=False)
def hpc_status(target):
    """
    Show a job's state, an output folder's timings, or your SLURM queue.

    \b
    Examples:
      mbo hpc status 5162141                          # job state + exit + diagnosis
      mbo hpc status /data/results/2025_07_27_mk355   # timings.json summary
      mbo hpc status                                  # squeue -u $USER
    """
    import json
    import os
    import subprocess

    from mbo_utilities.hpc import slurm

    if target and slurm.is_job_id(target):
        click.echo(slurm.job_report(target))
        return

    if target:
        logs = Path(target) / "logs"
        failures = sorted(logs.glob("FAILURE_*.log")) if logs.is_dir() else []
        timings = Path(target) / "timings.json"
        if not timings.exists():
            click.secho(f"No timings.json under {target} (run not finished?)", fg="yellow")
            errs = sorted(logs.glob("*.err")) if logs.is_dir() else []
            if errs:
                click.echo(f"\nLogs in {logs}:")
                for f in errs:
                    click.echo(f"  {f.name}  ({f.stat().st_size} bytes)")
                click.echo(f"\nRead the newest error log:\n  tail -n 80 {errs[-1]}")
            if failures:
                click.secho(f"\nFAILURE report(s) in {logs}:", fg="red")
                for f in failures:
                    click.echo(f"  {f}")
            return
        report = json.loads(timings.read_text())
        totals = report.get("totals", {})
        if totals:
            click.echo(f"{'stage':<12}{'sum':>10}{'mean':>10}{'max':>10}")
            for stage, t in totals.items():
                click.echo(f"{stage:<12}{t['sum']:>10.1f}{t['mean']:>10.1f}{t['max']:>10.1f}")
        for k, v in (report.get("wall") or {}).items():
            click.echo(f"wall.{k}: {v:.1f}s")
        return

    user = os.environ.get("USER", "")
    try:
        subprocess.run(["squeue", "-u", user], check=False)
    except FileNotFoundError:
        click.secho("squeue not found (not on a SLURM login node?)", fg="yellow")


@hpc.command("watch")
@click.argument("target", required=False, default="hpc.toml", type=click.Path())
@click.option("-o", "--out", "stream_out", is_flag=True,
              help="Start on stdout (.out); default is stderr (.err).")
@click.option("--no-follow", is_flag=True, help="Print the tail once and exit.")
@click.option("-n", "--lines", default=40, show_default=True, help="Initial tail lines.")
def hpc_watch(target, stream_out, no_follow, lines):
    """
    Follow a run's .err/.out logs, from a job id, a config, or an output dir.

    \b
    Examples:
      mbo hpc watch 5162141               # by SLURM job id (prints state first)
      mbo hpc watch                       # newest run from hpc.toml, follow .err
      mbo hpc watch hpc.toml -o           # follow .out instead
      mbo hpc watch /data/results/2025_07_27_mk355
      mbo hpc watch --no-follow           # tail once, don't stream

    A job id resolves exact log paths via scontrol and shows job state before any
    logs exist. While following (terminal): o/e switch out/err, n/p switch task
    logs, q quit.
    """
    from mbo_utilities.hpc.logs import watch

    try:
        watch(target, stream="out" if stream_out else "err",
              follow=not no_follow, lines=lines)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except KeyboardInterrupt:
        pass


@hpc.command("info")
@click.argument("pattern", required=False, default="hpc")
def hpc_info(pattern):
    """
    Show cluster partitions (nodes, CPUs, GPUs, memory) matching PATTERN.

    PATTERN is a regex on the partition name (default 'hpc'), so it isn't tied
    to one cluster. Use it to size a job: NODES tells you whether --mode array
    can spread; FREE / GPUS tell you what to request.

    \b
    Examples:
      mbo hpc info               # partitions matching 'hpc'
      mbo hpc info a100          # only a100 partitions
      mbo hpc info '.'           # everything
    """
    from mbo_utilities.hpc import cluster

    if not cluster.sinfo_available():
        click.secho("sinfo not found (not on a SLURM login node?)", fg="yellow")
        return
    parts = cluster.query_partitions(pattern)
    if not parts:
        click.secho(f"no partitions match /{pattern}/", fg="yellow")
        return
    click.echo(cluster.format_partitions(parts))


@hpc.command("check")
@click.argument("config_path", type=click.Path(exists=True), default="hpc.toml")
@click.option("--mode", type=click.Choice(["single", "array", "local"]), default="single",
              help="Mode to check the request against (CPU packing depends on it).")
def hpc_check(config_path, mode):
    """
    Check a config's requested resources against the data and the partition.

    Reads the input to size per-plane memory, queries sinfo for the partition,
    and reports the memory math plus suggested fixes for structural problems
    (array on a single node, cpus_per_task x tasks > node CPUs, gres > node GPUs).

    \b
    Examples:
      mbo hpc check hpc.toml
      mbo hpc check hpc.toml --mode array
    """
    from mbo_utilities.hpc.config import HpcConfig
    from mbo_utilities.hpc.check import run_check

    try:
        cfg = HpcConfig.from_toml(config_path)
    except ValueError as e:
        raise click.ClickException(str(e))
    try:
        run_check(cfg, mode=mode)
    except (ValueError, OSError, RuntimeError) as e:
        raise click.ClickException(str(e))


@hpc.command("bench")
@click.argument("output_dir", type=click.Path(exists=True))
def hpc_bench(output_dir):
    """
    Join an array run's per-plane `io` to the node each task ran on.

    Tells you whether SLURM actually spread the tasks across nodes, and whether
    `io` is lower on less-loaded nodes (packing hurts -> spreading helps) or
    uniform regardless (a lustre/OST or external-load limit). Run it on the
    output dir of a finished `--mode array` run.

    \b
    Examples:
      mbo hpc bench /lustre/.../2026_06_14_s2p
    """
    from mbo_utilities.hpc.bench import run_bench

    try:
        run_bench(output_dir)
    except (FileNotFoundError, RuntimeError, OSError) as e:
        raise click.ClickException(str(e))


@hpc.command("compare")
@click.argument("output_dirs", nargs=-1, type=click.Path(exists=True), required=True)
def hpc_compare(output_dirs):
    """
    Tabulate timings.json across runs side-by-side (no SLURM needed).

    Works for local runs, so it's the way to compare a local sweep:
    run `mbo hpc run hpc.toml --local --planes-per-gpu N` for N=1,2,4 (different
    --name each), then compare their `io` to see if the tiff->bin step
    parallelizes on local disk.

    \b
    Examples:
      mbo hpc compare ./cmp/*_f1 ./cmp/*_f2 ./cmp/*_f4
    """
    from mbo_utilities.hpc.bench import run_compare

    run_compare(output_dirs)


@hpc.command("ioprobe")
@click.argument("raw", type=click.Path(exists=True))
@click.option("--plane", default=1, show_default=True, help="1-based z-plane to read.")
@click.option("--frames", default=2000, show_default=True, help="Frames to read cold.")
def hpc_ioprobe(raw, plane, frames):
    """
    Decompose tiff->bin into read / phase-correct / write on real data.

    Proves whether `io` is read-bound (strided filesystem reads -> staging helps)
    or phase-bound (FFT scan-phase correction -> drop use_fft). Reads N frames of
    one plane cold, then times phase-apply and write on that in-memory chunk.
    Run it on a fresh range so the read isn't served from page cache.

    \b
    Examples:
      mbo hpc ioprobe /lustre/.../raw
      mbo hpc ioprobe /lustre/.../raw --plane 3 --frames 4000
    """
    from mbo_utilities.hpc.bench import run_ioprobe

    try:
        run_ioprobe(raw, plane=plane, frames=frames)
    except (OSError, ValueError, RuntimeError) as e:
        raise click.ClickException(str(e))


@hpc.command("iobench")
@click.argument("raw", type=click.Path(exists=True))
@click.option("--planes", default="",
              help="1-based z-planes, comma-separated (e.g. 1,7,14). Default: a few evenly-spaced.")
@click.option("--frames", default=500, show_default=True, help="Frames per plane to read.")
def hpc_iobench(raw, planes, frames):
    """
    Estimate the full run's io from a few planes, without processing all of them.

    Reads `frames` of each chosen plane (strided -> the io bottleneck), then
    scales to all planes x all frames. Compare a few planes against the
    full-dataset estimate without the multi-hour pipeline. Run on a fresh range
    so the read isn't served from page cache.

    \b
    Examples:
      mbo hpc iobench /lustre/.../raw
      mbo hpc iobench /lustre/.../raw --planes 1,7,13 --frames 1000
    """
    from mbo_utilities.hpc.bench import run_iobench

    pl = [int(p) for p in planes.split(",") if p.strip()] if planes else None
    try:
        run_iobench(raw, frames=frames, planes=pl)
    except (OSError, ValueError, RuntimeError) as e:
        raise click.ClickException(str(e))
