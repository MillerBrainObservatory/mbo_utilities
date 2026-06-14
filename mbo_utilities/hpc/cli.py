"""`mbo hpc` command group: init a config, submit it, check status."""

from __future__ import annotations

from pathlib import Path

import click


@click.group("hpc")
def hpc():
    """Run the Suite2p pipeline on SLURM (or locally) from a TOML config.

    \b
    Typical flow:
      mbo hpc init /data/raw      write hpc.toml (edit input/output/params)
      mbo hpc run hpc.toml --dry-run   preview the jobs
      mbo hpc run hpc.toml --mode array   submit array + dependent aggregate
    """


@hpc.command("init")
@click.argument("data_path", required=False, type=click.Path())
@click.option("-o", "--config", "config_path", type=click.Path(), default="hpc.toml",
              help="Config file to write.")
@click.option("-O", "--output", "output_root", type=click.Path(), default=None,
              help="Results root (default: <data_path>/../results).")
@click.option("--overwrite/--no-overwrite", default=False)
def hpc_init(data_path, config_path, output_root, overwrite):
    r"""
    Write a commented TOML config next to your data.

    \b
    Examples:
      mbo hpc init                       # hpc.toml in the current directory
      mbo hpc init /data/raw             # fills input + a results/ output
      mbo hpc init /data/raw -o run.toml
    """
    from mbo_utilities.hpc.config import render_template

    cfg_file = Path(config_path).expanduser()
    if cfg_file.exists() and not overwrite:
        click.secho(f"Exists: {cfg_file}  (--overwrite to replace)", fg="yellow")
        return

    # Absolute paths so the config works from any directory. Results default next
    # to the config (writable by construction), never the data tree — acquisition
    # directories are commonly read-only to the submitting user.
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
            "The current directory may be read-only (data directories often are). "
            "Re-run from a writable directory, or pass -o <writable-dir>/hpc.toml."
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
    r"""
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
@click.argument("target", required=False, type=click.Path())
def hpc_status(target):
    r"""
    Show timings for an output folder, or your SLURM queue.

    \b
    Examples:
      mbo hpc status /data/results/2025_07_27_mk355   # timings.json summary
      mbo hpc status                                  # squeue -u $USER
    """
    import json
    import os
    import subprocess

    if target:
        timings = Path(target) / "timings.json"
        if not timings.exists():
            click.secho(f"No timings.json under {target} (run not finished?)", fg="yellow")
            logs = Path(target) / "logs"
            errs = sorted(logs.glob("*.err")) if logs.is_dir() else []
            if errs:
                click.echo(f"\nLogs in {logs}:")
                for f in errs:
                    click.echo(f"  {f.name}  ({f.stat().st_size} bytes)")
                click.echo(f"\nRead the newest error log:\n  tail -n 80 {errs[-1]}")
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
