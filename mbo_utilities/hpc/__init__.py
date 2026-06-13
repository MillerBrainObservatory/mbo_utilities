"""HPC Suite2p runner: config-driven SLURM submission via submitit.

Python API:
    from mbo_utilities.hpc import HpcConfig, submit
    cfg = HpcConfig.from_toml("hpc.toml")
    submit(cfg, mode="array")          # or "single" / "local"

CLI:
    mbo hpc init /data/raw
    mbo hpc run hpc.toml --mode array
"""

from .config import DEFAULT_OPS, HpcConfig, render_template
from .pipeline import run_job
from .submit import plan, resolve_output_dir, submit

__all__ = [
    "DEFAULT_OPS",
    "HpcConfig",
    "render_template",
    "run_job",
    "plan",
    "resolve_output_dir",
    "submit",
]
