"""HPC Suite2p runner: config-driven SLURM submission via submitit.

Python API:
    from mbo_utilities.hpc import HpcConfig, submit
    cfg = HpcConfig.from_toml("hpc.toml")
    submit(cfg, mode="array")          # or "single" / "local"

CLI:
    mbo hpc init /data/raw
    mbo hpc run hpc.toml --mode array

Imports are lazy (PEP 562): pulling in the `hpc` package to register the CLI
group must not drag in config/pipeline/submit (and their deps), so a bare
`mbo hpc ... --help` stays fast on slow network filesystems.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_OPS",
    "HpcConfig",
    "render_template",
    "run_job",
    "plan",
    "resolve_output_dir",
    "submit",
]


def __getattr__(name):
    if name in ("DEFAULT_OPS", "HpcConfig", "render_template"):
        from . import config
        return getattr(config, name)
    if name == "run_job":
        from .pipeline import run_job
        return run_job
    if name in ("plan", "resolve_output_dir", "submit"):
        from . import submit as _submit
        return getattr(_submit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
