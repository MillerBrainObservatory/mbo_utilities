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


# attribute -> submodule it lives in. `submit` the function collides with
# submit.py the module, so resolve via importlib.import_module (which imports the
# submodule without probing this package's __getattr__) and cache the function in
# globals so it shadows the submodule attribute the import sets.
_SOURCE = {
    "DEFAULT_OPS": ".config",
    "HpcConfig": ".config",
    "render_template": ".config",
    "run_job": ".pipeline",
    "plan": ".submit",
    "resolve_output_dir": ".submit",
    "submit": ".submit",
}


def __getattr__(name):
    if name not in _SOURCE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    val = getattr(importlib.import_module(_SOURCE[name], __name__), name)
    globals()[name] = val
    return val
