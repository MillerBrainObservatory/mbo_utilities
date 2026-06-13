"""Declarative config for the HPC Suite2p runner.

One source of truth: the dataclass fields below define defaults and types; the
``HELP`` table supplies the comment shown in the generated TOML template. The
loader merges a user TOML over the defaults, coerces types, and validates.

Sections map to TOML tables:
  [io]        input/output/name
  [slurm]     scheduler resources (partition, gres, cpus, mem, time)
  [pipeline]  pack factor F and worker/IO behaviour
  [ops]       suite2p ops overrides (open-ended)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

# suite2p ops overrides applied to every plane (was DEFAULT_OPS in run_pipeline.py).
DEFAULT_OPS: dict = {
    "anatomical_only": 4,
    "diameter": 2,
    "cellprob_threshold": -4,
    "flow_threshold": 0,
    "spatial_hp_cp": 3,
    "niter": 200,
    "do_registration": 1,
    "two_step_registration": 1,
    "do_regmetrics": False,
    "lam_percentile": 0,
    "min_neuropil_pixels": 0,
    "max_overlap": 0.99,
}


@dataclass
class IOConfig:
    input: str = ""
    output: str = ""
    name: str = "s2p"
    dated_subfolder: bool = True


@dataclass
class SlurmConfig:
    partition: str = "hpc_a100_a"
    gres: str = "gpu:a100:1"
    cpus_per_task: int = 16
    mem_gb: int = 128
    time: str = "24:00:00"
    exclusive: bool = False
    array_parallelism: int = 0
    account: str = ""
    qos: str = ""


@dataclass
class PipelineConfig:
    planes_per_gpu: int = 4
    threads_per_worker: int = 0
    node_local: bool = True


# Comment shown next to each key in the generated template.
HELP: dict = {
    "io": {
        "input": "directory of ScanImage TIFFs",
        "output": "results root; final folder is <output>/<date>_<name>",
        "name": "label for the dated output subfolder",
        "dated_subfolder": "false to write straight into <output>",
    },
    "slurm": {
        "partition": "scheduler partition (sinfo -s)",
        "gres": "GPUs per job, e.g. gpu:1 or gpu:a100:1 (sinfo -o '%P %G')",
        "cpus_per_task": "CPUs per job; workers derive from this and F",
        "mem_gb": "memory per job in GB",
        "time": "wall-time limit HH:MM:SS or D-HH:MM:SS",
        "exclusive": "reserve the whole node (uniform timing)",
        "array_parallelism": "max concurrent array tasks (0 = scheduler default)",
        "account": "SLURM account (blank = default)",
        "qos": "SLURM QOS (blank = default)",
    },
    "pipeline": {
        "planes_per_gpu": "pack factor F: most planes that fit one GPU before OOM",
        "threads_per_worker": "BLAS/OMP threads per worker (0 = cpus // workers)",
        "node_local": "compute on node-local NVMe, copy results back",
    },
}

_SECTIONS = {"io": IOConfig, "slurm": SlurmConfig, "pipeline": PipelineConfig}


@dataclass
class HpcConfig:
    io: IOConfig = field(default_factory=IOConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ops: dict = field(default_factory=lambda: dict(DEFAULT_OPS))

    @classmethod
    def from_dict(cls, raw: dict) -> "HpcConfig":
        """Merge a parsed TOML mapping over defaults, coercing field types."""
        kw = {}
        for name, klass in _SECTIONS.items():
            section = dict(raw.get(name, {}) or {})
            valid = {f.name for f in fields(klass)}
            unknown = set(section) - valid
            if unknown:
                raise ValueError(
                    f"[{name}] unknown key(s): {', '.join(sorted(unknown))}"
                )
            coerced = {
                f.name: _coerce(section[f.name], f.type)
                for f in fields(klass)
                if f.name in section
            }
            kw[name] = klass(**coerced)
        ops = dict(DEFAULT_OPS)
        ops.update(raw.get("ops", {}) or {})
        kw["ops"] = ops
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path) -> "HpcConfig":
        return cls.from_dict(tomllib.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if not self.io.input:
            raise ValueError("[io] input is required (path to ScanImage TIFFs)")
        if not self.slurm.gres:
            raise ValueError("[slurm] gres is required (e.g. gpu:1)")
        if self.pipeline.planes_per_gpu < 1:
            raise ValueError("[pipeline] planes_per_gpu must be >= 1")
        _time_to_minutes(self.slurm.time)  # raises on bad format

    def timeout_min(self) -> int:
        return _time_to_minutes(self.slurm.time)

    def to_dict(self) -> dict:
        d = {name: asdict(getattr(self, name)) for name in _SECTIONS}
        d["ops"] = dict(self.ops)
        return d


def _coerce(value, typ):
    """Coerce a TOML scalar to the dataclass field's annotated type."""
    if typ in ("int", int):
        return int(value)
    if typ in ("bool", bool):
        return bool(value)
    if typ in ("str", str):
        return str(value)
    return value


def _time_to_minutes(t) -> int:
    """SLURM wall-time -> whole minutes. Accepts int minutes or [D-]HH:MM:SS."""
    if isinstance(t, (int, float)):
        return max(1, int(t))
    s = str(t).strip()
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        raise ValueError(f"[slurm] bad time {t!r}; use HH:MM:SS or D-HH:MM:SS") from None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, parts[0], 0
    else:
        raise ValueError(f"[slurm] bad time {t!r}; use HH:MM:SS or D-HH:MM:SS")
    return max(1, days * 1440 + h * 60 + m + (1 if sec else 0))


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_template(input_path: str = "", output_path: str = "") -> str:
    """A commented TOML template seeded from the dataclass defaults."""
    defaults = HpcConfig()
    if input_path:
        defaults.io.input = str(input_path)
    if output_path:
        defaults.io.output = str(output_path)

    lines = [
        "# mbo hpc config",
        "# edit, then submit with:  mbo hpc run <this-file>",
        "# preview without submitting:  mbo hpc run <this-file> --dry-run",
        "",
    ]
    for name, klass in _SECTIONS.items():
        lines.append(f"[{name}]")
        section = getattr(defaults, name)
        width = max(len(f.name) for f in fields(klass))
        for f in fields(klass):
            val = _toml_value(getattr(section, f.name))
            comment = HELP.get(name, {}).get(f.name, "")
            row = f"{f.name:<{width}} = {val}"
            lines.append(f"{row}  # {comment}" if comment else row)
        lines.append("")

    lines.append("[ops]")
    lines.append("# suite2p ops overrides applied to every plane")
    for k, v in defaults.ops.items():
        lines.append(f"{k} = {_toml_value(v)}")
    lines.append("")
    return "\n".join(lines)
