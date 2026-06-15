"""Declarative config for the HPC Suite2p runner.

One source of truth: the dataclass fields below define defaults and types; the
``HELP`` table supplies the comment shown in the generated TOML template. The
loader merges a user TOML over the defaults, coerces types, and validates.

Sections map to TOML tables:
  [io]          input/output/name
  [slurm]       scheduler resources (partition, gres, cpus, mem, time)
  [pipeline]    cluster/runner knobs (pack factor F, threads, node-local staging)
  [parameters]  one flat table forwarded to processing: suite2p ops AND lbm
                pipeline() knobs (keep_reg, norm_method, ...). split_parameters()
                routes each key to the right place.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

# suite2p ops overrides applied to every plane (was DEFAULT_OPS in run_pipeline.py).
DEFAULT_OPS: dict = {
    "algorithm": "cellpose",
    "img": "max_proj",
    "diameter": 2,
    "cellprob_threshold": -4,
    "flow_threshold": 0,
    "spatial_hp_cp": 3,
    "niter": 200,
    "do_registration": 1,
    "two_step_registration": 0,
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


# [parameters] keys routed to lbm pipeline() top-level kwargs; everything else
# in [parameters] is treated as a suite2p ops key.
_PIPELINE_PARAM_KEYS = frozenset({
    "keep_reg", "keep_raw", "norm_method", "correct_neuropil",
    "dff_window_size", "dff_percentile", "dff_smooth_window",
    "cell_filters", "accept_all_cells", "rastermap_kwargs", "save_json",
    "reader_kwargs", "roi_mode", "planes", "num_zplanes",
    "timepoints", "num_timepoints",
    "frames", "frame_indices",  # deprecated aliases of timepoints
})
# routed into writer_kwargs (phase correction).
_WRITER_PARAM_KEYS = frozenset({"fix_phase", "use_fft"})
# owned by the runner / [pipeline]; rejected if set in [parameters].
_MANAGED_PARAM_KEYS = frozenset({
    "save_path", "ops", "workers", "threads_per_worker",
    "skip_volumetric", "force_reg", "force_detect", "replot", "writer_kwargs",
    "planes_per_gpu", "node_local",
})

# pipeline-behaviour defaults merged into every run (see from_dict).
DEFAULT_PIPELINE_PARAMS: dict = {
    "keep_reg": False,
    "keep_raw": False,
    "fix_phase": True,
    "use_fft": True,
}

# inline comments for the keys shown in the generated template.
PARAM_HELP: dict = {
    "algorithm": "detection: cellpose | sourcery | sparsery",
    "img": "cellpose image: max_proj | meanImg | 'max_proj / meanImg'",
    "do_regmetrics": "registration PCA metrics, computationally intensive",
    "keep_reg": "keep registered data.bin (false will delete after processing)",
    "keep_raw": "keep raw pre-registration data_raw.bin",
    "fix_phase": "bidirectional scan-phase correction",
    "use_fft": "FFT phase correction (vs integer)",
}

# keys written into the generated [parameters] block. The rest of DEFAULT_OPS /
# DEFAULT_PIPELINE_PARAMS still applies at runtime (from_dict seeds them); they're
# just not surfaced in the file. Add any by hand to override.
TEMPLATE_PARAM_KEYS: tuple = (
    "algorithm", "img", "diameter", "cellprob_threshold", "flow_threshold",
    "do_registration", "two_step_registration", "do_regmetrics",
    "keep_reg", "keep_raw",
)

# subset knobs surfaced as commented hints. "Everything" is the default,
# expressed canonically as None (omit the key); a list restricts. [] is also
# accepted. None / [] / omitted all mean every plane / every timepoint.
TEMPLATE_COMMENTED: tuple = (
    ("planes", "[1, 7, 13]", "z-planes (1-based); omit for all. None or [] = all"),
    ("num_zplanes", "3", "first N z-planes; omit for all"),
    ("timepoints", "[1, 2, 3]", "timepoints (1-based); omit for all. None or [] = all"),
    ("num_timepoints", "500", "first N timepoints; omit for all"),
)


def split_parameters(params: dict) -> tuple[dict, dict]:
    """Route a flat [parameters] table into (suite2p ops, lbm pipeline kwargs).

    Dispatch by key name: known pipeline kwargs -> pipeline(); fix_phase/use_fft
    -> writer_kwargs; runner-owned keys are rejected; all other keys are suite2p
    ops parameters (so a typo silently becomes an ignored ops key).
    """
    ops = dict(DEFAULT_OPS)
    pipe: dict = {}
    writer = {"fix_phase": True, "use_fft": True}
    for k, v in (params or {}).items():
        if k in _MANAGED_PARAM_KEYS:
            raise ValueError(
                f"[parameters] '{k}' is managed by the runner; put cluster knobs "
                f"in [pipeline], not [parameters]"
            )
        if k in _WRITER_PARAM_KEYS:
            writer[k] = v
        elif k in _PIPELINE_PARAM_KEYS:
            pipe[k] = v
        else:
            ops[k] = v
    pipe["writer_kwargs"] = writer
    return ops, pipe


# Comment shown next to each key in the generated template.
HELP: dict = {
    "io": {
        "input": "directory of ScanImage TIFFs",
        "output": "WRITABLE results root; final folder is <output>/<date>_<name>",
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
    parameters: dict = field(
        default_factory=lambda: {**DEFAULT_OPS, **DEFAULT_PIPELINE_PARAMS}
    )

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
        params = {**DEFAULT_OPS, **DEFAULT_PIPELINE_PARAMS}
        params.update(raw.get("ops", {}) or {})  # legacy [ops] alias
        params.update(raw.get("parameters", {}) or {})
        kw["parameters"] = params
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
        split_parameters(self.parameters)  # raises on a runner-managed [parameters] key

    def timeout_min(self) -> int:
        return _time_to_minutes(self.slurm.time)

    def ops(self) -> dict:
        """suite2p ops dict routed out of [parameters]."""
        return split_parameters(self.parameters)[0]

    def pipeline_kwargs(self) -> dict:
        """kwargs forwarded to lbm_suite2p_python.pipeline() routed out of
        [parameters] (keep_reg, norm_method, writer_kwargs, ...)."""
        return split_parameters(self.parameters)[1]

    def to_dict(self) -> dict:
        d = {name: asdict(getattr(self, name)) for name in _SECTIONS}
        d["parameters"] = dict(self.parameters)
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
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        return "{" + ", ".join(f"{k} = {_toml_value(val)}" for k, val in v.items()) + "}"
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

    lines.append("[parameters]")
    allp = {**DEFAULT_OPS, **DEFAULT_PIPELINE_PARAMS}
    shown = [(k, _toml_value(allp[k])) for k in TEMPLATE_PARAM_KEYS if k in allp]
    width = max((len(f"{k} = {v}") for k, v in shown if PARAM_HELP.get(k)), default=0)
    for k, v in shown:
        row = f"{k} = {v}"
        comment = PARAM_HELP.get(k, "")
        lines.append(f"{row:<{width}}  # {comment}" if comment else row)
    for k, example, comment in TEMPLATE_COMMENTED:
        lines.append(f"# {k} = {example}   # {comment}")
    lines.append("")
    return "\n".join(lines)
