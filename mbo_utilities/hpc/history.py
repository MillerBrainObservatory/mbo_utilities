"""Recent-run registry and the one file-rotation primitive for ``mbo hpc``.

``prune_dir`` keeps any directory capped at its ``keep`` newest files (by mtime,
oldest evicted) — the single rotation used for both the recent-run registry
(``~/.mbo/hpc/runs``, last 10) and the central log dir (``~/.mbo/logs``).
``record_run`` drops one JSON per launch so ``mbo hpc watch`` / ``status`` with
no argument can resolve the last run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_RUNS = 10
MAX_LOGS = 20

# process-local monotonic counter: guarantees unique, creation-ordered record
# filenames even when the OS clock (Windows: ~ms) repeats within a tight loop.
_seq = 0


def _mtime(f: Path) -> float:
    try:
        return f.stat().st_mtime
    except OSError:
        return 0.0


def prune_dir(directory, keep: int, pattern: str = "*") -> None:
    """Keep the ``keep`` newest files (by mtime) under ``directory``; delete the
    rest. Newest kept, oldest evicted. Best-effort — never raises.
    """
    d = Path(directory)
    if keep < 0 or not d.is_dir():
        return
    try:
        files = [f for f in d.glob(pattern) if f.is_file()]
    except OSError:
        return
    # name as a tiebreak so a coarse-mtime filesystem still evicts deterministically
    # (record filenames embed a high-res timestamp, so name order ~ chronological).
    files.sort(key=lambda f: (_mtime(f), f.name), reverse=True)
    for f in files[keep:]:
        try:
            f.unlink()
        except OSError:
            pass


def logs_dir() -> Path:
    """Central HPC log dir (~/.mbo/logs), created if missing."""
    from mbo_utilities.preferences import get_mbo_dirs

    return get_mbo_dirs()["logs"]


def _runs_dir() -> Path:
    from mbo_utilities.preferences import get_mbo_dirs

    d = get_mbo_dirs()["base"] / "hpc" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_run(*, target=None, mode: str = "single", output_dir=None,
               job_id=None, config_path=None) -> None:
    """Append a run record to the registry and trim to the newest ``MAX_RUNS``.

    ``target`` is what ``watch`` / ``status`` resolve later — a SLURM job id when
    there is one, else the output dir. Best-effort: never raises.
    """
    try:
        rec = {
            "time": time.time(),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "job_id": str(job_id) if job_id else None,
            "output_dir": str(output_dir) if output_dir else None,
            "config_path": str(config_path) if config_path else None,
            "target": (
                str(target) if target
                else (str(job_id) if job_id else str(output_dir) if output_dir else None)
            ),
        }
        if rec["target"] is None:
            return
        global _seq
        _seq += 1
        runs = _runs_dir()
        # readable date + pid (cross-process unique) + monotonic seq (intra-process
        # unique and creation-ordered, so the name tiebreak in last_run/prune_dir
        # is chronological even when mtimes tie on a coarse-resolution fs).
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{_seq:09d}.json"
        (runs / name).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        prune_dir(runs, MAX_RUNS, "*.json")
    except OSError:
        pass


def last_run() -> dict | None:
    """The most recent run record, or None if the registry is empty."""
    try:
        runs = _runs_dir()
        files = sorted((f for f in runs.glob("*.json") if f.is_file()),
                       key=lambda f: (_mtime(f), f.name), reverse=True)
    except OSError:
        return None
    for f in files:
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None


def last_run_target() -> str | None:
    """watch/status target (job id or output dir) of the most recent run."""
    rec = last_run()
    if not rec:
        return None
    return rec.get("target") or rec.get("job_id") or rec.get("output_dir")
