"""Ask SLURM about a job: its state and the exact stdout/stderr paths.

``scontrol show job`` reports the authoritative ``StdOut``/``StdErr`` paths SLURM
streams to (on the shared FS, live) — better than reconstructing submitit's
filenames, and unaffected by node-local tmp staging. Once a job leaves the queue
scontrol forgets it, so ``sacct`` is the fallback for finished jobs.

Parsing is split from the subprocess calls so it can be tested without a cluster.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_JOB_ID_RE = re.compile(r"\d+(_\d+)?$")


def is_job_id(s) -> bool:
    """True for ``12345`` or array element ``12345_0`` (not a path)."""
    return bool(_JOB_ID_RE.fullmatch(str(s).strip()))


def slurm_available() -> bool:
    return shutil.which("scontrol") is not None or shutil.which("sacct") is not None


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return ""


def parse_scontrol(text: str) -> dict:
    """Flatten ``scontrol show job`` (Key=Value tokens) into a dict.

    The keys we read (JobState, StdOut, StdErr, NodeList, RunTime, Reason,
    WorkDir) are single-token values, so whitespace splitting is safe.
    """
    if not text or "Invalid job id" in text or "not found" in text:
        return {}
    d: dict[str, str] = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d.setdefault(k, v)
    return d


def scontrol_job(job_id) -> dict:
    return parse_scontrol(_run(["scontrol", "show", "job", str(job_id)]))


_SACCT_COLS = ["JobID", "State", "ExitCode", "Elapsed", "MaxRSS", "ReqMem", "NodeList"]


def parse_sacct(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        c = line.split("|")
        if len(c) >= len(_SACCT_COLS):
            rows.append(dict(zip(_SACCT_COLS, c)))
    return rows


def sacct_job(job_id) -> list[dict]:
    return parse_sacct(_run([
        "sacct", "-j", str(job_id), "-n", "-P",
        "-o", ",".join(_SACCT_COLS),
    ]))


def job_log_paths(job_id) -> tuple[Path | None, Path | None]:
    """(stdout, stderr) paths SLURM writes for a live job, via scontrol."""
    d = scontrol_job(job_id)
    out, err = d.get("StdOut"), d.get("StdErr")
    return (Path(out) if out else None, Path(err) if err else None)


def state_line(job_id) -> str:
    """One-line human state from scontrol (live) or sacct (finished)."""
    d = scontrol_job(job_id)
    if d:
        st = d.get("JobState", "?")
        extra = []
        node = d.get("NodeList", "")
        if d.get("RunTime") and d["RunTime"] not in ("00:00:00",):
            extra.append(f"runtime {d['RunTime']}")
        if node and node not in ("(null)", "None", ""):
            extra.append(node)
        reason = d.get("Reason")
        if st == "PENDING" and reason and reason not in ("None", ""):
            extra.append(f"reason {reason}")
        return f"job {job_id}: {st}" + (f"  ({', '.join(extra)})" if extra else "")
    rows = sacct_job(job_id)
    if rows:
        r = rows[0]
        node = r.get("NodeList", "")
        return (f"job {job_id}: {r['State']}  (exit {r['ExitCode']}"
                + (f", {node}" if node else "") + ")")
    return f"job {job_id}: not found (scontrol/sacct)"


def _table(rows: list[dict], cols: list[str]) -> str:
    head = [cols] + [[r.get(c, "") for c in cols] for r in rows]
    w = [max(len(str(row[i])) for row in head) for i in range(len(cols))]
    return "\n".join("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in head)


def job_report(job_id) -> str:
    """State + per-step sacct table + a one-line diagnosis for a finished job.

    For an out-of-the-queue job scontrol is gone, so this leans on sacct; point
    the user at the output dir (`mbo hpc status <dir>`) for the .err / FAILURE log.
    """
    lines = [state_line(job_id)]
    rows = sacct_job(job_id)
    if rows:
        lines.append(_table(rows, ["JobID", "State", "ExitCode", "Elapsed",
                                   "MaxRSS", "ReqMem"]))
        states = {r["State"].split()[0] for r in rows}
        if any("OUT_OF_ME" in s for s in states):
            lines.append("-> out of memory: raise [slurm] mem_gb or lower "
                         "planes_per_gpu (see `mbo hpc check`)")
        elif states - {"COMPLETED", "RUNNING", "PENDING"}:
            lines.append("-> a step failed; read its .err (`mbo hpc watch "
                         f"{job_id}`) and any FAILURE_*.log in the output dir")
    return "\n".join(lines)
