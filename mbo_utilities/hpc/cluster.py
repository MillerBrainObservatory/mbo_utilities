"""Parse `sinfo` into per-partition resource summaries for `mbo hpc info`/`check`.

Kept dependency-light (stdlib only) so importing it doesn't slow CLI startup.
`parse_sinfo` is pure (takes the raw text) so it can be tested without a cluster.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field

# one space-free token per field (every value is single-token, so split() parses):
# partition cpus mem(MB) free(MB) state cpus_a/i/o/t gres gres_used tmpdisk(MB)
_SINFO_FIELDS = (
    "Partition:24,CPUs:8,Memory:12,FreeMem:12,StateCompact:14,"
    "CPUsState:18,Gres:28,GresUsed:32,TmpDisk:12"
)


def _int(s) -> int:
    try:
        return int(float(str(s).strip()))
    except (ValueError, TypeError):
        return 0


def _clean_gres(g: str) -> str:
    g = re.sub(r"\(.*?\)", "", g or "")  # drop (S:0-1) socket suffixes
    g = g.strip()
    return "" if g in ("(null)", "N/A", "") else g


def _gpu_count(g: str) -> int:
    g = re.sub(r"\(.*?\)", "", g or "")  # drop (IDX:...) suffixes
    total = 0
    for tok in g.split(","):
        tok = tok.strip()
        if tok.startswith("gpu"):
            m = re.search(r"(\d+)\s*$", tok)
            total += int(m.group(1)) if m else 0
    return total


def _cpus_alloc_idle(s: str) -> tuple[int, int]:
    cols = (s or "").split("/")  # allocated/idle/other/total
    return (_int(cols[0]), _int(cols[1])) if len(cols) >= 2 else (0, 0)


@dataclass
class Partition:
    name: str
    nodes: int = 0
    cpus_per_node: int = 0
    mem_mb: int = 0
    free_mb_min: int = 0
    free_mb_max: int = 0
    gres: str = ""
    cpus_alloc: int = 0  # CPUs allocated across the partition's nodes
    cpus_idle: int = 0   # CPUs idle across the partition's nodes
    gpus_used: int = 0   # GPUs in use across the partition's nodes
    gpus_total: int = 0  # GPUs present across the partition's nodes
    tmp_mb: int = 0  # configured node-local /tmp (sinfo TmpDisk), 0 = unreported
    states: set = field(default_factory=set)

    @property
    def gpus_per_node(self) -> int:
        return _gpu_count(self.gres)


def parse_sinfo(text: str, pattern: str = "hpc") -> list[Partition]:
    """Aggregate per-node `sinfo` lines into partitions whose name matches
    the regex `pattern` (default 'hpc' so it isn't tied to one cluster)."""
    rx = re.compile(pattern)
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        cols = line.split()
        if len(cols) < 8:
            continue
        name = cols[0].rstrip("*")  # default partition is marked with a trailing *
        if not rx.search(name):
            continue
        p = parts.setdefault(name, Partition(name))
        p.nodes += 1
        p.cpus_per_node = max(p.cpus_per_node, _int(cols[1]))
        p.mem_mb = max(p.mem_mb, _int(cols[2]))
        free = _int(cols[3])
        p.free_mb_min = free if p.free_mb_min == 0 else min(p.free_mb_min, free)
        p.free_mb_max = max(p.free_mb_max, free)
        if cols[4]:
            p.states.add(cols[4])
        a, i = _cpus_alloc_idle(cols[5])
        p.cpus_alloc += a
        p.cpus_idle += i
        g = _clean_gres(cols[6])
        if g:
            p.gres = g
        p.gpus_total += _gpu_count(cols[6])
        p.gpus_used += _gpu_count(cols[7])
        if len(cols) > 8:
            p.tmp_mb = max(p.tmp_mb, _int(cols[8]))
    return sorted(parts.values(), key=lambda p: p.name)


def sinfo_available() -> bool:
    return shutil.which("sinfo") is not None


def query_partitions(pattern: str = "hpc") -> list[Partition]:
    out = subprocess.run(
        ["sinfo", "-h", "-N", "-O", _SINFO_FIELDS],
        capture_output=True, text=True, check=False,
    ).stdout
    return parse_sinfo(out, pattern)


def _gb(mb: int) -> str:
    return f"{mb / 1024:.0f} GB" if mb else "?"


def format_partitions(parts: list[Partition]) -> str:
    rows = [("PARTITION", "NODES", "CPUS(A/I)", "MEM", "FREE", "GPU USE", "STATE")]
    for p in parts:
        if p.free_mb_min and p.free_mb_max and p.free_mb_min != p.free_mb_max:
            free = f"{p.free_mb_min / 1024:.0f}-{p.free_mb_max / 1024:.0f} GB"
        else:
            free = _gb(p.free_mb_max)
        gpu = f"{p.gpus_used}/{p.gpus_total}" if p.gpus_total else "-"
        rows.append((
            p.name, str(p.nodes), f"{p.cpus_alloc}/{p.cpus_idle}",
            _gb(p.mem_mb), free, gpu,
            ",".join(sorted(p.states)),
        ))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)
