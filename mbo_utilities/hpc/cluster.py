"""Parse `sinfo` into per-partition resource summaries for `mbo hpc info`/`check`.

Kept dependency-light (stdlib only) so importing it doesn't slow CLI startup.
`parse_sinfo` is pure (takes the raw text) so it can be tested without a cluster.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field

# one line per node: partition|host|cpus|mem(MB)|free(MB)|gres|state
_SINFO_FMT = "%P|%n|%c|%m|%e|%G|%t"


def _int(s) -> int:
    try:
        return int(float(str(s).strip()))
    except (ValueError, TypeError):
        return 0


def _clean_gres(g: str) -> str:
    g = re.sub(r"\(.*?\)", "", g or "")  # drop (S:0-1) socket suffixes
    g = g.strip()
    return "" if g in ("(null)", "N/A", "") else g


@dataclass
class Partition:
    name: str
    nodes: int = 0
    cpus_per_node: int = 0
    mem_mb: int = 0
    free_mb_min: int = 0
    free_mb_max: int = 0
    gres: str = ""
    states: set = field(default_factory=set)

    @property
    def gpus_per_node(self) -> int:
        m = re.search(r":(\d+)\s*$", _clean_gres(self.gres))
        return int(m.group(1)) if m else 0


def parse_sinfo(text: str, pattern: str = "hpc") -> list[Partition]:
    """Aggregate per-node `sinfo` lines into partitions whose name matches
    the regex `pattern` (default 'hpc' so it isn't tied to one cluster)."""
    rx = re.compile(pattern)
    parts: dict[str, Partition] = {}
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 7:
            continue
        name = cols[0].rstrip("*")  # default partition is marked with a trailing *
        if not rx.search(name):
            continue
        p = parts.setdefault(name, Partition(name))
        p.nodes += 1
        p.cpus_per_node = max(p.cpus_per_node, _int(cols[2]))
        p.mem_mb = max(p.mem_mb, _int(cols[3]))
        free = _int(cols[4])
        p.free_mb_min = free if p.free_mb_min == 0 else min(p.free_mb_min, free)
        p.free_mb_max = max(p.free_mb_max, free)
        g = _clean_gres(cols[5])
        if g:
            p.gres = g
        if cols[6]:
            p.states.add(cols[6])
    return sorted(parts.values(), key=lambda p: p.name)


def sinfo_available() -> bool:
    return shutil.which("sinfo") is not None


def query_partitions(pattern: str = "hpc") -> list[Partition]:
    out = subprocess.run(
        ["sinfo", "-h", "-N", "-o", _SINFO_FMT],
        capture_output=True, text=True, check=False,
    ).stdout
    return parse_sinfo(out, pattern)


def _gb(mb: int) -> str:
    return f"{mb / 1024:.0f} GB" if mb else "?"


def format_partitions(parts: list[Partition]) -> str:
    rows = [("PARTITION", "NODES", "CPUS", "MEM", "FREE", "GPUS", "STATE")]
    for p in parts:
        if p.free_mb_min and p.free_mb_max and p.free_mb_min != p.free_mb_max:
            free = f"{p.free_mb_min / 1024:.0f}-{p.free_mb_max / 1024:.0f} GB"
        else:
            free = _gb(p.free_mb_max)
        rows.append((
            p.name, str(p.nodes), str(p.cpus_per_node or "?"),
            _gb(p.mem_mb), free, p.gres or "-", ",".join(sorted(p.states)),
        ))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows)
