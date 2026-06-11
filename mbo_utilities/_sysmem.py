"""Cheap system-memory sampling for pipeline logging.

One snapshot returns the Task-Manager headline number (system RAM percent
and used/total GB) plus a cheap breakdown of what this process tree is
using. Designed to be called every 1-2 seconds: the cost is one
``virtual_memory()`` call plus one recursive children walk (~8 ms on a
typical Windows box), dominated by the child enumeration.
"""

from __future__ import annotations

from typing import Any

_GB = 1024 ** 3


def mem_snapshot(proc: Any | None = None) -> dict[str, Any]:
    """System RAM headline plus this process tree's usage.

    Keys:
        sys_pct   system memory in use, matches Task Manager's percentage
        used_gb   system memory in use (total - available)
        total_gb  total physical memory
        proc_gb   summed RSS of this worker and its children
        nproc     number of processes in that tree
        top       (pid, name, gb) of the largest single process, or None
    """
    import psutil

    vm = psutil.virtual_memory()
    snap: dict[str, Any] = {
        "sys_pct": vm.percent,
        "used_gb": (vm.total - vm.available) / _GB,
        "total_gb": vm.total / _GB,
    }

    p = proc or psutil.Process()
    try:
        procs = [p] + p.children(recursive=True)
    except psutil.Error:
        procs = [p]

    rss = 0
    top = None
    top_rss = -1
    for c in procs:
        try:
            m = c.memory_info().rss
        except psutil.Error:
            continue
        rss += m
        if m > top_rss:
            top_rss, top = m, c

    snap["proc_gb"] = rss / _GB
    snap["nproc"] = len(procs)
    if top is not None:
        try:
            name = top.name()
        except psutil.Error:
            name = "?"
        snap["top"] = (top.pid, name, top_rss / _GB)
    else:
        snap["top"] = None
    return snap


def format_mem_line(snap: dict[str, Any]) -> str:
    """One-line human form of a snapshot for the task log."""
    line = (
        f"mem {snap['sys_pct']:.1f}% "
        f"{snap['used_gb']:.1f}/{snap['total_gb']:.1f} GB"
    )
    if "proc_gb" in snap:
        line += f" | pipeline {snap['proc_gb']:.2f} GB/{snap.get('nproc', 0)} proc"
        top = snap.get("top")
        if top:
            line += f", top {top[0]} {top[1]} {top[2]:.2f} GB"
    return line
