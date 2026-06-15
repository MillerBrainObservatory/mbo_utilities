"""`mbo hpc bench`: join per-plane `io` timings to the SLURM node each array task
ran on, to test whether spreading tasks across nodes changes I/O time.

The question this answers: when `io` is high, is it because many tasks were
packed onto one node (its NIC/tmp shared) — in which case spreading to more
nodes helps — or is it uniform regardless of node (a lustre/OST-level limit) —
in which case it doesn't. We read `timings.json` for the per-plane `io` and
`sacct` for the node each task landed on, group by node, and look at whether
`io` rises with tasks-per-node.

Parsing is split from the queries so the join logic is testable without a cluster.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import slurm

_TASK_RE = re.compile(r"(\d+)_(\d+)_log\.(?:err|out)$")
_ALLOC_RE = re.compile(r"^\d+_(\d+)$")  # array allocation row, not a .batch/.0 step


def array_id_and_tasks(logs_dir: Path) -> tuple[str | None, list[int]]:
    """From submitit logs `<arrayid>_<task>_log.err`, return (arrayid, task idxs)."""
    ids: set[str] = set()
    tasks: set[int] = set()
    if logs_dir.is_dir():
        for f in logs_dir.iterdir():
            m = _TASK_RE.search(f.name)
            if m:
                ids.add(m.group(1))
                tasks.add(int(m.group(2)))
    return (sorted(ids)[0] if ids else None), sorted(tasks)


def task_nodes(array_id: str) -> dict[int, dict]:
    """sacct -> {task_idx: {'node':, 'elapsed':, 'maxrss':, 'state':}} (alloc rows only)."""
    out: dict[int, dict] = {}
    for r in slurm.sacct_job(array_id):
        m = _ALLOC_RE.match(r.get("JobID", ""))
        if m:
            out[int(m.group(1))] = {
                "node": r.get("NodeList", ""),
                "elapsed": r.get("Elapsed", ""),
                "maxrss": r.get("MaxRSS", ""),
                "state": r.get("State", ""),
            }
    return out


def analyze(timings: dict, nodes: dict[int, dict], n_tasks: int):
    """Join each plane's io/reg/detect to the node its task ran on.

    Returns (per_task_rows, per_node_summary). pack (planes per task) is inferred
    from plane and task counts; for the spread test (planes_per_gpu=1) it is 1.
    """
    planes = timings.get("planes", [])
    n_planes = len(planes)
    pack = max(1, -(-n_planes // n_tasks)) if n_tasks else 1
    rows = []
    for i, p in enumerate(planes):
        task = i // pack
        rows.append({
            "task": task,
            "node": nodes.get(task, {}).get("node", "?"),
            "plane": p.get("plane", str(i)),
            "io": p.get("io", 0.0),
            "reg": p.get("reg", 0.0),
            "detect": p.get("detect", 0.0),
            "total": p.get("total", 0.0),
        })
    by_node: dict[str, list] = {}
    for r in rows:
        by_node.setdefault(r["node"], []).append(r)
    summary = []
    for node, rs in by_node.items():
        n_t = len({r["task"] for r in rs})
        summary.append({
            "node": node,
            "tasks": n_t,
            "planes": len(rs),
            "mean_io": sum(r["io"] for r in rs) / len(rs),
            "mean_total": sum(r["total"] for r in rs) / len(rs),
        })
    summary.sort(key=lambda s: (-s["tasks"], s["node"]))
    return rows, summary


def _verdict(summary: list[dict]) -> str:
    """Read the spreading effect off the per-node summary."""
    nodes = [s for s in summary if s["node"] not in ("", "?")]
    if len(nodes) < 2:
        return ("all tasks on one node - this run did NOT spread; "
                "rerun on a multi-node partition (e.g. hpc_a10_a) to test.")
    multi = [s for s in nodes if s["tasks"] > 1]
    single = [s for s in nodes if s["tasks"] == 1]
    if single and multi:
        io_single = sum(s["mean_io"] for s in single) / len(single)
        io_multi = sum(s["mean_io"] for s in multi) / len(multi)
        if io_single < 0.7 * io_multi:
            return (f"io is lower on 1-task nodes ({io_single:.0f}s) than packed "
                    f"nodes ({io_multi:.0f}s) -> packing hurts, spreading helps.")
        return (f"io similar on 1-task ({io_single:.0f}s) and packed nodes "
                f"({io_multi:.0f}s) -> bottleneck is NOT node packing "
                f"(lustre/OST or external load).")
    return (f"tasks spread across {len(nodes)} node(s), "
            f"mean io {sum(s['mean_io'] for s in nodes)/len(nodes):.0f}s; "
            f"compare this number to a single-node (packed) run.")


def compare(dirs) -> list[dict]:
    """Read each run's timings.json into a flat row for side-by-side display.

    Works for any run (local, single, array) — reads only timings.json, no SLURM.
    Use it to compare e.g. `--local --planes-per-gpu 1/2/4` to see whether the
    tiff->bin step parallelizes on local disk (isolating the writer from lustre).
    """
    rows = []
    for d in dirs:
        d = Path(d)
        t_path = d / "timings.json"
        if not t_path.exists():
            rows.append({"run": d.name, "ok": False})
            continue
        t = json.loads(t_path.read_text())
        tot = t.get("totals", {})

        def g(key, stat="sum"):
            return tot.get(key, {}).get(stat, 0.0)

        rows.append({
            "run": d.name, "ok": True,
            "planes": len(t.get("planes", [])),
            "workers": t.get("n_workers"),
            "io_sum": g("io"), "io_mean": g("io", "mean"),
            "reg_sum": g("reg"), "detect_sum": g("detect"),
            "total_sum": g("total"),
        })
    return rows


def run_compare(dirs) -> None:
    import click

    rows = compare(dirs)
    click.echo(f"{'RUN':<28}{'PLN':>4}{'WRK':>4}{'IO_SUM':>9}{'IO_MEAN':>9}"
               f"{'REG_SUM':>9}{'DET_SUM':>9}{'TOTAL':>10}")
    for r in rows:
        if not r.get("ok"):
            click.echo(f"{r['run'][:28]:<28}  (no timings.json)")
            continue
        click.echo(f"{r['run'][:28]:<28}{r['planes']:>4}{str(r['workers'] or '-'):>4}"
                   f"{r['io_sum']:>9.1f}{r['io_mean']:>9.1f}{r['reg_sum']:>9.1f}"
                   f"{r['detect_sum']:>9.1f}{r['total_sum']:>10.1f}")


def run_iobench(raw, frames: int = 500, planes=None) -> None:
    """Time the strided read (the tiff->bin bottleneck) of a frame subset for a
    few planes, then extrapolate to the full dataset.

    Lets you estimate the full run from a few planes without ever processing all
    of them: reads `frames` of each chosen plane (one at a time, low memory),
    reports per-plane, then scales to all planes x all frames. Read only — phase
    and write are ~1% (see `mbo hpc ioprobe`). `planes` are 1-based (mbo
    convention: plane 1 = zplane01); when omitted, a few evenly-spaced planes are
    sampled (first, middle, last) — not all.
    """
    import time

    import click
    import numpy as np
    from mbo_utilities import imread

    arr = imread(raw, fix_phase=False)
    nt = int(arr.shape[0])
    nz = int(arr.shape[2])
    if planes:
        plist = [p for p in planes if 1 <= p <= nz]  # 1-based (mbo convention)
    else:
        plist = sorted({1, (nz + 1) // 2, nz})  # a few evenly-spaced planes, not all
    k = min(int(frames), nt)

    rows = []
    for p in plist:
        s = time.perf_counter()
        chunk = np.asarray(arr[0:k, 0, p - 1, :, :])  # Z axis is 0-based
        rows.append((p, time.perf_counter() - s))
        del chunk

    sampled = sum(dt for _, dt in rows)
    full_sampled = sampled / k * nt              # the sampled planes, full frames
    per_plane = full_sampled / len(plist)        # mean full-plane read
    full_dataset = per_plane * nz                # all planes, full frames

    click.echo(f"read {k} of {nt} frames for planes {plist} of {nz}, "
               f"{arr.shape[-2]}x{arr.shape[-1]}  (read only = the io bottleneck)")
    click.echo(f"\n{'plane':>6}{'sampled':>10}{'per-frame':>12}{'full plane':>12}")
    for p, dt in rows:
        click.echo(f"{p:>6}{dt:>9.1f}s{dt / k * 1000:>10.2f}ms{dt / k * nt:>11.0f}s")
    click.secho(
        f"\n{len(plist)} sampled plane(s) at full {nt} frames: "
        f"{full_sampled:.0f}s ({full_sampled / 3600:.2f}h)",
        fg="cyan",
    )
    click.secho(
        f"full dataset ({nz} planes x {nt} frames): ~{full_dataset:.0f}s "
        f"({full_dataset / 3600:.2f}h)  [1 reader; pipeline parallelizes across workers]",
        fg="cyan",
    )


def run_ioprobe(raw, plane: int = 0, frames: int = 2000) -> None:
    """Decompose tiff->bin into read / phase-apply / write on real data.

    Reads `frames` of one plane COLD (no phase), then times the phase-correct and
    write on that in-memory chunk — so read is the only step that touches the
    filesystem and there's no page-cache confound on the phase numbers. Extrapolates
    each step to the full plane so you can see which dominates `io`.
    """
    import os
    import tempfile
    import time

    import click
    import numpy as np
    from mbo_utilities import imread
    from mbo_utilities.analysis.phasecorr import _apply_offset, bidir_phasecorr

    arr = imread(raw, fix_phase=False)  # raw, phase correction OFF for the read
    nt = int(arr.shape[0])
    nz = int(arr.shape[2])
    z = int(plane) - 1  # planes are 1-based (mbo convention); Z axis is 0-based
    if not 0 <= z < nz:
        raise ValueError(f"plane {plane} out of range 1..{nz}")
    k = min(int(frames), nt)

    t = time.perf_counter()
    chunk = np.asarray(arr[0:k, 0, z, :, :])  # COLD strided read from the FS
    t_read = time.perf_counter() - t

    # Phase/write timed on a bounded sub-sample so the probe's own memory stays
    # small (per-frame cost is linear, so it extrapolates the same). The read
    # still uses the full k frames — that's the read measurement.
    m = min(k, 512)
    sub = np.ascontiguousarray(chunk[:m])

    def _t(fn, *a, **kw):
        c = sub.copy()
        s = time.perf_counter()
        fn(c, *a, **kw)
        return time.perf_counter() - s

    t_fft = _t(_apply_offset, 2.3, use_fft=True)   # subpixel -> the FFT path
    t_int = _t(_apply_offset, 2.0, use_fft=False)  # integer -> np.roll
    s = time.perf_counter()
    _, off = bidir_phasecorr(sub, method="mean", use_fft=True)
    t_compute = time.perf_counter() - s

    tmp = os.path.join(os.environ.get("TMPDIR") or tempfile.gettempdir(),
                       f"ioprobe_{os.getpid()}.bin")
    s = time.perf_counter()
    sub.astype(np.int16).tofile(tmp)
    t_write = time.perf_counter() - s
    try:
        os.remove(tmp)
    except OSError:
        pass

    try:
        import resource
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB->GB (Linux)
    except Exception:
        peak_gb = None

    def full(v, n):
        return v / n * nt

    click.echo(f"plane {plane}: read {k} frames (phase timed on {m}), "
               f"{chunk.shape[-2]}x{chunk.shape[-1]}, offset={off:.2f}"
               + (f", peak RSS {peak_gb:.1f} GB" if peak_gb else ""))
    click.echo(f"\n{'step':<22}{'sampled':>10}{'per-frame':>12}{'full plane':>12}")
    for name, v, n in (("read (filesystem)", t_read, k),
                       ("phase FFT (subpix)", t_fft, m),
                       ("phase int (roll)", t_int, m),
                       ("write (tmp)", t_write, m)):
        click.echo(f"{name:<22}{v:>9.1f}s{v / n * 1000:>10.2f}ms{full(v, n):>11.0f}s")
    click.echo(f"{'compute offset(once)':<22}{t_compute:>9.1f}s")

    rd, ph = full(t_read, k), full(t_fft, m)
    click.secho(
        f"\n-> read {rd:.0f}s/plane vs phase {ph:.0f}s/plane: "
        + (f"read is {rd / max(ph, 0.01):.0f}x the phase -> read-bound (stage input)"
           if rd > ph else
           f"phase is {ph / max(rd, 0.01):.0f}x the read -> phase-bound (optimize the FFT apply)"),
        fg="cyan",
    )


def run_bench(output_dir) -> None:
    import click

    out = Path(output_dir).expanduser()
    tp = out / "timings.json"
    if not tp.exists():
        raise FileNotFoundError(f"no timings.json in {out} (finished array run?)")
    timings = json.loads(tp.read_text())

    array_id, tasks = array_id_and_tasks(out / "logs")
    if array_id is None:
        raise RuntimeError(
            f"no submitit array logs under {out / 'logs'} — bench needs an array run"
        )
    if not slurm.slurm_available():
        raise RuntimeError("scontrol/sacct not found — run on a SLURM login node")

    nodes = task_nodes(array_id)
    rows, summary = analyze(timings, nodes, len(tasks))

    click.echo(f"array {array_id}: {len(tasks)} task(s), {len(rows)} plane(s), "
               f"{len({s['node'] for s in summary if s['node'] not in ('', '?')})} distinct node(s)")
    click.echo(f"\n{'NODE':<16}{'TASKS':>6}{'PLANES':>7}{'MEAN_IO':>9}{'MEAN_TOTAL':>11}")
    for s in summary:
        click.echo(f"{s['node']:<16}{s['tasks']:>6}{s['planes']:>7}"
                   f"{s['mean_io']:>9.1f}{s['mean_total']:>11.1f}")

    click.echo(f"\n{'TASK':>4}  {'NODE':<16}{'IO':>8}{'REG':>8}{'DETECT':>8}  PLANE")
    for r in sorted(rows, key=lambda r: r["task"]):
        click.echo(f"{r['task']:>4}  {r['node']:<16}{r['io']:>8.1f}{r['reg']:>8.1f}"
                   f"{r['detect']:>8.1f}  {r['plane']}")

    click.secho(f"\n-> {_verdict(summary)}", fg="cyan")
