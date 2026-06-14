"""Find and follow a run's SLURM logs from its hpc.toml (or output dir).

Submitit streams stdout/stderr straight into ``<output_dir>/logs`` as
``*.out`` / ``*.err`` files; SLURM writes them live. They are NEVER in
``$TMPDIR`` — node-local staging only holds the *results* (data.bin, ops.npy,
timings.json) in tmp until copy-back, so logs are always read from the output
dir even while a node-local run is mid-flight.

``resolve_output_dir`` in submit.py returns the next free ``_N`` dir, so it
cannot point at an existing run. Here we instead glob ``<output>/<date>_<name>*``
and pick the one with the most recent log activity.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .config import HpcConfig


def _recency(p: Path) -> float:
    """Newest .out/.err mtime under ``p/logs``, else ``p``'s own mtime."""
    logs = p / "logs"
    files = list(logs.glob("*.out")) + list(logs.glob("*.err")) if logs.is_dir() else []
    try:
        if files:
            return max(f.stat().st_mtime for f in files)
        return p.stat().st_mtime
    except OSError:
        return 0.0


def candidate_output_dirs(cfg: HpcConfig) -> list[Path]:
    """Existing ``<output>/<date>_<name>*`` dirs, newest log activity first."""
    root = Path(cfg.io.output or ".").expanduser()
    if not cfg.io.dated_subfolder:
        return [root] if root.is_dir() else []
    name = cfg.io.name
    found: dict[Path, bool] = {}
    for pat in (f"*_{name}", f"*_{name}_*"):
        for p in root.glob(pat):
            if p.is_dir():
                found[p] = True
    return sorted(found, key=_recency, reverse=True)


def resolve_logs_dir(target) -> tuple[Path | None, Path | None]:
    """Map a config file or directory to ``(logs_dir, output_dir)``.

    - a .toml file -> newest output dir it describes that has a logs/ subdir
    - an output dir -> its logs/ subdir
    - a logs dir itself -> used directly
    - a results root -> newest child output dir with a logs/ subdir
    """
    p = Path(target).expanduser()
    if p.is_file():
        cfg = HpcConfig.from_toml(p)
        cands = candidate_output_dirs(cfg)
        for d in cands:
            if (d / "logs").is_dir():
                return d / "logs", d
        if cands:
            return cands[0] / "logs", cands[0]
        return None, None
    if p.is_dir():
        if (p / "logs").is_dir():
            return p / "logs", p
        if any(p.glob("*.out")) or any(p.glob("*.err")):
            return p, p.parent
        children = [c for c in p.iterdir() if c.is_dir() and (c / "logs").is_dir()]
        if children:
            d = max(children, key=_recency)
            return d / "logs", d
    return None, None


def list_logs(logs_dir: Path, stream: str) -> list[Path]:
    """Sorted (oldest->newest) .out or .err files in ``logs_dir``."""
    ext = "out" if stream == "out" else "err"
    files = [f for f in logs_dir.glob(f"*.{ext}") if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime)
    return files


def tail_lines(path: Path, n: int = 40) -> str:
    """Last ``n`` lines of ``path`` without reading the whole file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                step = min(8192, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        return "\n".join(data.decode("utf-8", "replace").splitlines()[-n:])
    except OSError:
        return ""


class _KeyReader:
    """Non-blocking single-key reader; a no-op when stdin isn't a TTY."""

    def __enter__(self):
        self.enabled = False
        self._fd = None
        self._old = None
        try:
            if not sys.stdin.isatty():
                return self
        except Exception:
            return self
        if os.name == "nt":
            self.enabled = True
            return self
        try:
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except Exception:
            self.enabled = False
        return self

    def __exit__(self, *exc):
        if self._fd is not None and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get(self):
        if not self.enabled:
            return None
        if os.name == "nt":
            import msvcrt
            return msvcrt.getwch() if msvcrt.kbhit() else None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None


def _header(path: Path, stream: str, interactive: bool, n_files: int) -> str:
    bar = f"--- {path.name}  [{stream}]"
    if interactive:
        keys = "o/e=out/err  q=quit"
        if n_files > 1:
            keys = "o/e=out/err  n/p=file  q=quit"
        bar += f"  ({keys})"
    return bar + " ---"


def watch(target="hpc.toml", stream="err", follow=True, lines=40) -> None:
    """Tail (and optionally follow) a run's .err/.out logs.

    Raises FileNotFoundError if no run/logs can be located.
    """
    logs_dir, output_dir = resolve_logs_dir(target)
    if logs_dir is None or not logs_dir.is_dir():
        raise FileNotFoundError(
            f"no logs for {target}. Run may not have started; "
            f"check `mbo hpc status`."
        )

    n_out = len(list_logs(logs_dir, "out"))
    n_err = len(list_logs(logs_dir, "err"))
    print(f"watching {output_dir}  ({n_err} .err, {n_out} .out)")

    files = list_logs(logs_dir, stream)
    cur = files[-1] if files else None

    if not follow:
        if cur is None:
            print(f"no .{stream} logs yet in {logs_dir}")
            return
        print(_header(cur, stream, False, len(files)))
        print(tail_lines(cur, lines))
        other = "out" if stream == "err" else "err"
        n_other = n_out if other == "out" else n_err
        if n_other:
            print(f"\n({n_other} .{other} log(s); -o for .out, drop --no-follow to follow)")
        return

    with _KeyReader() as keys:
        interactive = keys.enabled
        pos = 0
        if cur is not None:
            print(_header(cur, stream, interactive, len(files)))
            print(tail_lines(cur, lines))
            pos = cur.stat().st_size
        else:
            print(f"waiting for .{stream} logs in {logs_dir} ...")

        while True:
            k = keys.get()
            if k:
                k = k.lower()
                if k == "q":
                    return
                if k in ("o", "e"):
                    want = "out" if k == "o" else "err"
                    if want != stream:
                        stream = want
                        files = list_logs(logs_dir, stream)
                        cur = files[-1] if files else None
                        pos = 0
                        if cur is not None:
                            print("\n" + _header(cur, stream, interactive, len(files)))
                            print(tail_lines(cur, lines))
                            pos = cur.stat().st_size
                        else:
                            print(f"\nno .{stream} logs yet")
                    continue
                if k in ("n", "p") and files:
                    i = files.index(cur) if cur in files else len(files) - 1
                    i = min(i + 1, len(files) - 1) if k == "n" else max(i - 1, 0)
                    cur = files[i]
                    pos = 0
                    print("\n" + _header(cur, stream, interactive, len(files)))
                    print(tail_lines(cur, lines))
                    pos = cur.stat().st_size
                    continue

            if cur is None:
                files = list_logs(logs_dir, stream)
                if files:
                    cur = files[-1]
                    pos = 0
                    print(_header(cur, stream, interactive, len(files)))
            elif cur.exists():
                size = cur.stat().st_size
                if size > pos:
                    with open(cur, "rb") as fh:
                        fh.seek(pos)
                        chunk = fh.read()
                    pos = size
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
                elif size < pos:
                    pos = 0
            # also refresh the file list so n/p sees newly spawned task logs
            files = list_logs(logs_dir, stream) or files
            time.sleep(0.3)
