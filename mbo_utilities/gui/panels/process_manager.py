"""
Process manager panel.

Tracks and displays background processes that can survive GUI closure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imgui_bundle import imgui

from . import BasePanel

if TYPE_CHECKING:
    from mbo_utilities.gui.viewers import BaseViewer

__all__ = ["ProcessInfo", "ProcessManager", "ProcessPanel", "get_process_manager"]


@dataclass
class ProcessInfo:
    """Information about a background process."""

    pid: int
    description: str
    start_time: float
    task_type: str
    output_path: str | None = None
    args: dict | None = None
    # Runtime fields (not persisted in main list, but read from sidecar)
    status: str = "running"
    progress: float = 0.0
    status_message: str = ""
    error_details: dict | str | None = None

    def elapsed_seconds(self) -> float:
        """Seconds since process started."""
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        """Human-readable elapsed time."""
        elapsed = self.elapsed_seconds()
        if elapsed < 60:
            return f"{int(elapsed)} seconds ago"
        if elapsed < 3600:
            return f"{int(elapsed / 60)} minutes ago"
        return f"{elapsed / 3600:.1f} hours ago"

    def update_from_sidecar(self, verbose: bool = False) -> None:
        """Read progress_{uuid}.json (preferred) or progress_{pid}.json."""
        log_dir = Path.home() / "mbo" / "logs"
        uuid = self.args.get("_uuid") if self.args else None

        sidecar = None
        # Try UUID first
        if uuid:
            candidate = log_dir / f"progress_{uuid}.json"
            if candidate.exists():
                sidecar = candidate

        # Fallback to PID if UUID sidecar missing
        if not sidecar:
            candidate = log_dir / f"progress_{self.pid}.json"
            if candidate.exists():
                sidecar = candidate

        if sidecar and sidecar.exists():
            try:
                with open(sidecar) as f:
                    data = json.load(f)

                    # Validate: either UUID matches or PID matches
                    valid = False
                    if (uuid and data.get("uuid") == uuid) or data.get("pid") == self.pid:
                        valid = True

                    if valid:
                        self.status = data.get("status", self.status)
                        self.progress = data.get("progress", self.progress)
                        self.status_message = data.get("message", self.status_message)
                        self.error_details = data.get("details", self.error_details)
            except Exception:
                pass

    def is_alive(self) -> bool:
        """Check if process is still running."""
        try:
            if sys.platform == "win32":
                import ctypes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, self.pid)
                if handle:
                    k32.CloseHandle(handle)
                    return True
                err = ctypes.get_last_error()
                if err == 5:  # ERROR_ACCESS_DENIED
                    return True
                return False
            os.kill(self.pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def kill(self) -> bool:
        """Attempt to kill the process."""
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(self.pid)], check=True, capture_output=True)
            else:
                os.kill(self.pid, 9)  # SIGKILL
            return True
        except Exception:
            return False

    def tail_log(self, n: int = 50) -> list[str]:
        """Read the last n lines from the log file."""
        if not self.output_path:
            return []
        p = Path(self.output_path)
        if not p.is_file():
            return []
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.readlines()[-n:]
        except Exception:
            return []


class ProcessManager:
    """
    Manages background subprocesses that can survive GUI closure.

    Spawns processes using subprocess.Popen with no connection to parent.
    Tracks process info in a JSON file so processes can be monitored
    even after GUI restart.
    """

    PROCESS_FILE = Path.home() / "mbo" / "cache" / "running_processes.json"

    def __init__(self):
        self._processes: dict[int, ProcessInfo] = {}
        self._load()

    def _load(self) -> None:
        """Load process info from disk."""
        if not self.PROCESS_FILE.exists():
            return

        try:
            with open(self.PROCESS_FILE) as f:
                data = json.load(f)

            for entry in data.get("processes", []):
                info = ProcessInfo(**entry)
                self._processes[info.pid] = info
        except Exception:
            pass

    def _save(self) -> None:
        """Save process info to disk."""
        self.PROCESS_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "processes": [asdict(p) for p in self._processes.values()]
        }

        try:
            with open(self.PROCESS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def spawn(
        self,
        task_type: str,
        args: dict[str, Any],
        description: str,
        output_path: str | None = None,
    ) -> int | None:
        """Spawn a subprocess for a background task."""
        try:
            python_exe = sys.executable
            if sys.platform == "win32" and python_exe.endswith("python.exe"):
                pythonw = python_exe[:-10] + "pythonw.exe"
                if Path(pythonw).exists():
                    python_exe = pythonw

            from uuid import uuid4
            from datetime import datetime

            task_uuid = str(uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log_dir = Path.home() / ".mbo" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{timestamp}_{task_type}_{task_uuid[:8]}.log"

            args["_uuid"] = task_uuid
            args["_log_file"] = str(log_file)

            args_json = json.dumps(args)

            cmd = [
                python_exe,
                "-m", "mbo_utilities.gui._worker",
                task_type,
                args_json,
            ]

            f_out = open(log_file, "a")

            if sys.platform == "win32":
                creationflags = 0x00000008 | 0x00000200 | 0x08000000
                proc = subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                )

            info = ProcessInfo(
                pid=proc.pid,
                description=description,
                start_time=time.time(),
                task_type=task_type,
                output_path=str(log_file),
                args=args,
            )
            self._processes[proc.pid] = info
            self._save()

            return proc.pid

        except Exception:
            return None

    def get_all(self) -> list[ProcessInfo]:
        """Get all tracked processes."""
        return list(self._processes.values())

    def get_running(self) -> list[ProcessInfo]:
        """Get active processes (running or failed/pending view)."""
        active = []
        for p in self._processes.values():
            p.update_from_sidecar()
            if p.is_alive() or p.status == "error":
                active.append(p)
        return active

    def cleanup_finished(self) -> int:
        """Remove entries for processes that have finished successfully."""
        to_remove = []
        for pid, p in list(self._processes.items()):
            p.update_from_sidecar()
            if not p.is_alive():
                p.update_from_sidecar(verbose=True)

                retries = 0
                while p.status == "running" and retries < 10:
                    time.sleep(0.1)
                    p.update_from_sidecar(verbose=False)
                    retries += 1

                if p.status == "error":
                    continue

                if p.status != "completed":
                    p.status = "error"
                    p.status_message = "Process crashed unexpectedly"
                    continue

                to_remove.append(pid)

        for pid in to_remove:
            del self._processes[pid]

        if to_remove:
            self._save()

        return len(to_remove)

    def kill(self, pid: int) -> bool:
        """Kill a tracked process by pid."""
        if pid not in self._processes:
            return False

        info = self._processes[pid]
        if info.kill():
            del self._processes[pid]
            self._save()
            return True
        return False

    def has_running(self) -> bool:
        """Check if any tracked processes are still running."""
        return len(self.get_running()) > 0


# Global process manager instance
_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager


class ProcessPanel(BasePanel):
    """
    Panel for managing background processes.

    Shows:
    - Active tasks with progress
    - Background processes with logs
    - Kill/dismiss controls
    """

    name = "Processes"

    def __init__(self, viewer: BaseViewer):
        super().__init__(viewer)
        self._pm = get_process_manager()

    def draw(self) -> None:
        """Draw the process panel."""
        if not self._visible:
            return

        expanded, opened = imgui.begin("Processes", self._visible)
        self._visible = opened

        if expanded:
            self._pm.cleanup_finished()
            running = self._pm.get_running()

            if not running:
                imgui.text_disabled("No active tasks or background processes.")
            else:
                for proc in running:
                    self._draw_process(proc)

        imgui.end()

    def _draw_process(self, proc: ProcessInfo) -> None:
        """Draw a single process entry."""
        imgui.push_id(f"proc_{proc.pid}")
        imgui.bullet()

        # Color code status
        if proc.status == "error":
            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"[ERROR] {proc.description}")
        elif proc.status == "completed":
            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[DONE] {proc.description}")
        else:
            imgui.text(proc.description)

        imgui.indent()
        imgui.text_disabled(f"PID: {proc.pid} | Started: {proc.elapsed_str()}")

        # Kill button if active
        if proc.is_alive():
            imgui.same_line()
            if imgui.small_button(f"Kill##{proc.pid}"):
                self._pm.kill(proc.pid)

        # Progress bar
        if proc.progress > 0:
            imgui.progress_bar(proc.progress, imgui.ImVec2(-1, 0), f"{int(proc.progress * 100)}%")

        # Error message
        if proc.status == "error" and proc.status_message:
            imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.6, 1.0), f"Error: {proc.status_message}")

        # Log output
        if proc.output_path and Path(proc.output_path).is_file():
            if imgui.tree_node(f"Output##proc_{proc.pid}"):
                lines = proc.tail_log(20)
                line_height = imgui.get_text_line_height_with_spacing()
                output_height = min(len(lines) * line_height + 10, 150) if lines else line_height + 10

                if imgui.begin_child(f"##proc_output_{proc.pid}", imgui.ImVec2(0, output_height), imgui.ChildFlags_.borders):
                    for line in lines:
                        line_stripped = line.strip()
                        if "error" in line_stripped.lower():
                            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                        elif "warning" in line_stripped.lower():
                            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                        else:
                            imgui.text(line_stripped)
                    imgui.end_child()
                imgui.tree_pop()

        # Dismiss button for completed/failed
        if not proc.is_alive() and imgui.small_button(f"Dismiss##{proc.pid}"):
            if proc.pid in self._pm._processes:
                del self._pm._processes[proc.pid]
                self._pm._save()

        imgui.unindent()
        imgui.pop_id()
