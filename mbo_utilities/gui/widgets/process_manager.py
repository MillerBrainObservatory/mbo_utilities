"""
process manager for background operations.

tracks subprocesses that can survive gui closure. stores process info
to disk so orphaned processes can be detected on next gui launch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from mbo_utilities import log

logger = log.get("gui.process_manager")



@dataclass
class ProcessInfo:
    pid: int
    description: str
    start_time: float
    task_type: str
    output_path: str | None = None
    args: dict | None = None
    # runtime fields (not persisted in main list, but read from sidecar)
    status: str = "running"
    progress: float = 0.0
    status_message: str = ""
    error_details: dict | str | None = None

    def elapsed_seconds(self) -> float:
        """Seconds since process started."""
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        """human-readable elapsed time."""
        elapsed = self.elapsed_seconds()
        if elapsed < 60:
            return f"{int(elapsed)} seconds ago"
        if elapsed < 3600:
            return f"{int(elapsed / 60)} minutes ago"
        return f"{elapsed / 3600:.1f} hours ago"

    def update_from_sidecar(self, verbose=False):
        """Read progress_{uuid}.json (preferred) or progress_{pid}.json."""
        # Assume log dir is standard
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
            # Check PID sidecar
            candidate = log_dir / f"progress_{self.pid}.json"
            if candidate.exists():
                sidecar = candidate
            elif uuid:
                 # If neither exists and we have UUID, candidate for error reporting is UUID file
                 # (or we report missing for both)
                 pass

        if verbose:
            logger.debug(f"Checking sidecar for PID {self.pid} (UUID={uuid})")
            if uuid:
                logger.debug(f"  UUID path: {log_dir / f'progress_{uuid}.json'} (Exists: {(log_dir / f'progress_{uuid}.json').exists()})")
            logger.debug(f"  PID path:  {log_dir / f'progress_{self.pid}.json'} (Exists: {(log_dir / f'progress_{self.pid}.json').exists()})")

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
                    else:
                        logger.debug(f"Identity mismatch in sidecar {sidecar}. Found pid={data.get('pid')}, uuid={data.get('uuid')}")
            except Exception as e:
                # Silently ignore read errors - atomic writes should prevent most issues
                logger.debug(f"Failed to read sidecar for pid {self.pid}: {e}")
        elif verbose:
            logger.debug(f"No sidecar found for PID {self.pid}.")

    def is_alive(self) -> bool:
        """Check if process is still running."""
        try:
            # cross-platform check using os.kill with signal 0
            # on windows, this uses ctypes internally
            if sys.platform == "win32":
                import ctypes
                # Use WinDLL with use_last_error=True to correctly capture error codes
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, self.pid)
                if handle:
                    k32.CloseHandle(handle)
                    return True
                err = ctypes.get_last_error()
                if err == 5:
                    # ERROR_ACCESS_DENIED: Process exists but we can't query it. Assume alive.
                    return True
                # ERROR_INVALID_PARAMETER (87) usually means PID found no process (Dead).
                if err != 87:
                    # Log unusual errors
                    logger.debug(f"PID {self.pid} is_alive check failed. OpenProcess err={err}")
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
        except Exception as e:
            logger.warning(f"Failed to kill process {self.pid}: {e}")
            return False

    def get_last_log_line(self) -> str | None:
        """Read the last non-empty line from the log file."""
        if not self.output_path:
            return None
        p = Path(self.output_path)
        if not p.is_file():
            return None
        try:
            with open(p, "rb") as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                if pos == 0:
                    return None
                # Read backwards to find a newline
                buffer = 4096
                data = b""
                while pos > 0 and (len(data) < 2 or b"\n" not in data[1:]):
                    read_size = min(pos, buffer)
                    pos -= read_size
                    f.seek(pos)
                    data = f.read(read_size) + data
                lines = data.decode("utf-8", errors="replace").splitlines()
                return lines[-1] if lines else None
        except Exception:
            return None

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
    manages background subprocesses that can survive gui closure.

    spawns processes using subprocess.Popen with no connection to parent.
    tracks process info in a json file so processes can be monitored
    even after gui restart.
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

            # note: don't auto-cleanup on load so user can see finished/failed processes
        except Exception as e:
            logger.warning(f"Failed to load process file: {e}")

    def _save(self) -> None:
        """Save process info to disk."""
        self.PROCESS_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "processes": [asdict(p) for p in self._processes.values()]
        }

        try:
            with open(self.PROCESS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save process file: {e}")

    def spawn(
        self,
        task_type: str,
        args: dict[str, Any],
        description: str,
        output_path: str | None = None,
    ) -> int | None:
        """
        Spawn a subprocess for a background task.

        Parameters
        ----------
        task_type : str
            type of task: "save_zarr", "suite2p", etc.
        args : dict
            arguments to pass to the worker.
        description : str
            human-readable description for display.
        output_path : str, optional
            output path for display purposes.

        Returns
        -------
        int or None
            pid of spawned process, or None if spawn failed.
        """
        try:
            # Use pythonw.exe on Windows to prevent console window from appearing
            # Output still goes to log file via stdout/stderr redirection
            python_exe = sys.executable
            if sys.platform == "win32" and python_exe.endswith("python.exe"):
                pythonw = python_exe[:-10] + "pythonw.exe"
                if Path(pythonw).exists():
                    python_exe = pythonw

            # Generate UUID for the task
            from uuid import uuid4
            from datetime import datetime

            task_uuid = str(uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Central logging directory
            log_dir = Path.home() / ".mbo" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Single log file per task: {timestamp}_{task_type}_{short_uuid}.log
            log_file = log_dir / f"{timestamp}_{task_type}_{task_uuid[:8]}.log"

            # Pass UUID and log file to arguments
            args["_uuid"] = task_uuid
            args["_log_file"] = str(log_file) # worker will pick this up

            # serialize args to json string
            args_json = json.dumps(args)

            # construct command AFTER args_json is defined
            cmd = [
                python_exe,
                "-m", "mbo_utilities.gui._worker",
                task_type,
                args_json,
            ]

            # spawn detached process
            # redirect output to log file
            # open log file for redirection
            f_out = open(log_file, "a")  # Append mode

            if sys.platform == "win32":
                # DETACHED_PROCESS (0x00000008) | CREATE_NEW_PROCESS_GROUP (0x00000200) | CREATE_NO_WINDOW (0x08000000)
                creationflags = 0x00000008 | 0x00000200 | 0x08000000
                proc = subprocess.Popen(
                    cmd,
                    creationflags=creationflags,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                )
            else:
                # on unix, use start_new_session to detach
                proc = subprocess.Popen(
                    cmd,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=f_out,
                    stderr=f_out,
                )

            # track the process
            info = ProcessInfo(
                pid=proc.pid,
                description=description,
                start_time=time.time(),
                task_type=task_type,
                output_path=str(log_file), # Store log path here
                args=args,
            )
            self._processes[proc.pid] = info
            self._save()

            self._save()

            logger.info(f"Spawned background process {proc.pid}: {description}")
            logger.debug(f"Process log: {log_file}")
            return proc.pid

        except Exception as e:
            logger.exception(f"Failed to spawn process: {e}")
            return None

    def get_all(self) -> list[ProcessInfo]:
        """Get all tracked processes."""
        return list(self._processes.values())

    def get_running(self) -> list[ProcessInfo]:
        """Get active processes (running or failed/pending view)."""
        active = []
        for p in self._processes.values():
            # Always update from sidecar first to get latest status
            p.update_from_sidecar()

            # if alive, it's running.
            if p.is_alive() or p.status == "error":
                active.append(p)
        return active

    def cleanup_finished(self) -> int:
        """
        Remove entries for processes that have finished successfully.
        failed processes are kept until dismissed (killed) or cleared.

        returns number of entries removed.
        """
        # check all processes
        to_remove = []
        for pid, p in list(self._processes.items()):
            # check sidecar one last time to capture final state
            p.update_from_sidecar()

            if not p.is_alive():
                # verify status from sidecar one last time
                p.update_from_sidecar()

                if p.status == "error":
                    # Keep errors visible
                    continue

                if p.status != "completed":
                    # Process died without reporting completion or error -> Crash
                    # But if we just missed the update, give it one more grace check?
                    # No, update_from_sidecar() above should have caught it.
                    p.status = "error"
                    p.status_message = "Process crashed unexpectedly"
                    p.error_details = {"traceback": "Process exited without reporting results. Check worker logs."}
                    continue

                to_remove.append(pid)

        for pid in to_remove:
            del self._processes[pid]

        if to_remove:
            self._save()
            logger.debug(f"Cleaned up {len(to_remove)} finished processes")

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

    def kill_all(self) -> int:
        """Kill all tracked processes. returns count killed."""
        killed = 0
        for pid in list(self._processes.keys()):
            if self.kill(pid):
                killed += 1
        return killed

    def has_running(self) -> bool:
        """Check if any tracked processes are still running."""
        return len(self.get_running()) > 0


# global process manager instance
_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get the global process manager instance."""
    global _manager
    if _manager is None:
        _manager = ProcessManager()
    return _manager
