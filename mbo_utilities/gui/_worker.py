"""
worker entry point for background subprocess tasks.

this module is invoked via:
    python -m mbo_utilities.gui._worker <task_type> <args_json>

it runs independently of the gui and can survive gui closure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from mbo_utilities.gui.tasks import TASKS


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Setup logging for worker process."""
    logger = logging.getLogger("mbo.worker")
    logger.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler if explicit log file provided
    if log_file:
        try:
            # log file path is providing directly
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a") # Append to existing log
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            # print to stderr so it's captured in the redirect if setup fails
            pass

    return logger


def _update_status(pid: int, status: str, message: str | None = None, details: str | dict | None = None, uuid: str | None = None):
    """Ensure process status is reported to sidecar file."""
    try:
        # Match location used by TaskMonitor and ProcessManager
        log_dir = Path.home() / "mbo" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if uuid:
            sidecar = log_dir / f"progress_{uuid}.json"
        else:
            sidecar = log_dir / f"progress_{pid}.json"

        current_data = {}
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    current_data = json.load(f)
            except Exception:
                pass

        # If already set to final state by task, respect it
        # (unless we are reporting error which overrides success-in-progress)
        if status == "completed" and current_data.get("status") == "completed":
            return

        data = current_data.copy()
        data["pid"] = pid
        data["uuid"] = uuid
        data["timestamp"] = time.time()
        data["status"] = status

        if message:
            data["message"] = message
        if details:
            data["details"] = details

        if status == "completed":
            data["progress"] = 1.0

        # Write atomically to avoid race conditions with readers
        tmp_file = sidecar.with_suffix(".tmp")
        with open(tmp_file, "w") as f:
            json.dump(data, f)
        tmp_file.replace(sidecar)

    except Exception:
        pass


def main():
    """Main entry point for worker subprocess."""
    # Disable tqdm's dynamic display for file output (no terminal = no \r updates)
    # This prevents the flood of progress bar lines in log files
    os.environ["TQDM_DISABLE"] = "1"

    # early print so we can see the process started even if logging fails

    if len(sys.argv) < 3:
        sys.exit(1)

    task_type = sys.argv[1]
    args_json = sys.argv[2]


    try:
        args = json.loads(args_json)
    except json.JSONDecodeError:
        sys.exit(1)

    # setup logging
    log_file = args.get("_log_file")
    uuid = args.get("_uuid")
    logger = setup_logging(log_file)

    logger.info(f"Worker started: task={task_type}, pid={os.getpid()}")

    # get task function
    if task_type not in TASKS:
        logger.error(f"Unknown task type: {task_type}")
        _update_status(os.getpid(), "error", f"Unknown task type: {task_type}", uuid=uuid)
        sys.exit(1)

    task_func = TASKS[task_type]

    # run the task
    try:
        # Task functions are expected to accept (args, logger)
        task_func(args, logger)

        logger.info("Task completed successfully")

        # Explicitly mark as completed in sidecar so ProcessManager knows
        _update_status(os.getpid(), "completed", uuid=uuid)

        sys.exit(0)
    except Exception as e:
        logger.exception(f"Task failed: {e}")
        logger.exception(traceback.format_exc())

        # Explicitly mark as error
        _update_status(os.getpid(), "error", message=str(e), details=traceback.format_exc(), uuid=uuid)

        sys.exit(1)


if __name__ == "__main__":
    main()
