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
        except Exception as e:
            # print to stderr so it's captured in the redirect if setup fails
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

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

    except Exception as e:
        print(f"Failed to update status sidecar: {e}", file=sys.stderr)


def main():
    """Main entry point for worker subprocess."""
    # disable tqdm dynamic display for file output (no terminal = no \r updates)
    os.environ["TQDM_DISABLE"] = "1"

    # early print so we can see the process started even if logging fails
    print(f"Worker starting (pid={os.getpid()})", file=sys.stderr, flush=True)

    if len(sys.argv) < 3:
        print("Usage: python -m mbo_utilities.gui._worker <task_type> <args_json>", file=sys.stderr)
        sys.exit(1)

    task_type = sys.argv[1]
    args_json = sys.argv[2]

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON args: {e}", file=sys.stderr)
        sys.exit(1)

    # setup logging
    log_file = args.get("_log_file")
    uuid = args.get("_uuid")
    logger = setup_logging(log_file)

    logger.info(f"Worker started: task={task_type}, pid={os.getpid()}")

    # get task function
    if task_type not in TASKS:
        logger.error(f"Unknown task type: {task_type}")
        print(f"Unknown task type: {task_type}", file=sys.stderr)
        _update_status(os.getpid(), "error", f"Unknown task type: {task_type}", uuid=uuid)
        sys.exit(1)

    task_func = TASKS[task_type]

    # run the task
    try:
        task_func(args, logger)
        logger.info("Task completed successfully")
        _update_status(os.getpid(), "completed", uuid=uuid)
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Task failed: {e}")
        _update_status(os.getpid(), "error", message=str(e), details=traceback.format_exc(), uuid=uuid)
        sys.exit(1)


if __name__ == "__main__":
    main()
