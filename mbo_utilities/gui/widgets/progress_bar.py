import re
import sys
import time
from collections import defaultdict, deque


class OutputCapture:
    """Capture stdout/stderr to a buffer while still printing to console."""

    _instance = None
    _max_lines = 200

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._lines = deque(maxlen=cls._max_lines)
            cls._instance._original_stdout = None
            cls._instance._original_stderr = None
            cls._instance._capturing = False
        return cls._instance

    def start(self):
        """Start capturing stdout/stderr."""
        if self._capturing:
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._original_stdout, self._lines, "stdout")
        sys.stderr = _TeeWriter(self._original_stderr, self._lines, "stderr")
        self._capturing = True


class _TeeWriter:
    """Write to both the original stream and a capture buffer."""

    _ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")
    _BOX_REPLACEMENTS = {
        "█": "#", "▏": "|", "▎": "|", "▍": "|", "▌": "|",
        "▋": "|", "▊": "|", "▉": "|", "░": "-", "▒": "=",
        "▓": "#", "━": "-", "┃": "|", "╸": ">", "╺": "<",
    }
    _TQDM_PATTERNS = [
        re.compile(r"\d+%\|"),
        re.compile(r"it/s"),
        re.compile(r"[0-9]+/[0-9]+\s*\["),
    ]

    def __init__(self, original, buffer: deque, stream_name: str):
        self._original = original
        self._buffer = buffer
        self._stream_name = stream_name

    def _clean(self, text: str) -> str | None:
        text = self._ANSI_ESCAPE.sub("", text).replace("\r", "")
        if not text.strip():
            return None
        for old, new in self._BOX_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text.strip()

    def _tqdm_key(self, text: str) -> str | None:
        for pattern in self._TQDM_PATTERNS:
            if pattern.search(text):
                match = re.match(r"^([^:]+):", text)
                return f"tqdm_{match.group(1).strip() if match else 'tqdm'}"
        return None

    def write(self, text):
        if self._original:
            self._original.write(text)

        if not text:
            return

        cleaned = self._clean(text)
        if cleaned is None:
            return

        timestamp = time.strftime("%H:%M:%S")
        full_key = self._tqdm_key(cleaned)

        if full_key:
            # update last entry with same key in place to avoid log spam
            for i in range(len(self._buffer) - 1, -1, -1):
                entry = self._buffer[i]
                if len(entry) >= 4 and entry[3] == full_key:
                    self._buffer[i] = (timestamp, self._stream_name, cleaned, full_key)
                    return
            self._buffer.append((timestamp, self._stream_name, cleaned, full_key))
        else:
            self._buffer.append((timestamp, self._stream_name, cleaned, ""))

    def flush(self):
        if self._original:
            self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


_output_capture = OutputCapture()

_progress_state = defaultdict(
    lambda: {
        "hide_time": None,
        "is_showing_done": False,
        "done_shown_once": False,
        "done_cleared": False,
    }
)


def reset_progress_state(key: str):
    """Reset progress state for a given key to allow re-display."""
    if key in _progress_state:
        _progress_state[key] = {
            "hide_time": None,
            "is_showing_done": False,
            "done_shown_once": False,
            "done_cleared": False,
        }


def _get_active_progress_items(self) -> list[dict]:
    """Collect active progress operations from the widget state.

    Returns list of dicts with: key, text, progress, done.
    """
    items = []

    saveas_running = getattr(self, "_saveas_running", False)
    saveas_progress = getattr(self, "_saveas_progress", 0.0)
    saveas_current = getattr(self, "_saveas_current_index", 0)
    saveas_done = getattr(self, "_saveas_done", False)

    if saveas_running or (0.0 < saveas_progress < 1.0):
        text = "Starting save..." if saveas_progress == 0.0 else f"Saving z-plane {saveas_current}"
        items.append({
            "key": "saveas",
            "text": text,
            "progress": max(0.01, saveas_progress),
            "done": False,
        })
    elif saveas_done:
        items.append({
            "key": "saveas",
            "text": "Save complete",
            "progress": 1.0,
            "done": True,
        })

    num_graphics = getattr(self, "num_graphics", 1)
    zstats_running = getattr(self, "_zstats_running", [])
    zstats_progress = getattr(self, "_zstats_progress", [])
    zstats_current_z = getattr(self, "_zstats_current_z", [])
    nz = getattr(self, "nz", 1)

    for i in range(num_graphics):
        running = zstats_running[i] if isinstance(zstats_running, list) and i < len(zstats_running) else False
        progress = zstats_progress[i] if isinstance(zstats_progress, list) and i < len(zstats_progress) else 0.0
        current_z = zstats_current_z[i] if isinstance(zstats_current_z, list) and i < len(zstats_current_z) else 0

        if running or (0.0 < progress < 1.0):
            text = f"Z-stats {i+1}: starting..." if progress == 0.0 else f"Z-stats: plane {current_z + 1}/{nz}"
            items.append({
                "key": f"zstats_{i}",
                "text": text,
                "progress": max(0.01, progress),
                "done": False,
            })

    register_running = getattr(self, "_register_z_running", False)
    register_progress = getattr(self, "_register_z_progress", 0.0)
    register_msg = getattr(self, "_register_z_current_msg", None)
    register_done = getattr(self, "_register_z_done", False)

    if register_running or (0.0 < register_progress < 1.0):
        msg = register_msg if register_msg else "Starting..."
        items.append({
            "key": "register_z",
            "text": f"Z-Reg: {msg}",
            "progress": max(0.01, register_progress),
            "done": False,
        })
    elif register_done and register_msg:
        items.append({
            "key": "register_z",
            "text": f"Z-Reg: {register_msg}",
            "progress": 1.0,
            "done": True,
        })

    return items


def start_output_capture():
    """Start capturing stdout/stderr."""
    _output_capture.start()
