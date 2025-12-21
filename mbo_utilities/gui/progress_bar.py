import sys
import time
from collections import defaultdict, deque

from imgui_bundle import (
    imgui,
    hello_imgui,
)


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

    def stop(self):
        """Stop capturing and restore original streams."""
        if not self._capturing:
            return
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._capturing = False

    @property
    def lines(self) -> list[tuple[str, str, str]]:
        """Return captured lines as list of (timestamp, stream, text)."""
        # Convert 4-tuples to 3-tuples (drop the tqdm key)
        result = []
        for entry in self._lines:
            if len(entry) >= 3:
                result.append((entry[0], entry[1], entry[2]))
        return result

    def clear(self):
        """Clear captured lines."""
        self._lines.clear()


class _TeeWriter:
    """Write to both the original stream and a capture buffer."""

    def __init__(self, original, buffer: deque, stream_name: str):
        self._original = original
        self._buffer = buffer
        self._stream_name = stream_name
        self._last_tqdm_key = None  # Track last tqdm line to update in-place

    def _clean_tqdm_output(self, text: str) -> str | None:
        """
        Clean tqdm progress bar output for display in imgui.

        Returns cleaned text, or None if the line should be skipped.
        """
        import re

        # Remove ANSI escape codes (colors, cursor movement)
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07')
        text = ansi_escape.sub('', text)

        # Remove carriage return (tqdm uses \r to update in place)
        text = text.replace('\r', '')

        # Skip empty lines after cleaning
        if not text.strip():
            return None

        # Clean up tqdm progress bar characters
        # Replace box drawing characters with simpler alternatives
        replacements = {
            '█': '#',
            '▏': '|',
            '▎': '|',
            '▍': '|',
            '▌': '|',
            '▋': '|',
            '▊': '|',
            '▉': '|',
            '░': '-',
            '▒': '=',
            '▓': '#',
            '━': '-',
            '┃': '|',
            '╸': '>',
            '╺': '<',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def _is_tqdm_line(self, text: str) -> tuple[bool, str | None]:
        """
        Check if this is a tqdm progress line and extract a key for deduplication.

        Returns (is_tqdm, key) where key can be used to identify the same progress bar.
        """
        import re

        # tqdm lines typically have patterns like "  0%|" or "100%|" or contain "it/s"
        # They also often have the format: "description:  XX%|###..."
        tqdm_patterns = [
            r'\d+%\|',  # "50%|"
            r'it/s',     # iterations per second
            r'[0-9]+/[0-9]+\s*\[',  # "10/100 ["
        ]

        for pattern in tqdm_patterns:
            if re.search(pattern, text):
                # Extract a key from the description (text before the percentage)
                match = re.match(r'^([^:]+):', text)
                key = match.group(1).strip() if match else "tqdm"
                return True, key

        return False, None

    def write(self, text):
        # Always write to original stream
        if self._original:
            self._original.write(text)

        if not text:
            return

        # Clean and filter the text for our buffer
        cleaned = self._clean_tqdm_output(text)
        if cleaned is None:
            return

        timestamp = time.strftime("%H:%M:%S")

        # Check if this is a tqdm progress update
        is_tqdm, tqdm_key = self._is_tqdm_line(cleaned)

        if is_tqdm and tqdm_key:
            # For tqdm lines, update the last entry with the same key instead of appending
            # This prevents the log from filling up with hundreds of progress updates
            full_key = f"tqdm_{tqdm_key}"

            # Look for existing entry with this key and update it
            for i in range(len(self._buffer) - 1, -1, -1):
                entry = self._buffer[i]
                if len(entry) >= 4 and entry[3] == full_key:
                    # Replace this entry with updated progress
                    self._buffer[i] = (timestamp, self._stream_name, cleaned, full_key)
                    return

            # No existing entry, append new one with key
            self._buffer.append((timestamp, self._stream_name, cleaned, full_key))
        else:
            # Regular line, just append (with empty key)
            self._buffer.append((timestamp, self._stream_name, cleaned, ""))

    def flush(self):
        if self._original:
            self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


# Global output capture instance
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


def draw_progress(
    key: str,
    current_index: int,
    total_count: int,
    percent_complete: float,
    running_text: str = "Processing",
    done_text: str = "Completed",
    done: bool = False,
    custom_text: str | None = None,
):
    now = time.time()
    state = _progress_state[key]

    # if already cleared, never draw again
    if state["done_cleared"]:
        return

    if done and not state["done_shown_once"]:
        state["hide_time"] = now + 3
        state["is_showing_done"] = True
        state["done_shown_once"] = True
        state["done_cleared"] = False

    if not done and not state["is_showing_done"]:
        state["hide_time"] = None
        state["done_shown_once"] = False
    # elif not done:
    #     state["hide_time"] = None
    #     state["is_showing_done"] = False
    #     state["done_shown_once"] = False
    #     state["done_cleared"] = False

    if state["is_showing_done"] and state["hide_time"] and now >= state["hide_time"]:
        state["hide_time"] = None
        state["is_showing_done"] = False
        state["done_cleared"] = True
        return

    if not done and state["done_cleared"]:
        return  # prevent flashing previous bar

    bar_height = hello_imgui.em_size(1.4)
    imgui.spacing()

    p = min(max(percent_complete, 0.0), 1.0)
    w = imgui.get_content_region_avail().x

    bar_color = (
        imgui.ImVec4(0.0, 0.8, 0.0, 1.0)
        if state["is_showing_done"]
        else imgui.ImVec4(0.2, 0.5, 0.9, 1.0)
    )
    if state["is_showing_done"]:
        text = done_text
    elif custom_text:
        text = custom_text
    elif current_index is not None and total_count is not None:
        text = f"{running_text} {current_index + 1} of {total_count} [{int(p * 100)}%]"
    else:
        text = f"{running_text} [{int(p * 100)}%]"

    imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))
    imgui.progress_bar(p, imgui.ImVec2(w, bar_height), "")
    imgui.begin_group()

    if text:
        ts = imgui.calc_text_size(text)
        x = (w - ts.x) / 2
        imgui.set_cursor_pos_x(x)
        imgui.text_colored(imgui.ImVec4(1, 1, 1, 1), text)

    imgui.pop_style_var()
    imgui.pop_style_color()
    imgui.end_group()


def draw_saveas_progress(self):
    key = "saveas"
    state = _progress_state[key]
    if state["is_showing_done"]:
        draw_progress(
            key=key,
            current_index=self._saveas_current_index,
            total_count=self._saveas_total,
            percent_complete=self._saveas_progress,
            running_text="Saving",
            done_text="Completed",
            done=True,
        )
    elif 0.0 < self._saveas_progress < 1.0:
        draw_progress(
            key=key,
            current_index=self._saveas_current_index,
            total_count=self._saveas_total,
            percent_complete=self._saveas_progress,
            running_text="Saving",
            custom_text=f"Saving z-plane {self._saveas_current_index} [{int(self._saveas_progress * 100)}%]",
        )


def draw_zstats_progress(self):
    for i in range(self.num_graphics):
        key = f"zstats_{i}"
        state = _progress_state[key]

        if state["done_cleared"]:
            continue

        # Make sure these are valid per-graphic lists
        current_z = (
            self._zstats_current_z[i] if isinstance(self._zstats_current_z, list) else 0
        )
        progress = (
            self._zstats_progress[i] if isinstance(self._zstats_progress, list) else 0.0
        )
        done = self._zstats_done[i] if isinstance(self._zstats_done, list) else False

        draw_progress(
            key=key,
            current_index=current_z,
            total_count=self.nz,
            percent_complete=progress,
            running_text=f"Computing stats: graphic {i + 1}, plane(s)",
            done_text=f"Z-stats complete (graphic {i + 1})",
            done=done,
        )


def draw_register_z_progress(self):
    key = "register_z"
    state = _progress_state[key]

    # fully skip if cleared
    if state["done_cleared"]:
        return

    done = self._register_z_done
    progress = self._register_z_progress
    msg = self._register_z_current_msg

    if done:
        draw_progress(
            key=key,
            current_index=int(progress * 100),
            total_count=100,
            percent_complete=progress,
            running_text="Z-Registration",
            done_text="Z-Registration Complete!",
            done=True,
        )
    elif 0.0 < progress < 1.0:
        draw_progress(
            key=key,
            current_index=int(progress * 100),
            total_count=100,
            percent_complete=progress,
            running_text="Z-Registration",
            custom_text=f"Z-Registration: {msg} [{int(progress * 100)}%]",
        )


def _get_active_progress_items(self) -> list[dict]:
    """
    Collect all active progress operations from the widget state.

    Checks widget attributes directly for active operations, using _running flags
    for immediate feedback when tasks start.

    Returns list of dicts with: key, text, progress, done
    """
    items = []

    # Check saveas progress - use _running flag for immediate feedback
    saveas_running = getattr(self, '_saveas_running', False)
    saveas_progress = getattr(self, '_saveas_progress', 0.0)
    saveas_current = getattr(self, '_saveas_current_index', 0)

    if saveas_running or (0.0 < saveas_progress < 1.0):
        # show "Starting..." if progress is still 0
        if saveas_progress == 0.0:
            text = "Starting save..."
        else:
            text = f"Saving z-plane {saveas_current}"
        items.append({
            "key": "saveas",
            "text": text,
            "progress": max(0.01, saveas_progress),  # show small progress when starting
            "done": False,
        })

    # Check zstats progress for each graphic - use _running flags
    num_graphics = getattr(self, 'num_graphics', 1)
    zstats_running = getattr(self, '_zstats_running', [])
    zstats_progress = getattr(self, '_zstats_progress', [])
    zstats_current_z = getattr(self, '_zstats_current_z', [])
    nz = getattr(self, 'nz', 1)

    for i in range(num_graphics):
        running = zstats_running[i] if isinstance(zstats_running, list) and i < len(zstats_running) else False
        progress = zstats_progress[i] if isinstance(zstats_progress, list) and i < len(zstats_progress) else 0.0
        current_z = zstats_current_z[i] if isinstance(zstats_current_z, list) and i < len(zstats_current_z) else 0

        if running or (0.0 < progress < 1.0):
            # show "Starting..." if progress is still 0
            if progress == 0.0:
                text = f"Z-stats {i+1}: starting..."
            else:
                text = f"Z-stats: plane {current_z + 1}/{nz}"
            items.append({
                "key": f"zstats_{i}",
                "text": text,
                "progress": max(0.01, progress),
                "done": False,
            })

    # Check register_z progress - use _running flag
    register_running = getattr(self, '_register_z_running', False)
    register_progress = getattr(self, '_register_z_progress', 0.0)
    register_msg = getattr(self, '_register_z_current_msg', None)

    if register_running or (0.0 < register_progress < 1.0):
        msg = register_msg if register_msg else "Starting..."
        items.append({
            "key": "register_z",
            "text": f"Z-Reg: {msg}",
            "progress": max(0.01, register_progress),
            "done": False,
        })

    return items


def draw_status_bar(self):
    """
    Draw a compact status bar showing active operations.

    Displays a thin colored bar that turns green when operations are running.
    Hover over it to see detailed progress information in a tooltip.

    DEPRECATED: Use draw_status_indicator() instead for a compact text-based indicator.
    """
    items = _get_active_progress_items(self)

    if not items:
        return False  # Nothing active

    # Calculate overall progress (average of all items)
    total_progress = sum(item["progress"] for item in items) / len(items)
    all_done = all(item["done"] for item in items)

    # Bar color: green if all done, blue if in progress
    if all_done:
        bar_color = imgui.ImVec4(0.2, 0.8, 0.2, 1.0)  # Green
    else:
        bar_color = imgui.ImVec4(0.2, 0.6, 0.9, 1.0)  # Blue

    # Draw a thin progress bar spanning available width
    avail_width = imgui.get_content_region_avail().x
    bar_height = 4

    imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)
    imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.15, 0.15, 0.15, 1.0))
    imgui.progress_bar(total_progress, imgui.ImVec2(avail_width, bar_height), "")
    imgui.pop_style_color(2)

    # Tooltip on hover with detailed progress info
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text_colored(imgui.ImVec4(0.7, 0.9, 1.0, 1.0), f"Active Operations ({len(items)})")
        imgui.separator()

        for item in items:
            pct = int(item["progress"] * 100)
            if item["done"]:
                color = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)  # Green
                status = "Done"
            else:
                color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)  # White
                status = f"{pct}%"

            # Show progress bar for each item
            imgui.push_style_color(imgui.Col_.plot_histogram,
                imgui.ImVec4(0.2, 0.8, 0.2, 1.0) if item["done"] else imgui.ImVec4(0.2, 0.5, 0.9, 1.0))
            imgui.progress_bar(item["progress"], imgui.ImVec2(200, 14), "")
            imgui.pop_style_color()

            imgui.same_line()
            imgui.text_colored(color, f"{item['text']} [{status}]")

        imgui.end_tooltip()

    return True  # Something is active


def start_output_capture():
    """Start capturing stdout/stderr for display in the status indicator."""
    _output_capture.start()


def stop_output_capture():
    """Stop capturing stdout/stderr."""
    _output_capture.stop()


def draw_status_indicator(self):
    """
    Draw a compact colored status indicator with a button to show logs.

    Shows colored text:
    - Green "Background tasks" when idle (no active operations)
    - Orange "Background tasks (N running)" when operations are running

    Click the (?) button to open a popup with stdout/stderr log output.
    """
    items = _get_active_progress_items(self)

    # Determine status color and text
    if items:
        # In progress - orange with count
        text_color = imgui.ImVec4(1.0, 0.6, 0.2, 1.0)  # Orange
        # Calculate average progress
        avg_progress = sum(item["progress"] for item in items) / len(items)
        status_text = f"Background tasks ({len(items)} running, {int(avg_progress * 100)}%)"
    else:
        # Idle - green
        text_color = imgui.ImVec4(0.4, 0.8, 0.4, 1.0)  # Green
        status_text = "Background tasks"

    # Draw the status text
    imgui.text_colored(text_color, status_text)

    # Draw (?) button on same line with transparent style
    imgui.same_line()
    # Make button background transparent
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.3, 0.3, 0.5))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.2, 0.2, 0.2, 0.5))
    if imgui.small_button("(?)"):
        imgui.open_popup("Console Output")
        if not hasattr(self, "_log_popup_open"):
            self._log_popup_open = True
        self._log_popup_open = True
    imgui.pop_style_color(3)

    # Tooltip for the button
    if imgui.is_item_hovered():
        imgui.set_tooltip("Click to view console output")

    # Show Metadata button on the right side of the same line
    avail_width = imgui.get_content_region_avail().x
    button_width = imgui.calc_text_size("Show Metadata").x + imgui.get_style().frame_padding.x * 2
    imgui.same_line(imgui.get_window_width() - button_width - imgui.get_style().window_padding.x)
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.0, 0.0, 0.0, 1.0))
    imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
    if imgui.button("Show Metadata"):
        self.show_metadata_viewer = not self.show_metadata_viewer
    imgui.pop_style_var()
    imgui.pop_style_color(2)

    # Log output popup (resizable with close button)
    imgui.set_next_window_size(imgui.ImVec2(600, 500), imgui.Cond_.first_use_ever)

    opened, visible = imgui.begin_popup_modal(
        "Console Output",
        p_open=True if getattr(self, "_log_popup_open", False) else None,
        flags=imgui.WindowFlags_.no_saved_settings
    )

    if opened:
        if not visible:
            # user closed via X button
            self._log_popup_open = False
            imgui.close_current_popup()
            imgui.end_popup()
        else:
            imgui.text_colored(imgui.ImVec4(0.7, 0.9, 1.0, 1.0), "Console Output")
            imgui.same_line()
            imgui.text_disabled(f"({len(_output_capture.lines)} lines)")
            imgui.separator()

            # Show active operations if any
            if items:
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), f"Active Operations ({len(items)})")
                for item in items:
                    pct = int(item["progress"] * 100)
                    color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
                    imgui.text_colored(color, f"  {item['text']} [{pct}%]")
                imgui.separator()

            # Scrollable output area - show ALL lines, not just last 100
            # use -35 for button row height
            if imgui.begin_child("##LogScroll", imgui.ImVec2(-1, -35), imgui.ChildFlags_.borders):
                # Get captured stdout/stderr - show all lines (up to max_lines in deque)
                captured_lines = _output_capture.lines
                if captured_lines:
                    # Show in chronological order (oldest first, scroll to see newest)
                    for timestamp, stream, text in captured_lines:
                        if stream == "stderr":
                            col = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)  # Red for stderr
                        else:
                            col = imgui.ImVec4(0.4, 0.9, 0.4, 1.0)  # Green for stdout
                        # Truncate very long lines
                        display_text = text[:150] + "..." if len(text) > 150 else text
                        imgui.text_colored(col, f"[{timestamp}] {display_text}")
                    # Auto-scroll to bottom on first open
                    if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 20:
                        imgui.set_scroll_here_y(1.0)
                else:
                    imgui.text_disabled("No console output captured yet.")
                    imgui.text_disabled("Output will appear here when operations run.")
            imgui.end_child()

            # Buttons row
            if imgui.button("Save to File"):
                _save_log_to_file()
            imgui.same_line()
            if imgui.button("Clear"):
                _output_capture.clear()
            imgui.same_line()
            if imgui.button("Close"):
                self._log_popup_open = False
                imgui.close_current_popup()

            imgui.end_popup()

    return len(items) > 0


def _save_log_to_file():
    """Save captured log output to a file."""
    from pathlib import Path
    import datetime

    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / ".mbo" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"console_output_{timestamp}.log"

    # Write all captured lines
    lines = _output_capture.lines
    if not lines:
        return

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"MBO Console Output Log\n")
        f.write(f"Saved: {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'=' * 60}\n\n")
        for timestamp, stream, text in lines:
            stream_label = "ERR" if stream == "stderr" else "OUT"
            f.write(f"[{timestamp}] [{stream_label}] {text}\n")

    # silently save without console output


def draw_progress_overlay(self):
    """
    Draw a floating progress overlay in the bottom-right corner.

    This overlay is always visible regardless of which tab is active,
    showing all active progress operations in a compact format.

    DEPRECATED: Use draw_status_bar() instead for a less intrusive UI.
    """
    items = _get_active_progress_items(self)

    if not items:
        return

    # Calculate overlay position (bottom-right corner with padding)
    viewport = imgui.get_main_viewport()
    padding = 16
    overlay_width = 280
    overlay_height = len(items) * 38 + 32  # 38px per item + header

    pos_x = viewport.work_pos.x + viewport.work_size.x - overlay_width - padding
    pos_y = viewport.work_pos.y + viewport.work_size.y - overlay_height - padding

    imgui.set_next_window_pos(imgui.ImVec2(pos_x, pos_y), imgui.Cond_.always)
    imgui.set_next_window_size(imgui.ImVec2(overlay_width, 0))
    imgui.set_next_window_bg_alpha(0.85)

    window_flags = (
        imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_nav
        | imgui.WindowFlags_.always_auto_resize
    )

    if imgui.begin("##progress_overlay", flags=window_flags):
        # Header
        imgui.text_colored(imgui.ImVec4(0.7, 0.9, 1.0, 1.0), "Progress")
        imgui.same_line()
        imgui.text_disabled(f"({len(items)} active)")
        imgui.separator()

        for item in items:
            # Color based on done status
            if item["done"]:
                bar_color = imgui.ImVec4(0.2, 0.8, 0.2, 1.0)  # Green
                text_color = imgui.ImVec4(0.6, 1.0, 0.6, 1.0)
            else:
                bar_color = imgui.ImVec4(0.2, 0.5, 0.9, 1.0)  # Blue
                text_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

            # Progress bar
            imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)
            imgui.progress_bar(item["progress"], imgui.ImVec2(-1, 16), "")
            imgui.pop_style_color()

            # Text label
            pct = int(item["progress"] * 100)
            imgui.text_colored(text_color, f"{item['text']} [{pct}%]")

            imgui.spacing()

    imgui.end()
