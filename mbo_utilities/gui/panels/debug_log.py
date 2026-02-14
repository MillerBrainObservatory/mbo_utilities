"""
Debug log panel.

Displays log messages from all mbo.* loggers with filtering,
level control, and message management.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from . import BasePanel
from mbo_utilities.gui._imgui_helpers import fmt_value

if TYPE_CHECKING:
    from mbo_utilities.gui.viewers import BaseViewer

__all__ = ["DebugPanel", "GuiLogHandler", "draw_scope"]

LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL_VAL = {n: getattr(logging, n) for n in LEVELS}
LEVEL_COLORS = {
    "debug": imgui.ImVec4(0.5, 0.5, 0.5, 1.0),      # gray
    "info": imgui.ImVec4(0.8, 0.8, 0.8, 1.0),       # light gray
    "warning": imgui.ImVec4(1.0, 0.7, 0.2, 1.0),    # orange
    "error": imgui.ImVec4(1.0, 0.3, 0.3, 1.0),      # red
    "critical": imgui.ImVec4(1.0, 0.1, 0.5, 1.0),   # magenta
}


class _GuiNameFilter(logging.Filter):
    """Filter log records by logger name based on panel toggle state."""

    def __init__(self, panel: DebugPanel):
        super().__init__()
        self.panel = panel

    def filter(self, record: logging.LogRecord) -> bool:
        return self.panel.active_loggers.get(record.name, True)


class GuiLogHandler(logging.Handler):
    """Handler that routes log records to the debug panel."""

    def __init__(self, panel: DebugPanel):
        super().__init__()
        self.panel = panel
        self.addFilter(_GuiNameFilter(panel))

    def emit(self, record: logging.LogRecord) -> None:
        t = time.strftime("%H:%M:%S")
        lvl_map = {10: "debug", 20: "info", 30: "warning", 40: "error", 50: "critical"}
        lvl = lvl_map.get(record.levelno, "info")
        msg = self.format(record)
        self.panel.add_message(t, lvl, record.name, msg)


class DebugPanel(BasePanel):
    """
    Debug panel with per-level filtering and search.

    Features:
    - Filter messages by level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    - Toggle individual loggers on/off
    - Set log level globally or per-logger
    - Auto-scroll to newest messages
    - Search/filter messages
    - Clear messages
    - Message limit to prevent memory issues
    """

    name = "Debug Log"
    MAX_MESSAGES = 1000

    def __init__(self, viewer: BaseViewer):
        super().__init__(viewer)

        # Import log utilities
        try:
            from mbo_utilities import log
            self._log = log
            initial_loggers = log.get_package_loggers()
        except ImportError:
            self._log = None
            initial_loggers = []

        # Message storage
        self.messages: list[tuple[str, str, str, str]] = []  # (timestamp, level, name, msg)

        # Filter state
        self.filters = {
            "debug": False,  # off by default - too noisy
            "info": True,
            "warning": True,
            "error": True,
            "critical": True,
        }

        # Logger toggles
        self.active_loggers = dict.fromkeys(initial_loggers, True)
        self.levels = dict.fromkeys(self.active_loggers, "INFO")
        self.master_level = "INFO"

        # UI state
        self.auto_scroll = True
        self._last_msg_count = 0  # tracks message count for scroll-on-new-message
        self.search_text = ""
        self.show_loggers = False  # collapsed by default
        self.show_controls = True

    def add_message(self, timestamp: str, level: str, logger_name: str, message: str) -> None:
        """Add a message, enforcing max limit."""
        self.messages.append((timestamp, level, logger_name, message))
        if len(self.messages) > self.MAX_MESSAGES:
            # Remove oldest 10%
            trim = self.MAX_MESSAGES // 10
            self.messages = self.messages[trim:]

    def _apply_level(self, name: str, lvl_name: str) -> None:
        """Apply log level to a logger."""
        logging.getLogger(name).setLevel(LEVEL_VAL[lvl_name])

    def _refresh_loggers(self) -> None:
        """Refresh active_loggers list to include newly created loggers."""
        if self._log is None:
            return
        current = self._log.get_package_loggers()
        for name in current:
            if name not in self.active_loggers:
                self.active_loggers[name] = True
                self.levels[name] = self.master_level

    def _draw_controls(self) -> None:
        """Draw level filter checkboxes and global controls."""
        # Level filters
        imgui.text("Levels:")
        imgui.same_line()
        for lvl in ["debug", "info", "warning", "error", "critical"]:
            col = LEVEL_COLORS[lvl]
            imgui.push_style_color(imgui.Col_.text, col)
            changed, val = imgui.checkbox(lvl.capitalize(), self.filters[lvl])
            imgui.pop_style_color()
            if changed:
                self.filters[lvl] = val
            imgui.same_line()

        # Global level dropdown
        imgui.same_line(imgui.get_window_width() - 180)
        imgui.set_next_item_width(80)
        combo_open = imgui.begin_combo("##global_level", self.master_level)
        if isinstance(combo_open, tuple):
            combo_open = combo_open[0]
        if combo_open:
            for lvl in LEVELS:
                sel = imgui.selectable(lvl, self.master_level == lvl)
                clicked = sel[0] if isinstance(sel, tuple) else sel
                if clicked:
                    self.master_level = lvl
                    if self._log:
                        self._log.set_global_level(LEVEL_VAL[lvl])
                    for name in self.active_loggers:
                        self.levels[name] = lvl
            imgui.end_combo()
        imgui.same_line()

        # Clear button
        if imgui.button("Clear"):
            self.messages.clear()

    def _draw_search(self) -> None:
        """Draw search bar and auto-scroll toggle."""
        imgui.set_next_item_width(200)
        changed, val = imgui.input_text_with_hint(
            "##search", "Search messages...", self.search_text
        )
        if changed:
            self.search_text = val

        imgui.same_line()
        _, self.auto_scroll = imgui.checkbox("Auto-scroll", self.auto_scroll)

        imgui.same_line()
        imgui.text(f"({len(self.messages)} msgs)")

    def _draw_logger_toggles(self) -> None:
        """Draw collapsible section with per-logger controls."""
        self._refresh_loggers()

        flags = imgui.TreeNodeFlags_.default_open if self.show_loggers else 0
        expanded = imgui.collapsing_header("Loggers", flags)
        if isinstance(expanded, tuple):
            expanded = expanded[0]
        if expanded:
            self.show_loggers = True

            # Group loggers by prefix
            groups: dict[str, list[str]] = {}
            for name in sorted(self.active_loggers.keys()):
                parts = name.split(".")
                if len(parts) > 1:
                    prefix = parts[1]  # e.g., "arrays" from "mbo.arrays.tiff"
                else:
                    prefix = "core"
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(name)

            # Draw grouped loggers in columns
            columns = min(3, len(groups))
            if columns > 0 and imgui.begin_table("logger_table", columns, imgui.TableFlags_.borders_inner_v):
                for group_name in sorted(groups.keys()):
                    imgui.table_next_column()
                    imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), f"[{group_name}]")

                    for name in groups[group_name]:
                        short = name.split(".")[-1]
                        imgui.push_id(f"lg_{name}")

                        changed, state = imgui.checkbox(short, self.active_loggers[name])
                        if changed:
                            self.active_loggers[name] = state

                        imgui.same_line()
                        imgui.set_next_item_width(70)
                        cur = self.levels.get(name, "INFO")
                        combo_open = imgui.begin_combo("##lvl", cur)
                        if isinstance(combo_open, tuple):
                            combo_open = combo_open[0]
                        if combo_open:
                            for lvl in LEVELS:
                                sel = imgui.selectable(lvl, cur == lvl)
                                clicked = sel[0] if isinstance(sel, tuple) else sel
                                if clicked:
                                    self.levels[name] = lvl
                                    self._apply_level(name, lvl)
                            imgui.end_combo()

                        imgui.pop_id()

                imgui.end_table()
        else:
            self.show_loggers = False

    def _draw_messages(self) -> None:
        """Draw scrollable message list."""
        # Calculate available height
        avail = imgui.get_content_region_avail()
        child_height = max(100, avail.y - 4)

        imgui.begin_child(
            "##debug_scroll",
            imgui.ImVec2(0, child_height),
            imgui.ChildFlags_.borders,
            imgui.WindowFlags_.horizontal_scrollbar,
        )

        search_lower = self.search_text.lower()

        for timestamp, lvl, full_name, msg in reversed(self.messages):
            # Filter by level
            if not self.filters.get(lvl, False):
                continue

            # Filter by logger toggle
            if not self.active_loggers.get(full_name, True):
                continue

            # Filter by search
            if search_lower and search_lower not in msg.lower():
                continue

            short = full_name.split(".")[-1]
            col = LEVEL_COLORS.get(lvl, LEVEL_COLORS["info"])

            # Format: [HH:MM:SS] [logger] message
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), f"[{timestamp}]")
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.4, 0.7, 1.0, 1.0), f"[{short}]")
            imgui.same_line()
            imgui.text_colored(col, msg)

        # auto-scroll: only jump to bottom when new messages arrive.
        # if user scrolls away from bottom, disable auto-scroll until
        # they scroll back down or re-enable the checkbox.
        has_new = len(self.messages) != self._last_msg_count
        self._last_msg_count = len(self.messages)

        if self.auto_scroll and has_new:
            imgui.set_scroll_here_y(1.0)

        # disable auto-scroll when user scrolls up, re-enable at bottom
        max_scroll = imgui.get_scroll_max_y()
        if max_scroll > 0:
            at_bottom = imgui.get_scroll_y() >= max_scroll - imgui.get_text_line_height()
            if not at_bottom and not has_new:
                self.auto_scroll = False
            elif at_bottom:
                self.auto_scroll = True

        imgui.end_child()

    def draw(self) -> None:
        """Draw the full debug panel."""
        if not self._visible:
            return

        # Use a separate window
        expanded, opened = imgui.begin("Debug Log", self._visible)
        self._visible = opened

        if expanded:
            self._draw_controls()
            self._draw_search()
            imgui.separator()
            self._draw_logger_toggles()
            imgui.separator()
            self._draw_messages()

        imgui.end()


# Scope colors
_NAME_COLOR = imgui.ImVec4(0.95, 0.80, 0.30, 1.0)
_VALUE_COLOR = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)


def draw_scope():
    """Draw a scope inspector showing local variables from the calling frame."""
    with imgui_ctx.begin_child("Scope Inspector"):
        frame = inspect.currentframe().f_back
        vars_all = {**frame.f_locals}
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4))
        try:
            for name, val in sorted(vars_all.items()):
                if (
                    inspect.ismodule(val)
                    or (name.startswith("_") or name.endswith("_"))
                    or callable(val)
                ):
                    continue
                imgui.text_colored(_NAME_COLOR, name)
                imgui.same_line(spacing=16)
                imgui.text_colored(_VALUE_COLOR, fmt_value(val))
        finally:
            imgui.pop_style_var()
