# gui_logger.py
"""
debug panel for mbo_utilities GUI.

displays log messages from all mbo.* loggers with filtering,
level control, and message management.
"""
import logging
import time
from imgui_bundle import imgui
from .. import log

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
    """filter log records by logger name based on gui toggle state."""
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger

    def filter(self, record):
        return 1 if self.gui_logger.active_loggers.get(record.name, True) else 0


class GuiLogHandler(logging.Handler):
    """handler that routes log records to the gui debug panel."""
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger
        self.addFilter(_GuiNameFilter(gui_logger))

    def emit(self, record):
        t = time.strftime("%H:%M:%S")
        lvl_map = {10: "debug", 20: "info", 30: "warning", 40: "error", 50: "critical"}
        lvl = lvl_map.get(record.levelno, "info")
        msg = self.format(record)
        self.gui_logger.add_message(t, lvl, record.name, msg)


class GuiLogger:
    """
    debug panel widget for imgui.

    features:
    - filter messages by level (debug/info/warning/error/critical)
    - toggle individual loggers on/off
    - set log level globally or per-logger
    - auto-scroll to newest messages
    - search/filter messages
    - clear messages
    - message limit to prevent memory issues
    """
    MAX_MESSAGES = 1000

    def __init__(self):
        self.show = True
        self.filters = {
            "debug": False,  # off by default - too noisy
            "info": True,
            "warning": True,
            "error": True,
            "critical": True,
        }
        self.messages = []
        self.window_flags = imgui.WindowFlags_.none
        self.active_loggers = {name: True for name in log.get_package_loggers()}
        self.levels = {name: "INFO" for name in self.active_loggers}
        self.master_level = "INFO"

        # ui state
        self.auto_scroll = True
        self.search_text = ""
        self.show_loggers = False  # collapsed by default
        self.show_controls = True

    def add_message(self, timestamp, level, logger_name, message):
        """add a message, enforcing max limit."""
        self.messages.append((timestamp, level, logger_name, message))
        if len(self.messages) > self.MAX_MESSAGES:
            # remove oldest 10%
            trim = self.MAX_MESSAGES // 10
            self.messages = self.messages[trim:]

    @staticmethod
    def _apply_level(name: str, lvl_name: str):
        logging.getLogger(name).setLevel(LEVEL_VAL[lvl_name])

    def _refresh_loggers(self):
        """refresh active_loggers list to include newly created loggers."""
        current = log.get_package_loggers()
        for name in current:
            if name not in self.active_loggers:
                self.active_loggers[name] = True
                self.levels[name] = self.master_level

    def _draw_controls(self):
        """draw level filter checkboxes and global controls."""
        # level filters
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

        # global level dropdown
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
                    log.set_global_level(LEVEL_VAL[lvl])
                    for name in self.active_loggers:
                        self.levels[name] = lvl
            imgui.end_combo()
        imgui.same_line()

        # clear button
        if imgui.button("Clear"):
            self.messages.clear()

    def _draw_search(self):
        """draw search bar and auto-scroll toggle."""
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

    def _draw_logger_toggles(self):
        """draw collapsible section with per-logger controls."""
        self._refresh_loggers()

        flags = imgui.TreeNodeFlags_.default_open if self.show_loggers else 0
        expanded = imgui.collapsing_header("Loggers", flags)
        # handle both tuple return (older imgui) and bool return (newer imgui)
        if isinstance(expanded, tuple):
            expanded = expanded[0]
        if expanded:
            self.show_loggers = True

            # group loggers by prefix
            groups = {}
            for name in sorted(self.active_loggers.keys()):
                parts = name.split(".")
                if len(parts) > 1:
                    prefix = parts[1]  # e.g., "arrays" from "mbo.arrays.tiff"
                else:
                    prefix = "core"
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append(name)

            # draw grouped loggers in columns
            columns = min(3, len(groups))
            if imgui.begin_table("logger_table", columns, imgui.TableFlags_.borders_inner_v):
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

    def _draw_messages(self):
        """draw scrollable message list."""
        # calculate available height
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
            # filter by level
            if not self.filters.get(lvl, False):
                continue

            # filter by logger toggle
            if not self.active_loggers.get(full_name, True):
                continue

            # filter by search
            if search_lower and search_lower not in msg.lower():
                continue

            short = full_name.split(".")[-1]
            col = LEVEL_COLORS.get(lvl, LEVEL_COLORS["info"])

            # format: [HH:MM:SS] [logger] message
            imgui.text_colored(imgui.ImVec4(0.5, 0.5, 0.5, 1.0), f"[{timestamp}]")
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.4, 0.7, 1.0, 1.0), f"[{short}]")
            imgui.same_line()
            imgui.text_colored(col, msg)

        if self.auto_scroll:
            imgui.set_scroll_here_y(1.0)

        imgui.end_child()

    def draw(self):
        """draw the full debug panel."""
        self._draw_controls()
        self._draw_search()
        imgui.separator()
        self._draw_logger_toggles()
        imgui.separator()
        self._draw_messages()
