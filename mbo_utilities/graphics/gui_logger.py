import logging
import time
from imgui_bundle import imgui
from .. import log

GUI_LOGGERS = log.get_package_loggers()


class GuiLogHandler(logging.Handler):
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger

    def emit(self, record):
        t = time.strftime("%H:%M:%S")
        level_str = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "error",
        }.get(record.levelno, "info")
        # extract only the last segment of the logger name, e.g. "mbo.scan" → "scan"
        name = record.name.split(".")[-1]
        self.gui_logger.messages.append((t, level_str, name, self.format(record)))


class GuiLogger:
    def __init__(self):
        self.show = True
        self.filters = {"debug": True, "info": True, "warning": True}
        self.messages = []  # now holds tuples (time, level, logger_name, text)
        self.window_flags = imgui.WindowFlags_.none
        self.active_loggers = {name: True for name in GUI_LOGGERS}
        # self.sub_loggers = log.get_all_loggers()

    def draw(self):
        # Log‐level checkboxes
        _, self.filters["debug"] = imgui.checkbox("Debug", self.filters["debug"])
        imgui.same_line()
        _, self.filters["info"] = imgui.checkbox("Info", self.filters["info"])
        imgui.same_line()
        _, self.filters["warning"] = imgui.checkbox("warning", self.filters["warning"])

        imgui.separator()

        # Sub‐logger toggles
        for name in list(self.active_loggers):
            imgui.push_id(f"logger_{name}")
            changed, state = imgui.checkbox(
                f"Logger: {name}", self.active_loggers[name]
            )
            if changed:
                self.active_loggers[name] = state
                if state:
                    log.enable(name)
                else:
                    log.disable(name)
            imgui.pop_id()

        imgui.separator()
        imgui.begin_child("##debug_scroll", imgui.ImVec2(0, 0), False)

        # iterate newest‐first
        for t, lvl, logger_name, m in reversed(self.messages):
            if not self.filters.get(lvl, False):
                continue
            if not self.active_loggers.get(logger_name, False):
                continue

            # color by severity
            if lvl == "debug":
                col = imgui.ImVec4(0.6, 0.6, 0.6, 1)
            elif lvl == "info":
                col = imgui.ImVec4(1.0, 1.0, 1.0, 1)
            else:  # "warn" or "error"
                col = imgui.ImVec4(1.0, 0.3, 0.3, 1)
            imgui.text_colored(col, f"[{t}] [{logger_name}] {m}")
        imgui.end_child()

    def draw2(self):
        # Log‐level checkboxes
        _, self.filters["debug"] = imgui.checkbox("Debug", self.filters["debug"])
        imgui.same_line()
        _, self.filters["info"] = imgui.checkbox("Info", self.filters["info"])
        imgui.same_line()
        _, self.filters["warning"] = imgui.checkbox("warning", self.filters["warning"])

        imgui.separator()

        # Sub‐logger toggles
        for name in list(self.active_loggers):
            imgui.push_id(f"logger_{name}")
            changed, state = imgui.checkbox(
                f"Logger: {name}", self.active_loggers[name]
            )
            if changed:
                self.active_loggers[name] = state
                if state:
                    log.enable(name)
                else:
                    log.disable(name)
            imgui.pop_id()

        imgui.separator()
        imgui.begin_child("##debug_scroll", imgui.ImVec2(0, 0), False)
        for t, lvl, logger_name, m in self.messages:
            # skip if this level is unchecked or this logger is disabled
            if not self.filters.get(lvl, False):
                continue
            if not self.active_loggers.get(logger_name, False):
                continue
            col = {
                "debug": imgui.ImVec4(0.8, 0.8, 0.8, 1),
                "info": imgui.ImVec4(1.0, 1.0, 1.0, 1),
                "warning": imgui.ImVec4(1.0, 0.3, 0.3, 1),
            }[lvl]
            imgui.text_colored(col, f"[{t}] [{logger_name}] {m}")
        imgui.end_child()
