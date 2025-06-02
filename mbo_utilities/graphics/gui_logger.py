import logging
import time
from imgui_bundle import imgui
from .. import log


class GuiLogHandler(logging.Handler):
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger

    def emit(self, record):
        msg = self.format(record)
        self.gui_logger.add_message(record.levelno, msg)


class GuiLogger:
    def __init__(self):
        self.show = True
        self.filters = {"debug": True, "info": True, "error": True}
        self.messages = []
        self.window_flags = imgui.WindowFlags_.none
        self.active_loggers = {
            "mbo": True,
            "gui": True,
            "scan": True,
            "io": True
        }

    def add_message(self, levelno, msg):
        t = time.strftime("%H:%M:%S")
        level_str = {
            logging.DEBUG: "debug",
            logging.INFO: "info",
            logging.WARNING: "warn",
            logging.ERROR: "error",
            logging.CRITICAL: "error"
        }.get(levelno, "info")
        self.messages.append((t, level_str, msg))

    def draw(self):
        # Log level filters
        _, self.filters["debug"] = imgui.checkbox("Debug", self.filters["debug"])
        imgui.same_line()
        _, self.filters["info"] = imgui.checkbox("Info", self.filters["info"])
        imgui.same_line()
        _, self.filters["error"] = imgui.checkbox("Error", self.filters["error"])

        imgui.separator()

        # Toggle specific sub-loggers
        for name in list(self.active_loggers):
            imgui.push_id(f"logger_{name}")
            changed, state = imgui.checkbox(f"Logger: {name}", self.active_loggers[name])
            if changed:
                self.active_loggers[name] = state
                if state:
                    log.enable(name)
                else:
                    log.disable(name)
            imgui.pop_id()

        imgui.separator()
        imgui.begin_child("##debug_scroll", imgui.ImVec2(0, 0), False)
        for t, lvl, m in self.messages:
            if not self.filters.get(lvl, False):
                continue
            col = {
                "debug": imgui.ImVec4(0.8, 0.8, 0.8, 1),
                "info": imgui.ImVec4(1.0, 1.0, 1.0, 1),
                "error": imgui.ImVec4(1.0, 0.3, 0.3, 1),
            }[lvl]
            imgui.text_colored(col, f"[{t}] {m}")
        imgui.end_child()
