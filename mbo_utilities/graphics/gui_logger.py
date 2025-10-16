# gui_logger.py
import logging, time
from imgui_bundle import imgui
from .. import log

GUI_LOGGERS = log.get_package_loggers()
LEVELS = ["DEBUG","INFO","WARNING","ERROR","CRITICAL"]
LEVEL_VAL = {n: getattr(logging, n) for n in LEVELS}

class _GuiNameFilter(logging.Filter):
    def __init__(self, gui_logger): super().__init__(); self.gui_logger = gui_logger
    def filter(self, record): return 1 if self.gui_logger.active_loggers.get(record.name, True) else 0

class GuiLogHandler(logging.Handler):
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger
        self.addFilter(_GuiNameFilter(gui_logger))  # key line
    def emit(self, record):
        t = time.strftime("%H:%M:%S")
        lvl = {10:"debug",20:"info",30:"warning",40:"error",50:"error"}.get(record.levelno,"info")
        self.gui_logger.messages.append((t, lvl, record.name, self.format(record)))

class GuiLogger:
    def __init__(self):
        self.show = True
        self.filters = {"debug": True, "info": True, "warning": True, "error": True}
        self.messages = []
        self.window_flags = imgui.WindowFlags_.none
        self.active_loggers = {name: True for name in log.get_package_loggers()}
        self.levels = {name: "INFO" for name in self.active_loggers}
        self.master_level = "INFO"

    @staticmethod
    def _apply_level(name: str, lvl_name: str):
        logging.getLogger(name).setLevel(LEVEL_VAL[lvl_name])

    def draw(self):
        _, self.filters["debug"] = imgui.checkbox("Debug", self.filters["debug"])
        imgui.same_line()
        _, self.filters["info"] = imgui.checkbox("Info", self.filters["info"])
        imgui.same_line()
        _, self.filters["warning"] = imgui.checkbox("Warning", self.filters["warning"])
        imgui.same_line()
        _, self.filters["error"] = imgui.checkbox("Error", self.filters["error"])

        imgui.same_line()
        if imgui.begin_combo("Level (all)", self.master_level):
            for lvl in LEVELS:
                if imgui.selectable(lvl, self.master_level == lvl)[0]:
                    self.master_level = lvl
                    # apply only once when changed
                    logging.getLogger("mbo").setLevel(LEVEL_VAL[lvl])
                    for name in self.active_loggers:
                        self.levels[name] = lvl
                        self._apply_level(name, lvl)
            imgui.end_combo()

        imgui.separator()

        for name in list(self.active_loggers):
            imgui.push_id(f"logger_{name}")

            changed, state = imgui.checkbox(f"{name}", self.active_loggers[name])
            imgui.same_line()

            cur = self.levels.get(name, "INFO")
            if imgui.begin_combo("##lvl", cur):
                for lvl in LEVELS:
                    if imgui.selectable(lvl, cur == lvl)[0]:
                        self.levels[name] = lvl
                        self._apply_level(name, lvl)
                imgui.end_combo()

            if changed:
                self.active_loggers[name] = state

            imgui.pop_id()

        imgui.separator()

        imgui.begin_child("##debug_scroll", imgui.ImVec2(0, 0), False)
        for t, lvl, full, m in reversed(self.messages):
            if not self.filters.get(lvl, False):
                continue
            if not self.active_loggers.get(full, True):
                continue
            short = full.split(".")[-1]
            col = {
                "debug": imgui.ImVec4(0.6, 0.6, 0.6, 1),
                "info": imgui.ImVec4(1.0, 1.0, 1.0, 1),
                "warning": imgui.ImVec4(1.0, 0.6, 0.2, 1),
                "error": imgui.ImVec4(1.0, 0.3, 0.3, 1),
            }[lvl]
            imgui.text_colored(col, f"[{t}] [{short}] {m}")
        imgui.end_child()
