import time
from imgui_bundle import imgui


class GuiLogger:
    def __init__(self):
        self.show = True
        self.filters = {"debug": True, "info": True, "error": True}
        self.messages = []

    def log(self, level, msg):
        t = time.strftime("%H:%M:%S")
        self.messages.append((t, level, msg))

    def draw(self):
        opened, self.show = imgui.begin("Debug Panel", self.show)
        if not opened:
            imgui.end()
            return
        _, self.filters["debug"] = imgui.checkbox("Debug", self.filters["debug"])
        imgui.same_line()
        _, self.filters["info"] = imgui.checkbox("Info", self.filters["info"])
        imgui.same_line()
        _, self.filters["error"] = imgui.checkbox("Error", self.filters["error"])
        imgui.separator()
        imgui.begin_child("##debug_scroll", imgui.ImVec2(0, 0), False)
        for t, lvl, m in self.messages:
            if not self.filters[lvl]:
                continue
            col = {
                "debug": imgui.ImVec4(0.8, 0.8, 0.8, 1),
                "info": imgui.ImVec4(1.0, 1.0, 1.0, 1),
                "error": imgui.ImVec4(1.0, 0.3, 0.3, 1),
            }[lvl]
            imgui.text_colored(col, f"[{t}] {m}")
        imgui.end_child()
        imgui.end()
