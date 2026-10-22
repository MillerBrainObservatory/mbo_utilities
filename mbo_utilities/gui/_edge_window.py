"""EdgeWindow: ImguiWindow that self-registers on a figure edge."""

from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui


class EdgeWindow(ImguiWindow):
    def __init__(
        self,
        figure,
        size: int,
        location: str,
        title: str,
        window_flags=imgui.WindowFlags_.no_collapse
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_title_bar,
    ):
        super().__init__()
        figure.add_imgui_window(
            self,
            location=location,
            size=size,
            title=title,
            window_flags=window_flags,
        )
