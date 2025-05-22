from .run_gui import run_gui
from .imgui import SummaryDataWidget, PreviewDataWidget
from imgui_bundle import hello_imgui

from pathlib import Path

_cwd = Path().cwd()

hello_imgui.set_assets_folder(
    "./assets/JetBrainsMono/JetBrainsMono-Bold.ttf",
)

__all__ = [
    "SummaryDataWidget",
    "PreviewDataWidget",
    "run_gui",
]
