from ..util import is_imgui_installed

__all__ = []

if is_imgui_installed():
    HAS_IMGUI = True
    from .imgui import SummaryDataWidget, PreviewDataWidget
else:
    HAS_IMGUI = False
    PreviewDataWidget = None
    SummaryDataWidget = None

from .run_gui import run_gui

__all__ += [
    "SummaryDataWidget",
    "PreviewDataWidget",
    "HAS_IMGUI",
    "run_gui",
]
