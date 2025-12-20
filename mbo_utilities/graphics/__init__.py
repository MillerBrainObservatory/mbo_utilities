"""
Graphics module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p, wgpu, imgui_bundle) until actually needed.

The CLI entry point (mbo command) imports this module, so we must keep
top-level imports minimal for fast startup of light operations like
--download-notebook and --check-install.
"""

__all__ = [
    "PreviewDataWidget",
    "run_gui",
    "download_notebook",
    "GridSearchViewer",
    "setup_imgui",
    "set_qt_icon",
    "get_default_ini_path",
]


def __getattr__(name):
    """Lazy import heavy graphics modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    elif name == "download_notebook":
        from .run_gui import download_notebook
        return download_notebook
    elif name == "PreviewDataWidget":
        from . import _setup  # triggers setup on import
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    elif name == "GridSearchViewer":
        from .grid_search_viewer import GridSearchViewer
        return GridSearchViewer
    elif name == "setup_imgui":
        from ._setup import setup_imgui
        return setup_imgui
    elif name == "set_qt_icon":
        from ._setup import set_qt_icon
        return set_qt_icon
    elif name == "get_default_ini_path":
        from ._setup import get_default_ini_path
        return get_default_ini_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
