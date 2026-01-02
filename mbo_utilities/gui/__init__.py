"""
GUI module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p, wgpu, imgui_bundle) until actually needed.

The CLI entry point (mbo command) imports this module, so we must keep
top-level imports minimal for fast startup of light operations like
--download-notebook and --check-install.

Architecture
------------
The GUI is organized into these components:

- **Viewers**: Standalone application windows (TimeSeriesViewer, etc.)
- **Panels**: Reusable UI sections (DebugPanel, MetadataPanel, etc.)
- **Widgets**: Capability-based UI components (widgets/)

See docs/gui_refactor_plan.md for the full architecture documentation.
"""

__all__ = [
    # Panels
    "BasePanel",
    # Viewer architecture
    "BaseViewer",
    "DebugPanel",
    "GridSearchViewer",
    "MetadataPanel",
    "PollenCalibrationViewer",
    # Legacy exports (backwards compatibility)
    "PreviewDataWidget",
    "ProcessPanel",
    "TimeSeriesViewer",
    "download_notebook",
    "get_default_ini_path",
    "run_gui",
    "set_qt_icon",
    "setup_imgui",
]


def __getattr__(name):
    """Lazy import heavy GUI modules only when accessed."""
    # === Legacy exports (backwards compatibility) ===

    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    if name == "download_notebook":
        from .run_gui import download_notebook
        return download_notebook
    if name == "PreviewDataWidget":
        from . import _setup  # triggers setup on import
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    if name == "GridSearchViewer":
        from .widgets.grid_search import GridSearchViewer
        return GridSearchViewer
    if name == "setup_imgui":
        from ._setup import setup_imgui
        return setup_imgui
    if name == "set_qt_icon":
        from ._setup import set_qt_icon
        return set_qt_icon
    if name == "get_default_ini_path":
        from ._setup import get_default_ini_path
        return get_default_ini_path

    # === New architecture: Viewers ===

    if name == "BaseViewer":
        from .viewers import BaseViewer
        return BaseViewer
    if name == "TimeSeriesViewer":
        from .viewers import TimeSeriesViewer
        return TimeSeriesViewer
    if name == "PollenCalibrationViewer":
        from .viewers import PollenCalibrationViewer
        return PollenCalibrationViewer

    # === Panels ===

    if name == "BasePanel":
        from .panels import BasePanel
        return BasePanel
    if name == "DebugPanel":
        from .panels import DebugPanel
        return DebugPanel
    if name == "ProcessPanel":
        from .panels import ProcessPanel
        return ProcessPanel
    if name == "MetadataPanel":
        from .panels import MetadataPanel
        return MetadataPanel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
