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
- **Features**: Auto-discovered capability-based controls (RasterScanFeature, etc.)
- **Widgets**: Generic composable UI building blocks (ListSelectorWidget, etc.)

See docs/gui_refactor_plan.md for the full architecture documentation.
"""

__all__ = [
    # Legacy exports (backwards compatibility)
    "PreviewDataWidget",
    "run_gui",
    "download_notebook",
    "GridSearchViewer",
    "setup_imgui",
    "set_qt_icon",
    "get_default_ini_path",
    # New architecture exports
    "BaseViewer",
    "TimeSeriesViewer",
    "PollenCalibrationViewer",
    "DiagnosticsViewer",
    "Suite2pResultsViewer",
    "BasePanel",
    "DebugPanel",
    "ProcessPanel",
    "MetadataPanel",
    "BaseFeature",
    "get_supported_features",
    "ListSelectorWidget",
    "CheckboxGridWidget",
    "SliderGroupWidget",
]


def __getattr__(name):
    """Lazy import heavy GUI modules only when accessed."""

    # === Legacy exports (backwards compatibility) ===

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

    # === New architecture: Viewers ===

    elif name == "BaseViewer":
        from .viewers import BaseViewer
        return BaseViewer
    elif name == "TimeSeriesViewer":
        from .viewers import TimeSeriesViewer
        return TimeSeriesViewer
    elif name == "PollenCalibrationViewer":
        from .viewers import PollenCalibrationViewer
        return PollenCalibrationViewer
    elif name == "DiagnosticsViewer":
        from .viewers import DiagnosticsViewer
        return DiagnosticsViewer
    elif name == "Suite2pResultsViewer":
        from .viewers import Suite2pResultsViewer
        return Suite2pResultsViewer

    # === New architecture: Panels ===

    elif name == "BasePanel":
        from .panels import BasePanel
        return BasePanel
    elif name == "DebugPanel":
        from .panels import DebugPanel
        return DebugPanel
    elif name == "ProcessPanel":
        from .panels import ProcessPanel
        return ProcessPanel
    elif name == "MetadataPanel":
        from .panels import MetadataPanel
        return MetadataPanel

    # === New architecture: Features ===

    elif name == "BaseFeature":
        from .features import BaseFeature
        return BaseFeature
    elif name == "get_supported_features":
        from .features import get_supported_features
        return get_supported_features

    # === New architecture: Widgets ===

    elif name == "ListSelectorWidget":
        from .widgets_new import ListSelectorWidget
        return ListSelectorWidget
    elif name == "CheckboxGridWidget":
        from .widgets_new import CheckboxGridWidget
        return CheckboxGridWidget
    elif name == "SliderGroupWidget":
        from .widgets_new import SliderGroupWidget
        return SliderGroupWidget

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
