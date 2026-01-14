"""
pipeline widget registry.

pipelines are processing workflows (suite2p, masknmf, etc) that can be
run on imaging data. each pipeline has config and results views.

imports are done in a background thread to avoid blocking the GUI.
"""

from typing import Any
import threading

from imgui_bundle import imgui

from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
import contextlib

# registry of available pipeline classes
_PIPELINE_CLASSES: list[type[PipelineWidget]] = []
_REGISTRATION_LOCK = threading.Lock()
_REGISTRATION_STARTED = False
_REGISTRATION_COMPLETE = False


def _register_pipelines_sync() -> None:
    """Register pipeline widgets (called from background thread)."""
    global _PIPELINE_CLASSES, _REGISTRATION_COMPLETE

    with _REGISTRATION_LOCK:
        if _PIPELINE_CLASSES:
            _REGISTRATION_COMPLETE = True
            return

        # preload settings module first (it's imported by Suite2pPipelineWidget.__init__)
        try:
            from mbo_utilities.gui.widgets.pipelines import settings as _  # noqa: F401
        except Exception:
            pass

        # import pipeline widgets - they register themselves based on availability
        try:
            from mbo_utilities.gui.widgets.pipelines.suite2p import Suite2pPipelineWidget
            _PIPELINE_CLASSES.append(Suite2pPipelineWidget)
        except Exception:
            pass

        # future: add more pipelines here
        # from .masknmf import MaskNMFPipelineWidget
        # _PIPELINE_CLASSES.append(MaskNMFPipelineWidget)

        _REGISTRATION_COMPLETE = True


def start_preload() -> None:
    """Start background preloading of pipeline widgets.

    Call this early (e.g., on GUI startup) to warm up imports
    before the user clicks the Run tab.
    """
    global _REGISTRATION_STARTED

    if _REGISTRATION_STARTED:
        return

    _REGISTRATION_STARTED = True
    thread = threading.Thread(target=_register_pipelines_sync, daemon=True)
    thread.start()


def is_ready() -> bool:
    """Check if pipeline registration is complete."""
    return _REGISTRATION_COMPLETE


def _register_pipelines() -> None:
    """Register available pipeline widgets (blocking if not preloaded)."""
    if not _REGISTRATION_STARTED:
        start_preload()

    # if already complete, return immediately
    if _REGISTRATION_COMPLETE:
        return

    # wait for background thread to complete (blocking)
    _register_pipelines_sync()


def get_available_pipelines() -> list[type[PipelineWidget]]:
    """Get list of all registered pipeline classes."""
    _register_pipelines()
    return _PIPELINE_CLASSES.copy()


def get_pipeline_names() -> list[str]:
    """Get names of all registered pipelines."""
    _register_pipelines()
    return [p.name for p in _PIPELINE_CLASSES]


def any_pipeline_available() -> bool:
    """Check if any pipeline is available (installed)."""
    _register_pipelines()
    return any(p.is_available for p in _PIPELINE_CLASSES)


def draw_run_tab(parent: Any) -> None:
    """
    Draw the run tab content.

    shows pipeline selector and the selected pipeline's widget.
    if no pipelines available, shows install message.
    """
    _register_pipelines()

    # initialize state
    if not hasattr(parent, "_selected_pipeline_idx"):
        parent._selected_pipeline_idx = 0
    if not hasattr(parent, "_pipeline_instances"):
        parent._pipeline_instances = {}

    # check if any pipelines available
    if not _PIPELINE_CLASSES:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "No pipelines available."
        )
        imgui.text("Install a pipeline package:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install mbo_utilities[suite2p]"
        )
        return

    # get first available pipeline (currently only suite2p)
    pipeline_cls = _PIPELINE_CLASSES[0]
    parent._selected_pipeline_idx = 0

    # if not available, show install message
    if not pipeline_cls.is_available:
        imgui.spacing()
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            f"{pipeline_cls.name} is not installed."
        )
        imgui.spacing()
        imgui.text("Install with:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            pipeline_cls.install_command
        )
        return

    # get or create pipeline instance
    pipeline_key = pipeline_cls.name
    if pipeline_key not in parent._pipeline_instances:
        parent._pipeline_instances[pipeline_key] = pipeline_cls(parent)

    pipeline = parent._pipeline_instances[pipeline_key]

    # draw the pipeline widget
    try:
        pipeline.draw()
    except Exception as e:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
            f"Error: {e}"
        )


def cleanup_pipelines(parent: Any) -> None:
    """Clean up all pipeline instances when gui is closing.

    calls cleanup() on each pipeline to release resources like
    open windows, background threads, etc.
    """
    if not hasattr(parent, "_pipeline_instances"):
        return

    for pipeline in parent._pipeline_instances.values():
        with contextlib.suppress(Exception):
            pipeline.cleanup()

    parent._pipeline_instances.clear()


# lazy imports for settings - use __getattr__ for module-level lazy loading
_settings_cache = {}


def __getattr__(name: str):
    """Lazy load settings module exports on first access."""
    if name in ("Suite2pSettings", "draw_suite2p_settings_panel", "draw_section_suite2p"):
        if name not in _settings_cache:
            from mbo_utilities.gui.widgets.pipelines import settings
            _settings_cache["Suite2pSettings"] = settings.Suite2pSettings
            _settings_cache["draw_suite2p_settings_panel"] = settings.draw_suite2p_settings_panel
            _settings_cache["draw_section_suite2p"] = settings.draw_section_suite2p
        return _settings_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineWidget",
    "Suite2pSettings",
    "any_pipeline_available",
    "cleanup_pipelines",
    "draw_run_tab",
    "draw_section_suite2p",
    "draw_suite2p_settings_panel",
    "get_available_pipelines",
    "get_pipeline_names",
    "is_ready",
    "start_preload",
]
