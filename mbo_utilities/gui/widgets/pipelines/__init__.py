"""
pipeline widget registry.

pipelines are processing workflows (suite2p, masknmf, etc) that can be
run on imaging data. each pipeline has config and results views.

imports are done in a background thread to avoid blocking the GUI.
"""

from typing import Any
import threading
import time

from imgui_bundle import imgui

from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
import contextlib

# registry of available pipeline classes
_PIPELINE_CLASSES: list[type[PipelineWidget]] = []
_REGISTRATION_LOCK = threading.Lock()
_REGISTRATION_STARTED = False
_REGISTRATION_COMPLETE = False

# delay before the bg thread starts heavy imports. lets the main thread
# finish painting the first frame and initializing fastplotlib/qt
# without GIL contention from a multi-second suite2p import. tune via
# start_preload(delay_s=...) if needed.
_PRELOAD_DELAY_S = 1.0


def _register_pipelines_sync() -> None:
    """Register pipeline widgets (called from background thread)."""
    global _PIPELINE_CLASSES, _REGISTRATION_COMPLETE

    with _REGISTRATION_LOCK:
        if _PIPELINE_CLASSES:
            _REGISTRATION_COMPLETE = True
            return

        # preload settings module first (it's imported by Suite2pPipelineWidget.__init__).
        # _s2p_schema is now lazy, so this import is cheap — it does NOT
        # transitively pull in suite2p. the actual suite2p load happens
        # in the warm_up step below.
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

    # NOTE: do NOT eagerly import suite2p.parameters here. doing so
    # pulls torch/numba/scipy/cellpose into the bg thread and the
    # python GIL held during that import freezes the imgui main loop
    # for 10–20s. _s2p_schema._ensure_loaded() runs lazily on first
    # actual use (settings panel / Run tab), which restores the v2.7.7
    # startup behavior (instant) at the cost of a one-time freeze when
    # the user first opens settings — same trade as before.


def _delayed_preload(delay_s: float) -> None:
    """Sleep, then run the heavy preload. Runs in a daemon thread."""
    if delay_s > 0:
        time.sleep(delay_s)
    _register_pipelines_sync()


def start_preload(delay_s: float | None = None) -> None:
    """Start background preloading of pipeline widgets + suite2p schema.

    Call this early (e.g., on GUI startup) to warm up imports
    before the user clicks the Run tab. The bg thread sleeps briefly
    before doing heavy work so it doesn't contend with the main
    thread during first paint / fastplotlib init.
    """
    global _REGISTRATION_STARTED

    if _REGISTRATION_STARTED:
        return

    _REGISTRATION_STARTED = True
    delay = _PRELOAD_DELAY_S if delay_s is None else delay_s
    thread = threading.Thread(
        target=_delayed_preload, args=(delay,), daemon=True
    )
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
            "uv pip install mbo_utilities"
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
    lazy_names = (
        "Suite2pSettings",
        "Suite2pDB",
        "MboSuite2pExtras",
        "draw_suite2p_settings_panel",
        "draw_section_suite2p",
    )
    if name in lazy_names:
        if name not in _settings_cache:
            from mbo_utilities.gui.widgets.pipelines import settings
            for attr in lazy_names:
                _settings_cache[attr] = getattr(settings, attr)
        return _settings_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineWidget",
    "Suite2pSettings",
    "Suite2pDB",
    "MboSuite2pExtras",
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
