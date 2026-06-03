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

        try:
            from mbo_utilities.gui.widgets.pipelines.isoview import IsoviewPipelineWidget
            _PIPELINE_CLASSES.append(IsoviewPipelineWidget)
        except Exception:
            pass

        # future: add more pipelines here
        # from .masknmf import MaskNMFPipelineWidget
        # _PIPELINE_CLASSES.append(MaskNMFPipelineWidget)

        _REGISTRATION_COMPLETE = True

    # suite2p.parameters.SETTINGS is loaded out-of-process by
    # `_s2p_schema`'s module-level daemon (cached to
    # `~/.mbo/cache/s2p_settings_<version>.json`). Importing the widget
    # classes above triggers that daemon transitively; the subprocess
    # never enters this interpreter's address space.


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


def _active_array(parent: Any) -> Any:
    """Return the currently-loaded array (or ``None``).

    Used to filter pipelines via :meth:`PipelineWidget.applies_to`.
    """
    iw = getattr(parent, "image_widget", None)
    if iw is None or not iw.data:
        return None
    return iw.data[0]


def _is_pipeline_available(cls: type) -> bool:
    """Resolve ``is_available`` whether it's a class attr or a property.

    Suite2p declares ``is_available`` as a ``@property`` (instance-bound),
    so reading it off the class returns the descriptor (truthy) and not
    the value. We instantiate temporarily if needed — Suite2p widgets
    are heavy to construct so we cache the result on the class.
    """
    cached = getattr(cls, "_is_available_cached", None)
    if cached is not None:
        return cached
    val = cls.__dict__.get("is_available")
    if isinstance(val, property):
        try:
            result = bool(val.fget(cls.__new__(cls)))
        except Exception:
            # property reads parent state — give up and assume available
            result = True
    else:
        result = bool(getattr(cls, "is_available", True))
    cls._is_available_cached = result  # type: ignore[attr-defined]
    return result


def draw_run_tab(parent: Any) -> None:
    """Draw the run tab content.

    Renders a pipeline selector at the top (when more than one
    applicable pipeline is available), then the selected widget's
    config UI. Pipelines are filtered by ``is_available`` (deps
    installed) AND ``applies_to(active_array)`` (data type matches).
    """
    _register_pipelines()

    # Persist selection by pipeline NAME, not list index — the list of
    # applicable pipelines changes between datasets, and an int index
    # silently shifts to a different pipeline when the list shrinks.
    if not hasattr(parent, "_selected_pipeline_name"):
        parent._selected_pipeline_name = None
    if not hasattr(parent, "_pipeline_instances"):
        parent._pipeline_instances = {}

    arr = _active_array(parent)

    if not _PIPELINE_CLASSES:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "No pipelines registered.",
        )
        imgui.text("Install a pipeline package:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install mbo_utilities",
        )
        return

    # partition: applicable to current data (and installed) vs. not.
    applicable: list[type[PipelineWidget]] = []
    not_applicable: list[type[PipelineWidget]] = []
    not_installed: list[type[PipelineWidget]] = []
    for cls in _PIPELINE_CLASSES:
        installed = _is_pipeline_available(cls)
        try:
            applies = cls.applies_to(arr)
        except Exception:
            applies = False
        if installed and applies:
            applicable.append(cls)
        elif not installed:
            not_installed.append(cls)
        else:
            not_applicable.append(cls)

    # Isoview before Suite2p in the selector when both apply (Suite2p
    # applies to any array, so it would otherwise lead by registration
    # order). Stable: only Isoview is hoisted; the rest keep their order.
    applicable.sort(key=lambda c: 0 if c.name == "Isoview" else 1)

    if not applicable:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "No pipeline applies to the loaded data.",
        )
        imgui.spacing()
        if not_applicable:
            imgui.text("Installed but not applicable to this data:")
            for cls in not_applicable:
                imgui.bullet_text(f"{cls.name}")
        if not_installed:
            imgui.spacing()
            imgui.text("Not installed:")
            for cls in not_installed:
                imgui.bullet_text(cls.name)
                imgui.indent(16)
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
                    cls.install_command,
                )
                imgui.unindent(16)
        return

    # selector — combo when more than one option. Resolve persisted
    # name → list index each frame so a user's choice survives switching
    # between datasets where the applicable set changes.
    labels = [c.name for c in applicable]
    try:
        idx = labels.index(parent._selected_pipeline_name)
    except (ValueError, TypeError):
        idx = 0
    if len(applicable) > 1:
        imgui.set_next_item_width(220)
        changed, new_idx = imgui.combo("Pipeline##run_tab", idx, labels)
        if changed:
            idx = new_idx
        imgui.separator()
    pipeline_cls = applicable[idx]
    parent._selected_pipeline_name = pipeline_cls.name

    pipeline_key = pipeline_cls.name
    if pipeline_key not in parent._pipeline_instances:
        parent._pipeline_instances[pipeline_key] = pipeline_cls(parent)
    pipeline = parent._pipeline_instances[pipeline_key]

    try:
        pipeline.draw()
    except Exception as e:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
            f"Error: {e}",
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
