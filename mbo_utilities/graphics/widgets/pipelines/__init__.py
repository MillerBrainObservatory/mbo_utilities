"""
pipeline widget registry.

pipelines are processing workflows (suite2p, masknmf, etc) that can be
run on imaging data. each pipeline has config and results views.
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from mbo_utilities.graphics.widgets.pipelines._base import PipelineWidget

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget

# registry of available pipeline classes
_PIPELINE_CLASSES: list[type[PipelineWidget]] = []


def _register_pipelines() -> None:
    """register available pipeline widgets."""
    global _PIPELINE_CLASSES

    if _PIPELINE_CLASSES:
        return

    # import pipeline widgets - they register themselves based on availability
    try:
        from mbo_utilities.graphics.widgets.pipelines.suite2p import Suite2pPipelineWidget
        _PIPELINE_CLASSES.append(Suite2pPipelineWidget)
    except Exception:
        pass

    # future: add more pipelines here
    # from .masknmf import MaskNMFPipelineWidget
    # _PIPELINE_CLASSES.append(MaskNMFPipelineWidget)


def get_available_pipelines() -> list[type[PipelineWidget]]:
    """get list of all registered pipeline classes."""
    _register_pipelines()
    return _PIPELINE_CLASSES.copy()


def get_pipeline_names() -> list[str]:
    """get names of all registered pipelines."""
    _register_pipelines()
    return [p.name for p in _PIPELINE_CLASSES]


def any_pipeline_available() -> bool:
    """check if any pipeline is available (installed)."""
    _register_pipelines()
    return any(p.is_available for p in _PIPELINE_CLASSES)


def draw_run_tab(parent: "PreviewDataWidget") -> None:
    """
    draw the run tab content.

    shows pipeline selector and the selected pipeline's widget.
    if no pipelines available, shows install message.
    """
    _register_pipelines()

    # initialize state
    if not hasattr(parent, '_selected_pipeline_idx'):
        parent._selected_pipeline_idx = 0
    if not hasattr(parent, '_pipeline_instances'):
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


def cleanup_pipelines(parent: "PreviewDataWidget") -> None:
    """clean up all pipeline instances when gui is closing.

    calls cleanup() on each pipeline to release resources like
    open windows, background threads, etc.
    """
    if not hasattr(parent, '_pipeline_instances'):
        return

    for name, pipeline in parent._pipeline_instances.items():
        try:
            pipeline.cleanup()
        except Exception as e:
            print(f"Warning: cleanup failed for pipeline {name}: {e}")

    parent._pipeline_instances.clear()


__all__ = [
    "PipelineWidget",
    "get_available_pipelines",
    "get_pipeline_names",
    "any_pipeline_available",
    "draw_run_tab",
    "cleanup_pipelines",
]
