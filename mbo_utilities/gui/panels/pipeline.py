"""
Pipeline panel for running processing pipelines.

Wraps the pipeline widget registry to provide a panel-based interface
for selecting and running pipelines (suite2p, masknmf, etc).
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.panels import BasePanel

__all__ = ["PipelinePanel"]


class PipelinePanel(BasePanel):
    """
    Panel for selecting and running processing pipelines.

    Provides a tab-based interface for pipeline configuration and execution.
    Each pipeline (suite2p, masknmf, etc) has its own configuration UI.

    Attributes
    ----------
    name : str
        Panel name shown in UI.
    """

    name = "Pipelines"

    def __init__(self, viewer: Any):
        super().__init__(viewer)
        self._visible = True  # Pipeline tab is visible by default

        # Pipeline state - managed here, synced to widgets as needed
        self._selected_pipeline_idx: int = 0
        self._pipeline_instances: dict[str, Any] = {}

    @property
    def image_widget(self) -> Any:
        """Get image widget from viewer."""
        return getattr(self.viewer, "image_widget", None)

    @property
    def fpath(self) -> Any:
        """Get file path from viewer."""
        return getattr(self.viewer, "fpath", "")

    @property
    def nz(self) -> int:
        """Get number of z-planes from viewer."""
        return getattr(self.viewer, "nz", 1)

    @property
    def shape(self) -> tuple:
        """Get data shape from viewer."""
        return getattr(self.viewer, "shape", ())

    @property
    def num_graphics(self) -> int:
        """Get number of graphics from viewer."""
        return getattr(self.viewer, "num_graphics", 1)

    @property
    def logger(self) -> Any:
        """Get logger from viewer."""
        return getattr(self.viewer, "logger", None)

    def draw(self) -> None:
        """Draw the pipeline panel content."""
        if not self._visible:
            return

        from mbo_utilities.gui.widgets.pipelines import (
            get_available_pipelines,
            any_pipeline_available,
        )

        # Register pipelines if needed
        pipeline_classes = get_available_pipelines()

        # Check if any pipelines available
        if not pipeline_classes:
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

        # Get first available pipeline (currently only suite2p)
        pipeline_cls = pipeline_classes[0]
        self._selected_pipeline_idx = 0

        # If not available, show install message
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

        # Get or create pipeline instance
        # Pass self as the parent - we proxy viewer attributes
        pipeline_key = pipeline_cls.name
        if pipeline_key not in self._pipeline_instances:
            self._pipeline_instances[pipeline_key] = pipeline_cls(self)

        pipeline = self._pipeline_instances[pipeline_key]

        # Draw the pipeline widget
        try:
            pipeline.draw()
        except Exception as e:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                f"Error: {e}"
            )

    def cleanup(self) -> None:
        """Clean up all pipeline instances."""
        import contextlib

        for pipeline in self._pipeline_instances.values():
            with contextlib.suppress(Exception):
                pipeline.cleanup()

        self._pipeline_instances.clear()
