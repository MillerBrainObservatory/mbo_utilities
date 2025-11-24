"""
base class for pipeline widgets.

pipeline widgets have two modes:
- configure: set up and run processing
- results: view processing results

each pipeline is self-contained with its own settings dataclass,
config ui, and results ui.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget


class PipelineWidget(ABC):
    """base class for pipeline widgets."""

    # human-readable name shown in pipeline selector
    name: str = "Pipeline"

    # whether this pipeline's dependencies are installed
    is_available: bool = False

    # install command to show when not available
    install_command: str = "uv pip install mbo_utilities"

    def __init__(self, parent: "PreviewDataWidget"):
        self.parent = parent
        # track current mode: True = results, False = configure
        self._show_results = False

    @property
    def mode_label(self) -> str:
        """current mode label for toggle."""
        return "Results" if self._show_results else "Process"

    def toggle_mode(self) -> None:
        """toggle between configure and results mode."""
        self._show_results = not self._show_results

    def draw(self) -> None:
        """draw the pipeline widget with mode toggle."""
        from imgui_bundle import imgui

        # mode toggle switch
        imgui.text("Mode:")
        imgui.same_line()

        # toggle button styled as switch
        if self._show_results:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.6, 0.2, 1.0))
        else:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.3, 0.3, 0.3, 1.0))

        if imgui.button("Process" if not self._show_results else "Results"):
            self.toggle_mode()
        imgui.pop_style_color()

        imgui.same_line()
        imgui.text_disabled("(click to switch)")

        imgui.separator()
        imgui.spacing()

        # draw appropriate content
        if self._show_results:
            self.draw_results()
        else:
            self.draw_config()

    @abstractmethod
    def draw_config(self) -> None:
        """draw the configuration/processing ui."""
        ...

    @abstractmethod
    def draw_results(self) -> None:
        """draw the results viewer ui."""
        ...
