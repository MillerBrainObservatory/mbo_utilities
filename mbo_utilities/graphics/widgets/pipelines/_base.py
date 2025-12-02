"""
base class for pipeline widgets.

each pipeline is self-contained with its own settings dataclass and config ui.
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

    def draw(self) -> None:
        """draw the pipeline widget."""
        self.draw_config()

    @abstractmethod
    def draw_config(self) -> None:
        """draw the configuration/processing ui."""
        ...
