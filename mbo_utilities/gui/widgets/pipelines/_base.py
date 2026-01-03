"""
base class for pipeline widgets.

each pipeline is self-contained with its own settings dataclass and config ui.
"""

from abc import ABC, abstractmethod
from typing import Any


class PipelineWidget(ABC):
    """base class for pipeline widgets."""

    # human-readable name shown in pipeline selector
    name: str = "Pipeline"

    # whether this pipeline's dependencies are installed
    is_available: bool = False

    # install command to show when not available
    install_command: str = "uv pip install mbo_utilities"

    def __init__(self, parent: Any):
        self.parent = parent

    def draw(self) -> None:
        """Draw the pipeline widget."""
        self.draw_config()

    @abstractmethod
    def draw_config(self) -> None:
        """Draw the configuration/processing ui."""
        ...

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """
