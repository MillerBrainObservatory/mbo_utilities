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

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        """True iff this pipeline can be run against ``arr``.

        Called by the Run-tab selector BEFORE instantiation, so it
        must be safe to call without spinning up the widget. Override
        for pipelines tied to a specific array type (e.g. Isoview
        consolidator only applies to ``IsoviewArray`` instances).

        ``arr`` may be ``None`` when no data is loaded.

        Default: returns ``True`` (pipeline works on any data).
        """
        return True

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """
