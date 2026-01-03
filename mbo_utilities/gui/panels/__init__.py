"""
Panel classes - reusable UI sections.

A Panel is a UI section that:
- Can be shown/hidden
- Is reusable across different viewers
- Manages its own state
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.gui.viewers import BaseViewer

__all__ = [
    "BasePanel",
    "DebugPanel",
    "MetadataPanel",
    "PipelinePanel",
    "ProcessPanel",
    "StatsPanel",
]


class BasePanel(ABC):
    """
    Base class for reusable UI panels.

    A Panel is a UI section that:
    - Can be shown/hidden
    - Is reusable across different viewers
    - Manages its own state

    Attributes
    ----------
    name : str
        Human-readable name for this panel.
    viewer : BaseViewer
        The parent viewer containing this panel.
    """

    name: str = "Base Panel"

    def __init__(self, viewer: BaseViewer):
        """
        Initialize the panel.

        Parameters
        ----------
        viewer : BaseViewer
            The parent viewer containing this panel.
        """
        self.viewer = viewer
        self._visible = False

    @property
    def visible(self) -> bool:
        """Whether the panel is currently visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value

    def show(self) -> None:
        """Show the panel."""
        self._visible = True

    def hide(self) -> None:
        """Hide the panel."""
        self._visible = False

    def toggle(self) -> None:
        """Toggle the panel visibility."""
        self._visible = not self._visible

    @abstractmethod
    def draw(self) -> None:
        """
        Render the panel UI.

        This is called every frame. Implementation should check
        self._visible and skip rendering if not visible.
        """
        ...

    def cleanup(self) -> None:
        """
        Clean up resources when the panel is destroyed.

        Override in subclasses to release resources.
        """


# Lazy imports for panel classes
def __getattr__(name: str):
    if name == "DebugPanel":
        from .debug_log import DebugPanel
        return DebugPanel
    if name == "ProcessPanel":
        from .process_manager import ProcessPanel
        return ProcessPanel
    if name == "MetadataPanel":
        from .metadata import MetadataPanel
        return MetadataPanel
    if name == "PipelinePanel":
        from .pipeline import PipelinePanel
        return PipelinePanel
    if name == "StatsPanel":
        from .stats import StatsPanel
        return StatsPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
