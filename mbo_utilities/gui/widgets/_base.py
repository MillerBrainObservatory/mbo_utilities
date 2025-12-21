"""
base class for ui widgets that control image display.

widgets are self-contained ui components that:
- check if they should display based on data/parent attributes
- draw imgui controls
- modify image_widget.data or window funcs directly

no processors needed - all manipulation happens via data attributes
or imagewidget's built-in window_funcs/spatial_func apis.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


class Widget(ABC):
    """base class for ui widgets."""

    # human-readable name
    name: str = "Widget"

    # priority for ordering (lower = rendered first)
    priority: int = 100

    def __init__(self, parent: "PreviewDataWidget"):
        self.parent = parent

    @classmethod
    @abstractmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """
        check if this widget should show for the given parent.

        override to check data metadata, attributes, etc.
        """
        ...

    @abstractmethod
    def draw(self) -> None:
        """draw the imgui ui for this widget."""
        ...

    def cleanup(self) -> None:
        """clean up resources when widget is destroyed.

        override in subclasses to release resources like open windows,
        background threads, file handles, etc.
        """
        pass
