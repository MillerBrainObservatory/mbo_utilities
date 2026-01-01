"""
Base class for main widgets.

Main widgets are the primary content area of the data viewer GUI.
They render into the imgui sidebar and can interact with the fastplotlib viewer.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


class MainWidget(ABC):
    """
    Abstract base class for main viewer widgets.

    Main widgets handle the primary data visualization and interaction
    for a specific type of data. They are instantiated by the DataViewerGUI
    based on the loaded array type.

    Attributes
    ----------
    parent : PreviewDataWidget
        The parent GUI that contains this widget.
    name : str
        Human-readable name for this widget type.

    Notes
    -----
    Subclasses must implement:
    - draw(): Render the imgui UI
    - draw_viewer(): Configure the fastplotlib viewer
    - cleanup(): Release resources when closing
    """

    name: str = "MainWidget"

    def __init__(self, parent: "PreviewDataWidget"):
        """
        Initialize the main widget.

        Parameters
        ----------
        parent : PreviewDataWidget
            The parent GUI containing this widget.
        """
        self.parent = parent

    @property
    def image_widget(self):
        """Access the fastplotlib ImageWidget."""
        return self.parent.image_widget

    @property
    def data(self):
        """Access the loaded data arrays."""
        return self.parent.image_widget.data if self.parent.image_widget else None

    @property
    def logger(self):
        """Access the GUI logger."""
        return self.parent.logger

    @abstractmethod
    def draw(self) -> None:
        """
        Draw the imgui UI for this widget.

        This is called every frame to render the sidebar controls.
        """
        ...

    def draw_menu_items(self) -> None:
        """
        Draw additional menu items specific to this widget.

        Override to add widget-specific menu items to File/View/Help menus.
        Called from the parent GUI's menu bar rendering.
        """
        pass

    def draw_tabs(self) -> None:
        """
        Draw tab bar content for this widget.

        Override to provide tabbed interface. Default is no tabs.
        """
        pass

    def on_data_loaded(self) -> None:
        """
        Called when new data is loaded.

        Override to perform widget-specific initialization after data loads.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up resources when the widget is destroyed.

        Override to release threads, close windows, etc.
        """
        pass
