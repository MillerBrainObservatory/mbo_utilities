"""
Grid search viewer for Suite2p parameter comparison.

This viewer provides side-by-side comparison of Suite2p runs
with different parameter combinations.

NOTE: This wraps the existing GridSearchViewer functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["GridSearchViewer"]


class GridSearchViewer(BaseViewer):
    """
    Viewer for comparing Suite2p parameter grid search results.

    This viewer provides:
    - Side-by-side comparison of different parameter combinations
    - Statistics table for each combination
    - Launch external Suite2p viewer for detailed inspection

    NOTE: The full implementation is in grid_search_viewer.py.
    This class provides the new BaseViewer interface.
    """

    name = "Grid Search Viewer"

    def __init__(
        self,
        image_widget: "ImageWidget" = None,
        fpath: str | list[str] = None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, **kwargs)
        self._features = []

    def draw(self) -> None:
        """Main render callback."""
        imgui.text("Grid Search Viewer")
        imgui.text("See grid_search_viewer.py for full implementation")

    def on_data_loaded(self) -> None:
        """Handle data loading."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
