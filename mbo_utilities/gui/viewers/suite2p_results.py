"""
Suite2p results viewer.

This viewer provides visualization of Suite2p processing results
including cell traces and ROI display.

NOTE: This wraps the existing Suite2pResultsViewer functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["Suite2pResultsViewer"]


class Suite2pResultsViewer(BaseViewer):
    """
    Viewer for Suite2p processing results.

    This viewer provides:
    - Cell trace visualization
    - ROI selection and display
    - Results navigation

    NOTE: The full implementation is in suite2p_results.py.
    This class provides the new BaseViewer interface.
    """

    name = "Suite2p Results Viewer"

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
        imgui.text("Suite2p Results Viewer")
        imgui.text("See suite2p_results.py for full implementation")

    def on_data_loaded(self) -> None:
        """Handle data loading."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
