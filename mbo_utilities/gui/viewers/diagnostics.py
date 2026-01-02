"""
Diagnostics viewer for ROI filtering and visualization.

This viewer provides Suite2p results analysis with:
- ROI filtering based on metrics
- dF/F trace visualization
- Histogram visualization with threshold sliders
- Bidirectional sync with Suite2p GUI

NOTE: This wraps the existing DiagnosticsWidget functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["DiagnosticsViewer"]


class DiagnosticsViewer(BaseViewer):
    """
    Viewer for Suite2p ROI diagnostics and filtering.

    This viewer is a standalone application for analyzing Suite2p
    results. It provides:
    - ROI filtering by quality metrics
    - dF/F trace visualization
    - Histogram views with interactive thresholds
    - File watching for bidirectional sync with Suite2p GUI

    NOTE: The full implementation is in diagnostics_widget.py.
    This class provides the new BaseViewer interface.
    """

    name = "Diagnostics Viewer"

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
        imgui.text("Diagnostics Viewer")
        imgui.text("See diagnostics_widget.py for full implementation")

    def on_data_loaded(self) -> None:
        """Handle data loading."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
