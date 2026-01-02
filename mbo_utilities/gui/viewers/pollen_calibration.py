"""
Pollen calibration viewer for LBM beamlet calibration.

This viewer handles pollen calibration data (ZCYX) and provides:
- Info panel showing beamlet count, cavities, z-step, pixel size
- Automatic background calibration (detection + analysis)
- Manual calibration mode: click through beamlets in the viewer

NOTE: This is the new architecture. The main logic is still in
main_widgets/pollen_calibration.py. This viewer provides the
new BaseViewer interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseViewer
from ..panels import DebugPanel, ProcessPanel, MetadataPanel

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["PollenCalibrationViewer"]


class PollenCalibrationViewer(BaseViewer):
    """
    Viewer for pollen calibration data (ZCYX).

    This viewer is specialized for LBM beamlet calibration using
    pollen grain stacks. It provides:
    - Automatic bead detection and calibration
    - Interactive manual calibration mode
    - Results visualization and comparison

    The heavy lifting is done in main_widgets/pollen_calibration.py
    for now - this viewer provides the new interface structure.
    """

    name = "Pollen Calibration Viewer"

    def __init__(
        self,
        image_widget: "ImageWidget",
        fpath: str | list[str],
        **kwargs,
    ):
        super().__init__(image_widget, fpath, **kwargs)

        # Initialize panels
        self._panels["debug"] = DebugPanel(self)
        self._panels["processes"] = ProcessPanel(self)
        self._panels["metadata"] = MetadataPanel(self)

        # Pollen calibration has its own specialized UI
        # so we don't use the generic feature system
        self._features = []

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up log handler to route to debug panel."""
        try:
            import logging
            from ..panels.debug_log import GuiLogHandler
            handler = GuiLogHandler(self._panels["debug"])
            logging.getLogger("mbo_utilities").addHandler(handler)
        except Exception:
            pass

    def draw(self) -> None:
        """Main render callback."""
        self.draw_menu_bar()

        # Pollen calibration has its own specialized UI
        # The actual implementation is delegated to the PollenCalibrationWidget
        # in main_widgets/pollen_calibration.py for now

        imgui.text("Pollen Calibration Viewer")
        imgui.text("(Implementation in progress)")

        # Draw visible panels
        for panel in self._panels.values():
            panel.draw()

    def draw_menu_bar(self) -> None:
        """Render the menu bar."""
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Open File", "Ctrl+O")[0]:
                    pass
                imgui.end_menu()

            if imgui.begin_menu("View"):
                if imgui.menu_item("Metadata", "M")[0]:
                    self._panels["metadata"].toggle()
                if imgui.menu_item("Debug Log")[0]:
                    self._panels["debug"].toggle()
                imgui.end_menu()

            if imgui.begin_menu("Help"):
                if imgui.menu_item("Documentation")[0]:
                    import webbrowser
                    webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/")
                imgui.end_menu()

            imgui.end_menu_bar()

    def on_data_loaded(self) -> None:
        """Initialize calibration when data loads."""
        pass

    def cleanup(self) -> None:
        """Clean up resources when viewer closes."""
        super().cleanup()
