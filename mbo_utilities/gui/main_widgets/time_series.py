"""
Time series widget for standard imaging data.

This widget handles time-series volumetric data (TZYX) and provides:
- Preview tab with window functions, spatial functions, phase correction
- Signal Quality tab with z-stats plots
- Run tab for Suite2p/processing pipelines
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from mbo_utilities.gui.main_widgets._base import MainWidget

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


class TimeSeriesWidget(MainWidget):
    """
    Main widget for time-series imaging data.

    This is the default widget for TZYX data. It wraps the existing
    PreviewDataWidget functionality, which will be gradually migrated here.

    For now, this is a thin wrapper that delegates to the parent's methods.
    """

    name = "Time Series"

    def __init__(self, parent: "PreviewDataWidget"):
        super().__init__(parent)

    def draw(self) -> None:
        """
        Draw the time series viewer UI.

        Delegates to parent's existing tab drawing logic.
        """
        # The parent already handles drawing via draw_tabs() in update()
        # This method is for future use when we fully migrate the logic here
        pass

    def draw_tabs(self) -> None:
        """
        Draw the tabbed interface for time series data.

        Tabs:
        - Preview: Window functions, phase correction, etc.
        - Signal Quality: Z-stats plots
        - Run: Processing pipelines
        """
        # Currently handled by draw_tabs() function in imgui.py
        # Will be migrated here in future refactor
        pass

    def on_data_loaded(self) -> None:
        """Initialize time series specific features when data loads."""
        # Trigger z-stats computation
        if hasattr(self.parent, 'refresh_zstats'):
            self.parent.refresh_zstats()

    def cleanup(self) -> None:
        """Clean up time series widget resources."""
        # Cleanup is handled by parent's cleanup() for now
        pass
