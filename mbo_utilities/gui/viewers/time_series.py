"""
Time series viewer for standard calcium imaging data.

This viewer handles time-series volumetric data (TZYX) and provides:
- Preview tab with window functions, spatial functions, phase correction
- Signal Quality tab with z-stats plots
- Run tab for Suite2p/processing pipelines

NOTE: This is the new architecture. The current implementation still uses
PreviewDataWidget from imgui.py. This viewer will gradually take over
functionality as we migrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseViewer
from ..features import get_supported_features, draw_all_features, cleanup_all_features
from ..panels import DebugPanel, ProcessPanel, MetadataPanel

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["TimeSeriesViewer"]


class TimeSeriesViewer(BaseViewer):
    """
    Viewer for time-series calcium imaging data (TZYX).

    This is the default viewer for standard imaging data. It provides:
    - Temporal projections (mean/max/std over sliding window)
    - Spatial filtering (gaussian blur, mean subtraction)
    - Raster scan phase correction (for bidirectional scanning)
    - Frame averaging (for piezo stacks)
    - Suite2p pipeline integration

    Attributes
    ----------
    name : str
        Display name for this viewer type.
    proj : str
        Current projection type ("mean", "max", "std").
    window_size : int
        Temporal window size in frames.
    gaussian_sigma : float
        Gaussian blur sigma (0 = disabled).
    mean_subtraction : bool
        Whether mean subtraction is enabled.
    fix_phase : bool
        Whether raster scan phase correction is enabled.
    use_fft : bool
        Whether to use FFT for subpixel phase correction.
    """

    name = "Time Series Viewer"

    def __init__(
        self,
        image_widget: "ImageWidget",
        fpath: str | list[str],
        **kwargs,
    ):
        super().__init__(image_widget, fpath, **kwargs)

        # Display state
        self._proj = "mean"
        self._window_size = 10
        self._gaussian_sigma = 0.0
        self._mean_subtraction = False

        # Phase correction state
        self._fix_phase = False
        self._use_fft = True
        self._border = 10
        self._max_offset = 10
        self._phase_upsample = 10
        self._current_offset = []

        # Z-stats tracking
        self._zstats = []
        self._zstats_done = []
        self._zstats_running = []
        self._zstats_progress = []

        # Capability cache
        self._has_raster_scan_support = None

        # Initialize panels
        self._panels["debug"] = DebugPanel(self)
        self._panels["processes"] = ProcessPanel(self)
        self._panels["metadata"] = MetadataPanel(self)

        # Discover and initialize features
        self._features = get_supported_features(self)

        # Setup logging handler
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

    # === Properties for display state ===

    @property
    def proj(self) -> str:
        return self._proj

    @proj.setter
    def proj(self, value: str) -> None:
        self._proj = value
        self._refresh_image_widget()

    @property
    def window_size(self) -> int:
        return self._window_size

    @window_size.setter
    def window_size(self, value: int) -> None:
        self._window_size = max(1, value)
        self._refresh_image_widget()

    @property
    def gaussian_sigma(self) -> float:
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float) -> None:
        self._gaussian_sigma = max(0.0, value)
        self._refresh_image_widget()

    @property
    def mean_subtraction(self) -> bool:
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool) -> None:
        self._mean_subtraction = value
        self._refresh_image_widget()

    # === Properties for phase correction ===

    @property
    def fix_phase(self) -> bool:
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool) -> None:
        self._fix_phase = value
        self._apply_phase_correction()

    @property
    def use_fft(self) -> bool:
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value: bool) -> None:
        self._use_fft = value
        self._apply_phase_correction()

    @property
    def border(self) -> int:
        return self._border

    @border.setter
    def border(self, value: int) -> None:
        self._border = max(0, value)

    @property
    def max_offset(self) -> int:
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value: int) -> None:
        self._max_offset = max(1, value)

    @property
    def phase_upsample(self) -> int:
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value: int) -> None:
        self._phase_upsample = max(1, value)

    @property
    def current_offset(self) -> list:
        return self._current_offset

    @property
    def has_raster_scan_support(self) -> bool:
        """Check if any loaded array supports raster scan correction."""
        if self._has_raster_scan_support is None:
            from .._protocols import supports_raster_scan
            arrays = self._get_data_arrays()
            self._has_raster_scan_support = any(supports_raster_scan(arr) for arr in arrays)
        return self._has_raster_scan_support

    # === Rendering ===

    def draw(self) -> None:
        """Main render callback."""
        self.draw_menu_bar()

        # Draw features (window functions, raster scan, etc.)
        draw_all_features(self, self._features)

        # Draw visible panels
        for panel in self._panels.values():
            panel.draw()

    def draw_menu_bar(self) -> None:
        """Render the menu bar."""
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Open File", "Ctrl+O")[0]:
                    pass  # TODO: file open dialog
                if imgui.menu_item("Save As...", "Ctrl+S")[0]:
                    pass  # TODO: save as dialog
                imgui.end_menu()

            if imgui.begin_menu("View"):
                if imgui.menu_item("Metadata", "M")[0]:
                    self._panels["metadata"].toggle()
                if imgui.menu_item("Debug Log")[0]:
                    self._panels["debug"].toggle()
                if imgui.menu_item("Processes")[0]:
                    self._panels["processes"].toggle()
                imgui.end_menu()

            if imgui.begin_menu("Help"):
                if imgui.menu_item("Documentation")[0]:
                    import webbrowser
                    webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/")
                imgui.end_menu()

            imgui.end_menu_bar()

    # === Internal methods ===

    def _refresh_image_widget(self) -> None:
        """Refresh the display after settings change."""
        if self.image_widget is None:
            return

        # Apply window functions to image widget
        # This depends on fastplotlib ImageWidget API
        try:
            # The actual implementation will depend on how ImageWidget handles
            # window_funcs and spatial_func
            pass
        except Exception:
            pass

    def _apply_phase_correction(self) -> None:
        """Apply phase correction settings to data arrays."""
        arrays = self._get_data_arrays()
        self._current_offset = []

        for arr in arrays:
            if hasattr(arr, "fix_phase"):
                arr.fix_phase = self._fix_phase
            if hasattr(arr, "use_fft"):
                arr.use_fft = self._use_fft
            if hasattr(arr, "border"):
                arr.border = self._border
            if hasattr(arr, "phase_upsample"):
                arr.phase_upsample = self._phase_upsample

            # Get current offset
            if hasattr(arr, "offset"):
                self._current_offset.append(arr.offset)

        self._refresh_image_widget()

    def on_data_loaded(self) -> None:
        """Initialize time-series specific features when data loads."""
        # Reset capability cache
        self._has_raster_scan_support = None

        # Reinitialize features for new data
        cleanup_all_features(self._features)
        self._features = get_supported_features(self)

        # Initialize z-stats tracking
        num_graphics = len(self._get_data_arrays())
        self._zstats = [None] * num_graphics
        self._zstats_done = [False] * num_graphics
        self._zstats_running = [False] * num_graphics
        self._zstats_progress = [0.0] * num_graphics

        # TODO: Start z-stats computation in background

    def cleanup(self) -> None:
        """Clean up resources when viewer closes."""
        cleanup_all_features(self._features)
        super().cleanup()
