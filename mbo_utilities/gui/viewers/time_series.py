"""
Time series viewer for standard calcium imaging data.

This viewer handles time-series volumetric data (TZYX) and provides:
- Preview tab with window functions, spatial functions, phase correction
- Signal Quality tab with z-stats plots
- Run tab for Suite2p/processing pipelines

Integration Pattern
-------------------
During migration, this viewer delegates to the existing PreviewDataWidget
and TimeSeriesWidget implementations. The delegation works as follows:

1. TimeSeriesViewer.draw() is called by PreviewDataWidget.update()
2. If parent is set, draw() delegates to the existing draw_tabs() function
3. If no parent, draw() uses direct rendering

This allows gradual migration of functionality from imgui.py into this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from . import BaseViewer
from mbo_utilities.gui.panels import (
    DebugPanel,
    MetadataPanel,
    PipelinePanel,
    ProcessPanel,
    StatsPanel,
)
import contextlib

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
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)

        # Display state (will be synced with parent if available)
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
        self._has_pipeline = None

        # Only initialize panels when not using legacy delegation
        if parent is None:
            # start pipeline preload early (before user clicks Run tab)
            from mbo_utilities.gui.widgets.pipelines import start_preload
            start_preload()

            self._panels["debug"] = DebugPanel(self)
            self._panels["processes"] = ProcessPanel(self)
            self._panels["metadata"] = MetadataPanel(self)
            self._panels["stats"] = StatsPanel(self)
            self._panels["pipeline"] = PipelinePanel(self)
            self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up log handler to route to debug panel."""
        try:
            import logging
            from mbo_utilities.gui.panels.debug_log import GuiLogHandler
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
            from mbo_utilities.gui._protocols import supports_raster_scan
            arrays = self._get_data_arrays()
            self._has_raster_scan_support = any(supports_raster_scan(arr) for arr in arrays)
        return self._has_raster_scan_support

    def draw(self) -> None:
        """
        Main render callback.

        If a parent PreviewDataWidget is set, this delegates to the existing
        draw_tabs() function for backwards compatibility. Otherwise, it uses
        direct rendering.
        """
        if self.parent is not None:
            # Delegate to legacy draw_tabs() function
            self._draw_legacy()
        else:
            # Use new direct rendering
            self._draw_new()

    def _draw_legacy(self) -> None:
        """
        Draw using the legacy draw_tabs() function.

        This maintains backwards compatibility during migration.
        """
        # Standard time-series tabs
        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    self.parent.draw_preview_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()

            # Signal Quality tab - disabled until zstats are computed
            imgui.begin_disabled(not all(self.parent._zstats_done))
            if imgui.begin_tab_item("Signal Quality")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                        self.parent.draw_stats_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            imgui.end_disabled()

            # Run tab for processing pipelines
            if self._has_pipeline is None:
                from mbo_utilities.gui.widgets.pipelines import any_pipeline_available
                self._has_pipeline = any_pipeline_available()
            if not self._has_pipeline:
                imgui.begin_disabled()
            if imgui.begin_tab_item("Run")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    from mbo_utilities.gui.widgets.pipelines import draw_run_tab
                    draw_run_tab(self.parent)
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            if not self._has_pipeline:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        "Suite2p not installed.\n"
                        'Install with: uv pip install "mbo_utilities[suite2p]"'
                    )
            imgui.end_tab_bar()

    def _draw_new(self) -> None:
        """
        Draw using direct rendering.

        This is the target architecture for fully-migrated viewers.
        """
        from mbo_utilities.gui.widgets import (
            get_supported_widgets,
            draw_all_widgets,
        )

        self.draw_menu_bar()

        # Main tab bar
        if imgui.begin_tab_bar("MainPreviewTabs"):
            # Preview tab - core preview widgets
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

                # Get and draw supported widgets for this viewer
                if not hasattr(self, "_widgets") or self._widgets is None:
                    self._widgets = get_supported_widgets(self)
                draw_all_widgets(self, self._widgets)

                imgui.pop_style_var(2)
                imgui.end_tab_item()

            # Signal Quality tab - z-stats visualization
            stats_panel = self._panels.get("stats")
            stats_computed = stats_panel and any(getattr(stats_panel, "_zstats_done", []))
            imgui.begin_disabled(not stats_computed)
            if imgui.begin_tab_item("Signal Quality")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

                with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                    if stats_panel:
                        stats_panel.draw()

                imgui.pop_style_var(2)
                imgui.end_tab_item()
            imgui.end_disabled()

            # Run tab - processing pipelines
            if self._has_pipeline is None:
                from mbo_utilities.gui.widgets.pipelines import any_pipeline_available
                self._has_pipeline = any_pipeline_available()
            if not self._has_pipeline:
                imgui.begin_disabled()
            if imgui.begin_tab_item("Run")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

                pipeline_panel = self._panels.get("pipeline")
                if pipeline_panel:
                    pipeline_panel.draw()

                imgui.pop_style_var(2)
                imgui.end_tab_item()
            if not self._has_pipeline:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        "Suite2p not installed.\n"
                        'Install with: uv pip install "mbo_utilities[suite2p]"'
                    )

            imgui.end_tab_bar()

        # Draw popup panels (metadata viewer, debug log, process console)
        self._panels["metadata"].draw()
        self._panels["debug"].draw()
        self._panels["processes"].draw()

    def draw_menu_bar(self) -> None:
        """Render the menu bar."""
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Open File", "Ctrl+O")[0]:
                    pass
                if imgui.menu_item("Save As...", "Ctrl+S")[0]:
                    pass
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

        with contextlib.suppress(Exception):
            pass

    def _apply_phase_correction(self) -> None:
        """Apply phase correction settings to data arrays."""
        arrays = self._get_data_arrays()

        for arr in arrays:
            if hasattr(arr, "fix_phase"):
                arr.fix_phase = self._fix_phase
            if hasattr(arr, "use_fft"):
                arr.use_fft = self._use_fft
            if hasattr(arr, "border"):
                arr.border = self._border
            if hasattr(arr, "phase_upsample"):
                arr.phase_upsample = self._phase_upsample

        self._refresh_image_widget()

        # read offsets after refresh so computed values are available
        self._current_offset = []
        for arr in arrays:
            if hasattr(arr, "offset"):
                self._current_offset.append(arr.offset)

    def on_data_loaded(self) -> None:
        """Initialize time-series specific features when data loads."""
        # Reset capability cache
        self._has_raster_scan_support = None

        # Initialize z-stats tracking
        num_graphics = len(self._get_data_arrays())
        self._zstats = [None] * num_graphics
        self._zstats_done = [False] * num_graphics
        self._zstats_running = [False] * num_graphics
        self._zstats_progress = [0.0] * num_graphics

    def cleanup(self) -> None:
        """Clean up resources when viewer closes."""
        # Clean up widgets if using new architecture
        if hasattr(self, "_widgets") and self._widgets:
            from mbo_utilities.gui.widgets import cleanup_all_widgets
            cleanup_all_widgets(self._widgets)
            self._widgets = None

        super().cleanup()
