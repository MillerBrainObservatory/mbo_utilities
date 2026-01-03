"""
Stats panel for z-stats visualization.

Displays mean, std, and SNR statistics across z-planes with ImPlot charts.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.panels import BasePanel

__all__ = ["StatsPanel"]


class StatsPanel(BasePanel):
    """
    Panel for z-stats visualization.

    Computes and displays mean, std, and SNR statistics across z-planes
    with interactive ImPlot charts.

    Attributes
    ----------
    name : str
        Panel name shown in UI.
    """

    name = "Z-Stats"

    def __init__(self, viewer: Any):
        super().__init__(viewer)
        self._visible = True  # Stats section is visible by default

        # Z-stats state - will be initialized when data is available
        self._zstats: list[dict[str, list[float]]] = []
        self._zstats_means: list[Any] = []
        self._zstats_mean_scalar: list[float] = []
        self._zstats_done: list[bool] = []
        self._zstats_running: list[bool] = []
        self._zstats_progress: list[float] = []
        self._zstats_current_z: list[int] = []
        self._selected_array: int = 0

    def _init_stats_state(self, n: int) -> None:
        """Initialize z-stats state for n arrays."""
        self._zstats = [{"mean": [], "std": [], "snr": []} for _ in range(n)]
        self._zstats_means = [None] * n
        self._zstats_mean_scalar = [0.0] * n
        self._zstats_done = [False] * n
        self._zstats_running = [False] * n
        self._zstats_progress = [0.0] * n
        self._zstats_current_z = [0] * n

    @property
    def num_graphics(self) -> int:
        """Get number of graphics/arrays from viewer."""
        return getattr(self.viewer, "num_graphics", 1)

    @property
    def nz(self) -> int:
        """Get number of z-planes from viewer."""
        return getattr(self.viewer, "nz", 1)

    @property
    def shape(self) -> tuple:
        """Get data shape from viewer."""
        return getattr(self.viewer, "shape", ())

    @property
    def image_widget(self) -> Any:
        """Get image widget from viewer."""
        return getattr(self.viewer, "image_widget", None)

    @property
    def logger(self) -> Any:
        """Get logger from viewer."""
        return getattr(self.viewer, "logger", None)

    def compute(self) -> None:
        """Compute z-stats for all arrays."""
        from mbo_utilities.gui._stats import compute_zstats

        # Initialize state if needed
        n = self.num_graphics
        if len(self._zstats) != n:
            self._init_stats_state(n)

        # Mark all as running
        for i in range(n):
            self._zstats_running[i] = True

        # Compute using existing function - pass self as the "parent"
        # since we have all the required attributes
        compute_zstats(self)

    def refresh(self) -> None:
        """Reset and recompute z-stats."""
        from mbo_utilities.gui._stats import refresh_zstats
        refresh_zstats(self)

    def draw(self) -> None:
        """Draw the z-stats visualization section."""
        if not self._visible:
            return

        # Check if any stats are computed
        if not self._zstats or not any(self._zstats_done):
            # Show compute button if data is available
            if self.image_widget and getattr(self.image_widget, "data", None):
                if imgui.button("Compute Z-Stats"):
                    self.compute()
                imgui.same_line()
                imgui.text_disabled("Click to compute statistics")
            return

        # Draw using existing function
        from mbo_utilities.gui._stats import draw_stats_section
        draw_stats_section(self)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._zstats.clear()
        self._zstats_means.clear()
        self._zstats_mean_scalar.clear()
        self._zstats_done.clear()
        self._zstats_running.clear()
        self._zstats_progress.clear()
        self._zstats_current_z.clear()
