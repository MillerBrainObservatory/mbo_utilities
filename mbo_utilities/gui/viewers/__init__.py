"""
Viewer classes - standalone GUI applications.

A Viewer is a complete GUI window that:
- Manages its own data and state
- Contains Panels (reusable UI sections)
- Contains Features (capability-based controls)
- Uses Widgets (generic UI building blocks)
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from imgui_bundle import imgui

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = [
    "BaseViewer",
    "get_viewer_class",
    "TimeSeriesViewer",
    "PollenCalibrationViewer",
    "DiagnosticsViewer",
    "GridSearchViewer",
    "Suite2pResultsViewer",
]


class BaseViewer(ABC):
    """
    Base class for all viewer applications.

    A Viewer is a standalone GUI window that:
    - Manages its own data and state
    - Contains Panels (reusable UI sections)
    - Contains Features (capability-based controls)
    - Uses Widgets (generic UI building blocks)

    Attributes
    ----------
    name : str
        Human-readable name for this viewer type.
    image_widget : ImageWidget
        The fastplotlib ImageWidget for display.
    fpath : str | list[str]
        Path(s) to the loaded data file(s).

    Notes
    -----
    Subclasses must implement:
    - draw(): Main render callback
    """

    name: str = "Base Viewer"

    def __init__(
        self,
        image_widget: "ImageWidget",
        fpath: str | list[str],
        **kwargs,
    ):
        """
        Initialize the viewer.

        Parameters
        ----------
        image_widget : ImageWidget
            The fastplotlib ImageWidget for display.
        fpath : str | list[str]
            Path(s) to the loaded data file(s).
        **kwargs
            Additional keyword arguments for subclasses.
        """
        self.image_widget = image_widget
        self.fpath = fpath
        self._panels: dict = {}
        self._features: list = []
        self._kwargs = kwargs

    @property
    def data(self):
        """Access the loaded data arrays."""
        if self.image_widget is None:
            return None
        return self.image_widget.data

    def _get_data_arrays(self) -> list:
        """Get the loaded data arrays as a list."""
        if self.image_widget is None or self.image_widget.data is None:
            return []
        return list(self.image_widget.data)

    @abstractmethod
    def draw(self) -> None:
        """Main render callback. Must be implemented by subclasses."""
        ...

    def draw_menu_bar(self) -> None:
        """
        Render the menu bar.

        Override to add viewer-specific menus. Base implementation provides
        common File/View/Help menus.
        """
        pass

    def on_data_loaded(self) -> None:
        """
        Called when new data is loaded.

        Override to perform viewer-specific initialization after data loads.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up resources when the viewer closes.

        Override to release threads, close windows, etc.
        """
        for feature in self._features:
            if hasattr(feature, "cleanup"):
                feature.cleanup()
        for panel in self._panels.values():
            if hasattr(panel, "cleanup"):
                panel.cleanup()


def get_viewer_class(data_array) -> type[BaseViewer]:
    """
    Select the appropriate viewer class based on data type.

    Parameters
    ----------
    data_array : array-like
        The data array to display.

    Returns
    -------
    type[BaseViewer]
        The viewer class to use.
    """
    # Import here to avoid circular imports
    from .time_series import TimeSeriesViewer

    # Check for pollen calibration data
    if hasattr(data_array, "metadata"):
        meta = data_array.metadata
        if hasattr(meta, "get"):
            exp_type = meta.get("experiment_type", "")
            if exp_type == "pollen_calibration":
                from .pollen_calibration import PollenCalibrationViewer
                return PollenCalibrationViewer

    # Default to time-series
    return TimeSeriesViewer


# Lazy imports for viewer classes
def __getattr__(name: str):
    if name == "TimeSeriesViewer":
        from .time_series import TimeSeriesViewer
        return TimeSeriesViewer
    if name == "PollenCalibrationViewer":
        from .pollen_calibration import PollenCalibrationViewer
        return PollenCalibrationViewer
    if name == "DiagnosticsViewer":
        from .diagnostics import DiagnosticsViewer
        return DiagnosticsViewer
    if name == "GridSearchViewer":
        from .grid_search import GridSearchViewer
        return GridSearchViewer
    if name == "Suite2pResultsViewer":
        from .suite2p_results import Suite2pResultsViewer
        return Suite2pResultsViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
