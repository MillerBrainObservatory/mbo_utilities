"""
Viewer classes - standalone GUI applications.

A Viewer is a complete GUI window that:
- Manages its own data and state
- Contains Panels (reusable UI sections)
- Contains Features (capability-based controls)
- Uses Widgets (generic UI building blocks)

Integration with Legacy Code
----------------------------
The new viewer classes integrate with the existing PreviewDataWidget through
a delegation pattern. The PreviewDataWidget creates a viewer instance and
delegates rendering to it. This allows gradual migration of functionality.

To use new viewers from PreviewDataWidget:
1. PreviewDataWidget.update() calls viewer.draw() instead of draw_tabs()
2. Viewer.draw() delegates to existing main_widget.draw() for now
3. Gradually migrate logic from main_widgets into viewers
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
    "TimeSeriesViewer",
    "get_viewer_class",
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
    parent : PreviewDataWidget | None
        Optional parent widget for legacy integration.

    Notes
    -----
    Subclasses must implement:
    - draw(): Main render callback
    """

    name: str = "Base Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
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
        parent : PreviewDataWidget, optional
            Parent widget for legacy integration. When set, the viewer
            can delegate to existing main_widget implementations.
        **kwargs
            Additional keyword arguments for subclasses.
        """
        self.image_widget = image_widget
        self.fpath = fpath
        self.parent = parent
        self._panels: dict = {}
        self._features: list = []
        self._kwargs = kwargs

        # Legacy main_widget for delegation during migration
        self._main_widget = None

    @property
    def data(self):
        """Access the loaded data arrays."""
        if self.image_widget is None:
            return None
        return self.image_widget.data

    @property
    def logger(self):
        """Access the logger (from parent if available)."""
        if self.parent is not None and hasattr(self.parent, "logger"):
            return self.parent.logger
        # Fallback to module logger
        import logging
        return logging.getLogger("mbo_utilities.gui")

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

    def on_data_loaded(self) -> None:
        """
        Called when new data is loaded.

        Override to perform viewer-specific initialization after data loads.
        """
        # If using legacy main_widget, delegate
        if self._main_widget is not None:
            self._main_widget.on_data_loaded()

    def _on_data_shape_changed(self) -> None:
        """
        Called when the data shape changes dynamically.

        This can happen when toggling frame averaging on piezo stacks.
        Subclasses should override to update sliders or other UI elements.
        """
        if self.image_widget is None:
            return

        try:
            data_arrays = self._get_data_arrays()
            if not data_arrays:
                return

            arr = data_arrays[0]
            new_shape = arr.shape
            n_dims = len(new_shape) - 2  # exclude Y, X

            # clamp current indices to new valid range
            if self.image_widget.slider_dims:
                current = dict(self.image_widget.current_index)
                for dim in list(current.keys()):
                    # find the axis for this dim in the data shape
                    idx = self.image_widget.slider_dims.index(dim) if dim in self.image_widget.slider_dims else None
                    if idx is not None and idx < n_dims:
                        max_val = new_shape[idx] - 1
                        if current[dim] > max_val:
                            current[dim] = max_val
                self.image_widget.current_index = current
        except Exception:
            pass

    def cleanup(self) -> None:
        """
        Clean up resources when the viewer closes.

        Override to release threads, close windows, etc.
        """
        # Clean up legacy main_widget
        if self._main_widget is not None:
            self._main_widget.cleanup()

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

    Notes
    -----
    Detection logic must match main_widgets.get_main_widget_class() to ensure
    consistency during migration.
    """
    # Import here to avoid circular imports
    from .time_series import TimeSeriesViewer

    # Check for pollen calibration data (matches main_widgets detection)
    if hasattr(data_array, "stack_type") and data_array.stack_type == "pollen":
        from .pollen_calibration import PollenCalibrationViewer
        return PollenCalibrationViewer

    # Fallback: also check metadata for experiment_type
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
