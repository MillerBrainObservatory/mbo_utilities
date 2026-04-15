"""Viewer classes - standalone GUI applications.

A Viewer is a complete GUI window embedded inside PreviewDataWidget.
It owns the tab bar and delegates the actual rendering to the parent
widget's draw methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = [
    "BaseViewer",
    "TimeSeriesViewer",
    "get_viewer_class",
]


class BaseViewer(ABC):
    """Base class for all viewer applications."""

    name: str = "Base Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        self.image_widget = image_widget
        self.fpath = fpath
        self.parent = parent

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
        import logging
        return logging.getLogger("mbo_utilities.gui")

    @abstractmethod
    def draw(self) -> None:
        """Main render callback. Must be implemented by subclasses."""
        ...

    def on_data_loaded(self) -> None:
        """Called by _dialogs when new data is loaded. Override as needed."""

    def cleanup(self) -> None:
        """Clean up resources when the viewer closes. Override as needed."""


def get_viewer_class(data_array) -> type[BaseViewer]:
    """Select the appropriate viewer class based on data type."""
    from .time_series import TimeSeriesViewer

    if hasattr(data_array, "stack_type") and data_array.stack_type == "pollen":
        from .pollen_calibration import PollenCalibrationViewer
        return PollenCalibrationViewer

    return TimeSeriesViewer


def __getattr__(name: str):
    if name == "TimeSeriesViewer":
        from .time_series import TimeSeriesViewer
        return TimeSeriesViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
