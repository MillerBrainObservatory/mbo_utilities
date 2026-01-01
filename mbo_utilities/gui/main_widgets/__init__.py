"""
Main widgets for the MBO data viewer GUI.

Main widgets are the primary content area that depends on the loaded data type.
Each widget type handles a specific kind of data visualization and interaction.

- TimeSeriesWidget: Standard time-series imaging data (TZYX)
- PollenCalibrationWidget: Pollen calibration data (ZCYX)
"""

from mbo_utilities.gui.main_widgets._base import MainWidget
from mbo_utilities.gui.main_widgets.time_series import TimeSeriesWidget
from mbo_utilities.gui.main_widgets.pollen_calibration import PollenCalibrationWidget


def get_main_widget_class(array) -> type[MainWidget]:
    """
    Select appropriate main widget class based on array type.

    Parameters
    ----------
    array : LazyArray
        The loaded data array.

    Returns
    -------
    type[MainWidget]
        The widget class to instantiate for this data type.
    """
    # Check for pollen calibration data
    if hasattr(array, "stack_type") and array.stack_type == "pollen":
        return PollenCalibrationWidget

    # Default to time series viewer
    return TimeSeriesWidget


__all__ = [
    "MainWidget",
    "TimeSeriesWidget",
    "PollenCalibrationWidget",
    "get_main_widget_class",
]
