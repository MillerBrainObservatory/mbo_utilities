from pathlib import Path
import numpy as np

from ..util import is_imgui_installed, is_qt_installed, is_running_jupyter
from ..file_io import ScanMultiROIReordered, load_data_path

if is_imgui_installed():
    import fastplotlib as fpl
if is_qt_installed():
    from .qt import render_qt_widget


def run_gui(
        data_in: None | str | Path | ScanMultiROIReordered | np.ndarray = None,
):
    """Open a GUI to preview data."""
    # Handle data_in, which can be a path to files
    if data_in is None:
        print('No data provided')
        if not is_qt_installed():
            raise ValueError(f"No `data_in` argument provided and no qt installation. "
                             f"Support for file loading is only available with Qt installs."
                             f"Install with `pip install -U 'mbo_utilities[all]'`.")
        else:
            # set to None, we will load a dialog folder in QT later
            print('Setting data to None')
            data = None
    else:
        print('Data provided and set.')
        data = data_in if isinstance(data_in, ScanMultiROIReordered) else load_data_path(data_in)

    if is_running_jupyter():
        print('Is running jupyter')
        # TODO: load dialog when qt isn't installed
        if data_in is None:
            print("Running jupyter, no data provided")
            if not is_qt_installed():
                raise ValueError(f"No `data_in` argument provided and no qt installation. "
                                 f"Support for file loading is only available with Qt installs."
                                 f"Install with `pip install -U 'mbo_utilities[all]'`.")
        else:
            # no data, in Jupyter, we need a QT dialog to load a data path
            # can remove this once we have a means to load a native file-dialog from within jupyter
            iw = fpl.ImageWidget(data=data, histogram_widget=True,)
            iw.show()
            return iw
    else:  # not runniing jupyter
            print(f"Not running Jupyter, Rendering qt widget")
            render_qt_widget(data=data) 
