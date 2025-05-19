from pathlib import Path
import numpy as np

try:
    from imgui_bundle import portable_file_dialogs as pdf
except ImportError:
    MBO_HAS_IMGUI = False

from ..util import is_imgui_installed
from .imgui import PreviewDataWidget
from ..file_io import (
    ScanMultiROIReordered,
    to_lazy_array,
    get_files,
    read_scan,
    mbo_home
)

if is_imgui_installed():
    import fastplotlib as fpl

def run_gui(
    data_in: None | str | Path | ScanMultiROIReordered | np.ndarray = None, **kwargs
):
    """Open a GUI to preview data."""
    # Handle data_in, which can be a path to files
    if data_in is None:
        fd = pdf.select_folder(str(mbo_home))
        fpath = fd.result()
        files = get_files(fpath)
        data_in = read_scan(files)
    if isinstance(data_in, ScanMultiROIReordered):
        data = data_in
    else:
        data = to_lazy_array(data_in)

    iw = fpl.ImageWidget(data=data, histogram_widget=True, **kwargs)
    gui = PreviewDataWidget()
    iw.figure.add_gui()
    iw.show()
    fpl.loop.run()
    return None