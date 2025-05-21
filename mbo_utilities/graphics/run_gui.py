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
    mbo_home,
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
        # all logic for what this function can support is handled here
        data = to_lazy_array(data_in)

    if isinstance(data_in, list):
        vmin, vmax = data[0].min, data[0].max
        nx, ny = data[0].shape[-2:]
    else:
        vmin, vmax = data.min, data.max
        nx, ny = data.shape[-2:]

    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={
            "size": (nx, ny),
        },
        graphic_kwargs={"vmin": vmin, "vmax": vmax},
        window_funcs={"t": (np.mean, 0)},
    )

    add_gui = kwargs.get("gui", None)
    if add_gui:
        gui = PreviewDataWidget(iw=iw)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()
    return None
