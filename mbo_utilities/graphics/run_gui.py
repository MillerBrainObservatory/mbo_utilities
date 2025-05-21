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
    _is_arraylike,
    to_lazy_array,
    get_files,
    read_scan,
    mbo_home,
)

if is_imgui_installed():
    import fastplotlib as fpl

from pathlib import Path
import numpy as np

try:
    from imgui_bundle import portable_file_dialogs as pdf
except ImportError:
    MBO_HAS_IMGUI = False

from ..util import is_imgui_installed
from .imgui import PreviewDataWidget
from ..file_io import (
    to_lazy_array,
    get_files,
    read_scan,
    mbo_home
)

if is_imgui_installed():
    import fastplotlib as fpl


def run_gui(data_in=None, **kwargs):
    """Open a GUI to preview data of any supported type."""

    if data_in is None:
        fd = pdf.select_folder(str(mbo_home))
        fpath = Path(fd.result())
        files = get_files(fpath)
        data = read_scan(files)
    elif _is_arraylike(data_in):
        data = data_in
    else:
        data = to_lazy_array(data_in)

    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    if sample.ndim < 2:
        raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")

    vmin, vmax = sample.min(), sample.max()
    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx, ny)},
        graphic_kwargs={"vmin": vmin, "vmax": vmax},
        window_funcs={"t": (np.mean, 0)},
    )

    if kwargs.get("gui"):
        gui = PreviewDataWidget(iw=iw)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()

