import numpy as np
from mbo_utilities.file_io import get_files, read_scan
from mbo_utilities.graphics.imgui import PreviewDataWidget
from imgui_bundle import imgui, portable_file_dialogs as pfd, hello_imgui, immapp
from wgpu.utils.imgui import ImguiRenderer, Stats
import fastplotlib as fpl
from rendercanvas.auto import RenderCanvas

canvas = RenderCanvas()

fpath = "/home/flynn/lbm_data/raw"
data = read_scan(fpath)
nx, ny = data.shape[-2:]
iw = fpl.ImageWidget(
    data=data,
    histogram_widget=False,
    figure_kwargs={
       "size": (nx * 2, ny * 2),
       "canvas": canvas,
    },
    graphic_kwargs={"vmin": data.min(), "vmax": data.max()},
    window_funcs={"t": (np.mean, 1)},
)
edge_gui = PreviewDataWidget(iw=iw, fpath=fpath,)
iw.figure.add_gui(edge_gui)
iw.show()

fpl.loop.run()
