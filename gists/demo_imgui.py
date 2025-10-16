import time
from pathlib import Path

import numpy as np
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.file_io import merge_zarr_rois, group_plane_rois, load_zarr_grouped
# from imgui_bundle import hello_imgui
# from mbo_utilities.graphics.imgui import PreviewDataWidget
import fastplotlib as fpl

fpath = Path(r"D:\demo\mrois_fft")
# data = imread(fpath)
start = time.time()
data = load_zarr_grouped(fpath)
print(f"Time to load: {time.time() - start:.2f} seconds")

start = time.time()
iw = fpl.ImageWidget(
    data=data,
    histogram_widget=True,
    figure_kwargs={"size": (800, 1000)},
    graphic_kwargs={"vmin": -400, "vmax": 4000},
    window_funcs={"t": (np.mean, 0)},
)
print(f"Time to create widget: {time.time() - start:.2f} seconds")
iw.show()
fpl.loop.run()

imwrite(data, fpath.parent / f"{fpath.stem}_copy.tif", register_z=True)

# start = time.time()
# merge_zarr_rois(fpath, output_dir=fpath.parent / f"{fpath.stem}_merged")
# print(f"Time to complete: {time.time() - start:.2f} seconds")

x = 5