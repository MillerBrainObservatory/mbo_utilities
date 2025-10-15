import time
from pathlib import Path

import numpy as np
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.file_io import merge_zarr_rois, group_plane_rois
# from imgui_bundle import hello_imgui
# from mbo_utilities.graphics.imgui import PreviewDataWidget
# import fastplotlib as fpl

fpath = Path(r"D:\demo\raw")
data = imread(fpath)

imwrite(data, fpath.parent / f"{fpath.stem}_copy.tif", register_z=True)


# start = time.time()
# merge_zarr_rois(fpath, output_dir=fpath.parent / f"{fpath.stem}_merged")
# print(f"Time to complete: {time.time() - start:.2f} seconds")

x = 5