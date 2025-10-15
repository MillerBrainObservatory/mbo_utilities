import time
from pathlib import Path

import numpy as np
from mbo_utilities.lazy_array import imread
from mbo_utilities.file_io import merge_zarr_rois, group_plane_rois
# from mbo_utilities.graphics.imgui import PreviewDataWidget
# import fastplotlib as fpl

fpath = Path(r"D:\demo\regz")
start = time.time()
merge_zarr_rois(fpath, output_dir=fpath.parent / f"{fpath.stem}_merged")
print(f"Time to complete: {time.time() - start:.2f} seconds")

x = 5