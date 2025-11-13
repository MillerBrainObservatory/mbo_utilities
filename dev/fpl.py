from pathlib import Path
import numpy as np
import time
import mbo_utilities as mbo
import zarr
import tifffile
import fastplotlib as fpl

data_path = Path(r"D:\demo\ome_v2\volume.zarr")
# data_path = Path(r"D:\demo\ome_v2\sharded\plane07_stitched.zarr")
files = list(data_path.glob("*.tif*"))

arr = mbo.imread(data_path)

iw = fpl.ImageWidget(arr)
iw.show()

fpl.loop.run()
