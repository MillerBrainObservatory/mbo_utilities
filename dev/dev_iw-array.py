import time
from pathlib import Path
from mbo_utilities.graphics.run_gui import run_gui
from mbo_utilities import imread
import fastplotlib as fpl
from mbo_utilities import subsample_array

# Initial data path
data_path = r"E:\tests\lbm\mbo_utilities\big_raw"
data = imread(data_path)
print(f"Initial data shape: {data.shape}")

# start = time.time()
# ss_arr = subsample_array(data, ignore_dims=(1, 2, 3))
# end = time.time()
# print(end - start)
# print(ss_arr.shape)
#
# del ss_arr
#
# start = time.time()
# ss_arr = data[::10, 0]
# end = time.time()
# print(end - start)
# print(ss_arr.shape)

# Create ImageWidget
pdw = run_gui(data)
x = 5
# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
