import time
from pathlib import Path
from mbo_utilities.graphics.run_gui import run_gui
from mbo_utilities import imread
import fastplotlib as fpl
from mbo_utilities import subsample_array

# Initial data
data_path = r"D:\cj\2025-11-21\results\grid_search\ana4_dia2_cel0.00_flo0.00_spa0.50"
data = imread(data_path)
pdw = run_gui(data)
x = 5
# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
