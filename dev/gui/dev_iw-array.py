import time
from pathlib import Path
from mbo_utilities.graphics.run_gui import run_gui
from mbo_utilities import imread
import fastplotlib as fpl
from mbo_utilities import subsample_array

# Initial data
data_path = r"D:/raw_scanimage_tiffs"
data = imread(data_path)
pdw = run_gui(data)
# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
