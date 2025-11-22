from pathlib import Path
from mbo_utilities.graphics.run_gui import run_gui
from mbo_utilities import imread
import fastplotlib as fpl

# Initial data path
data_path = r"E:\tests\lbm\mbo_utilities\big_raw"
data = imread(data_path)
print(f"Initial data shape: {data.shape}")

# Create ImageWidget
pdw = run_gui(data)
x = 5
# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
