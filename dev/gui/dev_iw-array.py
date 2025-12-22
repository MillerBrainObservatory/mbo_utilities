# import time
# from pathlib import Path
# from mbo_utilities.graphics.run_gui import run_gui
import mbo_utilities
# import fastplotlib as fpl
# from mbo_utilities import subsample_array

# Initial data
data_path = r"D:/demo/raw"
data = mbo_utilities.imread(data_path)
print(data.metadata)
# pdw = run_gui(data)
# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
