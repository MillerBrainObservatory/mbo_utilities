# import time
# from pathlib import Path
from mbo_utilities.gui.run_gui import run_gui
import mbo_utilities
# import fastplotlib as fpl
# from mbo_utilities import run_gui

# Initial data
data_path = r"E:\datasets\pollen\2026-03-31_thorlabs-obj-testing\Pollen_4p1x_5um-Step_Thorlabs28x"
data = mbo_utilities.imread(data_path)
print(data.metadata)
pdw = run_gui(data)

# iw = fpl.ImageWidget(
#     [data],
#     names=["My Custom Filetype"],
#     slider_dim_names=("t", "z"),
# )
