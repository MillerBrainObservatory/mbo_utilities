# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }

import mbo_utilities as mbo
import numpy as np
from pathlib import Path
import tifffile
import fastplotlib as fpl

fname_scan = r"D:\W2_DATA\kbarber\2025_03_01\mk301\green\*"
raw_scan = mbo.read_scan(fname_scan)

save_path = Path().home().joinpath("lbm_data/output")
mbo.save_as(raw_scan, save_path, planes=[0, 8], ext=".bin")