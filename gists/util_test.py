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
save_path = Path().home().joinpath("dev")
uncor_save = save_path.joinpath("raw")
corrected_save = save_path.joinpath("corrected")

raw_scan = mbo.read_scan(fname_scan)

mbo.save_as(raw_scan, uncor_save, planes=[7], fix_phase=False)
mbo.save_as(raw_scan, corrected_save, planes=[7], fix_phase=True)

x = 2