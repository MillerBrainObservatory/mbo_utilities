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

fname_scan = r"/home/flynn/lbm_data/raw"
save_path = Path("/home/flynn/lbm_data")

uncor_save = save_path.joinpath("uncorrected")
corrected_save = save_path.joinpath("corrected")

raw_scan = mbo.read_scan(fname_scan)

# mbo.save_as(raw_scan, uncor_save, planes=[11], fix_phase=False)
mbo.save_as(
    raw_scan,
    corrected_save,
    planes=[11],
    fix_phase=True,
    target_chunk_mb=50,
    debug=False,
    ext="bin",
)

x = 2
