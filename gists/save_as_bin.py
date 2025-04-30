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
import fastplotlib as fpl

fname_scan = Path().home().joinpath("lbm_data/demo/raw")
scan = mbo.read_scan(fname_scan)

files = mbo.get_files(fname_scan)
md = mbo.get_metadata(files)

if __name__ == "__main__":
    save_path = Path().home().joinpath("lbm_data/output")
    mbo.save_as(scan, save_path, planes=[0, 1, 2, 3], ext=".bin")
    data = np.memmap(save_path.joinpath("plane1/raw_data.bin"), dtype="int16").reshape((1437, 448, 448))
    fpl.ImageWidget(data).show()
    fpl.loop.run()