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
    mbo.save_as(scan, save_path, [6, 13], ext=".bin", overwrite=False)
    data = np.memmap(save_path.joinpath("plane0/raw_data.bin"), dtype="int16").reshape(
        (1437, 448, 448)
    )
    data2 = np.memmap(save_path.joinpath("plane6/raw_data.bin"), dtype="int16").reshape(
        (1437, 448, 448)
    )
    data3 = np.memmap(save_path.joinpath("plane9/raw_data.bin"), dtype="int16").reshape(
        (1437, 448, 448)
    )
    data4 = np.memmap(
        save_path.joinpath("plane13/raw_data.bin"), dtype="int16"
    ).reshape((1437, 448, 448))
    iw = fpl.ImageWidget(
        data=[data, data2, data3, data4],
        names=["Plane 1", "Plane 2", "Plane 3", "Plane 4"],
        histogram_widget=False,
    )
    for subplot in iw.figure:
        subplot.toolbar = False
    iw.show()
    fpl.loop.run()
