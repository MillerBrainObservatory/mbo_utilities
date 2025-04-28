# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "lbm_suite2p_python",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }
# lbm_suite2p_python = { git = "https://github.com/MillerBrainObservatory/lbm_suite2p_python", branch = "master" }
import mbo_utilities as mbo
from pathlib import Path
import fastplotlib as fpl

raw_scan = mbo.read_scan(r"D://demo//raw_data/*")
raw_scan[0, 0, :, :]

if __name__ == "__main__":
    iw = fpl.ImageWidget(data=raw_scan, histogram_widget=True, names=["raw_scan"])
    # gui = mbo.graphics.SummaryDataWidget(image_widget=iw, size=200, location="right")
    # iw.figure.add_gui(gui)
    iw.show()

    fpl.loop.run()

