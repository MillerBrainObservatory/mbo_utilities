# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }
import tifffile

import mbo_utilities as mbo
from pathlib import Path

if __name__ == "__main__":
    raw = Path(r"")
    # Values for argument `roi`:
    #
    # 0:                    Save all ROIs separately
    # 1:                    Save only the first ROI
    # 2:                    Save only the second ROI
    # (1, 2 ... num_rois):  Save only the specified ROIs
    # None:                 Assembly and save joined ROIs
    test_scan = mbo.read_scan("D://tests//data", roi=0)
    savedir = r"D:\demo\masknmf"
    mbo.save_as(
        test_scan,
        savedir,
        ext=".tiff",
        overwrite=False,
        debug=True,
        fix_phase=True,
        planes=[1, 5, 10, 14],
    )