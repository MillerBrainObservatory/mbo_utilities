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
import tifffile

import mbo_utilities as mbo
from pathlib import Path


if __name__ == "__main__":
    raw = Path(r"D://tests//data")
    test_scan = mbo.read_scan(raw, roi=(1,2)) # -1 or 0 both return a tuple of ROI's
    savedir = r"D:\tests\save_as_test"
    mbo.save_as(
        test_scan,
        savedir,
        ext=".tiff",
        overwrite=True,
        fix_phase=True,
        planes=[7],
        debug=True,
    )
    files = list(Path(savedir).glob("*.tif*"))[0]
    img = tifffile.imread(files)
    print(img.shape)
