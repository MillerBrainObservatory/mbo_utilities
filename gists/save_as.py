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
from pprint import pprint


if __name__ == "__main__":
    # raw = Path(r"D://demo//raw")
    raw = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301\green")
    # raw = Path(r"/home/flynn/lbm_data/raw")
    import tifffile

    test_scan = mbo.read_scan(raw)
    savedir = r"D:\W2_DATA\masknmf\mk301_barber"
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
