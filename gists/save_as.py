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

    test_scan = mbo.read_scan("/home/flynn/lbm_data/raw", roi=0)
    savedir = r"/home/flynn/lbm_data/saved"
    mbo.save_as(
        test_scan,
        savedir,
        ext=".bin",
        overwrite=True,
        debug=True,
        fix_phase=True,
        planes=[1, 5, 10, 14],
    )
