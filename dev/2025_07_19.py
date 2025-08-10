from pathlib import Path
import mbo_utilities as mbo
import numpy as np

path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355")
files = mbo.get_files(path.joinpath("green"), "tif")
lazy = mbo.imread(files[:4])

# planes = [1, 4, 7, 10, 14]
#
# lazy.fix_phase = False
# lazy.phasecorr_method = "frame"
# outpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_nophase")
# outpath.mkdir(exist_ok=True)
# mbo.imwrite(lazy, outpath, planes=planes, overwrite=True)
#
# outpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_frame")
# outpath.mkdir(exist_ok=True)
# lazy.fix_phase = True
# lazy.phasecorr_method = "frame"
# mbo.imwrite(lazy, outpath, planes=planes, overwrite=True)
#
# outpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_mean")
# outpath.mkdir(exist_ok=True)
# lazy.fix_phase = True
# lazy.phasecorr_method = "mean"
# mbo.imwrite(lazy, outpath, planes=planes, overwrite=True)

# load files and get mean image

dirs = ["out_nophase", "out_frame", "out_mean"]
for d in dirs:
    outpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355") / d
    lazy = mbo.imread(outpath / "mean_image.tif")
    if len(lazy.shape) == 4:
        mean_image = np.mean(lazy, axis=(0, 1))
    elif len(lazy.shape) == 3:
        mean_image = np.mean(lazy, axis=0)
    else:
        mean_image = lazy
    mbo.imwrite(mean_image.compute(), outpath / "mean_image.tif")

import suite2p
ops = suite2p.default_ops()
ops["roidetect"] = False
import lbm_suite2p_python as lsp
input_path = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\plane14\plane10_stitched.tif"
lsp.run_plane(input_path, ops=ops)

import mbo_utilities as mbo
data = mbo.imread(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\plane10\suite2p")

x = 4