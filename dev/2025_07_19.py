from pathlib import Path
import mbo_utilities as mbo

path = Path(r"D:\W2_DATA\santi\stitched\plane07_stitched/data.bin")
lazy = mbo.imread(path)

import suite2p
ops = suite2p.default_ops()
ops["roidetect"] = False
import lbm_suite2p_python as lsp
input_path = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\plane14\plane10_stitched.tif"
lsp.run_plane(input_path, ops=ops)

import mbo_utilities as mbo
data = mbo.imread(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\plane10\suite2p")



x = 4