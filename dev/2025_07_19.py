from pathlib import Path
import mbo_utilities as mbo

path = Path(r"D:\W2_DATA\santi\stitched\plane07_stitched/data.bin")
lazy = mbo.imread(path)

x = 4