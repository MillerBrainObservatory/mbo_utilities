from pathlib import Path
import fastplotlib as fpl
import mbo_utilities as mbo

fpath = Path(r"D:\W2_DATA\santi\assembled\roi2\plane5.tif")
md = mbo.get_metadata(fpath)
fpath = Path(r"D:\W2_DATA\kbarber\2025_07_17\m355\green")
md2 = mbo.get_metadata(fpath)

outpath = Path(r"D:\W2_DATA\kbarber\2025_07_16\m350\assembled_phase_frame")
outpath.mkdir(exist_ok=True)

# Assemble the data
data = mbo.imread(fpath)
data.fix_phase = True
data.phasecorr_method = "frame"
data.roi = 2
mbo.imwrite(data, outpath, planes=[1, 7, 14])

# run z-plane 1
data = mbo.imread(outpath.joinpath("plane01_stitched.tif"))
iw = fpl.ImageWidget(data)
iw.show()
fpl.loop.run()