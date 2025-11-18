from pathlib import Path
import numpy as np
import time
import mbo_utilities as mbo
import fastplotlib as fpl
import zarr
import tifffile

data_path = Path(r"D:\example_extraction\zarr_with_zreg")
data_path_og = Path(r"D:\example_extraction\zarr")

no_reg = mbo.imread(data_path_og)
with_reg = mbo.imread(data_path)

iw = fpl.ImageWidget([no_reg, with_reg], names=["No Plane Alignment", "Suite3D Plane Alignment"])
iw.show()
fpl.loop.run()

# job_path = align_zplanes(data_path)
# full_job_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw_suite3d_plane_alignment\s3d-v2_1-init-file_500-init-frames_gpu\summary\summary.npy")
# summary = np.load(job_path.joinpath("summary/summary.npy"), allow_pickle=True).item()
# tvecs = summary["tvecs"]

x = mbo.imread(files)
x.phasecorr_method = None
