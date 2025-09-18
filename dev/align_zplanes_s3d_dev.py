import os
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parents[2] / "suite3d"
sys.path.insert(0, str(repo_root))

import numpy as np
from suite3d.job import Job


fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\green")
job_path = fpath.parent.joinpath("suite3d_plane_alignment")

# Set the mandatory parameters
params = {
    # volume rate
    'fs': 17,
    'planes': np.arange(14),
    'n_ch_tif': 14,
    'tau': 0.7,
    'lbm': True,
    'fuse_strips': True,
    'subtract_crosstalk': False,
    'init_n_frames': None,
    'n_init_files': 5,
    'n_proc_corr': 12,
    'max_rigid_shift_pix': 150,
    '3d_reg': True,
    'gpu_reg': True,
    'block_size': [64, 64],
}

tifs = list(fpath.glob("*.tif*"))
job = Job(str(job_path), 'v2-5init_files', create=True, overwrite=True, verbosity = 1, tifs=tifs, params=params)

start = time.time()
job.run_init_pass()
end = time.time()
print(f"Initialization pass took {end - start:.2f} seconds")