from pathlib import Path
import numpy as np
import sys
import time

repo_root = Path(__file__).resolve().parents[2] / "suite3d"
sys.path.insert(0, str(repo_root))

from suite3d.job import Job

if __name__ == "__main__":
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
        'init_n_frames': 500,
        'n_init_files': 1,
        'n_proc_corr': 15,
        'max_rigid_shift_pix': 150,
        '3d_reg': True,
        'gpu_reg': True,
        'block_size': [64, 64],
    }

    tifs = list(fpath.glob("*.tif*"))
    job = Job(str(job_path), 'v2_1-init-file_500-init-frames_gpu', create=True, overwrite=True, verbosity=0, tifs=tifs, params=params)

    start = time.time()
    job.run_init_pass()
    end = time.time()
    print(f"Initialization pass took {end - start:.2f} seconds")