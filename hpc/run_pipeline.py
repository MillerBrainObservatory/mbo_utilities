from pathlib import Path

import lbm_suite2p_python as lsp
from mbo_utilities import imread

input_dir = Path("/lustre/fs4/mbo/scratch/mbo_data/mk355")
output_dir = Path("/lustre/fs4/mbo/scratch/mbo_data/mk355/results")
output_dir.mkdir(parents=True, exist_ok=True)

ops = {
    "anatomical_only": 4,
    "diameter": 2,
    "cellprob_threshold": -6,
    "spatial_hp_cp": 3,
    "niter": 200,
    "do_registration": 1,
    "two_step_registration": 1,
    "lam_percentile": 0,
    "min_neuropil_pixels": 0,
    "max_overlap": 0.99,
}

arr = imread(input_dir)
print(f"shape={arr.shape} dims={getattr(arr, 'dims', None)}", flush=True)

lsp.pipeline(
    input_data=arr,
    save_path=str(output_dir),
    ops=ops,
    planes=None,
    keep_reg=True,
    keep_raw=False,
    force_reg=True,
    force_detect=True,
    reader_kwargs={"fix_phase": True, "use_fft": True},
)
