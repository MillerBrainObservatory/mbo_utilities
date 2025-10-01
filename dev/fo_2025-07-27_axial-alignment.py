# /// script
# requires-python = ">=3.12.7, <3.12.10"
# dependencies = [
#     "mbo_utilities",
#     "numpy",
#     "tifffile",
#     "scikit-image",
#     "scipy",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }

from pathlib import Path
import time
import numpy as np
import tifffile

from tqdm.auto import tqdm
from mbo_utilities.pipelines.suite2p import shift_frame

input_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw_subsampled_t")
stitched_dir = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\stitching_test")
files = list(stitched_dir.glob("*.tif*"))

# load plane shifts
summary = np.load(
    r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite3d_plane_alignment\s3d-v1\summary\summary.npy",
    allow_pickle=True,
).item()
shifts = summary["plane_shifts"]  # shape (n_planes, 2) [dy, dx]

total_frames_all_files = sum(tifffile.imread(file).shape[0] for file in files)

start = time.time()
with tqdm(total=total_frames_all_files, desc="Shifting Frames") as pbar:
    for i, file in enumerate(files):
        out_file = (
            stitched_dir
            / "aligned_gpu"
            / file.name.replace("_stitched.tif", "_aligned.tif")
        )
        img = tifffile.imread(file)
        dy, dx = shifts[i]
        aligned = np.empty_like(img)
        for i, frame in tqdm(enumerate(img)):
            aligned[i] = shift_frame(frame, -dy, dx)
            pbar.update(1)
        tifffile.imwrite(out_file, aligned.astype(np.float32))

end = time.time()
print(f"Total alignment time: {end - start:.2f} seconds")
