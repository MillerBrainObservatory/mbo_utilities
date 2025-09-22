import time
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import tifffile
from scipy.ndimage import median_filter
import mbo_utilities as mbo
from scipy.ndimage import shift
from mbo_utilities.pipelines.suite2p import convolve, shift_frame
import fastplotlib as fpl

# paths
stitched_dir = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\stitching_test")
files = list(stitched_dir.glob("*.tif*"))

# load plane shifts
summary = np.load(
    r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite3d_plane_alignment\s3d-v1\summary\summary.npy",
    allow_pickle=True
).item()
shifts = summary["plane_shifts"]  # shape (n_planes, 2) [dy, dx]

total_frames_all_files = sum(tifffile.imread(file).shape[0] for file in files)

start = time.time()
with tqdm(total=total_frames_all_files, desc="Shifting Frames") as pbar:
    for i, file in (enumerate(files)):
        out_file = stitched_dir / "aligned_gpu" / file.name.replace("_stitched.tif", "_aligned.tif")
        img = tifffile.imread(file)
        dy, dx = shifts[i]
        aligned = np.empty_like(img)
        for i, frame in tqdm(enumerate(img)):
            aligned[i] = shift_frame(frame, -dy, dx)
            pbar.update(1)
        tifffile.imwrite(out_file, aligned.astype(np.float32))

end = time.time()
print(f"Total alignment time: {end - start:.2f} seconds")
x=2

def save_projections(fpath: Path | str, str_contains: str = "", max_depth: int = 1, kernel_size: int = 3):
    fpath = Path(fpath)
    files = mbo.get_files(fpath, str_contains=str_contains, max_depth=max_depth)
    outdir = fpath.joinpath("projections")
    outdir.mkdir(exist_ok=True)

    # NumPy-based projections
    projections = ["max", "mean", "std"]
    for f in files:
        data = mbo.imread(f)  # shape (T, Y, X)
        for fn in projections:
            proj = getattr(np, fn)(data, axis=0)
            out_file = outdir / f"{f.stem}_{fn}.tif"
            tifffile.imwrite(out_file, proj.astype(np.float32))
        # Median filter projection
        p_med = median_filter(np.median(data, axis=0), size=kernel_size)
        out_file = outdir / f"{f.stem}_median.tif"
        tifffile.imwrite(out_file, p_med.astype(np.float32))
        print(f"Saved projections for {f.name} → {outdir}")

# projections_dir = stitched_dir / "projections"
# save_projections(stitched_dir, str_contains="_aligned", max_depth=1, kernel_size=3)

#% Preview alignment

from pathlib import Path
import fastplotlib as fpl
import tifffile
import numpy as np

stitched_dir = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\stitching_test")
files = list(stitched_dir.glob("*.tif*"))

aligned_files = list((stitched_dir / "aligned_gpu").glob("*_aligned.tif"))
volume = np.stack([tifffile.imread(file) for file in aligned_files])

pre_aligned_files = list(stitched_dir.glob("*stitched.tif*"))
volume_pre_aligned = np.stack([tifffile.imread(file) for file in pre_aligned_files])

fpl.ImageWidget([volume_pre_aligned,volume], names=["No Spatial Registration", "Spatially Registered (Suite3D)"]).show()
fpl.loop.run()