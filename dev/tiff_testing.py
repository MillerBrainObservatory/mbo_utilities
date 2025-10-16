from pathlib import Path
import numpy as np
import time
import mbo_utilities as mbo
import zarr
import tifffile

data_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
files = list(data_path.glob("*.tif*"))

# job_path = align_zplanes(data_path)
# full_job_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw_suite3d_plane_alignment\s3d-v2_1-init-file_500-init-frames_gpu\summary\summary.npy")
# summary = np.load(job_path.joinpath("summary/summary.npy"), allow_pickle=True).item()
# tvecs = summary["tvecs"]

x = mbo.imread(files)
x.phasecorr_method = None

benchmark_dir = data_path.parent / "benchmark"
benchmark_dir.mkdir(exist_ok=True)

# zarr

outpath = benchmark_dir / "zarr"
outpath.mkdir(exist_ok=True, parents=True)

start = time.time()
mbo.imwrite(x, outpath, planes=[1, 2, 3], ext=".zarr")
end = time.time()
fname = "zarr_log.txt"

with open(outpath / fname, "a") as f:
    f.write(
        f"Time to write: {end - start:.2f}\n"
        f"Array shape: {x.shape}\n"
        f"Dtype: {x.dtype}\n"
        f"Phase correction: {x.phasecorr_method}\n\n"
        f"----------------------------------------\n"
    )
print(f"Time to write zarr: {end - start:.2f}")


# tiff


outpath = benchmark_dir / "tiff"
start = time.time()
mbo.imwrite(x, outpath, planes=[1, 2, 3], ext=".tif")
end = time.time()
fname = "zarr_log.txt"

with open(outpath / fname, "a") as f:
    f.write(
        f"Time to write: {end - start:.2f}\n"
        f"Array shape: {x.shape}\n"
        f"Dtype: {x.dtype}\n"
        f"Phase correction: {x.phasecorr_method}\n\n"
        f"----------------------------------------\n"
    )

print(f"Time to write tiff: {end - start:.2f}")


# binary
outpath = benchmark_dir / "s2p_bin"
outpath.mkdir(exist_ok=True, parents=True)
start = time.time()
mbo.imwrite(x, outpath, planes=[2], ext=".bin")
end = time.time()

with open(outpath / fname, "a") as f:
    f.write(
        f"Time to write: {time.time() - start:.2f}\n"
        f"Array shape: {x.shape}\n"
        f"Dtype: {x.dtype}\n"
        f"Phase correction: {x.phasecorr_method}\n\n"
        f"----------------------------------------\n"
    )

print(f"Time to write binary: {end - start:.2f}")

# h5
outpath = benchmark_dir / "h5"
outpath.mkdir(exist_ok=True, parents=True)
start = time.time()
mbo.imwrite(x, outpath, planes=[2], ext=".bin")
end = time.time()

with open(outpath / fname, "a") as f:
    f.write(
        f"Time to write: {end - start:.2f}\n"
        f"Array shape: {x.shape}\n"
        f"Dtype: {x.dtype}\n"
        f"Phase correction: {x.phasecorr_method}\n\n"
        f"----------------------------------------\n"
    )
print(f"Time to write h5: {end - start:.2f}")


# import zarr
# import fastplotlib as fpl
# arr = zarr.open(outpath)["plane02_stitched.zarr"]
# fpl.ImageWidget(arr).show(); fpl.loop.run()

# start = time.time()
# mbo.imwrite(x, outpath, planes=[2], ext=".bin")
# end = time.time()
# print(f"Time to write binary: {end - start:.2f}")
#
# start = time.time()
# mbo.imwrite(x, outpath, planes=[2], ext=".h5")
# end = time.time()
# print(f"Time to write h5: {end - start:.2f}")

x = 2
# mbo.imwrite(x, temp_outfile)
