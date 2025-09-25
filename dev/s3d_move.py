from pathlib import Path
import warnings
import lbm_suite2p_python as lsp
from mbo_utilities import get_files #, imread, get_metadata

warnings.simplefilter(action='ignore')

inpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered")
outpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered\mbo_v3")
model_path = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\suite2p\z_registered\mbo_v3.npy")

if model_path.is_file():
    ops = {"pretrained_model": model_path}
    aligned_files = get_files(inpath, "aligned", max_depth=3)
    lsp.run_volume(aligned_files[:3], save_path=outpath)

# tiffs = get_files(Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\test"), "stitched", max_depth=2)
# md = [get_metadata(tiffs[i]) for i in range(len(tiffs))]
# s3d_dir = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\test").rglob("*summary.npy*")
# summary = np.load(list(s3d_dir)[0], allow_pickle=True).item()
# plane_shifts = summary["plane_shifts"]
# print(plane_shifts)
# print(tiffs)
#
# import numpy as np
# from pathlib import Path
# import tifffile
# from scipy.ndimage import shift
#
# # Inputs
# base_dir = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\test")
# tiffs = sorted(base_dir.rglob("plane*_stitched.tif"))
# summary_file = list(base_dir.rglob("*summary.npy"))[0]
#
# # Load shifts
# summary = np.load(summary_file, allow_pickle=True).item()
# plane_shifts = summary["plane_shifts"]  # (N_planes, 2), (dy, dx)
#
# arrays = [tifffile.memmap(str(p)) for p in tiffs]
# nframes = [arr.shape[0] for arr in arrays]
# H, W = arrays[0].shape[-2:]  # assume 2D or (T, Y, X)
#
# # Compute required padding (based on max shifts)
# dy_min, dx_min = plane_shifts.min(axis=0)
# dy_max, dx_max = plane_shifts.max(axis=0)
#
# pad_top, pad_left = max(0, -dy_min), max(0, -dx_min)
# pad_bottom, pad_right = max(0, dy_max), max(0, dx_max)
#
# target_shape = (nframes[0], H + pad_top + pad_bottom, W + pad_left + pad_right)
# print("Final shape:", target_shape)
#
# out_arrays = []
# for arr, (dy, dx) in zip(arrays, plane_shifts):
#     padded = np.zeros(target_shape, dtype=arr.dtype)
#     yy, xx = slice(pad_top + dy, pad_top + dy + H), slice(pad_left + dx, pad_left + dx + W)
#     padded[:, yy, xx] = arr
#     out_arrays.append(padded)
#
# for i, (tif, arr) in enumerate(zip(tiffs, out_arrays)):
#     outpath = tif.with_name(tif.stem + "_aligned.tif")
#     tifffile.imwrite(outpath, arr, metadata=md[i])
#     print("Wrote:", outpath)
#
# aligned_files = get_files(outpath.parent, "aligned")
# x = tifffile.imread(aligned_files[0])
# arr = imread(aligned_files)
# fpl.ImageWidget(arr).show()
# fpl.loop.run()


    # fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw")
    # arr = imread(fpath)
    # imwrite(arr, fpath.parent.joinpath("test"))
    # x = 2

    # start = time.time()
    # arr.preprocess()
    # end = time.time()
    # print(f"Preprocessing time: {end - start:.2f} seconds")
    #
    # outpath = Path(str(fpath) + "_extracted")
    # start = time.time()
    # imwrite(arr, outpath)
    # end = time.time()
    # print(f"Writing time: {end - start:.2f} seconds")