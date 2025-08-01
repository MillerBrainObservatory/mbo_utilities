from pathlib import Path
import numpy as np
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p
from lbm_suite2p_python import load_ops, rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import fastplotlib as fpl



data = mbo.imread(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\testing")

x = 2
mbo.imwrite(data, r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed", planes=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], roi=0)

files = list(Path(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed").glob("*.tif"))
files

ops = suite2p.default_ops()
ops["roidetect"] = False

lsp.run_volume(files, ops=ops, )

def merge_rois(roi_left, roi_right, save_path):
    roi_left = Path(roi_left)
    roi_right = Path(roi_right)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    ops_roi_left = np.load(roi_left / "ops.npy", allow_pickle=True).item()
    ops_roi_right = np.load(roi_right / "ops.npy", allow_pickle=True).item()
    stat_roi_left = np.load(roi_left / "stat.npy", allow_pickle=True)
    stat_roi_right = np.load(roi_right / "stat.npy", allow_pickle=True)

    def uncorrect_crop(stat, xoff, yoff, H):
        for s in stat:
            s["xpix"] = np.array(s["xpix"]) + xoff
            s["ypix"] = np.array(s["ypix"]) + yoff
            if "ipix_neuropil" in s:
                s["ipix_neuropil"] = np.array(s["ipix_neuropil"]) + (xoff * H + yoff)
        return stat

    xoff1, yoff1 = ops_roi_left["xrange"][0], ops_roi_left["yrange"][0]
    xoff2, yoff2 = ops_roi_right["xrange"][0], ops_roi_right["yrange"][0]
    H = ops_roi_left["Ly"]

    stat_roi_left = uncorrect_crop(stat_roi_left, xoff1, yoff1, H)
    stat_roi_right = uncorrect_crop(stat_roi_right, xoff2, yoff2, H)

    h1, w1 = ops_roi_left["meanImg"].shape
    h2, w2 = ops_roi_right["meanImg"].shape
    assert h1 == h2, "Heights must match for horizontal stitching"
    H, W = h1, w1 + w2

    def shift_stat(stat, dx):
        shifted = []
        for s in stat:
            s2 = s.copy()
            s2["xpix"] = np.array(s2["xpix"]) + dx
            if "med" in s2:
                s2["med"] = [s2["med"][0], s2["med"][1] + dx]
            shifted.append(s2)
        return np.array(shifted, dtype=object)

    stat_roi_right = shift_stat(stat_roi_right, dx=w1)
    stat_merged = np.concatenate([stat_roi_left, stat_roi_right])
    np.save(save_path / "stat.npy", stat_merged)

    def cat(fname, astype=None):
        d1 = np.load(roi_left / fname)
        d2 = np.load(roi_right / fname)
        if astype:
            d1 = d1.astype(astype)
            d2 = d2.astype(astype)
        return np.concatenate([d1, d2], axis=0)

    np.save(save_path / "F.npy", cat("F.npy"))
    np.save(save_path / "Fneu.npy", cat("Fneu.npy"))
    np.save(save_path / "spks.npy", cat("spks.npy"))
    np.save(save_path / "iscell.npy", cat("iscell.npy", astype=bool))

    def hcat(a, b):
        return np.hstack([np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)])

    ops_merged = {}
    ops_merged["meanImg"] = hcat(ops_roi_left["meanImg"], ops_roi_right["meanImg"])
    if "meanImgE" in ops_roi_left and "meanImgE" in ops_roi_right:
        ops_merged["meanImgE"] = hcat(ops_roi_left["meanImgE"], ops_roi_right["meanImgE"])
    if "max_proj" in ops_roi_left and "max_proj" in ops_roi_right:
        ops_merged["max_proj"] = hcat(ops_roi_left["max_proj"], ops_roi_right["max_proj"])
    if "Vcorr" in ops_roi_left and "Vcorr" in ops_roi_right:
        ops_merged["Vcorr"] = hcat(ops_roi_left["Vcorr"], ops_roi_right["Vcorr"])
    if "refImg" in ops_roi_left and "refImg" in ops_roi_right:
        ops_merged["refImg"] = hcat(ops_roi_left["refImg"], ops_roi_right["refImg"])

    ops_merged["xrange"] = [0, W]
    ops_merged["yrange"] = [0, H]
    ops_merged["Lx"] = W
    ops_merged["Ly"] = H
    ops_merged["crop_offset_x"] = 0
    ops_merged["crop_offset_y"] = 0

    np.save(save_path / "ops.npy", ops_merged)

    redcell_path = save_path / "redcell.npy"
    if not redcell_path.exists():
        np.save(redcell_path, np.zeros((len(stat_merged), 2), dtype=np.float32))

folder1 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi1"
folder2 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi2"
output_folder = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane_01_roi1_roi2"
merge_rois(folder1, folder2, output_folder)

ops_merged = lsp.load_ops("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/ops.npy")
stat_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/stat.npy", allow_pickle=True)
iscell_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/iscell.npy")
ov(ops_merged, stat_merged, iscell_merged, proj="max_proj", color_mode="random")

ops1 = np.load(folder1 + "/ops.npy", allow_pickle=True).item()
stat1 = np.load(folder1 + "/stat.npy", allow_pickle=True)
iscell1 = np.load(folder1 + "/iscell.npy", allow_pickle=True)
ov(ops1, stat1, iscell1, proj="max_proj", color_mode="random")

ops2 = np.load(folder2 + "/ops.npy", allow_pickle=True).item()
stat2 = np.load(folder2 + "/stat.npy", allow_pickle=True)
iscell2 = np.load(folder2 + "/iscell.npy", allow_pickle=True)
ov(ops_merged, stat_merged, iscell_merged, proj="max_proj")

# lsp.suite2p_roi_overlay(ops2, stat2, iscell2, "max_proj", color_mode="random")
##


def ov(ops, stat, iscell, proj=None, plot_indices=None, savepath=None,
       color_mode='random', red_border=False, colors=None):
    ops = load_ops(ops)
    img = ops[proj]

    # Infer offsets from ops
    xr0, xr1 = ops.get("xrange", [0, ops["Lx"]])
    yr0, yr1 = ops.get("yrange", [0, ops["Ly"]])
    offset_x = xr0
    offset_y = yr0

    H_img, W_img = img.shape

    # Normalize image
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    norm_img = np.clip((img - p1) / (p99 - p1), 0, 1)

    H = np.zeros_like(norm_img)
    S = np.zeros_like(norm_img)
    mask = np.zeros_like(norm_img, dtype=bool)

    iscell = np.asarray(iscell)
    cell_mask = iscell if iscell.ndim == 1 else iscell[:, 0]
    indices = np.flatnonzero(cell_mask) if plot_indices is None else plot_indices

    for i, n in enumerate(indices):
        s = stat[n]
        ypix = np.array(s["ypix"]) - offset_y
        xpix = np.array(s["xpix"]) - offset_x

        valid = (ypix >= 0) & (ypix < H_img) & (xpix >= 0) & (xpix < W_img)
        ypix = ypix[valid]
        xpix = xpix[valid]

        mask[ypix, xpix] = True

        if colors is not None:
            hue = rgb_to_hsv(np.array([[colors[i][:3]]]))[0, 0, 0]
        elif color_mode == "random":
            hue = np.random.rand()
        elif color_mode == "uniform":
            hue = 0.6
        else:
            hue = (i / max(len(indices), 1)) % 1.0

        H[ypix, xpix] = hue
        S[ypix, xpix] = 1

    rgb = hsv_to_rgb(np.stack([H, S, norm_img], axis=-1))

    if red_border and mask.any():
        from skimage.segmentation import find_boundaries
        borders = find_boundaries(mask, mode="outer")
        rgb[borders] = [1, 0, 0]

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ov(ops1, stat1, iscell1, proj="max_proj", color_mode="random")
ov(ops_merged, stat_merged, iscell_merged, proj="max_proj", color_mode="random")

print(f"roi1 mean-image shape: {ops1['meanImg'].shape}")
print(f"roi2 mean-image shape: {ops2['meanImg'].shape}")
print(f"merged mean-image shape: {ops_merged['meanImg'].shape}")

print(f"roi1 xrange: {ops1['xrange']}, yrange: {ops1['yrange']}")
print(f"roi2 xrange: {ops2['xrange']}, yrange: {ops2['yrange']}")
print(f"merged xrange: {ops_merged['xrange']}, yrange: {ops_merged['yrange']}")

print(f"Example ROI 1 first 5 : {stat1[0]['xpix'][:5]}, {stat1[0]['ypix'][:5]}")
print(f"Example ROI 2 first 5 : {stat2[0]['xpix'][:5]}, {stat2[0]['ypix'][:5]}")

import suite3d
import os
os.chdir(os.path.dirname(os.path.abspath("suite3d")))

from suite3d.job import Job
tifs = mbo.get_files(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green", "tif")

job_params = {
    'n_ch_tif': 14,  # number of channels recorded in the tiff file, typically 30
    'cavity_size': 1,  # number of planes in the deeper cavity, typically 15
    'planes': np.arange(14),
    'voxel_size_um': (16, 2, 2),

    # number of files to use for the initial pass
    # usually, ~500 frames is a good rule of thumb
    # we will just use 200 here for speed
    'n_init_files': 1,

    # number of pixels to fuse between the ROI strips
    # 'fuse_shift_override': 7,
    # will try to automatically estimate crosstalk using
    # the shallowest crosstalk_n_planes planes. if you want to override,
    # set override_crosstalk = float between 0 and 1
    'subtract_crosstalk': False,
    'fs': 17,
    'tau': 0.8,
    '3d_reg': True,
    'gpu_reg': True,
}

# Create the job
from suite3d.job import Job
job_path = r"D://W2_DATA/kbarber/2025_07_17/mk355/green/processed/job"
job = suite3d.Job(job_path,'test1', tifs = tifs,
          params=job_params, create=True, overwrite=True, verbosity = 1)
