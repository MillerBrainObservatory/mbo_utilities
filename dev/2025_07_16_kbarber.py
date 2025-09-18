from pathlib import Path
# import numpy as np
import mbo_utilities as mbo
# import lbm_suite2p_python as lsp
# import suite2p
# from lbm_suite2p_python import load_ops, rgb_to_hsv, hsv_to_rgb
# import matplotlib.pyplot as plt
# import fastplotlib as fpl

data = mbo.imread(r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw_subsampled_t")
mbo.imwrite(data, r"D:\W2_DATA\kbarber\07_27_2025\mk355\stitching_test", roi=None)

# Commented out: FO 09/17/25
files = list(Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_mean\pre_reg").glob("*.tif"))
ops = suite2p.default_ops()
ops["roidetect"] = True
ops["keep_movie_raw"] = True
ops["tau"] = 0.7

lsp.run_volume(files, ops=ops,)

# def ov(ops, stat, iscell, proj=None, plot_indices=None, savepath=None,
#        color_mode='random', red_border=False, colors=None):
#     ops = load_ops(ops)
#     img = ops[proj]
#
#     # Infer offsets from ops
#     xr0, xr1 = ops.get("xrange", [0, ops["Lx"]])
#     yr0, yr1 = ops.get("yrange", [0, ops["Ly"]])
#     offset_x = xr0
#     offset_y = yr0
#
#     H_img, W_img = img.shape
#
#     # Normalize image
#     p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
#     norm_img = np.clip((img - p1) / (p99 - p1), 0, 1)
#
#     H = np.zeros_like(norm_img)
#     S = np.zeros_like(norm_img)
#     mask = np.zeros_like(norm_img, dtype=bool)
#
#     iscell = np.asarray(iscell)
#     cell_mask = iscell if iscell.ndim == 1 else iscell[:, 0]
#     indices = np.flatnonzero(cell_mask) if plot_indices is None else plot_indices
#
#     for i, n in enumerate(indices):
#         s = stat[n]
#         ypix = np.array(s["ypix"]) - offset_y
#         xpix = np.array(s["xpix"]) - offset_x
#
#         valid = (ypix >= 0) & (ypix < H_img) & (xpix >= 0) & (xpix < W_img)
#         ypix = ypix[valid]
#         xpix = xpix[valid]
#
#         mask[ypix, xpix] = True
#
#         if colors is not None:
#             hue = rgb_to_hsv(np.array([[colors[i][:3]]]))[0, 0, 0]
#         elif color_mode == "random":
#             hue = np.random.rand()
#         elif color_mode == "uniform":
#             hue = 0.6
#         else:
#             hue = (i / max(len(indices), 1)) % 1.0
#
#         H[ypix, xpix] = hue
#         S[ypix, xpix] = 1
#
#     rgb = hsv_to_rgb(np.stack([H, S, norm_img], axis=-1))
#
#     if red_border and mask.any():
#         from skimage.segmentation import find_boundaries
#         borders = find_boundaries(mask, mode="outer")
#         rgb[borders] = [1, 0, 0]
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(rgb)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()
