from pathlib import Path
import numpy as np
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import suite2p
from lbm_suite2p_python import load_ops, rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt
import fastplotlib as fpl
import numpy as np
from pathlib import Path
import mbo_utilities as mbo

file = mbo.get_metadata(r"D:\W2_DATA\santi\213107tUTC_Max15_depth400um_fov1908x2000um_res2p00x2p00umpx_fr02p605Hz_pow230p1mW_00001_00001.tif")

def embed_into_canvas(img, yrange, xrange, canvas_shape):
    """
    Crop an image by its yrange and xrange.
    """
    full = np.zeros(canvas_shape, dtype=img.dtype)
    y0, y1 = yrange
    x0, x1 = xrange
    target_shape = (y1 - y0, x1 - x0)
    img_cropped = img[:target_shape[0], :target_shape[1]]
    full[y0:y0 + img_cropped.shape[0], x0:x0 + img_cropped.shape[1]] = img_cropped
    return full

def concat_binfiles_and_merge_metadata(f1, f2, output_bin, output_ops):
    left = Suite2pArray(f1)
    right = Suite2pArray(f2)
    md_left = load_ops(f1)
    md_right = load_ops(f2)

    assert left.Ly == right.Ly, f"Ly mismatch: {left.Ly} vs {right.Ly}"
    assert left.nframes == right.nframes, f"nframes mismatch: {left.nframes} vs {right.nframes}"

    Ly = left.Ly
    Lx = left.Lx + right.Lx
    nframes = left.nframes
    dtype = left.dtype

    output_bin = Path(output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    with open(output_bin, "wb") as f_out:
        for i in range(nframes):
            frame = np.hstack([left[i], right[i]])
            f_out.write(frame.astype(dtype).tobytes())

    left.close()
    right.close()

    canvas_Ly = max(md_left["yrange"][1], md_right["yrange"][1])
    canvas_Lx = max(md_left["xrange"][1] + md_left["xrange"][0],
                    md_right["xrange"][1] + md_right["xrange"][0])

    merged_md = dict(md_left)
    merged_md["Ly"] = Ly
    merged_md["Lx"] = Lx
    merged_md["yrange"] = [0, Ly]
    merged_md["xrange"] = [0, Lx]
    merged_md["raw_file"] = str(output_bin.resolve())

    for key in ["meanImg", "meanImgE", "Vcorr"]:
        if key in md_left and key in md_right:
            canvas_shape = (canvas_Ly, canvas_Lx)
            left_full = embed_into_canvas(md_left[key], md_left["yrange"], md_left["xrange"], canvas_shape)
            right_full = embed_into_canvas(md_right[key], md_right["yrange"], md_right["xrange"], canvas_shape)
            merged_md[key] = np.hstack([left_full, right_full])

    output_ops = Path(output_ops)
    output_ops.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_ops, merged_md)

    print(f"Saved:\n  bin: {output_bin}\n  ops: {output_ops}")

base = Path("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed")
outpath = base.joinpath("output")
bin_files = list(base.rglob("data.bin"))
ops_files = list(base.rglob("ops.npy"))
plane_1_bins = bin_files[:2]
plane_1_ops = ops_files[:2]
from mbo_utilities.lazy_array import Suite2pArray
import lbm_suite2p_python as lsp
md1 = lsp.load_ops(ops_files[0])
md2 = lsp.load_ops(ops_files[1])
testing_dir = Path(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\testing")
output_bin = testing_dir.joinpath("concat.bin")
output_ops = testing_dir.joinpath("ops.npy")

concat_binfiles_and_merge_metadata(ops_files[0], ops_files[1], output_bin, output_ops)

data = mbo.imread(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\testing")
fpl.ImageWidget(data).show()
fpl.loop.run()

x = 2
mbo.imwrite(data, r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed", planes=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], roi=0)

files = list(Path(r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed").glob("*.tif"))

ops = suite2p.default_ops()
ops["roidetect"] = False

lsp.run_volume(files, ops=ops, )

folder1 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi1"
folder2 = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane01_roi2"
output_folder = r"D:\W2_DATA\kbarber\2025_07_17\mk355\green\processed\plane_01_roi1_roi2"

ops_merged = lsp.load_ops("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/ops.npy")
stat_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/stat.npy", allow_pickle=True)
iscell_merged = np.load("D:/W2_DATA/kbarber/2025_07_17/mk355/green/processed/plane_01_roi1_roi2/iscell.npy")

ops1 = np.load(folder1 + "/ops.npy", allow_pickle=True).item()
stat1 = np.load(folder1 + "/stat.npy", allow_pickle=True)
iscell1 = np.load(folder1 + "/iscell.npy", allow_pickle=True)

ops2 = np.load(folder2 + "/ops.npy", allow_pickle=True).item()
stat2 = np.load(folder2 + "/stat.npy", allow_pickle=True)
iscell2 = np.load(folder2 + "/iscell.npy", allow_pickle=True)

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
