from pathlib import Path
import numpy as np
import mbo_utilities as mbo
# import lbm_suite2p_python as lsp
# import suite2p
# import matplotlib.pyplot as plt
import fastplotlib as fpl
from tqdm.auto import tqdm
import tifffile
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

import numpy as np
import tifffile
from pathlib import Path
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import glob, h5py


# files = sorted(glob.glob(r"D:\W2_DATA\kbarber\03_01\raw\*.tif*"))
# stack = np.stack([tifffile.imread(f).squeeze() for f in files], axis=0)
#
# with h5py.File("mk301_plane7.h5", "w") as h5:
#     h5["mov"] = stack

x = 2

# data = mbo.imread(r"D:\W2_DATA\kbarber\03_01\plane01_roi1.tif")

def _cast(arr, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)

def register_planes(
        filepaths,
        anchor_idx=0,
        projection="max",
        upsample=10,
        save=True,
        out_suffix="registered.tif"):
    if not filepaths:
        raise ValueError("filepaths list is empty")
    if anchor_idx < 0 or anchor_idx >= len(filepaths):
        raise ValueError("anchor_idx out of range")
    try:
        proj_func = getattr(np, projection)
    except AttributeError:
        raise ValueError(f"'{projection}' is not a valid numpy function")

    projs = []
    stacks = []
    for f in filepaths:
        mem = tifffile.memmap(f)
        stacks.append(mem)
        projs.append(proj_func(mem, axis=0))

    n = len(filepaths)
    pairwise = []
    for i in tqdm(range(n - 1), desc="Pairwise shifts"):
        shift, _, _ = phase_cross_correlation(projs[i], projs[i + 1], upsample_factor=upsample)
        pairwise.append(np.asarray(shift, dtype=np.float64))

    cum = [None] * n
    cum[anchor_idx] = np.array([0.0, 0.0])
    for i in range(anchor_idx + 1, n):
        cum[i] = cum[i - 1] + pairwise[i - 1]
    for i in range(anchor_idx - 1, -1, -1):
        cum[i] = cum[i + 1] - pairwise[i]

    shifts = {filepaths[i]: (float(cum[i][0]), float(cum[i][1])) for i in range(n)}

    if save:
        for i, f in enumerate(filepaths):
            stack = stacks[i]
            out = np.empty_like(stack)
            for t in range(stack.shape[0]):
                F = np.fft.fftn(stack[t])
                aligned = np.fft.ifftn(fourier_shift(F, cum[i])).real
                out[t] = _cast(aligned, stack.dtype)
            p = Path(f)
            suf = out_suffix if out_suffix.lower().endswith(".tif") else out_suffix + ".tif"
            tifffile.imwrite(str(p.with_name(p.stem + suf)), out)

    return shifts

reg = True

if reg:
    files = list(Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_mean\pre_reg").glob("*.*")) # grab all files in directory
    dxdy = register_planes([str(f) for f in files], projection="mean")
    x = 2
else:
    reg_fpath = Path(r"D:\W2_DATA\kbarber\07_27_2025\mk355\out_mean")
    files = list(reg_fpath.glob("*registered*"))

# make a 3D max-projection stack
volume = []
for f in files:
    mem = tifffile.memmap(f)
    volume.append(mem.max(axis=0))
volume = np.stack(volume, axis=0)

fpl.ImageWidget(volume).show()
fpl.loop.run()

x = 2