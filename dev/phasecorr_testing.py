from pathlib import Path
import fastplotlib as fpl
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim

try:
    import mbo_utilities as mbo
except ImportError:
    print("uv pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@dev")

from mbo_utilities.phasecorr import phase_offsets_timecourse, apply_patchwise_offsets_v2

raw_files = [x for x in Path(r"D:\tests_bigmem\roi2").glob("*.tif*")]

# Load data
zplane_name = "plane10"
lazy = mbo.imread(r"D:\W2_DATA\kbarber\2025_03_01\mk301\green", fix_phase=False, roi=2)
# mbo.imwrite(lazy, r"D:\\phasecorr\\plane10_uncorrected",planes=[10])

data = mbo.imread(r"D:\\phasecorr\\plane10_corrected\\roi2\\plane10.tif")
xsplits, patch_offsets = phase_offsets_timecourse(data, n_parts=4, method="mean")
patch_corrected = apply_patchwise_offsets_v2(data, xsplits, patch_offsets)
mbo.imwrite(r"D:\\phasecorr", patch_corrected)

# tifffile.imwrite(r"D:\phasecorr\patched_corrected.tif", corrected, photometric="minisblack")
corrected = tifffile.imread(r"D:\phasecorr\\plane10_corrected\\roi2\\plane10.tif")
data = tifffile.imread(r"D:\phasecorr\\plane10_uncorrected\\roi2\\plane10.tif")
iw = fpl.ImageWidget([data, corrected])
iw.show()
fpl.loop.run()