# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }

import mbo_utilities as mbo
import numpy as np
from pathlib import Path
import tifffile
import fastplotlib as fpl

from mbo_utilities.phasecorr import nd_windowed_compute_optimal_offset, apply_scan_phase_offsets

def _phase_offset(frame, upsample=10, border=0, max_offset=3):
    if frame.ndim == 3:
        frame = frame.mean(axis=0)

    h, w = frame.shape

    if isinstance(border, int):
        t = b = l = r = border
    else:
        t, b, l, r = border

    pre, post = frame[::2], frame[1::2]
    m = min(pre.shape[0], post.shape[0])

    row_start = t
    row_end = m - b if b else m
    col_start = l
    col_end = w - r if r else w

    a = pre[row_start:row_end, col_start:col_end]
    b_ = post[row_start:row_end, col_start:col_end]

    from skimage.registration import phase_cross_correlation
    shift, *_ = phase_cross_correlation(a, b_, upsample_factor=upsample)
    dx = float(shift[1])
    if max_offset:
        return np.sign(dx) * min(abs(dx), max_offset)
    return dx


fname_scan = r"/home/flynn/lbm_data/raw"
save_path = Path("/home/flynn/lbm_data")

uncor_save = save_path.joinpath("uncorrected")
corrected_save = save_path.joinpath("corrected")

scan_roi_1 = mbo.read_scan(
    fname_scan,
    roi=1,
    fix_phase=True,
)
z11r1 = scan_roi_1[:, 11, :, :]

# # mbo.save_as(raw_scan, uncor_save, planes=[11], fix_phase=False)
# mbo.save_as(
#     raw_scan,
#     corrected_save,
#     planes=[11],
#     fix_phase=True,
#     target_chunk_mb=50,
#     debug=False,
#     ext="bin",
# )

x = 2
