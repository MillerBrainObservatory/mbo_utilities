import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


def _phase_offset(frame, upsample=10, border=0, max_offset=8):
    if frame.ndim == 3:
        frame = frame.mean(0)

    frame = frame.astype(np.float32, copy=False)
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

    shift, *_ = phase_cross_correlation(a, b_, upsample_factor=upsample)
    dx = float(shift[1])
    return np.sign(dx) * min(abs(dx), max_offset)

def _apply_offset(frame, off):
    rows = frame[1::2]
    f = np.fft.fftn(rows)
    f = fourier_shift(f, (0, off))
    frame[1::2] = np.fft.ifftn(f).real
    return frame


def compute_scan_phase_offsets(arr, *, method="subpix", upsample=10, max_offset=8, border=0):
    a = np.asarray(arr)
    if a.ndim == 2:
        return _phase_offset(a, upsample, max_offset, border)
    flat = a.reshape(a.shape[0], *a.shape[-2:])
    if method == "subpix":
        return np.array([_phase_offset(f, upsample, max_offset, border) for f in flat], dtype=np.float32)
    if method == "two_step":
        offs = []
        for f in flat:
            o1 = _phase_offset(f, upsample, max_offset, border)
            f2 = _apply_offset(f.copy(), o1)
            o2 = _phase_offset(f2, upsample, max_offset, border)
            offs.append(o1 + o2)
        return np.array(offs, dtype=np.float32)
    raise ValueError(method)


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out
