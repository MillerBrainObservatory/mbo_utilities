import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from . import log

TWO_DIM_PHASECORR_METHODS = {"frame", "mean", "max", "std", "mean-sub", "mean-sub-std"}
THREE_DIM_PHASECORR_METHODS = ["mean", "max", "std", "mean-sub"]

MBO_WINDOW_METHODS = {
    "mean": lambda X: np.mean(X, axis=0),
    "max": lambda X: np.max(X, axis=0),
    "std": lambda X: np.std(X, axis=0),
    "mean-sub": lambda X: X - np.mean(X, axis=0),
    "mean-sub-std": lambda X: (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8),
}

logger = log.get("phasecorr")

def _phase_offset(
        frame,
        upsample=10,
        border=0,
        max_offset=4
):
    if frame.ndim != 2:
        raise ValueError("Expected a 2D frame, got a 3D array.")

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
    if max_offset:
        return np.sign(dx) * min(abs(dx), max_offset)
    return dx

def _apply_offset(frame, shift):
    rows = frame[1::2]
    f = np.fft.fftn(rows)
    f = fourier_shift(f, (0, shift))
    frame[1::2] = np.fft.ifftn(f).real
    return frame


def compute_scan_phase_offsets(
        arr,
        method="mean",
        upsample=10,
        max_offset=4,
        border=2,
):
    """
    Compute scan‐phase offsets. If `arr` is 2D, always run a single‐image offset.
    If `arr` is 3D (time × height × width), one of:

      - "frame"      → compute offset frame‐by‐frame (returns a 1D array of length T)
      - "mean", "max", "std"       → collapse along time with np.mean/np.max/np.std first
      - "mean-sub"   → subtract the temporal mean from each frame, then run offset on that difference
      - "mean-sub-std" → first z‐score each pixel over time, then compute offset on that z‐scored image

    """
    a = np.asarray(arr)
    if a.ndim == 2:
        if method not in TWO_DIM_PHASECORR_METHODS:
            logger.debug(
                "Attempted to use a windowed phase-corr method on 2D data."
                f"Available 2D methods: {TWO_DIM_PHASECORR_METHODS}"
            )
        return _phase_offset(
            a,
            upsample=upsample,
            border=border,
            max_offset=max_offset
        )
    # flatten z/t
    flat = a.reshape(a.shape[0], *a.shape[-2:])

    # one offset per frame
    if method == "frame":
        return np.array([_phase_offset(
            f,
            upsample=upsample,
            border=border,
            max_offset=max_offset,
        ) for f in flat])  # dtype=np.float32)

    if method not in MBO_WINDOW_METHODS:
        raise ValueError(f"Unknown phase‐corr method: {method!r}")

    image = MBO_WINDOW_METHODS[method](flat)
    return _phase_offset(image, upsample=upsample, border=border, max_offset=max_offset)

def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out
