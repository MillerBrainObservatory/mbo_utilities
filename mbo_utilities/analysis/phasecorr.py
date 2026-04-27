import numpy as np

from mbo_utilities import log

TWO_DIM_PHASECORR_METHODS = {"frame", None}
THREE_DIM_PHASECORR_METHODS = ["mean", "max", "std", "mean-sub"]

MBO_WINDOW_METHODS = {
    "mean": lambda X: np.mean(X, axis=0),
    "max": lambda X: np.max(X, axis=0),
    "std": lambda X: np.std(X, axis=0),
    "mean-sub": lambda X: X[0] - np.mean(X, axis=0),
}

ALL_PHASECORR_METHODS = set(TWO_DIM_PHASECORR_METHODS) | set(THREE_DIM_PHASECORR_METHODS)

logger = log.get("phasecorr")


def _phase_corr_2d(frame, upsample=4, border=0, max_offset=4, use_fft=True):
    """Estimate horizontal shift between odd and even rows via 1D rFFT phase
    correlation along x. Parabolic peak refinement gives ~0.05 px precision.
    `use_fft=False` returns the integer-rounded result. `upsample` is ignored
    (kept for API stability)."""
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape {frame.shape}")

    if isinstance(border, int):
        t = b = l = r = border
    else:
        t, b, l, r = border

    # split into even/odd FIRST, then crop. cropping the full frame before
    # splitting flips parity when t is odd (rows 3,5,7,... become "even").
    h, w = frame.shape
    pre  = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    pre  = pre[t:m - b if b else m, l:w - r if r else w]
    post = post[t:m - b if b else m, l:w - r if r else w]
    even = pre.astype(np.float32, copy=False)
    odd  = post.astype(np.float32, copy=False)

    Lx = even.shape[-1]
    fe = np.fft.rfft(even, axis=-1); fe /= np.abs(fe) + 1e-5
    fo = np.fft.rfft(odd,  axis=-1); fo /= np.abs(fo) + 1e-5
    cc = np.fft.fftshift(np.fft.irfft(fe * np.conj(fo), n=Lx, axis=-1).mean(axis=0))

    win = max_offset or Lx // 2
    lo, hi = Lx // 2 - win, Lx // 2 + win + 1
    int_shift = int(np.argmax(cc[lo:hi]) - win)

    if not use_fft:
        return float(int_shift)

    idx = Lx // 2 + int_shift
    if 0 < idx < Lx - 1:
        y0, y1, y2 = cc[idx - 1], cc[idx], cc[idx + 1]
        denom = y0 - 2 * y1 + y2
        sub = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        return float(int_shift) + float(np.clip(sub, -0.5, 0.5))
    return float(int_shift)


def _apply_offset(img, offset, use_fft=False):
    """Shift every odd row of `img` by `offset` pixels along x, in place.
    Subpixel via 1D rFFT along x; otherwise integer np.roll."""
    if img.ndim < 2:
        return img
    rows = img[..., 1::2, :]
    if use_fft and offset != round(offset):
        n = rows.shape[-1]
        f = np.fft.rfft(rows.astype(np.float32, copy=False), axis=-1)
        k = np.fft.rfftfreq(n)
        phase = np.exp(-2j * np.pi * k * offset).astype(np.complex64)
        shifted = np.fft.irfft(f * phase, n=n, axis=-1)
        img[..., 1::2, :] = shifted.astype(img.dtype, copy=False)
    else:
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
    return img


def bidir_phasecorr(
    arr, *, method="mean", use_fft=False, upsample=4, max_offset=10, border=4, offset=None
):
    """Correct bidirectional scan offset on a 2D or 3D array.

    Returns (corrected, offset_or_offsets). Pass `offset=` to skip estimation.
    `method` controls the reduction for 3D inputs: 'mean'/'max'/'std'/'mean-sub'
    estimate one offset from the reduced image; 'frame' estimates per-frame.
    """
    if offset is not None:
        off = float(offset)
        return _apply_offset(arr.copy(), off, use_fft), off

    if arr.ndim == 2:
        offs = _phase_corr_2d(arr, upsample, border, max_offset, use_fft)
    else:
        flat = arr.reshape(arr.shape[0], *arr.shape[-2:])
        if method == "frame":
            offs = np.array([
                _phase_corr_2d(f, upsample, border, max_offset, use_fft) for f in flat
            ])
        elif method in MBO_WINDOW_METHODS:
            offs = _phase_corr_2d(
                MBO_WINDOW_METHODS[method](flat), upsample, border, max_offset, use_fft
            )
        else:
            raise ValueError(f"unknown method {method}")

    if np.ndim(offs) == 0:
        out = _apply_offset(arr.copy(), float(offs), use_fft)
    else:
        out = np.stack([
            _apply_offset(f.copy(), float(s), use_fft)
            for f, s in zip(arr, offs, strict=False)
        ])
    return out, offs


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs, use_fft=True)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off, use_fft=True)
    return out
