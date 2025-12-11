import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from mbo_utilities import log

TWO_DIM_PHASECORR_METHODS = {"frame", None}
THREE_DIM_PHASECORR_METHODS = ["mean", "max", "std", "mean-sub"]

MBO_WINDOW_METHODS = {
    "mean": lambda X: np.mean(X, axis=0),
    "max": lambda X: np.max(X, axis=0),
    "std": lambda X: np.std(X, axis=0),
    "mean-sub": lambda X: X[0] - np.mean(X, axis=0),
}

ALL_PHASECORR_METHODS = set(TWO_DIM_PHASECORR_METHODS) | set(
    THREE_DIM_PHASECORR_METHODS
)

logger = log.get("phasecorr")


def _phase_corr_2d(frame, upsample=4, border=0, max_offset=4, use_fft=False):
    """
    Estimate horizontal shift between even and odd rows of a 2D frame.

    Parameters
    ----------
    frame : ndarray (H, W)
        Input image.
    upsample : int
        Subpixel precision (only used if use_fft=True).
    border : int or tuple
        Number of pixels to crop from edges (t, b, l, r).
    max_offset : int
        Maximum shift allowed.
    use_fft : bool
        If True, use FFT-based 2D phase correlation (subpixel).
        If False, use fast integer-only correlation.
    """
    if frame.ndim != 2:
        raise ValueError("Expected 2D frame, got shape {}".format(frame.shape))

    h, w = frame.shape

    if isinstance(border, int):
        t = b = l = r = border
    else:
        t, b, l, r = border

    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])

    row_start = t
    row_end = m - b if b else m
    col_start = l
    col_end = w - r if r else w

    a = pre[row_start:row_end, col_start:col_end]
    b_ = post[row_start:row_end, col_start:col_end]

    if use_fft:
        _shift, *_ = phase_cross_correlation(a, b_, upsample_factor=upsample)
        dx = float(_shift[1])
        logger.debug(f"2D FFT phase correlation shift: {dx:.2f}")
    else:
        a_mean = a.mean(axis=0) - np.mean(a)
        b_mean = b_.mean(axis=0) - np.mean(b_)

        offsets = np.arange(-4, 4, 1)
        scores = np.empty_like(offsets, dtype=float)

        for i, k in enumerate(offsets):
            if k > 0:
                aa = a_mean[:-k]
                bb = b_mean[k:]
            elif k < 0:
                aa = a_mean[-k:]
                bb = b_mean[:k]
            else:
                aa = a_mean
                bb = b_mean
            num = np.dot(aa, bb)
            denom = np.linalg.norm(aa) * np.linalg.norm(bb)
            scores[i] = num / denom if denom else 0.0

        k_best = offsets[np.argmax(scores)]
        dx = -float(k_best)
        logger.debug(f"Integer phase correlation shift: {dx:.2f}")

    if max_offset:
        dx = np.sign(dx) * min(abs(dx), max_offset)
        logger.debug(f"Clipped shift to max_offset={max_offset}: {dx:.2f}")
    return dx


def _apply_offset(img, offset, use_fft=False):
    """
    Apply horizontal shift to every odd row of an (..., Y, X) array.

    Parameters
    ----------
    img : ndarray
        Image array to shift
    offset : float
        Horizontal shift in pixels
    use_fft : bool
        If True, use 2D FFT for subpixel shifting
    """
    if img.ndim < 2:
        return img

    rows = img[..., 1::2, :]

    if use_fft:
        f = np.fft.fftn(rows, axes=(-2, -1))
        shift_vec = (0,) * (f.ndim - 1) + (offset,)
        rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    else:
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
    return img


def bidir_phasecorr(
    arr, *, method="mean", use_fft=False, upsample=4, max_offset=10, border=4, offset=None
):
    """
    Correct for bi-directional scanning offsets in 2D or 3D array.

    Parameters
    ----------
    arr : ndarray
        Input array, either 2D (H, W) or 3D (N, H, W).
    method : str, optional
        Method to compute reference image for 3D arrays.
        Options: 'mean', 'max', 'std', 'mean-sub' or 'frame'.
        For 2D arrays, only 'frame' or None is used.
    use_fft : bool, optional
        If True, use FFT-based 2D phase correlation (subpixel).
    upsample : int, optional
        Subpixel precision for phase correlation.
    max_offset : int, optional
        Maximum allowed offset in pixels.
    border : int or tuple, optional
        Number of pixels to crop from edges (t, b, l, r).
    offset : float, optional
        If provided, skip offset computation and apply this precomputed offset directly.
        Useful for applying a consistent offset computed from a larger frame average.
    """
    logger.debug(f"bidir_phasecorr: arr={arr.shape}, method={method}, use_fft={use_fft}")

    if offset is not None:
        _offsets = float(offset)
        logger.debug(f"using precomputed offset: {_offsets}")
        out = _apply_offset(arr.copy(), _offsets, use_fft)
        return out, _offsets

    if arr.ndim == 2:
        _offsets = _phase_corr_2d(arr, upsample, border, max_offset, use_fft)
    else:
        flat = arr.reshape(arr.shape[0], *arr.shape[-2:])

        if method == "frame":
            logger.debug("Using individual frames for phase correlation")
            _offsets = np.array(
                [
                    _phase_corr_2d(
                        frame=f,
                        upsample=upsample,
                        border=border,
                        max_offset=max_offset,
                        use_fft=use_fft,
                    )
                    for f in flat
                ]
            )
        else:
            if method not in MBO_WINDOW_METHODS:
                raise ValueError(f"unknown method {method}")
            logger.debug(f"Using '{method}' window for phase correlation")
            _offsets = _phase_corr_2d(
                frame=MBO_WINDOW_METHODS[method](flat),
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft,
            )

    if np.ndim(_offsets) == 0:
        out = _apply_offset(arr.copy(), float(_offsets), use_fft)
    else:
        out = np.stack(
            [_apply_offset(f.copy(), float(s), use_fft) for f, s in zip(arr, _offsets)]
        )
    return out, _offsets


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out
