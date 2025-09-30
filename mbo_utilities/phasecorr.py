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
    "mean-sub": lambda X: X[0]
    - np.mean(X, axis=0),  # mostly for compatibility with gui window functions
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
        If True, use FFT-based phase correlation (subpixel).
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
    else:
        # fast 1D correlation along x (row averages)
        a_mean = a.mean(0)
        b_mean = b_.mean(0)
        offsets = range(-max_offset, max_offset + 1)
        scores = [
            np.dot(a_mean[max_offset:-max_offset], np.roll(b_mean, k)[max_offset:-max_offset])
            for k in offsets
        ]
        dx = float(offsets[int(np.argmax(scores))])

    if max_offset:
        dx = np.sign(dx) * min(abs(dx), max_offset)
    return dx


def _apply_offset(img, offset, use_fft=False):
    """
    Apply one scalar `shift` (in X) to every *odd* row of an
    (..., Y, X) array.  Works for 2-D or 3-D stacks.
    """
    if img.ndim < 2:
        return img

    rows = img[..., 1::2, :]

    if use_fft:
        f = np.fft.fftn(rows, axes=(-2, -1))
        shift_vec = (0,) * (f.ndim - 1) + (offset,)  # e.g. (0,0,dx) for 3-D
        rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    else:
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
    return img


def bidir_phasecorr(arr, *, method="mean", use_fft=False,upsample=4, max_offset=4, border=0):
    """
    Correct for bi-directional scanning offsets in 2D or 3D array.

    Parameters
    ----------
    arr : ndarray
        Input array, either 2D (H, W) or 3D (N, H, W).
    method : str, optional
        Method to compute reference image for 3D arrays.
        Options: 'mean', 'max', 'std', 'mean-sub' or 'frame
        (for 2D arrays, only 'frame' or None).
    use_fft : bool, optional
        If True, use FFT-based phase correlation (subpixel).
    upsample : int, optional
        Subpixel precision for phase correlation.
    max_offset : int, optional
        Maximum allowed offset in pixels.
    border : int or tuple, optional
        Number of pixels to crop from edges (t, b, l, r).
    """

    if arr.ndim == 2:
        _offsets = _phase_corr_2d(arr, upsample, border, max_offset)
    else:
        flat = arr.reshape(arr.shape[0], *arr.shape[-2:])
        if method == "frame":
            _offsets = np.array(
                [_phase_corr_2d(frame=f, upsample=upsample, border=border, max_offset=max_offset, use_fft=use_fft) for f in flat]
            )
        else:
            if method not in MBO_WINDOW_METHODS:
                raise ValueError(f"unknown method {method}")
            _offsets = _phase_corr_2d(
                frame=MBO_WINDOW_METHODS[method](flat),
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft
            )

    if np.ndim(_offsets) == 0:  # scalar
        out = _apply_offset(arr.copy(), float(_offsets), use_fft)
    else:
        out = np.stack(
            [
                _apply_offset(f.copy(), float(s))  # or _apply_offset
                for f, s in zip(arr, _offsets)
            ]
        )
    return out, _offsets


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out


if __name__ == "__main__":
    from mbo_utilities import get_files, imread

    files = get_files(r"D:\tests\data", "tif")
    fpath = r"D:\W2_DATA\kbarber\2025_03_01\mk301\green"

