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
        a_mean = a.mean(axis=0) - np.mean(a)
        b_mean = b_.mean(axis=0) - np.mean(b_)

        offsets = np.arange(-4, 4, 1)
        scores = np.empty_like(offsets, dtype=float)

        for i, k in enumerate(offsets):
            # valid overlap, no wrapping
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
        shift_vec = (0,) * (f.ndim - 1) + (offset,)
        rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    else:
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
    return img


def bidir_phasecorr(
    arr, *, method="mean", use_fft=False, upsample=4, max_offset=4, border=0
):
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
            _offsets = _phase_corr_2d(
                frame=MBO_WINDOW_METHODS[method](flat),
                upsample=upsample,
                border=border,
                max_offset=max_offset,
                use_fft=use_fft,
            )

    if np.ndim(_offsets) == 0:  # scalar
        out = _apply_offset(arr.copy(), float(_offsets), use_fft)
    else:
        out = np.stack(
            [
                _apply_offset(f.copy(), float(s))
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


def compute_scan_offsets(tiff_path, max_lag=8):
    """
    Compute scan phase offsets for each z-plane in a TIFF stack.

    Matches the matlab implementation of demas et.al. 2021.

    Parameters
    ----------
    tiff_path : str or Path
        Path to multi-plane TIFF file.
    max_lag : int, optional
        Maximum lag to search in cross-correlation (default 8).

    Returns
    -------
    offsets : np.ndarray, shape (n_planes,)
        Detected scan offsets (in pixels) for each plane.
    """
    import tifffile
    from pathlib import Path
    from scipy.signal import correlate

    tiff_path = Path(tiff_path)
    data = tifffile.imread(
        tiff_path
    )  # shape = (T, Y, X, C?) depending on ScanImage export
    if data.ndim == 2:
        raise ValueError("Expected multi-plane data, got single frame")

    # assume (frames, y, x) or (frames, y, x, planes)
    if data.ndim == 3:
        # no explicit plane axis, treat each frame as a plane
        n_planes = data.shape[0]
        vol = data
    elif data.ndim == 4:
        # (frames, y, x, planes)
        n_planes = data.shape[-1]
        vol = np.moveaxis(data, -1, 0)  # (planes, frames, y, x)
    else:
        raise ValueError(f"Unexpected TIFF shape {data.shape}")

    offsets = []
    for p in range(n_planes):
        plane_data = vol[p] if vol.ndim == 3 else vol[p].max(axis=0)
        if vol.ndim == 3:
            img = plane_data
        else:
            img = plane_data

        # odd/even line split
        v1 = img[:, ::2].astype(float).ravel()
        v2 = img[:, 1::2].astype(float).ravel()

        v1 -= v1.mean()
        v2 -= v2.mean()
        v1[v1 < 0] = 0
        v2[v2 < 0] = 0

        corr = correlate(v1, v2, mode="full", method="auto")
        mid = len(corr) // 2
        search = corr[mid - max_lag : mid + max_lag + 1]
        lags = np.arange(-max_lag, max_lag + 1)
        offsets.append(lags[np.argmax(search)])

    return np.array(offsets, dtype=int)


if __name__ == "__main__":

    import numpy as np
    import tifffile
    from pathlib import Path
    from mbo_utilities import imread

    data_path = Path(
        r"D:\W2_DATA\kbarber\07_27_2025\mk355\raw\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif")
    data = imread(data_path)
    data.fix_phase = False

    test = []
    for idx in range(5):
        frame = data[idx, 0, :, :]
        dx_int = _phase_corr_2d(frame, use_fft=False)
        print(f"no fft: {dx_int}")
        dx_fft = _phase_corr_2d(frame, use_fft=True)
        print(f"with fft: {dx_fft}")