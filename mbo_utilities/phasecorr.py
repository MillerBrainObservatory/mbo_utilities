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


def _phase_corr_1d_fft(a, b, upsample=10):
    """
    Fast 1D horizontal phase correlation using row-averaged signals.

    This is optimized for bi-directional scan phase correction where we only
    care about horizontal shift. Much faster than full 2D phase correlation.

    Parameters
    ----------
    a, b : ndarray (H, W)
        Even and odd row images to compare
    upsample : int
        Upsampling factor for subpixel precision

    Returns
    -------
    float
        Horizontal shift in pixels
    """
    # Average along rows to get 1D signals
    a_1d = a.mean(axis=0)
    b_1d = b.mean(axis=0)

    # 1D FFT (much faster than 2D)
    A = np.fft.fft(a_1d)
    B = np.fft.fft(b_1d)

    # Cross-power spectrum
    cross_power = A * np.conj(B)
    cross_power /= np.abs(cross_power) + 1e-10

    # Inverse FFT to get correlation
    correlation = np.fft.ifft(cross_power).real

    # Find peak with subpixel precision
    maxima = np.argmax(correlation)

    if upsample > 1:
        # Refine around peak
        # Create upsampled region around peak
        width = 3
        shift_range = np.linspace(maxima - width, maxima + width, width * upsample * 2)

        # Evaluate correlation at upsampled points
        upsampled_corr = np.zeros_like(shift_range)
        for i, shift in enumerate(shift_range):
            # Apply subpixel shift in frequency domain
            freq = np.fft.fftfreq(len(a_1d))
            phase_shift = np.exp(-2j * np.pi * freq * shift)
            shifted_B = np.fft.ifft(B * phase_shift).real
            upsampled_corr[i] = np.dot(a_1d, shifted_B)

        maxima = shift_range[np.argmax(upsampled_corr)]

    # Handle wrap-around
    shift = maxima
    if shift > len(a_1d) / 2:
        shift -= len(a_1d)

    return float(shift)


def _phase_corr_2d(frame, upsample=4, border=0, max_offset=4, use_fft=False, fft_method="1d"):
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
    fft_method : str
        FFT method to use if use_fft=True:
        - '1d': Fast 1D correlation (horizontal only, ~10x faster)
        - '2d': Full 2D correlation (scikit-image, more accurate)
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
        if fft_method == "1d":
            dx = _phase_corr_1d_fft(a, b_, upsample=upsample)
            logger.debug(f"1D FFT phase correlation shift: {dx:.2f}")
        else:  # fft_method == "2d"
            _shift, *_ = phase_cross_correlation(a, b_, upsample_factor=upsample)
            dx = float(_shift[1])
            logger.debug(f"2D FFT phase correlation shift: {dx:.2f}")
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
        # Integer method tests "how much is b already shifted?"
        # If b is shifted RIGHT by k, we need to shift LEFT by -k to correct
        dx = -float(k_best)
        logger.debug(f"Integer phase correlation shift: {dx:.2f}")

    if max_offset:
        dx = np.sign(dx) * min(abs(dx), max_offset)
        logger.debug(f"Clipped shift to max_offset={max_offset}: {dx:.2f}")
    return dx


def _apply_offset(img, offset, use_fft=False, fft_method="1d"):
    """
    Apply one scalar `shift` (in X) to every *odd* row of an
    (..., Y, X) array.  Works for 2-D or 3-D stacks.

    Parameters
    ----------
    img : ndarray
        Image array to shift
    offset : float
        Horizontal shift in pixels
    use_fft : bool
        If True, use FFT for subpixel shifting
    fft_method : str
        FFT method if use_fft=True:
        - '1d': Fast 1D FFT per row (horizontal only, ~5x faster)
        - '2d': Full 2D FFT (scipy.ndimage.fourier_shift)
    """
    if img.ndim < 2:
        return img

    rows = img[..., 1::2, :]

    if use_fft:
        if fft_method == "1d":
            # 1D FFT is much faster - only shift horizontally
            freq = np.fft.fftfreq(rows.shape[-1])
            phase_shift = np.exp(-2j * np.pi * freq * offset)

            # Apply to each row
            original_shape = rows.shape
            rows_2d = rows.reshape(-1, rows.shape[-1])
            for i in range(rows_2d.shape[0]):
                row_fft = np.fft.fft(rows_2d[i])
                rows_2d[i] = np.fft.ifft(row_fft * phase_shift).real

            rows[:] = rows_2d.reshape(original_shape)
        else:  # fft_method == "2d"
            f = np.fft.fftn(rows, axes=(-2, -1))
            shift_vec = (0,) * (f.ndim - 1) + (offset,)
            rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    else:
        rows[:] = np.roll(rows, shift=int(round(offset)), axis=-1)
    return img


def bidir_phasecorr(
    arr, *, method="mean", use_fft=False, upsample=4, max_offset=10, border=4, fft_method="2d",
    z_aware=False, num_z_planes=None, min_window_size=None
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
    fft_method : str, optional
        FFT method if use_fft=True:
        - '1d': Fast 1D correlation (horizontal only, ~10x faster, recommended)
        - '2d': Full 2D correlation (scikit-image, slightly more accurate)
    z_aware : bool, optional
        Deprecated - no longer used. Phase correction is applied consistently regardless of z-planes.
    num_z_planes : int, optional
        Deprecated - no longer used.
    min_window_size : int, optional
        Deprecated - no longer enforced. Phase correction is always applied.
    """
    if arr.ndim == 2:
        _offsets = _phase_corr_2d(arr, upsample, border, max_offset, use_fft, fft_method)
    else:
        flat = arr.reshape(arr.shape[0], *arr.shape[-2:])

        # Apply phase correction consistently
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
                        fft_method=fft_method,
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
                fft_method=fft_method,
            )

    if np.ndim(_offsets) == 0:  # scalar
        out = _apply_offset(arr.copy(), float(_offsets), use_fft, fft_method)
    else:
        out = np.stack(
            [_apply_offset(f.copy(), float(s), use_fft, fft_method) for f, s in zip(arr, _offsets)]
        )
    return out, _offsets


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out
