import numpy as np
from scipy.ndimage import sobel
from skimage.registration import phase_cross_correlation


def snr_roi(data, y0, y1, x0, x1):
    """Higher = Better."""
    roi = data[:, y0:y1, x0:x1].astype(np.float32)
    roi -= roi.min()
    mean_signal = np.mean(roi)
    noise = np.std(roi)
    return mean_signal / noise


def mean_row_misalignment(data):
    """Lower is better."""
    offsets = []
    for frame in data:
        even = frame[::2]
        odd = frame[1::2]
        m = min(len(even), len(odd))
        shift, _, _ = phase_cross_correlation(even[:m], odd[:m], upsample_factor=10)
        offsets.append(abs(shift[1]))  # X-axis
    return np.mean(offsets)


def temporal_corr(data, x0, x1, y0, y1):
    patch = data[:, y0:y1, x0:x1]
    t = patch.shape[0]
    corrs = [
        np.corrcoef(patch[i].ravel(), patch[i + 1].ravel())[0, 1] for i in range(t - 1)
    ]
    return np.nanmean(corrs)


def sharpness_metric(frame):
    gx = sobel(frame, axis=0)
    gy = sobel(frame, axis=1)
    return np.mean(np.sqrt(gx**2 + gy**2))


def avg_sharpness(data):
    return np.mean([sharpness_metric(f) for f in data])


def frame_correlations(data, subsample: int = 1):
    """
    Compute frame-to-frame correlations for an image stack.

    Parameters
    ----------
    data : array-like
        3D array of shape (t, y, x) or 4D array (t, z, y, x).
        If 4D, correlations are computed per z-plane.
    subsample : int
        Subsample factor for faster computation. Default 1 (no subsampling).

    Returns
    -------
    correlations : np.ndarray
        Array of correlation values between consecutive frames.
        Shape (t-1,) for 3D input, (t-1, z) for 4D input.
    """
    data = np.asarray(data)

    if data.ndim == 4:
        # (t, z, y, x) - compute per z-plane
        t, z, _y, _x = data.shape
        correlations = np.zeros((t - 1, z), dtype=np.float32)
        for zi in range(z):
            plane = data[:, zi, ::subsample, ::subsample]
            for i in range(t - 1):
                f1 = plane[i].ravel()
                f2 = plane[i + 1].ravel()
                correlations[i, zi] = np.corrcoef(f1, f2)[0, 1]
        return correlations

    if data.ndim == 3:
        # (t, y, x)
        t = data.shape[0]
        if subsample > 1:
            data = data[:, ::subsample, ::subsample]
        correlations = np.zeros(t - 1, dtype=np.float32)
        for i in range(t - 1):
            f1 = data[i].ravel()
            f2 = data[i + 1].ravel()
            correlations[i] = np.corrcoef(f1, f2)[0, 1]
        return correlations

    raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")


def detect_dropped_frames(
    data,
    threshold: float | None = None,
    n_std: float = 3.0,
    subsample: int = 1,
):
    """
    Detect potential dropped frames based on correlation anomalies.

    Dropped frames typically show lower correlation with their neighbors
    because they represent a temporal discontinuity.

    Parameters
    ----------
    data : array-like
        3D array (t, y, x) or 4D array (t, z, y, x).
    threshold : float, optional
        Absolute correlation threshold. Frames with correlation below
        this are flagged. If None, uses adaptive threshold based on n_std.
    n_std : float
        Number of standard deviations below mean to flag as dropped.
        Only used if threshold is None. Default 3.0.
    subsample : int
        Subsample factor for faster computation. Default 1.

    Returns
    -------
    dict with keys:
        - 'correlations': array of frame-to-frame correlations
        - 'dropped_indices': indices of potential dropped frames
        - 'threshold': the threshold used for detection
        - 'mean_corr': mean correlation value
        - 'std_corr': standard deviation of correlations
    """
    correlations = frame_correlations(data, subsample=subsample)

    # For 4D data, average across z-planes for detection
    if correlations.ndim == 2:
        corr_avg = np.nanmean(correlations, axis=1)
    else:
        corr_avg = correlations

    mean_corr = np.nanmean(corr_avg)
    std_corr = np.nanstd(corr_avg)

    if threshold is None:
        threshold = mean_corr - n_std * std_corr

    # Find frames where correlation drops below threshold
    # A low correlation at index i means frame i+1 is suspicious
    # (it doesn't correlate well with frame i)
    low_corr_indices = np.where(corr_avg < threshold)[0]

    # The dropped frame is likely at index i+1 (the frame after the low correlation)
    dropped_indices = low_corr_indices + 1

    return {
        "correlations": correlations,
        "dropped_indices": dropped_indices,
        "threshold": threshold,
        "mean_corr": mean_corr,
        "std_corr": std_corr,
    }
