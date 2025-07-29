import numpy as np
from scipy.ndimage import sobel
from skimage.registration import phase_cross_correlation


def snr_roi(data, y0, y1, x0, x1):
    """Higher = Better"""
    roi = data[:, y0:y1, x0:x1]
    mean_signal = np.mean(roi)
    noise = np.std(roi)
    return mean_signal / noise


def mean_row_misalignment(data):
    """Lower is better"""
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
