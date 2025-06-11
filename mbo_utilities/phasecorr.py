import numpy as np
from scipy.signal import correlate
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation

from mbo_utilities import log

TWO_DIM_PHASECORR_METHODS = {"frame"}
THREE_DIM_PHASECORR_METHODS = ["mean", "max", "std", "mean-sub"]

MBO_WINDOW_METHODS = {
    "frame":     lambda X: np.mean(X, axis=0),
    "mean":      lambda X: np.mean(X, axis=0),
    "max":       lambda X: np.max(X, axis=0),
    "std":       lambda X: np.std(X, axis=0),
    # pick first frame then subtract global mean
    "mean-sub":  lambda X: X[0] - np.mean(X, axis=0),
}

logger = log.get("phasecorr")


def _return_scan_offset(
        img: np.ndarray,
        nvals: int = 8
) -> int:
    """legacy cross-correlation offset on a 2-D image

    Translated to Python from Demas et al. 2021: https://www.nature.com/articles/s41592-021-01239-8#Sec2
    """
    in_pre, in_post = img[::2], img[1::2]
    n = min(in_pre.shape[0], in_post.shape[0])
    in_pre, in_post = in_pre[:n], in_post[:n]

    in_pre = np.hstack([np.zeros((n, nvals)), in_pre, np.zeros((n, nvals))]).T.ravel("F")
    in_post = np.hstack([np.zeros((n, nvals)), in_post, np.zeros((n, nvals))]).T.ravel("F")

    in_post -= in_post.mean()
    in_post[in_post < 0] = 0
    in_pre[in_pre < 0] = 0

    r = correlate(in_pre, in_post, mode="full") \
        / (len(in_pre) - np.abs(np.arange(-len(in_pre)+1, len(in_pre))))
    lags = np.arange(-nvals, nvals + 1)
    return lags[np.argmax(r[len(r)//2-nvals : len(r)//2+nvals+1])]  # noqa

def _fix_scan_phase(img: np.ndarray, offset: int) -> np.ndarray:
    """integer-pixel phase fix (even rows ←→ odd rows)

    Translated to Python from Demas et al. 2021: https://www.nature.com/articles/s41592-021-01239-8#Sec2
    """
    # flip the sign of the offset to match the original code
    if offset == 0:
        return img
    out = np.zeros_like(img)
    if img.ndim == 2:
        even, odd = img[0::2], img[1::2]
        if offset > 0:
            out[0::2,:-offset], out[1::2, offset:] = even[:, offset:], odd[:, :-offset]
        else:
            offset = -offset
            out[0::2, offset:], out[1::2, :-offset] = even[:, :-offset], odd[:, offset:]
    else:                               # 3-D or 4-D stack
        out[:] = np.stack([_fix_scan_phase(f, offset) for f in img])
    return out

def _phase_corr_2d(
        frame,
        upsample=1,
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

def _apply_offset(img, shift):
    """
    Apply one scalar `shift` (in X) to every *odd* row of an
    (..., Y, X) array.  Works for 2-D or 3-D stacks.
    """
    if img.ndim < 2:
        return img

    rows = img[..., 1::2, :]

    f = np.fft.fftn(rows, axes=(-2, -1))
    shift_vec = (0,) * (f.ndim - 1) + (shift,)   # e.g. (0,0,dx) for 3-D
    rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec),
                           axes=(-2, -1)).real
    return img

def nd_windowed(arr, *, method="mean", upsample=1,
                max_offset=4, border=2):
    """Return (corrected array, offsets)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        offs = _phase_corr_2d(a, upsample, border, max_offset)
    else:
        flat = a.reshape(a.shape[0], *a.shape[-2:])
        if method == "frame":
            offs = np.array(
                [_phase_corr_2d(f, upsample, border, max_offset) for f in flat]
            )
        else:
            if method not in MBO_WINDOW_METHODS:
                raise ValueError(f"unknown method {method}")
            img = MBO_WINDOW_METHODS[method](flat)
            offs = _phase_corr_2d(img, upsample, border, max_offset)
    if np.ndim(offs) == 0:  # scalar
        corrected = _apply_offset(a.copy(), float(offs))
    else:
        corrected = np.stack(
            [_apply_offset(f.copy(), float(s))  # or _apply_offset
             for f, s in zip(a, offs)]
        )
    return corrected, offs


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out

if __name__ == "__main__":
    from mbo_utilities import get_files
    from mbo_utilities.lazy_array import LazyArrayLoader

    files = get_files(r"D:\tests\data", "tif")
    if not files:
        raise ValueError("No files found matching '*.tif'")

    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    array_object = LazyArrayLoader(files[0])
    lazy_array = array_object.load()
    lazy_array.fix_phase = False
    array = lazy_array[:200, 11, :, :]

    # test legacy scanphase correction
    dx = _return_scan_offset(array[0])
    array_1 = _fix_scan_phase(array, dx)


    methods = ["frame", "mean", "max", "std", "mean-sub",]
    arrs = [array_1]
    ofs = [dx]
    for m in methods:
        corr, offs = nd_windowed(array, method=m, upsample=2)
        arrs.append(corr)
        ofs.append(offs)

    import fastplotlib as fpl
    iw = fpl.ImageWidget(
        data=arrs,
        names=["Cross-Corr"] + methods,
        histogram_widget=True,
        figure_kwargs={"size": (1200, 800)},
        graphic_kwargs={"vmin": array.min(), "vmax": array.max()},
        window_funcs={"t": (np.mean, 0)},
    )
    iw.show()
    fpl.loop.run()
    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # for ax, m in zip(axs.flat, methods):
    #     corr, offs = nd_windowed(array, method=m, upsample=2)
    #     ax.imshow(corr.mean(0)[150:170, 330:350], cmap="gray")
    #     ax.set_title(f"{m}\nμ={np.mean(offs):.2f}")
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()
