import numpy as np
from scipy.ndimage import fourier_shift, shift
from skimage.registration import phase_cross_correlation

from mbo_utilities import log

TWO_DIM_PHASECORR_METHODS = {"frame"}
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


def _phase_corr_2d(frame, upsample=4, border=0, max_offset=4):
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

    _shift, *_ = phase_cross_correlation(a, b_, upsample_factor=upsample)
    dx = float(_shift[1])
    if max_offset:
        return np.sign(dx) * min(abs(dx), max_offset)
    return dx


def _apply_offset(img, offset):
    """
    Apply one scalar `shift` (in X) to every *odd* row of an
    (..., Y, X) array.  Works for 2-D or 3-D stacks.
    """
    if img.ndim < 2:
        return img

    rows = img[..., 1::2, :]

    f = np.fft.fftn(rows, axes=(-2, -1))
    shift_vec = (0,) * (f.ndim - 1) + (offset,)  # e.g. (0,0,dx) for 3-D
    rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec), axes=(-2, -1)).real
    return img


def nd_windowed(arr, *, method="frame", upsample=4, max_offset=4, border=0):
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
            [
                _apply_offset(f.copy(), float(s))  # or _apply_offset
                for f, s in zip(a, offs)
            ]
        )
    return corrected, offs


def apply_scan_phase_offsets(arr, offs):
    out = np.asarray(arr).copy()
    if np.isscalar(offs):
        return _apply_offset(out, offs)
    for k, off in enumerate(offs):
        out[k] = _apply_offset(out[k], off)
    return out


def apply_patchwise_offsets(data, xsplits, offsets, blend=True, blend_width=16):
    t, h, w = data.shape
    n_parts = len(xsplits) - 1
    offsets = np.asarray(offsets)  # shape (n_parts, t)

    out = np.empty_like(data, dtype=np.float32)
    x = np.arange(w)

    # Centers for interpolation
    centers = xsplits[:-1] + np.diff(xsplits) // 2

    for frame_i in range(t):
        dx_parts = offsets[:, frame_i]
        dx_map = np.interp(x, centers, dx_parts, left=dx_parts[0], right=dx_parts[-1])

        if blend:
            # Linear ramp blend mask to reduce seam artifacts
            blend_mask = np.ones(w, dtype=np.float32)
            for i in range(1, n_parts):
                l = max(xsplits[i] - blend_width, xsplits[i - 1])
                r = min(xsplits[i] + blend_width, xsplits[i + 1])
                ramp = np.linspace(0, 1, r - l, dtype=np.float32)
                blend_mask[l : xsplits[i]] *= 1 - ramp[: xsplits[i] - l]
                blend_mask[xsplits[i] : r] *= ramp[xsplits[i] - l :]

        for col in range(w):
            col_data = data[frame_i, :, col]
            col_shifted = col_data.copy()
            col_shifted[1::2] = shift(
                col_data[1::2],
                shift=(dx_map[col],),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            if blend:
                out[frame_i, :, col] = (1 - blend_mask[col]) * col_data + blend_mask[
                    col
                ] * col_shifted
            else:
                out[frame_i, :, col] = col_shifted

    return out


def apply_patchwise_offsets_v2(data, xsplits, offsets, blend="edge", extrapolate=True):
    t, h, w = data.shape
    n_parts = len(xsplits) - 1
    offsets = np.asarray(offsets)  # (n_parts, t)
    out = data.copy().astype(np.float32)

    x = np.arange(w)

    for frame_i in range(t):
        dx_parts = offsets[:, frame_i]

        if blend == "center":
            centers = xsplits[:-1] + np.diff(xsplits) // 2
            dx_map = np.interp(
                x, centers, dx_parts, left=dx_parts[0], right=dx_parts[-1]
            )
        elif blend == "edge":
            dx_map = np.zeros_like(x, dtype=np.float32)
            for i in range(n_parts):
                x0, x1 = xsplits[i], xsplits[i + 1]
                dx_map[x0:x1] = dx_parts[i]
        else:
            raise ValueError("Invalid blend mode")

        for col in range(w):
            col_data = data[frame_i, :, col]
            col_shifted = col_data.copy()
            col_shifted[1::2] = shift(
                col_data[1::2],
                shift=(dx_map[col],),
                order=1,
                mode="nearest",
                prefilter=False,
            )
            out[frame_i, :, col] = col_shifted

    return out


def phase_offsets_timecourse(
    data, n_parts=3, upsample=10, max_offset=4, border=0, method="frame"
):
    t, h, w = data.shape
    xsplits = np.linspace(0, w, n_parts + 1, dtype=int)
    offsets = []

    flat = data.reshape(t, h, w)

    for i in range(n_parts):
        x0, x1 = xsplits[i], xsplits[i + 1]
        patch = flat[:, :, x0:x1]

        if method == "frame":
            patch_offsets = [
                _phase_corr_2d(f, upsample, border, max_offset) for f in patch
            ]
        else:
            if method not in MBO_WINDOW_METHODS:
                raise ValueError(f"unknown method: {method}")
            summary_img = MBO_WINDOW_METHODS[method](patch)
            patch_offsets = _phase_corr_2d(summary_img, upsample, border, max_offset)
            patch_offsets = [patch_offsets] * t

        offsets.append(patch_offsets)

    return xsplits, np.array(offsets)


if __name__ == "__main__":
    from mbo_utilities import get_files, imread

    files = get_files(r"D:\tests\data", "tif")
    fpath = r"D:\W2_DATA\kbarber\2025_03_01\mk301\green"
    if not files:
        raise ValueError("No files found matching '*.tif'")

    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    array_object = imread(fpath)
    lazy_array = array_object
    lazy_array.fix_phase = False
    array = lazy_array[:2000, 8, :, :]
    fig, ax = plt.subplots()
    for num in [1, 3, 9]:
        _, ofs = nd_windowed(array, method="frame", upsample=num)
        ax.plot(ofs, label=f"upsample={num}")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Offset (pixels)")
    ax.legend()
    plt.show()

    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    # for ax, m in zip(axs.flat, MBO_WINDOW_METHODS):
    #     corr, offs = nd_windowed(array, method=m, upsample=2)
    #     ax.imshow(corr.mean(0)[150:170, 330:350], cmap="gray")
    #     ax.set_title(f"{m}\nμ={np.mean(offs):.2f}")
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()
