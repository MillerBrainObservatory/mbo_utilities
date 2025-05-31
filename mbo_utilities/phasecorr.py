import numpy as np
from scipy.ndimage import fourier_shift
from skimage.registration import phase_cross_correlation


def _subpix(frames: np.ndarray, upsample: int, border: tuple[int, int, int, int]):
    out = frames.copy()
    offs = np.zeros(frames.shape[0], dtype=np.float32)

    t, b, l, r = border
    for i, fr in enumerate(frames):
        even, odd = fr[::2], fr[1::2]
        m = min(even.shape[0], odd.shape[0])
        even_crop = even[:m, t : even.shape[1] - b, l : even.shape[2] - r]
        odd_crop = odd[:m, t : odd.shape[1] - b, l : odd.shape[2] - r]
        shift, *_ = phase_cross_correlation(
            even_crop, odd_crop, upsample_factor=upsample
        )
        offs[i] = shift[1]

    if np.any(offs):
        rows = out[:, 1::2]
        fft = np.fft.fftn(rows, axes=(1, 2))
        shifted = np.array(
            [fourier_shift(fft[k], (0, offs[k])) for k in range(fft.shape[0])]
        )
        out[:, 1::2] = np.fft.ifftn(shifted, axes=(1, 2)).real
    return out


def _identity(frames: np.ndarray, *_, **__) -> np.ndarray:
    return frames


_METHODS = {
    "subpix": _subpix,
    "raw": _identity,  # no correction
    # "two_step": two_step_kernel,
    # "crosscorr": crosscorr_kernel,
}


def correct_scan_phase(
    arr: np.ndarray,
    *,
    method: str = "subpix",
    upsample: int = 10,
    border: int | tuple[int, int, int, int] = 0,
) -> np.ndarray:
    fn = _METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown phase-corr method {method!r}")

    if isinstance(border, int):
        border = (border, border, border, border)

    a = np.asarray(arr)
    y, x = a.shape[-2:]
    flat = a.reshape(-1, y, x)
    fixed = fn(flat, upsample, border)
    return fixed.reshape(a.shape)
