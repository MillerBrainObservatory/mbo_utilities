"""axial (plane-to-plane) rigid registration via phase correlation.

vendored and trimmed from suite3d (reference_image.py, register_gpu.py).
only keeps the path that produces per-plane (y, x) shift vectors from a
time-averaged 3d image. no reference-image iteration, no movie output,
no shared-memory machinery.

public api:
    align_planes(mov3D, ...)      -> tvecs (nz, 2)
    compute_plane_shifts(mov, ...) -> tvecs (nz, 2)  # accepts 4d, auto-means

both cpu and gpu backends. gpu requires cupy; falls back to cpu if
use_gpu=True but cupy is unavailable (raises explicitly).

original suite3d `align_planes` is cpu-only; the gpu variant here is new
but uses the same algorithm with cupy-backed ffts.
"""

from __future__ import annotations

import numpy as np

try:
    from mkl_fft import fft2 as _np_fft2, ifft2 as _np_ifft2
except ImportError:
    from scipy.fft import fft2 as _np_fft2, ifft2 as _np_ifft2

try:
    import cupy as _cp
    from cupyx.scipy.fft import fft2 as _cp_fft2, ifft2 as _cp_ifft2
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    _cp = None
    _cp_fft2 = None
    _cp_ifft2 = None


def _resolve_backend(use_gpu: bool):
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError(
                "use_gpu=True but cupy is not installed. "
                "install cupy-cuda12x or cupy-cuda11x, or pass use_gpu=False."
            )
        return _cp, _cp_fft2, _cp_ifft2
    return np, _np_fft2, _np_ifft2


# mask + reference builders

def _meshgrid_mean_centered_2d(nx, ny, xp):
    x = xp.arange(0, nx)
    y = xp.arange(0, ny)
    x = xp.abs(x - x.mean())
    y = xp.abs(y - y.mean())
    return xp.meshgrid(x, y)


def _meshgrid_mean_centered_3d(nz, ny, nx, xp):
    x = xp.arange(0, nx); x = xp.abs(x - x.mean())
    y = xp.arange(0, ny); y = xp.abs(y - y.mean())
    z = xp.arange(0, nz); z = xp.abs(z - z.mean())
    return xp.meshgrid(z, y, x, indexing="ij")


def _gaussian_fft(sig, ny, nx, xp, fft2):
    xx, yy = _meshgrid_mean_centered_2d(nx, ny, xp)
    hgx = xp.exp(-xp.square(xx / sig) / 2)
    hgy = xp.exp(-xp.square(yy / sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    return xp.real(fft2(xp.fft.ifftshift(hgg)))


def _spatial_taper_3d(sig, sigz, nz, ny, nx, xp):
    zz, yy, xx = _meshgrid_mean_centered_3d(nz, ny, nx, xp)
    mY = ((ny - 1) / 2) - 2 * sig
    mX = ((nx - 1) / 2) - 2 * sig
    mZ = ((nz - 1) / 2) - 2 * sigz
    maskY = 1.0 / (1.0 + xp.exp((yy - mY) / sig))
    maskX = 1.0 / (1.0 + xp.exp((xx - mX) / sig))
    if sigz == 0:
        return maskY * maskX
    maskZ = 1.0 / (1.0 + xp.exp((zz - mZ) / sigz))
    return maskY * maskX * maskZ


def _compute_masks_3d(ref_img, sigma, xp):
    sig, sigz = sigma
    nz, ny, nx = ref_img.shape
    mult = _spatial_taper_3d(sig, sigz, nz, ny, nx, xp)
    # mean of a complex array is complex; preserved from suite3d for parity
    offset = ref_img.mean() * (1.0 - mult)
    return mult, offset


def _phasecorr_ref(ref_img, smooth_sigma, xp, fft2):
    # fft, conjugate, phase-normalize, multiply by gaussian filter
    cf = xp.conj(fft2(ref_img))
    cf /= 1e-5 + xp.absolute(cf)
    cf = cf * _gaussian_fft(smooth_sigma, cf.shape[0], cf.shape[1], xp, fft2)
    return cf.astype("complex64")


# phase-corr primitives

def _clip_and_mask(mov, mult_mask, add_mask):
    # mov is complex; masks are applied in place
    mov *= mult_mask
    mov += add_mask
    return mov


def _convolve_2d(mov, ref_f, xp, fft2, ifft2):
    # phase-normalized cross-correlation in fft domain
    mov[:] = fft2(mov, axes=(1, 2))
    mov /= xp.abs(mov) + xp.complex64(1e-5)
    mov *= ref_f
    mov[:] = ifft2(mov, axes=(1, 2))
    return mov


def _unwrap_fft_2d(mov_float, nr, out):
    # reshuffle fft output so zero-shift peak lands in the center of `out`
    # mov_float shape: (nt, ny, nx); out shape: (nt, 2*nr+1, 2*nr+1)
    ny, nx = mov_float.shape[-2:]
    ncc = nr * 2 + 1
    out[:, :nr,   :nr]   = mov_float[:, -nr:,   -nr:]
    out[:,  nr:,  :nr]   = mov_float[:, :nr + 1, -nr:]
    out[:, :nr,   nr:]   = mov_float[:, -nr:,   :nr + 1]
    out[:,  nr:,  nr:]   = mov_float[:, :nr + 1, :nr + 1]
    return out


def _get_max_cc_coord(phase_corr, max_reg_xy, xp):
    nt, ncc, _ = phase_corr.shape
    flat = phase_corr.reshape(nt, ncc ** 2)
    argmaxs = xp.argmax(flat, axis=1)
    cmax = xp.max(flat, axis=1)
    ymax = (argmaxs // ncc) - max_reg_xy
    xmax = (argmaxs % ncc) - max_reg_xy
    return ymax, xmax, cmax


# public api

def align_planes(
    mov3D,
    sigma=(1.45, 0),
    smooth_sigma=1.15,
    max_reg_xy=50,
    use_gpu=False,
    progress_callback=None,
):
    """compute per-plane (y, x) shift vectors via adjacent-plane phase correlation.

    parameters
    ----------
    mov3D : ndarray (nz, ny, nx)
        time-averaged mean image per z-plane.
    sigma : (float, float)
        (xy_sigma, z_sigma) for the spatial taper. z_sigma=0 disables the z-taper
        (matches suite3d default for per-plane alignment).
    smooth_sigma : float
        gaussian width (in fft domain) for the phase-corr reference filter.
    max_reg_xy : int
        max shift search radius in pixels.
    use_gpu : bool
        use cupy if available; otherwise numpy + scipy/mkl fft.
    progress_callback : callable or None
        called as cb(fraction, message) after each plane.

    returns
    -------
    tvecs : ndarray (nz, 2) numpy
        cumulative (y_shift, x_shift) per plane, relative to plane 0.
        always returned as numpy even when computed on gpu.
    """
    xp, fft2, ifft2 = _resolve_backend(use_gpu)

    mov3D = xp.asarray(mov3D, dtype=xp.complex64)
    mov3D = xp.expand_dims(mov3D, axis=1)  # (nz, 1, ny, nx)
    nz, nt, ny, nx = mov3D.shape

    mult_mask, add_mask = _compute_masks_3d(mov3D.squeeze(axis=1), sigma, xp)

    ncc = max_reg_xy * 2 + 1
    ymaxs = xp.zeros((nz, nt), dtype=xp.int16)
    xmaxs = xp.zeros((nz, nt), dtype=xp.int16)
    phase_corr = xp.zeros((nt, ncc, ncc), dtype=xp.float32)

    refs_f = xp.zeros_like(mov3D)
    for z in range(nz):
        refs_f[z] = _phasecorr_ref(mov3D[z, 0], smooth_sigma, xp, fft2)

    for zidx in range(1, nz):
        mov3D[zidx] = _clip_and_mask(mov3D[zidx], mult_mask[zidx], add_mask[zidx])
        mov3D[zidx] = _convolve_2d(mov3D[zidx], refs_f[zidx - 1], xp, fft2, ifft2)
        _unwrap_fft_2d(mov3D[zidx].real, max_reg_xy, out=phase_corr)
        ymaxs[zidx], xmaxs[zidx], _ = _get_max_cc_coord(phase_corr, max_reg_xy, xp)
        if progress_callback:
            progress_callback(zidx / nz, f"aligning plane {zidx}/{nz}")

    tvec_y = -xp.cumsum(ymaxs)
    tvec_x = -xp.cumsum(xmaxs)
    tvecs = xp.stack((tvec_y, tvec_x), axis=1)

    if use_gpu:
        tvecs = _cp.asnumpy(tvecs)
    return np.asarray(tvecs)


def compute_plane_shifts(
    mov,
    sigma=(1.45, 0),
    smooth_sigma=1.15,
    max_reg_xy=50,
    use_gpu=False,
    progress_callback=None,
):
    """convenience wrapper that accepts a 4d movie and takes the time mean.

    parameters
    ----------
    mov : ndarray
        (nz, nt, ny, nx) 4d movie, or (nz, ny, nx) mean image.
        4d is time-averaged internally.
    sigma, smooth_sigma, max_reg_xy, use_gpu, progress_callback
        see `align_planes`.

    returns
    -------
    tvecs : ndarray (nz, 2) numpy
    """
    mov = np.asarray(mov)
    if mov.ndim == 4:
        mov = mov.mean(axis=1)
    if mov.ndim != 3:
        raise ValueError(f"expected 3d or 4d movie, got shape {mov.shape}")
    return align_planes(
        mov,
        sigma=sigma,
        smooth_sigma=smooth_sigma,
        max_reg_xy=max_reg_xy,
        use_gpu=use_gpu,
        progress_callback=progress_callback,
    )
