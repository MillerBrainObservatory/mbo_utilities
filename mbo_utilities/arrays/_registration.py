"""axial (plane-to-plane) rigid registration via phase correlation.

computes per-plane (y, x) shift vectors from a time-averaged 3d image
using fft-based phase correlation between adjacent z-planes.

public api:
    align_planes(mov3D, ...)        -> tvecs (nz, 2)
    compute_plane_shifts(mov, ...)  -> tvecs (nz, 2)  # accepts 4d, auto-means
    compute_axial_shifts(arr, ...)  -> tvecs (nz, 2)  # streams from a lazy array,
                                                      # writes shifts into metadata
    validate_axial_shifts(metadata) -> bool           # metadata-only check

backend is auto-detected: cupy if a cuda runtime is reachable, else numpy.
pass use_gpu=True or use_gpu=False to override.
"""

from __future__ import annotations

from typing import Any

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


def _auto_resolve_gpu() -> bool:
    """detect whether a cuda gpu is actually usable via cupy.

    returns true only if cupy imported AND a cuda device is reachable.
    falls back to false on any failure (no cupy, no driver, no device).
    """
    if not HAS_CUPY:
        return False
    try:
        _cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def _resolve_backend(use_gpu: bool | None):
    """pick numpy or cupy backend. None = auto-detect."""
    if use_gpu is None:
        use_gpu = _auto_resolve_gpu()
    if use_gpu:
        if not HAS_CUPY:
            raise RuntimeError(
                "use_gpu=True but cupy is not installed. "
                "install cupy-cuda12x or cupy-cuda11x, or pass use_gpu=False."
            )
        return _cp, _cp_fft2, _cp_ifft2, True
    return np, _np_fft2, _np_ifft2, False


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
    # mean of a complex array is complex — preserved for the complex add mask below
    offset = ref_img.mean() * (1.0 - mult)
    return mult, offset


def _phasecorr_ref(ref_img, smooth_sigma, xp, fft2):
    cf = xp.conj(fft2(ref_img))
    cf /= 1e-5 + xp.absolute(cf)
    cf = cf * _gaussian_fft(smooth_sigma, cf.shape[0], cf.shape[1], xp, fft2)
    return cf.astype("complex64")


# phase-corr primitives

def _clip_and_mask(mov, mult_mask, add_mask):
    mov *= mult_mask
    mov += add_mask
    return mov


def _convolve_2d(mov, ref_f, xp, fft2, ifft2):
    mov[:] = fft2(mov, axes=(1, 2))
    mov /= xp.abs(mov) + xp.complex64(1e-5)
    mov *= ref_f
    mov[:] = ifft2(mov, axes=(1, 2))
    return mov


def _unwrap_fft_2d(mov_float, nr, out):
    # reshuffle fft output so zero-shift peak lands in the center of `out`
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
    max_reg_xy=150,
    use_gpu=None,
    progress_callback=None,
):
    """compute per-plane (y, x) shift vectors via adjacent-plane phase correlation.

    parameters
    ----------
    mov3D : ndarray (nz, ny, nx)
        time-averaged mean image per z-plane.
    sigma : (float, float)
        (xy_sigma, z_sigma) for the spatial taper. z_sigma=0 disables the
        z-taper (the standard choice for per-plane alignment).
    smooth_sigma : float
        gaussian width (in fft domain) for the phase-corr reference filter.
    max_reg_xy : int
        max shift search radius in pixels.
    use_gpu : bool or None
        None (default) auto-detects; True/False override. gpu path uses cupy;
        raising if use_gpu=True and cupy is missing.
    progress_callback : callable or None
        called as cb(fraction, message) after each plane.

    returns
    -------
    tvecs : ndarray (nz, 2) numpy
        cumulative (y_shift, x_shift) per plane, relative to plane 0.
        always returned as numpy even when computed on gpu.
    """
    xp, fft2, ifft2, on_gpu = _resolve_backend(use_gpu)

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

    if on_gpu:
        tvecs = _cp.asnumpy(tvecs)
    return np.asarray(tvecs)


def compute_plane_shifts(
    mov,
    sigma=(1.45, 0),
    smooth_sigma=1.15,
    max_reg_xy=150,
    use_gpu=None,
    progress_callback=None,
):
    """accepts a 4d (nz, nt, ny, nx) movie or a 3d (nz, ny, nx) mean image.

    4d input is time-averaged internally. returns tvecs (nz, 2).
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


def _stream_plane_mean(arr, max_frames: int, chunk_frames: int) -> np.ndarray:
    """stream a (nz, ny, nx) time-mean image from a lazy array without
    materializing the full 4d movie in memory.

    supports both 5d arrays (T, C, Z, Y, X) with shape5d and 4d (T, Z, Y, X).
    """
    if hasattr(arr, "shape5d"):
        T, C, Z, Y, X = tuple(arr.shape5d)
        use_5d = True
    elif getattr(arr, "ndim", None) == 4:
        T, Z, Y, X = arr.shape
        C = 1
        use_5d = False
    else:
        raise ValueError(f"unsupported array shape {getattr(arr, 'shape', None)}")

    n_sub = min(max_frames, T) if max_frames else T
    frame_idx = np.linspace(0, T - 1, n_sub, dtype=int).tolist()

    im3d_sum = np.zeros((Z, Y, X), dtype=np.float64)
    count = 0
    for start in range(0, n_sub, chunk_frames):
        idx = frame_idx[start : start + chunk_frames]
        if use_5d:
            batch = np.asarray(arr[idx])
            if batch.ndim == 5:
                # collapse C axis (take first color channel if multiple)
                batch = batch[:, 0]
        else:
            batch = np.asarray(arr[idx])
        im3d_sum += batch.sum(axis=0, dtype=np.float64)
        count += batch.shape[0]

    if count == 0:
        raise RuntimeError("streaming mean: no frames were read")
    return (im3d_sum / count).astype(np.float32)


def compute_axial_shifts(
    arr,
    *,
    metadata: dict | None = None,
    max_frames: int = 200,
    chunk_frames: int = 10,
    max_reg_xy: int = 150,
    sigma: tuple = (1.45, 0),
    smooth_sigma: float = 1.15,
    use_gpu: bool | None = None,
    progress_callback=None,
) -> np.ndarray:
    """compute axial shift vectors from a lazy array; write them to metadata.

    streams a time-mean (nz, ny, nx) image from `arr` without materializing
    the full 4d movie, runs phase-correlation plane alignment, and (if
    `metadata` is passed) writes the result in place under the keys:

        metadata["plane_shifts"]        = tvecs.tolist()     (nz, 2)
        metadata["apply_shift"]         = True
        metadata["plane_shifts_params"] = {...}              (reproducibility)

    parameters
    ----------
    arr : lazy array
        any array with shape5d (T, C, Z, Y, X) or a 4d (T, Z, Y, X) shape.
    metadata : dict or None
        target metadata dict. mutated in place if not None.
    max_frames : int
        number of frames to subsample (evenly spaced) for the time-mean.
    chunk_frames : int
        frames loaded per chunk while streaming.
    max_reg_xy : int
        max shift search radius in pixels.
    sigma : (float, float)
        spatial-taper sigma; see `align_planes`.
    smooth_sigma : float
        gaussian fft-filter width; see `align_planes`.
    use_gpu : bool or None
        None = auto-detect cupy+cuda; True/False override.
    progress_callback : callable or None
        forwarded to `align_planes` (called per plane).

    returns
    -------
    tvecs : ndarray (nz, 2) int-valued float
        per-plane cumulative (y, x) shift relative to plane 0.
    """
    resolved_gpu = _auto_resolve_gpu() if use_gpu is None else bool(use_gpu)

    im3d = _stream_plane_mean(arr, max_frames=max_frames, chunk_frames=chunk_frames)

    tvecs = compute_plane_shifts(
        im3d,
        sigma=sigma,
        smooth_sigma=smooth_sigma,
        max_reg_xy=max_reg_xy,
        use_gpu=resolved_gpu,
        progress_callback=progress_callback,
    )

    if metadata is not None:
        metadata["plane_shifts"] = tvecs.tolist()
        metadata["apply_shift"] = True
        metadata["plane_shifts_params"] = {
            "max_reg_xy": int(max_reg_xy),
            "sigma": list(sigma),
            "smooth_sigma": float(smooth_sigma),
            "max_frames": int(max_frames),
            "chunk_frames": int(chunk_frames),
            "use_gpu": bool(resolved_gpu),
        }

    return tvecs


def validate_axial_shifts(metadata: dict | None, num_planes: int | None = None) -> bool:
    """true iff metadata contains a well-formed plane_shifts entry."""
    if not metadata:
        return False
    shifts = metadata.get("plane_shifts")
    if shifts is None:
        return False
    try:
        arr = np.asarray(shifts)
    except Exception:
        return False
    if arr.ndim != 2 or arr.shape[1] != 2:
        return False
    if num_planes is not None and len(arr) != num_planes:
        return False
    return True
