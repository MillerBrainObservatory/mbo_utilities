"""axial (plane-to-plane) rigid registration via phase correlation.

computes per-plane (y, x) shift vectors from a time-averaged 3d image
using fft-based phase correlation between adjacent z-planes.

public api:
    align_planes(mov3D, ...)        -> tvecs (nz, 2)
    compute_plane_shifts(mov, ...)  -> tvecs (nz, 2)  # accepts 4d, auto-means
    compute_axial_shifts(arr, ...)  -> tvecs (nz, 2)  # streams from a lazy array,
                                                      # writes shifts into metadata
    validate_axial_shifts(metadata) -> bool           # metadata-only check
    with_axial_shifts(arr, plane_shifts=None)         # apply shifts on read (5D)
    AxialShiftView(source, plane_shifts)              # lazy 5D padded-canvas view

backend is auto-detected: cupy if a cuda runtime is reachable, else numpy.
pass use_gpu=True or use_gpu=False to override.
"""

from __future__ import annotations

import math
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
        from mbo_utilities.gpu import gpu_compute_disabled
        if gpu_compute_disabled():
            return False
    except Exception:
        pass
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
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
    max_reg_xy=30,
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
    max_reg_xy=30,
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


# keep the working volume (mean image + align_planes' complex64 copies) near
# this many voxels. ScanImage planes (hundreds of px) stay below it and run at
# full resolution; large-FOV non-ScanImage stacks are downsampled to fit.
_AXIAL_TARGET_VOXELS = 150_000_000


def _auto_downsample(
    nz: int, ny: int, nx: int, target: int = _AXIAL_TARGET_VOXELS
) -> int:
    """Integer YX downsample factor keeping ``nz * ny * nx`` near ``target``."""
    voxels = int(nz) * int(ny) * int(nx)
    if voxels <= target:
        return 1
    return max(1, math.ceil(math.sqrt(voxels / target)))


def _block_mean_yx(a: np.ndarray, f: int) -> np.ndarray:
    """Block-mean the trailing (Y, X) axes by factor ``f``, accumulating at the
    reduced size so no full-resolution float copy is ever materialized."""
    *lead, ny, nx = a.shape
    yc, xc = (ny // f) * f, (nx // f) * f
    a = a[..., :yc, :xc]
    out = np.zeros((*lead, yc // f, xc // f), dtype=np.float32)
    for i in range(f):
        for j in range(f):
            out += a[..., i::f, j::f]
    out /= f * f
    return out


def _stream_plane_mean(
    arr, max_frames: int, chunk_frames: int, downsample: int = 1
) -> np.ndarray:
    """stream a (nz, ny, nx) time-mean image from a lazy array without
    materializing the full 4d movie in memory.

    supports both 5d arrays (T, C, Z, Y, X) and 4d (T, Z, Y, X). ``downsample``
    block-means the (Y, X) plane by that integer factor before accumulating, so
    a large-FOV stack never allocates a full-resolution float volume.
    """
    if hasattr(arr, "_shape5d"):
        T, C, Z, Y, X = tuple(arr._shape5d())
        use_5d = True
    elif getattr(arr, "ndim", None) == 4:
        T, Z, Y, X = arr.shape
        C = 1
        use_5d = False
    else:
        raise ValueError(f"unsupported array shape {getattr(arr, 'shape', None)}")

    f = max(1, int(downsample))
    yd, xd = (Y // f), (X // f)

    n_sub = min(max_frames, T) if max_frames else T
    frame_idx = np.linspace(0, T - 1, n_sub, dtype=int).tolist()

    im3d_sum = np.zeros((Z, yd, xd), dtype=np.float64)
    count = 0
    for start in range(0, n_sub, chunk_frames):
        idx = frame_idx[start : start + chunk_frames]
        batch = np.asarray(arr[idx])
        if use_5d and batch.ndim == 5:
            # collapse C axis (take first color channel if multiple)
            batch = batch[:, 0]
        if f > 1:
            batch = _block_mean_yx(batch, f)
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
    max_reg_xy: int = 30,
    sigma: tuple = (1.45, 0),
    smooth_sigma: float = 1.15,
    downsample: int | None = None,
    use_gpu: bool | None = None,
    progress_callback=None,
) -> np.ndarray:
    """
    Compute axial shift vectors from a lazy array; write them to metadata.

    Streams a time-mean (nz, ny, nx) image from `arr` without materializing
    the full 4d movie, runs phase-correlation plane alignment, and (if
    `metadata` is passed) writes the result in place under the keys:

        metadata["plane_shifts"]        = tvecs.tolist()     (nz, 2)
        metadata["plane_shifts_params"] = {...}              (reproducibility)

    parameters
    ----------
    arr : lazy array
        any 5D (T, C, Z, Y, X) or 4d (T, Z, Y, X) lazy array.
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
    downsample : int or None
        YX block-mean factor for the shift estimate. None (default) picks one
        from the FOV so large non-ScanImage stacks stay within a bounded
        working set; 1 forces full resolution. Shifts are scaled back to
        full-resolution pixels, so precision is +/- `downsample` px.
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

    if hasattr(arr, "_shape5d"):
        _, _, Z, Y, X = tuple(arr._shape5d())
    else:
        Z, Y, X = tuple(arr.shape)[-3:]
    f = _auto_downsample(Z, Y, X) if downsample is None else max(1, int(downsample))

    im3d = _stream_plane_mean(
        arr, max_frames=max_frames, chunk_frames=chunk_frames, downsample=f
    )

    tvecs = compute_plane_shifts(
        im3d,
        sigma=sigma,
        smooth_sigma=smooth_sigma,
        max_reg_xy=max_reg_xy,
        use_gpu=resolved_gpu,
        progress_callback=progress_callback,
    )
    if f > 1:
        # shifts were measured on the downsampled grid; rescale to full-res px
        tvecs = tvecs * f

    if metadata is not None:
        metadata["plane_shifts"] = tvecs.tolist()
        metadata["plane_shifts_params"] = {
            "max_reg_xy": int(max_reg_xy),
            "sigma": list(sigma),
            "smooth_sigma": float(smooth_sigma),
            "max_frames": int(max_frames),
            "chunk_frames": int(chunk_frames),
            "downsample": int(f),
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


_TCZYX = ("T", "C", "Z", "Y", "X")


def _validated_tczyx_shape(source) -> tuple[int, int, int, int, int]:
    """Return the source's 5D (T, C, Z, Y, X) shape, validating it is a
    genuine TCZYX layout so per-plane shifts apply to the Z axis.

    Prefers ``_shape5d()`` (the always-5D contract), else ``shape``. When the
    source exposes 5D ``dims``, they must equal ``("T","C","Z","Y","X")`` so
    ``shape``'s Z entry and the ``"Z"`` dim agree; a 4D-presenting wrapper
    (dims length != 5) is left to the shape-only path.
    """
    if hasattr(source, "_shape5d"):
        shape5d = tuple(source._shape5d())
    else:
        shape5d = tuple(getattr(source, "shape", ()))
    if len(shape5d) != 5:
        raise ValueError(
            f"axial shifts need a 5D TCZYX source; got shape {shape5d!r}"
        )
    dims = getattr(source, "dims", None)
    if dims is not None and len(dims) == 5 and tuple(dims) != _TCZYX:
        raise ValueError(
            f"axial shifts assume TCZYX (Z at axis 2); source dims are "
            f"{tuple(dims)}. shape and the 'Z' dim must agree."
        )
    return tuple(int(s) for s in shape5d)


def _is_int_index(k) -> bool:
    return isinstance(k, (int, np.integer))


def _idx_list(k, n: int) -> list[int]:
    """index (int / slice / list) -> list of non-negative ints over size n."""
    if _is_int_index(k):
        k = int(k)
        return [k if k >= 0 else n + k]
    if isinstance(k, slice):
        return list(range(*k.indices(n)))
    return [int(i) if int(i) >= 0 else n + int(i) for i in k]


class AxialShiftView:
    """5D TCZYX lazy view that applies per-plane axial shifts on read.

    Non-destructive companion to `compute_axial_shifts` /
    `imwrite(register_z=True)`: the source array is never modified, the
    shifts are applied only when a frame is read. Toggle `enabled` at any
    time to switch between the aligned (zero-padded canvas) view and the
    original source frames.

    Parameters
    ----------
    source : array-like
        Any 5D TCZYX lazy array (5D `shape`) supporting
        `source[t, c, z, :, :]` indexing. If it reports 5D `dims`, they
        must be `("T", "C", "Z", "Y", "X")` so the shift count is checked
        against the same Z axis the view indexes.
    plane_shifts : array-like (nz, 2)
        Per-plane integer (dy, dx) shifts. Either computed (e.g. by
        `compute_axial_shifts` / `imwrite(register_z=True)`) or supplied by
        the user. Length must equal the source Z size.
    enabled : bool, default True
        If True, reads return aligned frames on a zero-padded canvas of
        shape `(Y + pt + pb, X + pl + pr)`. If False, reads pass straight
        through and return the original source frames.

    Notes
    -----
    Shifts are integer pixel offsets, so applying them is a copy into an
    offset window of a zero canvas — no interpolation, O(Y*X) per plane.
    Reversal is exact: flipping `enabled` to False (or reading `source`)
    yields the original pixels; nothing is ever written back.

    The view is a transparent stand-in for the source: `.shape`, `.dims`,
    `.dtype` and `.metadata` are reported directly, and any other attribute
    (`nz`, `num_planes`, `filenames`, ...) is forwarded to the source.
    `.metadata` passes through live and writable; `plane_shifts` are dropped
    from it only while the view saves itself (so a reopened, already-aligned
    output is not shifted a second time).

    Examples
    --------
    >>> view = with_axial_shifts(arr)        # shifts applied on read
    >>> aligned_vol = np.asarray(view[0])    # (C, Z, Hc, Wc) aligned
    >>> view.enabled = False                 # back to raw, same object
    >>> raw_vol = np.asarray(view[0])        # original (C, Z, Y, X)
    """

    def __init__(self, source, plane_shifts, enabled: bool = True):
        self._source = source

        self._T, self._C, self._Z, self._Y, self._X = _validated_tczyx_shape(source)

        self._shifts = np.asarray(plane_shifts, dtype=int)
        if self._shifts.ndim != 2 or self._shifts.shape[1] != 2:
            raise ValueError(
                f"plane_shifts must be (nz, 2); got shape {self._shifts.shape}"
            )
        if len(self._shifts) != self._Z:
            raise ValueError(
                f"need one shift per plane: source has {self._Z} planes but "
                f"got {len(self._shifts)} shifts"
            )

        dy, dx = self._shifts[:, 0], self._shifts[:, 1]
        self._pt = max(0, -int(dy.min()))
        self._pb = max(0, int(dy.max()))
        self._pl = max(0, -int(dx.min()))
        self._pr = max(0, int(dx.max()))
        self._H = self._Y + self._pt + self._pb
        self._W = self._X + self._pl + self._pr

        self.enabled = bool(enabled)
        # set True only while this view saves itself (see _imwrite); strips
        # plane_shifts from the written metadata so the baked output isn't
        # re-shifted on reopen. False everywhere else, so metadata passes
        # through live and writable for GUI editing.
        self._hide_shifts = False

    @property
    def source(self):
        """the wrapped source array (never modified)."""
        return self._source

    @property
    def _arr(self):
        """the wrapped source, for one-level `_arr` unwrapping by callers
        (e.g. the GUI's isinstance peel). Mirrors `.source`."""
        return self._source

    @property
    def plane_shifts(self) -> np.ndarray:
        """the (nz, 2) integer shifts applied when enabled."""
        return self._shifts

    @property
    def padding(self) -> tuple[int, int, int, int]:
        """(pt, pb, pl, pr) canvas padding applied when enabled."""
        return (self._pt, self._pb, self._pl, self._pr)

    @property
    def dtype(self):
        return self._source.dtype

    @property
    def num_color_channels(self) -> int:
        return self._C

    @property
    def metadata(self):
        md = getattr(self._source, "metadata", None)
        if md is not None and self._hide_shifts:
            # only while this view saves itself: its pixels are already
            # aligned, so drop the still-pending shifts to avoid a reader
            # double-shifting the baked output. at all other times the source
            # dict passes through live (and writable), so GUI metadata edits
            # reach the underlying array.
            md = {
                k: v
                for k, v in md.items()
                if k not in ("plane_shifts", "plane_shifts_params")
            }
        return md

    @property
    def dims(self) -> tuple[str, ...]:
        return _TCZYX

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        if self.enabled:
            return (self._T, self._C, self._Z, self._H, self._W)
        return (self._T, self._C, self._Z, self._Y, self._X)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape5d()

    @property
    def ndim(self) -> int:
        return 5

    def __len__(self) -> int:
        return self._T

    def _key5(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            n_missing = 5 - (len(key) - 1)
            key = key[:i] + (slice(None),) * max(n_missing, 0) + key[i + 1:]
        if len(key) > 5:
            raise IndexError(f"too many indices for 5D array: {len(key)}")
        if len(key) < 5:
            key = key + (slice(None),) * (5 - len(key))
        return key

    def __getitem__(self, key):
        t_key, c_key, z_key, y_key, x_key = self._key5(key)
        # always read full spatial: shifts remap (y, x), so a padded-space
        # spatial key can't be pushed down to the source.
        raw = np.asarray(self._source[t_key, c_key, z_key, :, :])

        if self.enabled:
            raw = self._to_canvas(raw, t_key, c_key, z_key)

        if y_key == slice(None) and x_key == slice(None):
            return raw
        spatial = (slice(None),) * (raw.ndim - 2) + (y_key, x_key)
        return raw[spatial]

    def _to_canvas(self, raw, t_key, c_key, z_key):
        pt, pl = self._pt, self._pl
        Y, X = self._Y, self._X
        out = np.zeros(raw.shape[:-2] + (self._H, self._W), dtype=raw.dtype)

        if _is_int_index(z_key):
            z = _idx_list(z_key, self._Z)[0]
            dy, dx = int(self._shifts[z, 0]), int(self._shifts[z, 1])
            out[..., pt + dy: pt + dy + Y, pl + dx: pl + dx + X] = raw
            return out

        # z is a kept axis; its position = number of non-int axes before it.
        z_axis = (0 if _is_int_index(t_key) else 1) + (0 if _is_int_index(c_key) else 1)
        z_indices = _idx_list(z_key, self._Z)
        raw_zf = np.moveaxis(raw, z_axis, 0)
        out_zf = np.moveaxis(out, z_axis, 0)  # view into out
        for i, z in enumerate(z_indices):
            dy, dx = int(self._shifts[z, 0]), int(self._shifts[z, 1])
            out_zf[i, ..., pt + dy: pt + dy + Y, pl + dx: pl + dx + X] = raw_zf[i]
        return out

    def __array__(self, dtype=None, copy=None):
        # explicit (do not let __getattr__ leak the source's rank/shape).
        data = np.asarray(self[0])
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def astype(self, dtype, *args, **kwargs):
        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        # forward domain attributes (nz, num_planes, num_color_channels,
        # num_rois, filenames, roi_mode, iter_rois, stack_type, fs, ...) to
        # the wrapped source so the view is a transparent stand-in. shape /
        # dims / metadata / dtype and the numpy protocol are defined
        # explicitly above and never reach here. underscore names are not
        # forwarded so internal state and the `_arr` peel property resolve
        # normally (and __init__ stays recursion-safe before _source is set).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_source"), name)

    def _imwrite(self, outpath, **kwargs):
        """Stream this view (aligned when enabled) to disk via the writers.

        Lets `imwrite(view, out, ext=".zarr")` save the aligned, padded
        volume without materializing it in memory. Output metadata omits
        plane_shifts when enabled (they are baked into the saved pixels).
        """
        from mbo_utilities.arrays._base import _imwrite_base
        prev = self._hide_shifts
        self._hide_shifts = self.enabled
        try:
            return _imwrite_base(self, outpath, **kwargs)
        finally:
            self._hide_shifts = prev

    def save(self, outpath, **kwargs):
        return self._imwrite(outpath, **kwargs)

    def __repr__(self) -> str:
        return (
            f"AxialShiftView(shape={self.shape}, dtype={self.dtype}, "
            f"enabled={self.enabled}, "
            f"pad=T{self._pt}/B{self._pb}/L{self._pl}/R{self._pr})"
        )


def with_axial_shifts(arr, *, enabled: bool = True, plane_shifts=None) -> AxialShiftView:
    """Wrap a 5D TCZYX array so per-plane axial shifts are applied on read.

    Non-destructive: `arr` is never modified. Reversible: set
    `view.enabled = False` (or read `view.source`) to recover the
    original frames.

    Parameters
    ----------
    arr : array-like
        5D TCZYX lazy array (5D `shape`, supports `arr[t, c, z, :, :]`).
    enabled : bool, default True
        Initial state. True applies shifts on read; False passes through.
    plane_shifts : array-like (nz, 2), optional
        Shifts to use, one (dy, dx) row per z-plane; its length must equal
        the array's Z size. Defaults to `arr.metadata["plane_shifts"]`
        (written by `imwrite(register_z=True)` / `compute_axial_shifts`).

    Returns
    -------
    AxialShiftView

    Raises
    ------
    ValueError
        If `plane_shifts` is not given and no valid `plane_shifts` is
        present in `arr.metadata`, if the count does not match the Z size,
        or if `arr` is not a TCZYX 5D array.
    """
    if plane_shifts is None:
        md = getattr(arr, "metadata", None)
        # plane count from the validated TCZYX 'Z' axis, so the metadata
        # shift count is checked against the same Z the view will index.
        nz = _validated_tczyx_shape(arr)[2]
        if not validate_axial_shifts(md, nz):
            raw = md.get("plane_shifts") if md else None
            # shifts present and well-formed, but the count doesn't match the
            # array's planes — almost always a file written before the writer
            # subset plane_shifts to the exported planes (stale .zarr / kernel).
            if raw is not None and validate_axial_shifts(md, None):
                n = len(np.asarray(raw))
                raise ValueError(
                    f"metadata has plane_shifts for {n} planes but this array has "
                    f"{nz}. Re-run imwrite(register_z=True) with the current version "
                    f"(restart the kernel first so it reloads), or pass plane_shifts= "
                    f"with {nz} (dy, dx) rows."
                )
            raise ValueError(
                "no valid plane_shifts in metadata; pass plane_shifts=, or run "
                "compute_axial_shifts / imwrite(register_z=True) first"
            )
        plane_shifts = md["plane_shifts"]
    return AxialShiftView(arr, plane_shifts, enabled=enabled)
