"""Read-time bidirectional scan-phase correction over any 5D TCZYX lazy array.

Wraps a source array and applies phase correction on read, toggled by a
``PhaseCorrectionFeature``. Transparent passthrough when disabled, so the GUI
can wrap any time-dimension array once at load and flip correction on/off
without re-reading from disk. Mirrors ``AxialShiftView``: the source is never
modified and ``shape``/``dims``/``metadata``/``dtype`` forward through.

Correction uses ``bidir_phasecorr`` on whatever chunk is read: a single 2D
frame estimates one offset from that frame; a multi-frame chunk (e.g. a save
window) estimates one offset from its mean image and applies it to every frame
(``method="mean"``), matching ``ScanImageArray``.
"""

from __future__ import annotations

import numpy as np

from mbo_utilities.arrays._registration import (
    _TCZYX,
    _idx_list,
    _validated_tczyx_shape,
)
from mbo_utilities.arrays.features import (
    PhaseCorrectionFeature,
    PhaseCorrectionMixin,
)


class PhaseCorrectedView(PhaseCorrectionMixin):
    """5D TCZYX lazy view that applies bidirectional phase correction on read.

    Non-destructive and reversible: set ``view.fix_phase = False`` (or read
    ``view.source``) to recover the original frames. The ``phase_correction``
    feature is the same type ``ScanImageArray`` exposes, so the GUI's
    phase-correction controls work on the view via duck typing.
    """

    def __init__(
        self,
        source,
        *,
        enabled: bool = False,
        method: str = "mean",
        use_fft: bool = True,
        border: int = 4,
        max_offset: int = 10,
    ):
        self._source = source
        self._T, self._C, self._Z, self._Y, self._X = _validated_tczyx_shape(source)

        self.phase_correction = PhaseCorrectionFeature(
            enabled=enabled,
            method=method,
            shift=None,  # auto-compute on read
            use_fft=use_fft,
            border=border,
            max_offset=max_offset,
        )
        # per-(t, c, z) offset cache for the GUI's current-offset readout
        self._offset_cache: dict[tuple[int, int, int], float] = {}
        self.phase_correction.add_event_handler(self._on_feature_change)

    def _on_feature_change(self, event):
        self._invalidate_offset_cache()

    @property
    def source(self):
        """The wrapped source array (never modified)."""
        return self._source

    @property
    def _arr(self):
        """The wrapped source, for one-level `_arr` unwrapping by callers."""
        return self._source

    @property
    def dtype(self):
        return self._source.dtype

    @property
    def num_color_channels(self) -> int:
        return self._C

    @property
    def metadata(self):
        return getattr(self._source, "metadata", None)

    @property
    def dims(self) -> tuple[str, ...]:
        return _TCZYX

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return (self._T, self._C, self._Z, self._Y, self._X)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape5d()

    @property
    def ndim(self) -> int:
        return 5

    def __len__(self) -> int:
        return self._T

    def _invalidate_offset_cache(self) -> None:
        self._offset_cache.clear()

    def get_offset_at(self, t, c, z) -> float | None:
        """Cached phase offset for the (t, c, z) cell, or None if not read yet."""
        return self._offset_cache.get((int(t), int(c), int(z)))

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
        # read full spatial: the offset is estimated and applied across whole
        # rows, so a y/x sub-key is applied after correction, not pushed down.
        raw = np.asarray(self._source[t_key, c_key, z_key, :, :])

        if self.phase_correction.enabled and raw.size:
            raw = self._apply(raw, t_key, c_key, z_key)

        if y_key == slice(None) and x_key == slice(None):
            return raw
        spatial = (slice(None),) * (raw.ndim - 2) + (y_key, x_key)
        return raw[spatial]

    def _apply(self, raw, t_key, c_key, z_key):
        from mbo_utilities.analysis.phasecorr import _apply_offset, bidir_phasecorr

        pc = self.phase_correction
        flat = raw.reshape(-1, raw.shape[-2], raw.shape[-1])  # (N, Y, X)

        shift = pc.effective_shift
        if shift is not None:
            out = _apply_offset(flat.copy(), float(shift), pc.use_fft)
            offs = float(shift)
        else:
            out, offs = bidir_phasecorr(
                flat,
                method=pc.method.value,
                use_fft=pc.use_fft,
                max_offset=pc.max_offset,
                border=pc.border,
            )

        self._record_offsets(t_key, c_key, z_key, offs)
        return np.asarray(out).reshape(raw.shape)

    def _record_offsets(self, t_key, c_key, z_key, offs) -> None:
        # frames are flattened T-major then C then Z; size-1 (int-indexed)
        # axes drop out but keep the cartesian product aligned with that order.
        coords = [
            (t, c, z)
            for t in _idx_list(t_key, self._T)
            for c in _idx_list(c_key, self._C)
            for z in _idx_list(z_key, self._Z)
        ]
        if np.ndim(offs) == 0:
            for coord in coords:
                self._offset_cache[coord] = float(offs)
        else:
            for coord, o in zip(coords, np.ravel(offs), strict=False):
                self._offset_cache[coord] = float(o)

    def __array__(self, dtype=None, copy=None):
        # explicit so __getattr__ never leaks the source's rank via numpy.
        data = np.asarray(self[0])
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def astype(self, dtype, *args, **kwargs):
        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        # forward domain attributes (filenames, fs, num_planes, source_path,
        # ...) to the source. shape/dims/metadata/dtype and the phase-correction
        # properties are defined above and never reach here; underscore names
        # are not forwarded so __init__ stays recursion-safe before _source.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_source"), name)

    def _imwrite(self, outpath, **kwargs):
        """Stream this view to disk; correction is baked in when enabled."""
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(self, outpath, **kwargs)

    def save(self, outpath, **kwargs):
        return self._imwrite(outpath, **kwargs)

    def __repr__(self) -> str:
        pc = self.phase_correction
        return (
            f"PhaseCorrectedView(shape={self.shape}, dtype={self.dtype}, "
            f"enabled={pc.enabled}, method={pc.method.value})"
        )


def with_phasecorr(
    arr,
    *,
    enabled: bool = False,
    method: str = "mean",
    use_fft: bool = True,
    border: int = 4,
    max_offset: int = 10,
) -> PhaseCorrectedView:
    """Wrap a 5D TCZYX array so bidirectional phase correction applies on read.

    Non-destructive: ``arr`` is never modified. Reversible: set
    ``view.fix_phase = False`` (or read ``view.source``) to recover the
    original frames. Works on any array with a 5D TCZYX surface and a time
    dimension (zarr, h5, npy, mp4, plain tiff, ...).

    Parameters
    ----------
    arr : array-like
        5D TCZYX lazy array (5D ``shape``, supports ``arr[t, c, z, :, :]``).
    enabled : bool, default False
        Initial state. False passes through; toggle later via ``fix_phase``.
    method : str, default "mean"
        Reduction used to estimate the offset for a multi-frame read.
    use_fft : bool, default True
        Subpixel FFT-based estimation/application.
    border, max_offset : int
        Phase-estimation parameters (see ``bidir_phasecorr``).

    Returns
    -------
    PhaseCorrectedView
    """
    return PhaseCorrectedView(
        arr,
        enabled=enabled,
        method=method,
        use_fft=use_fft,
        border=border,
        max_offset=max_offset,
    )
