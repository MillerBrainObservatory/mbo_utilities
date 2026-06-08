"""4D single-channel view over a 5D TCZYX lazy array.

Lives at module top-level (not under ``gui.tasks``) so subprocess
workers spawned by ``lbm_suite2p_python.run_volume`` can re-import it
after their own ``imread`` call.
"""

from __future__ import annotations

from mbo_utilities.lazy_array import LazyArray


class _ChannelView(LazyArray):
    """4D TZYX view of a single channel from 5D TCZYX data.

    Wraps a lazy array and presents it as 4D by fixing the channel index.
    Used to feed single-channel data to pipelines that expect TZYX input.

    Subclasses ``LazyArray`` so ``isinstance(obj, LazyArray)`` recognizes it
    everywhere (including spawned pipeline workers), while overriding
    ``shape``/``ndim`` to keep the 4D surface.
    """

    def __init__(self, arr, channel_0idx: int):
        self._arr = arr
        self._ch = channel_0idx
        self._metadata_override = None

    @property
    def shape(self):
        s = self._arr.shape
        return (s[0], s[2], s[3], s[4])

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        s = self._arr._shape5d() if hasattr(self._arr, "_shape5d") else self._arr.shape
        return (s[0], 1, s[2], s[3], s[4])

    @property
    def ndim(self):
        return 4

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def metadata(self):
        if self._metadata_override is not None:
            return self._metadata_override
        md = dict(getattr(self._arr, "metadata", {}))
        md["num_color_channels"] = 1
        return md

    @metadata.setter
    def metadata(self, value):
        self._metadata_override = value

    @property
    def filenames(self):
        return getattr(self._arr, "filenames", [])

    @property
    def num_planes(self):
        return self._arr.shape[2]

    @property
    def num_color_channels(self):
        return 1

    @property
    def num_channels(self):
        return 1

    @property
    def dims(self):
        return ("T", "Z", "Y", "X")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 5:
            return self._arr[(key[0], self._ch, key[2], key[3], key[4])]
        key = key + (slice(None),) * (4 - len(key))
        return self._arr[(key[0], self._ch, key[1], key[2], key[3])]

    def _imwrite(self, outpath, planes=None, ext=".tiff", **kwargs):
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(
            self, outpath, planes=planes, ext=ext, **kwargs
        )
