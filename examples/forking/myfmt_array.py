"""
Example third-party LazyArray plugin for a contrived ".myfmt" format.

A real fork would ship this in its own package and declare the entry point:

    [project.entry-points."mbo_utilities.lazy_arrays"]
    myfmt = "mypkg.myfmt_array:MyFormatArray"

then `imread("data.myfmt")` resolves here with no edits to mbo_utilities.

For a quick demo without packaging, register at runtime:

    from mbo_utilities import imread, register_array_class
    from examples.forking.myfmt_array import MyFormatArray
    register_array_class(MyFormatArray)
    arr = imread("data.myfmt")   # -> MyFormatArray

A ".myfmt" file here is just a .npy renamed (np.save then rename), kept tiny
so the example stays self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mbo_utilities import LazyArray


class MyFormatArray(LazyArray):
    PRIORITY = 60  # beats the generic .npy reader (50) for this extension

    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self._data = np.load(self.path, allow_pickle=False)
        self._raw_shape = self._data.shape  # treat as TYX for the demo

    @classmethod
    def can_open(cls, path) -> bool:
        return Path(path).suffix.lower() == ".myfmt"

    def _shape5d(self):
        t, y, x = self._raw_shape
        return (t, 1, 1, y, x)

    def __getitem__(self, key):
        from mbo_utilities.arrays._base import _index_5d_into_raw
        return _index_5d_into_raw(self._data, key, len(self._raw_shape))

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def metadata(self) -> dict:
        return {}
