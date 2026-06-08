---
orphan: true
---

# Adding a format (forking the LazyArray API)

`imread()` dispatches to the highest-`PRIORITY` registered `LazyArray`
subclass whose `can_open(path)` returns `True`. A third-party package can add
or override a format by shipping one class and registering it — no edits to
`mbo_utilities`.

## Minimal class

```python
import numpy as np
from mbo_utilities import LazyArray

class MyFormatArray(LazyArray):
    PRIORITY = 50  # higher wins; built-ins range 30 (TiffArray) .. 100 (Suite2p)

    def __init__(self, path, **kwargs):
        self.path = path
        # open the file, read header, etc.

    @classmethod
    def can_open(cls, path) -> bool:
        from pathlib import Path
        return Path(path).suffix.lower() == ".myfmt"

    def _shape5d(self):
        return (self.nt, self.nc, self.nz, self.ny, self.nx)  # TCZYX

    def __getitem__(self, key):
        # key is whatever the caller passed; normalize to 5D and read only
        # the requested ranges. Do NOT iterate a full axis to satisfy one index.
        ...

    @property
    def dtype(self):
        return np.dtype("uint16")

    @property
    def metadata(self) -> dict:
        return {}
```

`shape`, `ndim` (== 5), `nt/nc/nz/ny/nx`, `squeeze()`, and the registry hooks
come from `LazyArray`. Everything else (reductions, frame rate, voxel size,
ROIs, phase correction) is opt-in via the mixins in `mbo_utilities.arrays` and
`mbo_utilities.arrays.features` — none are required.

## Register it

Either at runtime:

```python
from mbo_utilities import register_array_class
register_array_class(MyFormatArray)            # or priority=120 to override
```

…or, preferred for a package, via an entry point in your `pyproject.toml` so
`imread()` finds it with no import:

```toml
[project.entry-points."mbo_utilities.lazy_arrays"]
myfmt = "mypkg.arrays:MyFormatArray"
```

## Priority guide

| Range | Use |
|------|------|
| 90–100 | very specific layout marker (e.g. a directory sentinel) |
| 70–80  | format + sub-pattern detection |
| 50     | generic format detection (default) |
| 30     | last-resort fallback |

`can_open()` is called on every candidate during dispatch — keep it cheap and
exception-safe (a raised exception is treated as "no").
