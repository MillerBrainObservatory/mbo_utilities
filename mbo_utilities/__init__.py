import os
from pathlib import Path

from .file_io import (
    get_files,
    zstack_from_files,
    npy_to_dask,
    read_scan,
    save_png,
    save_mp4,
    expand_paths,
    _get_mbo_dirs,
)
from .assembly import save_as
from .metadata import is_raw_scanimage, get_metadata, params_from_metadata
from .util import (
    norm_minmax,
    smooth_data,
    is_running_jupyter,
    is_imgui_installed,
    subsample_array,
)

try:
    from icecream import ic, install

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
    install = None

if is_imgui_installed():
    from .graphics import run_gui
else:
    raise ImportError(f"This should be installed with mbo_utilities. Please report this [here](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or on slack.")

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()

# default to disabling all ic() calls
ic.enable() if os.environ.get("MBO_DEBUG", False) else ic.disable()


__all__ = [
    # file_io
    "scanreader",
    "npy_to_dask",
    "get_files",
    "zstack_from_files",
    "read_scan",
    "save_png",
    "save_mp4",
    "subsample_array",
    # metadata
    "is_raw_scanimage",
    "get_metadata",
    "params_from_metadata",
    # util
    "expand_paths",
    "norm_minmax",
    "smooth_data",
    "is_running_jupyter",
    "is_imgui_installed",  # we may just enforce imgui?
    # assembly
    "save_as",
]
