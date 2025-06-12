from pathlib import Path

from .file_io import (
    get_files,
    npy_to_dask,
    read_scan,
    expand_paths,
    get_mbo_dirs,
    normalize_file_url,
)
from .assembly import save_as, save_nonscan
from .metadata import is_raw_scanimage, get_metadata, params_from_metadata
from .util import (
    norm_minmax,
    smooth_data,
    is_running_jupyter,
    is_imgui_installed,
    subsample_array,
)

if is_imgui_installed():
    from .graphics import run_gui
else:
    raise ImportError(
        f"This should be installed with mbo_utilities. Please report this [here](https://github.com/MillerBrainObservatory/mbo_utilities/issues) or on slack."
    )

__version__ = (Path(__file__).parent / "VERSION").read_text().strip()


__all__ = [
    # file_io
    "run_gui",
    "get_mbo_dirs",
    "scanreader",
    "npy_to_dask",
    "get_files",
    "read_scan",
    "subsample_array",
    # metadata
    "is_raw_scanimage",
    "get_metadata",
    "params_from_metadata",
    # util
    "normalize_file_url",
    "expand_paths",
    "norm_minmax",
    "smooth_data",
    "is_running_jupyter",
    "is_imgui_installed",  # we may just enforce imgui?
    # assembly
    "save_as",
    "save_nonscan",
]
