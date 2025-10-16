from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mbo_utilities")
except PackageNotFoundError:
    # fallback for editable installs
    __version__ = "0.0.0"


from .file_io import (
    get_files,
    files_to_dask,
    expand_paths,
    get_mbo_dirs,
    load_ops,
    write_ops,
    get_plane_from_filename,
)
from .plot_util import save_png, save_mp4

from .metadata import is_raw_scanimage, get_metadata
from .util import (
    norm_minmax,
    smooth_data,
    is_running_jupyter,
    is_imgui_installed,
    subsample_array,
)
from .lazy_array import imread, imwrite


__all__ = [
    # file_io
    "imread",
    "imwrite",
    # "run_gui",
    "get_mbo_dirs",
    "scanreader",
    "files_to_dask",
    "get_files",
    "subsample_array",
    "load_ops",
    "write_ops",
    "get_plane_from_filename",
    # metadata
    "is_raw_scanimage",
    "get_metadata",
    # util
    "expand_paths",
    "norm_minmax",
    "smooth_data",
    "is_running_jupyter",
    "is_imgui_installed",  # we may just enforce imgui?
    # assembly
    "save_mp4",
    "save_png",
]
