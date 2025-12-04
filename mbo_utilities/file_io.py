from collections import defaultdict
from collections.abc import Sequence
import re

from pathlib import Path
import numpy as np

import dask.array as da
from tifffile import tifffile

from . import log
from .reader import PIPELINE_TAGS

try:
    from zarr import open as zarr_open
    from zarr.storage import FsspecStore
    from fsspec.implementations.reference import ReferenceFileSystem

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    zarr_open = None
    ReferenceFileSystem = None
    FsspecStore = None

CHUNKS = {0: 1, 1: "auto", 2: -1, 3: -1}

logger = log.get("file_io")


def files_to_dask(files: list[str | Path], astype=None, chunk_t=250):
    """
    Lazily build a Dask array or list of arrays depending on filename tags.

    - "plane", "z", or "chan" → stacked along Z (TZYX)
    - "roi" → list of 3D (T,Y,X) arrays, one per ROI
    - otherwise → concatenate all files in time (T)
    """
    files = [Path(f) for f in files]
    if not files:
        raise ValueError("No input files provided.")

    has_plane = any(re.search(r"(plane|z|chan)[_-]?\d+", f.stem, re.I) for f in files)
    has_roi = any(re.search(r"roi[_-]?\d+", f.stem, re.I) for f in files)

    # lazy-load utility inline
    def load_lazy(f):
        if f.suffix == ".npy":
            arr = np.load(f, mmap_mode="r")
        elif f.suffix in (".tif", ".tiff"):
            arr = tifffile.memmap(f, mode="r")
        else:
            raise ValueError(f"Unsupported file type: {f}")
        chunks = (min(chunk_t, arr.shape[0]),) + arr.shape[1:]
        return da.from_array(arr, chunks=chunks)

    if has_roi:
        roi_groups = defaultdict(list)
        for f in files:
            m = re.search(r"roi[_-]?(\d+)", f.stem, re.I)
            roi_idx = int(m.group(1)) if m else 0
            roi_groups[roi_idx].append(f)

        roi_arrays = []
        for roi_idx, group in sorted(roi_groups.items()):
            arrays = [load_lazy(f) for f in sorted(group)]
            darr = da.concatenate(arrays, axis=0)  # concat in time
            if astype:
                darr = darr.astype(astype)
            roi_arrays.append(darr)
        return roi_arrays

    # Plane or Z grouping case
    if has_plane:
        plane_groups = defaultdict(list)
        for f in files:
            m = re.search(r"(plane|z|chan)[_-]?(\d+)", f.stem, re.I)
            plane_idx = int(m.group(2)) if m else 0
            plane_groups[plane_idx].append(f)

        plane_stacks = []
        for z, group in sorted(plane_groups.items()):
            arrays = [load_lazy(f) for f in sorted(group)]
            plane = da.concatenate(arrays, axis=0)
            plane_stacks.append(plane)

        full = da.stack(plane_stacks, axis=1)  # (T,Z,Y,X)
        return full.astype(astype) if astype else full

    # Default: concatenate along time
    arrays = [load_lazy(f) for f in sorted(files)]
    full = da.concatenate(arrays, axis=0)  # (T,Y,X)
    return full.astype(astype) if astype else full


def expand_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """
    Expand a path, list of paths, or wildcard pattern into a sorted list of actual files.

    This is a handy wrapper for loading images or data files when you’ve got a folder,
    some wildcards, or a mix of both.

    Parameters
    ----------
    paths : str, Path, or list of (str or Path)
        Can be a single path, a wildcard pattern like "\\*.tif", a folder, or a list of those.

    Returns
    -------
    list of Path
        Sorted list of full paths to matching files.

    Examples
    --------
    >>> expand_paths("data/\\*.tif")
    [Path("data/img_000.tif"), Path("data/img_001.tif"), ...]

    >>> expand_paths(Path("data"))
    [Path("data/img_000.tif"), Path("data/img_001.tif"), ...]

    >>> expand_paths(["data/\\*.tif", Path("more_data")])
    [Path("data/img_000.tif"), Path("more_data/img_050.tif"), ...]
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected str, Path, or sequence of them, got {type(paths)}")

    result = []
    for p in paths:
        p = Path(p)
        if "*" in str(p):
            result.extend(p.parent.glob(p.name))
        elif p.is_dir():
            result.extend(p.glob("*"))
        elif p.exists() and p.is_file():
            result.append(p)

    return sorted(p.resolve() for p in result if p.is_file())


def sort_by_si_filename(filename):
    """
    Sort ScanImage files by the last number in the filename (e.g., _00001, _00002, etc.).
    """
    numbers = re.findall(r"\d+", str(filename))
    return int(numbers[-1]) if numbers else 0


def get_files(
    base_dir, str_contains="", max_depth=1, sort_ascending=True, exclude_dirs=None
) -> list | Path:
    """
    Recursively search for files in a specified directory whose names contain a given substring,
    limiting the search to a maximum subdirectory depth. Optionally, the resulting list of file paths
    is sorted in ascending order using numeric parts of the filenames when available.

    Parameters
    ----------
    base_dir : str or Path
        The base directory where the search begins. This path is expanded (e.g., '~' is resolved)
        and converted to an absolute path.
    str_contains : str, optional
        A substring that must be present in a file's name for it to be included in the result.
        If empty, all files are matched.
    max_depth : int, optional
        The maximum number of subdirectory levels (relative to the base directory) to search.
        Defaults to 1. If set to 0, it is automatically reset to 1.
    sort_ascending : bool, optional
        If True (default), the matched file paths are sorted in ascending alphanumeric order.
        The sort key extracts numeric parts from filenames so that, for example, "file2" comes
        before "file10".
    exclude_dirs : iterable of str or Path, optional
        An iterable of directories to exclude from the resulting list of file paths. By default
        will exclude ".venv/", "__pycache__/", ".git" and ".github"].

    Returns
    -------
    list of str
        A list of full file paths (as strings) for files within the base directory (and its
        subdirectories up to the specified depth) that contain the provided substring.

    Raises
    ------
    FileNotFoundError
        If the base directory does not exist.
    NotADirectoryError
        If the specified base_dir is not a directory.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> # Get all files that contain "ops.npy" in their names by searching up to 3 levels deep:
    >>> ops_files = mbo.get_files("path/to/files", "ops.npy", max_depth=3)
    >>> # Get only files containing "tif" in the current directory (max_depth=1):
    >>> tif_files = mbo.get_files("path/to/files", "tif")
    """
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory '{base_path}' does not exist.")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' is not a directory.")
    if max_depth == 0:
        max_depth = 1

    base_depth = len(base_path.parts)
    pattern = f"*{str_contains}*" if str_contains else "*"

    if exclude_dirs is None:
        exclude_dirs = [".venv", ".git", "__pycache__"]

    files = [
        file
        for file in base_path.rglob(pattern)
        if len(file.parts) - base_depth <= max_depth
        and file.is_file()
        and not any(excl in file.parts for excl in exclude_dirs)
    ]

    if sort_ascending:
        files.sort(key=sort_by_si_filename)
    return files


def derive_tag_from_filename(path):
    """
    Derive a folder tag from a filename based on “planeN”, “roiN”, or "tagN" patterns.

    Parameters
    ----------
    path : str or pathlib.Path
        File path or name whose stem will be parsed.

    Returns
    -------
    str
        If the stem starts with “plane”, “roi”, or “res” followed by an integer,
        returns that tag plus the integer (e.g. “plane3”, “roi7”, “res2”).
        Otherwise returns the original stem unchanged.

    Examples
    --------
    >>> derive_tag_from_filename("plane_01.tif")
    'plane1'
    >>> derive_tag_from_filename("plane2.bin")
    'plane2'
    >>> derive_tag_from_filename("roi5.raw")
    'roi5'
    >>> derive_tag_from_filename("ROI_10.dat")
    'roi10'
    >>> derive_tag_from_filename("res-3.h5")
    'res3'
    >>> derive_tag_from_filename("assembled_data_1.tiff")
    'assembled_data_1'
    >>> derive_tag_from_filename("file_12.tif")
    'file_12'
    """
    name = Path(path).stem
    for tag in PIPELINE_TAGS:
        low = name.lower()
        if low.startswith(tag):
            suffix = name[len(tag) :]
            if suffix and (suffix[0] in ("_", "-")):
                suffix = suffix[1:]
            if suffix.isdigit():
                return f"{tag}{int(suffix)}"
    return name


def _get_mbo_project_root() -> Path:
    """Return the root path of the mbo_utilities repository (based on this file)."""
    return Path(__file__).resolve().parent.parent


def get_mbo_dirs() -> dict:
    """
    Ensure ~/mbo and its subdirectories exist.

    Returns a dict with paths to the root, settings, and cache directories.
    """
    base = Path.home().joinpath("mbo")
    imgui = base.joinpath("imgui")
    cache = base.joinpath("cache")
    logs = base.joinpath("logs")
    tests = base.joinpath("tests")
    data = base.joinpath("data")

    assets = imgui.joinpath("assets")
    settings = assets.joinpath("app_settings")

    for d in (base, imgui, cache, logs, assets, data, tests):
        d.mkdir(exist_ok=True)

    return {
        "base": base,
        "imgui": imgui,
        "cache": cache,
        "logs": logs,
        "assets": assets,
        "settings": settings,
        "data": data,
        "tests": tests,
    }


def get_last_savedir_path() -> Path:
    """Return path to settings file tracking last saved folder.

    .. deprecated::
        Use :func:`mbo_utilities.preferences.get_last_save_dir` instead.
    """
    import warnings
    warnings.warn(
        "get_last_savedir_path() is deprecated. Use mbo_utilities.preferences.get_last_save_dir() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Path.home().joinpath("mbo", "settings", "last_savedir.json")


def load_last_savedir(default=None) -> Path:
    """Load last saved directory path if it exists.

    .. deprecated::
        Use :func:`mbo_utilities.preferences.get_last_save_dir` instead.
    """
    import warnings
    warnings.warn(
        "load_last_savedir() is deprecated. Use mbo_utilities.preferences.get_last_save_dir() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mbo_utilities.preferences import get_last_save_dir
    result = get_last_save_dir()
    if result:
        return result
    return Path(default or Path().cwd())


def save_last_savedir(path: Path):
    """Persist the most recent save directory path.

    .. deprecated::
        Use :func:`mbo_utilities.preferences.set_last_save_dir` instead.
    """
    import warnings
    warnings.warn(
        "save_last_savedir() is deprecated. Use mbo_utilities.preferences.set_last_save_dir() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mbo_utilities.preferences import set_last_save_dir
    set_last_save_dir(path)


def print_tree(path, max_depth=1, prefix=""):
    path = Path(path)
    if not path.is_dir():
        print(path)
        return

    entries = sorted([p for p in path.iterdir() if p.is_dir()])
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name + "/")

        if max_depth > 1:
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, max_depth=max_depth - 1, prefix=prefix + extension)
