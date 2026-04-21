from collections import defaultdict
from collections.abc import Sequence
import re

from pathlib import Path
import numpy as np

import dask.array as da
from tifffile import tifffile

from . import log

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


def expand_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    r"""
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
    """Sort ScanImage files by the last number in the filename (e.g., _00001, _00002, etc.)."""
    numbers = re.findall(r"\d+", str(filename))
    return int(numbers[-1]) if numbers else 0


def _is_leaf_zarr(path: Path) -> bool:
    """Check if a .zarr directory is a leaf (no nested .zarr subdirs).

    A leaf zarr store contains chunk data (numbered directories like 0/, 1/)
    and metadata files, but no nested .zarr directories.
    """
    if not path.is_dir() or path.suffix != ".zarr":
        return False
    # Check if any immediate subdirectory is also a .zarr
    try:
        for child in path.iterdir():
            if child.is_dir() and child.suffix == ".zarr":
                return False  # Has nested zarr, not a leaf
    except (PermissionError, OSError):
        pass
    return True


def _walk_with_zarr_filter(base_path: Path, max_depth: int, exclude_dirs: list):
    """Walk directory tree, stopping at leaf .zarr directories.

    Yields files found during traversal. Does not recurse into:
    - Leaf .zarr directories (those without nested .zarr subdirs)
    - Excluded directories
    - Directories beyond max_depth
    """
    len(base_path.parts)

    def _walk(current: Path, depth: int):
        if depth > max_depth:
            return

        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            return

        for entry in entries:
            # Skip excluded directories
            if entry.name in exclude_dirs:
                continue

            if entry.is_file():
                yield entry
            elif entry.is_dir():
                # If this is a leaf .zarr store, yield it as a "file" (the store itself)
                # and don't recurse into its chunk directories
                if _is_leaf_zarr(entry):
                    # Yield the zarr directory itself as if it were a file
                    # (zarr stores are treated as single units)
                    yield entry
                else:
                    # Recurse into non-leaf directories
                    yield from _walk(entry, depth + 1)

    yield from _walk(base_path, 0)


def get_files(
    base_dir, str_contains="", max_depth=1, sort_ascending=True, exclude_dirs=None
) -> list | Path:
    """
    Recursively search for files in a specified directory whose names contain a given substring,
    limiting the search to a maximum subdirectory depth. Optionally, the resulting list of file paths
    is sorted in ascending order using numeric parts of the filenames when available.

    This function intelligently handles zarr stores: it stops recursing into leaf .zarr
    directories (those that don't contain nested .zarr subdirs) to avoid traversing
    thousands of internal chunk directories.

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
    # Handle UNC paths carefully - resolve() can break them on Windows
    base_path = Path(base_dir).expanduser()
    # Only resolve non-UNC paths (UNC paths start with \\)
    path_str = str(base_path)
    if not path_str.startswith("\\\\"):
        base_path = base_path.resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory '{base_path}' does not exist.")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' is not a directory.")
    if max_depth == 0:
        max_depth = 1

    if exclude_dirs is None:
        exclude_dirs = [".venv", ".git", "__pycache__"]

    # Use custom walk that handles zarr stores properly
    files = []
    for entry in _walk_with_zarr_filter(base_path, max_depth, exclude_dirs):
        # Filter by str_contains if specified
        if str_contains and str_contains not in entry.name:
            continue
        files.append(entry)

    if sort_ascending:
        files.sort(key=sort_by_si_filename)
    return files


def load_npy(path):
    """Load .npy file with cross-platform Path object handling.

    Handles pickled PosixPath/WindowsPath objects that fail to load
    across operating systems.
    """
    import pathlib
    import sys
    import pickle

    class CrossPlatformUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "pathlib":
                if name in ("PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"):
                    return pathlib.Path
            return super().find_class(module, name)

    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            _shape, _fortran_order, dtype = np.lib.format._read_array_header(f, version)

            if dtype.hasobject:
                f.seek(0)
                np.lib.format.read_magic(f)
                np.lib.format._read_array_header(f, version)
                return CrossPlatformUnpickler(f).load()
            f.seek(0)
            return np.load(f, allow_pickle=True)
    except Exception:
        _original_posix = getattr(pathlib, "PosixPath", None)
        _original_windows = getattr(pathlib, "WindowsPath", None)
        try:
            if sys.platform == "win32":
                pathlib.PosixPath = pathlib.WindowsPath
            else:
                pathlib.WindowsPath = pathlib.PosixPath
            return np.load(path, allow_pickle=True)
        finally:
            if _original_posix is not None:
                pathlib.PosixPath = _original_posix
            if _original_windows is not None:
                pathlib.WindowsPath = _original_windows


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
    return Path.home().joinpath(".mbo", "settings", "last_savedir.json")


