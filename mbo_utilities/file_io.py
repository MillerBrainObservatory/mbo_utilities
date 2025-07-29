import json
from collections.abc import Sequence
from io import StringIO
from itertools import product
import re

from pathlib import Path
import numpy as np

from icecream import ic

import dask.array as da
from tifffile import TiffFile

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

MBO_SUPPORTED_FTYPES = [".tiff", ".bin", ".h5"]
MBO_PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")

logger = log.get("file_io")


def load_ops(ops_input: str | Path | list[str | Path]):
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning None.")
    return {}


def write_ops(metadata, raw_filename):
    """
    Write metadata to an ops file alongside the given filename.
    metadata must contain
    'shape'
    'pixel_resolution',
    'frame_rate' keys.
    """
    logger.debug(f"Writing ops file for {raw_filename} with metadata: {metadata}")
    assert isinstance(raw_filename, (str, Path)), (
        "filename must be a string or Path object"
    )
    filename = Path(raw_filename).expanduser().resolve()

    # this convention means input can be either
    if filename.is_file():
        root = filename.parent
    else:
        root = filename

    ops_path = root.joinpath("ops.npy")
    logger.debug(f"Writing ops file to {ops_path}.")

    shape = metadata["shape"]
    nt = shape[0]
    Lx = shape[-2]
    Ly = shape[-1]

    if "pixel_resolution" not in metadata:
        logger.warning("No pixel resolution found in metadata, using default [2, 2].")
    if "fs" not in metadata:
        if "frame_rate" in metadata:
            metadata["fs"] = metadata["frame_rate"]
        elif "framerate" in metadata:
            metadata["fs"] = metadata["framerate"]
        else:
            logger.debug("No frame rate found in metadata; defaulting fs=10")
            metadata["fs"] = 10

    dx, dy = metadata.get("pixel_resolution", [2, 2])
    ops = {
        # suite2p needs these
        "Ly": Ly,
        "Lx": Lx,
        "fs": metadata["fs"],
        "nframes": nt,
        "dx": dx,
        "dy": dy,
        "ops_path": str(ops_path),
        # and dump the rest of the metadata
        **metadata,
    }
    np.save(ops_path, ops)
    logger.debug(f"Ops file written to {ops_path} with metadata:\n {ops}")


def npy_to_dask(files, name="", axis=1, astype=None):
    """
    Creates a Dask array that lazily stacks multiple .npy files along a specified axis without fully loading them into memory.

    Taken from suite3d for convenience
    https://github.com/alihaydaroglu/suite3d/blob/py310/suite3d/utils.py
    To avoid the unnessessary import. Very nice function, thanks Ali!

    Parameters
    ----------
    files : list of str or Path
        A list of file paths pointing to .npy files containing array data. Each file must have the same shape except
        possibly along the concatenation axis.
    name : str, optional
        A string to be appended to a base name ("from-npy-stack-") to label the resulting Dask array. Default is an empty string.
    axis : int, optional
        The axis along which to stack/concatenate the arrays from the provided files. Default is 1.
    astype : numpy.dtype, optional
        If provided, the resulting Dask array will be cast to this data type. Otherwise, the data type is inferred
        from the first file.

    Returns
    -------
    dask.array.Array

    Examples
    --------
    >>> # https://www.fastplotlib.org/
    >>> import fastplotlib as fpl
    >>> import mbo_utilities as mbo
    >>> files = mbo.get_files("path/to/images/", 'fused', 3) # suite3D output
    >>> arr = npy_to_dask(files, name="stack", axis=1)
    >>> print(arr.shape)
    (nz, nt, ny, nx )
    >>> # Optionally, cast the array to float32
    >>> arr = npy_to_dask(files, axis=1, astype=np.float32)
    >>> fpl.ImageWidget(arr.transpose(1, 0, 2, 3)).show()
    """
    sample_mov = np.load(files[0], mmap_mode="r")
    file_ts = [np.load(f, mmap_mode="r").shape[axis] for f in files]
    nz, nt_sample, ny, nx = sample_mov.shape

    dtype = sample_mov.dtype
    chunks = [(nz,), (nt_sample,), (ny,), (nx,)]
    chunks[axis] = tuple(file_ts)
    chunks = tuple(chunks)
    name = "from-npy-stack-%s" % name

    keys = list(product([name], *[range(len(c)) for c in chunks]))
    values = [(np.load, files[i], "r") for i in range(len(chunks[axis]))]

    dsk = dict(zip(keys, values, strict=False))

    arr = da.Array(dsk, name, chunks, dtype)
    if astype is not None:
        arr = arr.astype(astype)

    return arr


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


def _tiff_to_fsspec(tif_path: Path, base_dir: Path) -> dict:
    """
    Create a kerchunk reference for a single TIFF file.

    Parameters
    ----------
    tif_path : Path
        Path to the TIFF file on disk.
    base_dir : Path
        Directory representing the “root” URI for the reference.

    Returns
    -------
    refs : dict
        A kerchunk reference dict (in JSON form) for this single TIFF.
    """
    with TiffFile(str(tif_path.expanduser().resolve())) as tif:
        with StringIO() as f:
            store = tif.aszarr()
            store.write_fsspec(f, url=base_dir.as_uri())
            refs = json.loads(f.getvalue())  # type: ignore
    return refs


def _multi_tiff_to_fsspec(tif_files: list[Path], base_dir: Path) -> dict:
    assert len(tif_files) > 1, "Need at least two TIFF files to combine."

    combined_refs: dict[str, str] = {}
    per_file_refs = []
    total_shape = None
    total_chunks = None
    zarr_meta = {}
    for tif_path in tif_files:
        # Create a json reference for each TIFF file
        inner_refs = _tiff_to_fsspec(tif_path, base_dir)
        zarr_meta = json.loads(inner_refs.pop(".zarray"))
        inner_refs.pop(".zattrs", None)

        shape = zarr_meta["shape"]
        chunks = zarr_meta["chunks"]

        if total_shape is None:
            total_shape = shape.copy()
            total_chunks = chunks
        else:
            assert shape[1:] == total_shape[1:], f"Shape mismatch in {tif_path}"
            assert chunks == total_chunks, f"Chunk mismatch in {tif_path}"
            total_shape[0] += shape[0]  # accumulate along axis 0

        per_file_refs.append((inner_refs, shape))

    combined_zarr_meta = {
        "shape": total_shape,  # total shape tracks the full-assembled image shape
        "chunks": total_chunks,
        "dtype": zarr_meta["dtype"],
        "compressor": zarr_meta["compressor"],
        "filters": zarr_meta.get("filters", None),
        "order": zarr_meta["order"],
        "zarr_format": zarr_meta["zarr_format"],
        "fill_value": zarr_meta.get("fill_value", 0),
    }

    combined_refs[".zarray"] = json.dumps(combined_zarr_meta)
    combined_refs[".zattrs"] = json.dumps(
        {"_ARRAY_DIMENSIONS": ["T", "C", "Y", "X"][: len(total_shape)]}
    )

    axis0_offset = 0
    # since we are combining along axis 0, we need to adjust the keys
    # in the inner_refs to account for the offset along that axis.
    for inner_refs, shape in per_file_refs:
        chunksize0 = total_chunks[0]
        for key, val in inner_refs.items():
            idx = list(map(int, key.strip("/").split(".")))
            idx[0] += axis0_offset // chunksize0
            new_key = ".".join(map(str, idx))
            combined_refs[new_key] = val
        axis0_offset += shape[0]

    return combined_refs


def read_scan():
    raise DeprecationWarning("read_scan is deprecated, use mbo.imread() instead.")


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
        ic("Max-depth of 0 is not allowed. Setting to 1.")
        max_depth = 1

    base_depth = len(base_path.parts)
    pattern = f"*{str_contains}*" if str_contains else "*"

    if exclude_dirs is None:
        exclude_dirs = [".venv", ".git", "__pycache__"]

    def is_excluded(path):
        return any(excl in path.parts for excl in exclude_dirs)

    files = [
        file
        for file in base_path.rglob(pattern)
        if len(file.parts) - base_depth <= max_depth
        and file.is_file()
        and not is_excluded(file)
    ]

    if sort_ascending:

        def numerical_sort_key(path):
            match = re.search(r"\d+", path.name)
            return int(match.group()) if match else float("inf")

        files.sort(key=numerical_sort_key)

    return [str(file) for file in files]


def get_plane_from_filename(path, fallback=None):
    path = Path(path)
    for part in path.stem.lower().split("_"):
        if part.startswith("plane"):
            suffix = part[5:]
            if suffix.isdigit():
                return int(suffix.lstrip("0") or "0")
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not extract plane number from filename: {path.name}")


def _is_arraylike(obj) -> bool:
    """
    Checks if the object is array-like.
    For now just checks if obj has `__getitem__()`
    """
    for attr in ["__getitem__", "shape", "ndim"]:
        if not hasattr(obj, attr):
            return False

    return True


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


def _convert_range_to_slice(k):
    return slice(k.start, k.stop, k.step) if isinstance(k, range) else k
