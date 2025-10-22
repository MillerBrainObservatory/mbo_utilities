import json
from collections import defaultdict
from collections.abc import Sequence
from io import StringIO
import re

from pathlib import Path
import numpy as np

import dask.array as da
from tifffile import TiffFile, tifffile

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

MBO_SUPPORTED_FTYPES = [".tiff", ".zarr", ".bin", ".h5"]
PIPELINE_TAGS = ("plane", "roi", "z", "plane_", "roi_", "z_")


logger = log.get("file_io")


def load_ops(ops_input: str | Path | list[str | Path]):
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning None.")
    return {}


def write_ops(metadata, raw_filename, **kwargs):
    """
    Write metadata to an ops file alongside the given filename.
    metadata must contain
    'shape', 'pixel_resolution', 'frame_rate' keys.
    """
    logger.debug(f"Writing ops file for {raw_filename} with metadata: {metadata}")
    assert isinstance(raw_filename, (str, Path))
    filename = Path(raw_filename).expanduser().resolve()

    structural = kwargs.get("structural", False)
    chan = 2 if structural or "data_chan2.bin" in str(filename) else 1
    logger.debug(f"Detected channel {chan}")

    root = filename.parent if filename.is_file() else filename
    ops_path = root / "ops.npy"
    logger.info(f"Writing ops file to {ops_path}")

    shape = metadata["shape"]
    nt, Lx, Ly = shape[0], shape[-2], shape[-1]

    if "pixel_resolution" not in metadata:
        logger.warning("No pixel resolution found in metadata, using default [2, 2].")
    if "fs" not in metadata:
        if "frame_rate" in metadata:
            metadata["fs"] = metadata["frame_rate"]
        elif "framerate" in metadata:
            metadata["fs"] = metadata["framerate"]
        else:
            logger.warning("No frame rate found; defaulting fs=10")
            metadata["fs"] = 10

    dx, dy = metadata.get("pixel_resolution", [2, 2])

    # Load or initialize ops
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
    else:
        from .metadata import default_ops
        ops = default_ops()

    # Update shared core fields
    ops.update({
        "Ly": Ly,
        "Lx": Lx,
        "fs": metadata["fs"],
        "dx": dx,
        "dy": dy,
        "ops_path": str(ops_path),
    })

    # Channel-specific entries
    if chan == 1:
        ops["nframes_chan1"] = nt
        ops["raw_file"] = str(filename)
    else:
        ops["nframes_chan2"] = nt
        ops["chan2_file"] = str(filename)

    ops["align_by_chan"] = chan

    # Compatibility: prefer functional channel for top-level nframes
    if "nframes_chan1" in ops:
        ops["nframes"] = ops["nframes_chan1"]
    elif "nframes_chan2" in ops:
        ops["nframes"] = ops["nframes_chan2"]

    # Merge extra metadata without overwriting
    ops = {**ops, **metadata}
    np.save(ops_path, ops)
    logger.debug(f"Ops file written to {ops_path} with metadata:\n{ops}")


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
    has_roi   = any(re.search(r"roi[_-]?\d+", f.stem, re.I) for f in files)

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


def sort_by_si_filename(filename):
    """
    Sort ScanImage files by the last number in the filename (e.g., _00001, _00002, etc.).
    """
    numbers = re.findall(r"\d+", str(filename))
    return int(numbers[-1]) if numbers else 0


def is_excluded(path, exclude_dirs=()):
    return any(excl in path.parts for excl in exclude_dirs)


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
        and not is_excluded(file, exclude_dirs)
    ]

    if sort_ascending:
        files.sort(key=sort_by_si_filename)
    return files


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


def group_plane_rois(input_dir):
    input_dir = Path(input_dir)
    grouped = defaultdict(list)

    for d in input_dir.iterdir():
        if (
                d.is_dir()
                and not d.name.endswith(".zarr")     # exclude zarr dirs
                and d.stem.startswith("plane")
                and "_roi" in d.stem
        ):
            parts = d.stem.split("_")
            if len(parts) == 2 and parts[1].startswith("roi"):
                plane = parts[0]  # e.g. "plane01"
                grouped[plane].append(d)

    return grouped


def merge_zarr_rois(
        input_dir,
        output_dir=None,
        overwrite=True
):
    """
    Concatenate roi1 + roi2 .zarr stores for each plane into a single planeXX.zarr.

    Parameters
    ----------
    input_dir : Path or str
        Directory containing planeXX_roi1, planeXX_roi2 subfolders with ops.npy + data.zarr.
    output_dir : Path or str, optional
        Where to write merged planeXX.zarr. Defaults to `input_dir`.
    overwrite : bool
        If True, existing outputs are replaced.
    """

    z_merged = None
    input_dir = Path(input_dir)
    output_dir = (
        Path(output_dir)
        if output_dir
        else input_dir.parent / (input_dir.name + "_merged")
    )
    output_dir.mkdir(exist_ok=True)
    logger.debug(f"Saving merged zarrs to {output_dir}")

    roi1_dirs = sorted(input_dir.glob("*plane*_roi1*"))
    roi2_dirs = sorted(input_dir.glob("*plane*_roi2*"))
    if not roi1_dirs or not roi2_dirs:
        logger.critical("No roi1 or roi2 in input dir")
        return None
    assert len(roi1_dirs) == len(roi2_dirs), "Mismatched ROI dirs"

    for roi1, roi2 in zip(roi1_dirs, roi2_dirs):
        zplane = roi1.stem.split("_")[0]  # "plane01"
        out_path = output_dir / f"{zplane}.zarr"
        if out_path.exists():
            if overwrite:
                logger.info(f"Overwriting {out_path}")
                import shutil

                shutil.rmtree(out_path)
            else:
                logger.info(f"Skipping {zplane}, {out_path} exists")
                continue

        # load ops
        z1 = da.from_zarr(roi1)
        z2 = da.from_zarr(roi2)

        assert z1.shape[0] == z2.shape[0], "Frame count mismatch"
        assert z1.shape[1] == z2.shape[1], "Height mismatch"

        # concatenate along width (axis=2)
        z_merged = da.concatenate([z1, z2], axis=2)
        z_merged.to_zarr(out_path, overwrite=overwrite)

    if z_merged is not None:
        logger.info(f"Merged shape: {z_merged.shape}")

    return None


def load_zarr_grouped(input_dir, ):
    """
    Discover and lazily concatenate ROI .zarr stores per plane.

    Returns
    -------
    dict[str, dask.array.Array]
        Mapping plane tag -> concatenated dask array
    """
    """
    Lazily load multiple planeN_roiN.zarr stores into a single 4D array (T, Z, Y, X).

    Each plane's ROIs are concatenated horizontally (along X),
    and all planes are stacked along the Z dimension.

    Returns
    -------
    dask.array.Array
        Lazy 4D array with shape (T, Z, Y, X_total)
    """
    input_dir = Path(input_dir)
    grouped = defaultdict(list)

    # collect plane-roi groups
    for d in sorted(input_dir.glob("plane*_roi*.zarr")):
        if not d.is_dir():
            continue
        parts = d.stem.split("_")
        if len(parts) == 2 and parts[1].startswith("roi"):
            grouped[parts[0]].append(d)

    if not grouped:
        raise ValueError(f"No plane*_roi*.zarr directories in {input_dir}")

    planes = []
    for plane, roi_dirs in sorted(grouped.items(), key=lambda kv: kv[0]):
        roi_dirs = sorted(roi_dirs, key=lambda p: p.stem.lower())
        arrays = [da.from_zarr(p, chunks=None) for p in roi_dirs]

        base_shape = arrays[0].shape
        for a in arrays[1:]:
            if a.shape[:2] != base_shape[:2]:
                raise ValueError(f"Shape mismatch in {plane}: {a.shape} vs {base_shape}")

        merged_plane = da.concatenate(arrays, axis=2)  # concat horizontally
        planes.append(merged_plane)
        logger.info(f"{plane}: concatenated {len(arrays)} ROIs → {merged_plane.shape}")

    arr_4d = da.stack(planes, axis=1)  # stack planes along Z
    logger.info(f"Final 4D array shape: {arr_4d.shape} (T, Z, Y, X)")
    return arr_4d

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


def get_last_savedir_path() -> Path:
    """Return path to settings file tracking last saved folder."""
    return Path.home().joinpath("mbo", "settings", "last_savedir.json")

def load_last_savedir(default=None) -> Path:
    """Load last saved directory path if it exists."""
    f = get_last_savedir_path()
    if f.is_file():
        try:
            path = Path(json.loads(f.read_text()).get("last_savedir", ""))
            if path.exists():
                return path
        except Exception:
            pass
    return Path(default or Path().cwd())

def save_last_savedir(path: Path):
    """Persist the most recent save directory path."""
    f = get_last_savedir_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"last_savedir": str(path)}))


def _convert_range_to_slice(k):
    return slice(k.start, k.stop, k.step) if isinstance(k, range) else k


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
