"""
directory scanner for discovering datasets.

uses pipeline_registry patterns to identify datasets.
handles scanimage raw tiffs (filename_00001.tif, filename_00002.tif)
by grouping them into a single acquisition.
"""

import os
import re
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

from mbo_utilities import log
from mbo_utilities.db.models import Dataset, DatasetStatus

logger = log.get("db.scanner")

# pattern to match scanimage numbered suffixes: _00001, _00002, etc.
# also handles _00001_00001 (acquisition_timepoint) pattern
# we want to strip the LAST _NNNNN (or _NNNNN_NNNNN) from the name
SCANIMAGE_SUFFIX_PATTERN = re.compile(r"(_\d{5})+$")


def _match_pattern(path: Path, pattern: str) -> bool:
    """check if path matches a glob pattern."""
    # convert path to posix for consistent matching
    path_str = path.as_posix()

    # handle ** patterns
    if "**" in pattern:
        # strip leading **/ for matching
        pattern_parts = pattern.split("**/")
        if len(pattern_parts) == 2:
            suffix_pattern = pattern_parts[1]
            return fnmatch(path.name, suffix_pattern) or fnmatch(path_str, pattern)
    else:
        return fnmatch(path.name, pattern)

    return False


def _get_directory_size(path: Path) -> int:
    """get total size of directory contents."""
    total = 0
    try:
        if path.is_file():
            return path.stat().st_size
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def _get_scanimage_base_name(path: Path) -> str | None:
    """
    extract base name from scanimage tiff, stripping _00001 suffix.

    returns None if not a scanimage-style filename.
    examples:
        pollen_00001.tif -> pollen
        scan_00001_00001.tif -> scan_00001 (first is acquisition, second is timepoint)
        plane01.tif -> None (not scanimage style)
    """
    stem = path.stem
    match = SCANIMAGE_SUFFIX_PATTERN.search(stem)
    if match:
        return stem[:match.start()]
    return None


def _group_scanimage_files(tiff_files: list[Path]) -> dict[str, list[Path]]:
    """
    group scanimage tiff files by their base acquisition name.

    returns dict mapping base_name -> list of files.
    non-scanimage tiffs are grouped by their full stem.
    """
    groups = {}
    for f in tiff_files:
        base = _get_scanimage_base_name(f)
        if base is None:
            # not a scanimage file, use full stem
            base = f.stem
        key = (str(f.parent), base)  # group by directory + base name
        if key not in groups:
            groups[key] = []
        groups[key].append(f)
    return groups


def _get_files_size(files: list[Path]) -> int:
    """get total size of a list of files."""
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except (OSError, PermissionError):
            pass
    return total


def _is_inside_discovered(path: Path, discovered: set) -> bool:
    """check if path is inside any discovered directory."""
    path_str = str(path)
    parent_str = str(path.parent)
    for d in discovered:
        # skip non-directory entries (like "dir:basename" keys)
        if ":" in d:
            continue
        # check if file's parent is or is inside the discovered dir
        if parent_str == d or parent_str.startswith(d + os.sep):
            return True
    return False


# suite2p output files that should not be indexed separately
SUITE2P_OUTPUT_FILES = {
    "stat.npy", "ops.npy", "iscell.npy",
    "F.npy", "Fneu.npy", "spks.npy", "dff.npy",
    "redcell.npy", "rastermap_model.npy",
}


def _extract_metadata(path: Path, pipeline_name: str) -> dict:
    """extract metadata from a dataset path."""
    metadata = {
        "num_frames": None,
        "num_zplanes": None,
        "num_rois": None,
        "shape": "",
        "dtype": "",
        "dx": None,
        "dy": None,
        "dz": None,
        "fs": None,
        "status": DatasetStatus.UNKNOWN,
    }

    try:
        # try to load with imread to get metadata
        from mbo_utilities import imread, get_metadata, get_voxel_size

        # for suite2p, load ops.npy directly for fast metadata
        if pipeline_name == "suite2p":
            ops_path = path if path.name == "ops.npy" else path / "ops.npy"
            if ops_path.exists():
                from mbo_utilities.util import load_npy
                ops = load_npy(ops_path).item()
                metadata["num_frames"] = ops.get("nframes")
                metadata["shape"] = f"({ops.get('nframes')}, {ops.get('Ly')}, {ops.get('Lx')})"
                metadata["dtype"] = "int16"
                metadata["status"] = DatasetStatus.SEGMENTED
                # check for registration
                if ops.get("iscell") is not None or (path.parent / "iscell.npy").exists():
                    metadata["status"] = DatasetStatus.COMPLETE
                return metadata

        # for zarr, check zarr.json
        if pipeline_name == "zarr":
            zarr_json = path / "zarr.json" if path.is_dir() else path.parent / "zarr.json"
            if zarr_json.exists():
                import json
                with open(zarr_json) as f:
                    zattrs = json.load(f)
                # zarr v3 format
                if "shape" in zattrs:
                    metadata["shape"] = str(tuple(zattrs["shape"]))
                metadata["status"] = DatasetStatus.REGISTERED
                return metadata

        # generic: try imread (may be slow for large files)
        # only do this for small files or when needed
        if path.is_file() and path.stat().st_size < 100 * 1024 * 1024:  # <100MB
            try:
                arr = imread(path)
                metadata["shape"] = str(arr.shape)
                metadata["dtype"] = str(arr.dtype)
                if hasattr(arr, "metadata"):
                    meta = arr.metadata
                    vs = get_voxel_size(meta)
                    metadata["dx"] = vs.dx
                    metadata["dy"] = vs.dy
                    metadata["dz"] = vs.dz
                    metadata["fs"] = meta.get("fs")
                    metadata["num_frames"] = meta.get("num_timepoints", meta.get("num_frames"))
                    metadata["num_zplanes"] = meta.get("num_zplanes")
                    metadata["num_rois"] = meta.get("num_rois")
            except Exception as e:
                logger.debug(f"could not load {path} for metadata: {e}")

    except Exception as e:
        logger.debug(f"metadata extraction failed for {path}: {e}")

    return metadata


def scan_for_datasets(
    root: Path,
    recursive: bool = True,
    progress_callback=None,
) -> Iterator[Dataset]:
    """
    scan a directory for datasets using pipeline registry patterns.

    yields Dataset objects for each discovered dataset.
    groups scanimage raw tiffs (filename_00001.tif, etc.) into single acquisitions.

    Parameters
    ----------
    root : Path
        root directory to scan
    recursive : bool
        whether to scan subdirectories
    progress_callback : callable, optional
        callback(current, total, path) for progress updates

    Yields
    ------
    Dataset
        discovered dataset objects
    """
    from mbo_utilities.pipeline_registry import get_all_pipelines
    from mbo_utilities.arrays import register_all_pipelines
    from mbo_utilities.metadata import is_raw_scanimage

    # ensure all pipelines are registered
    register_all_pipelines()
    pipelines = get_all_pipelines()

    root = Path(root).resolve()
    if not root.exists():
        logger.error(f"path does not exist: {root}")
        return

    # track discovered paths to avoid duplicates
    discovered = set()

    logger.info(f"scanning {root} for datasets...")

    if recursive:
        all_files = list(root.rglob("*"))
    else:
        all_files = list(root.iterdir())

    total_files = len(all_files)

    # separate files and directories
    files = []
    dirs = []
    for path in all_files:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            dirs.append(path)

    # pass 1: find suite2p directories (ops.npy marker)
    for idx, path in enumerate(dirs):
        if progress_callback:
            progress_callback(idx, len(dirs), path)

        ops_file = path / "ops.npy"
        if ops_file.exists():
            # check if this is a plane subdir or the main suite2p dir
            parent_ops = path.parent / "ops.npy"
            if parent_ops.exists():
                # this is a plane subdir, skip (will be handled by parent)
                continue
            if str(path) not in discovered:
                discovered.add(str(path))
                yield _create_dataset(path, "suite2p", pipelines)

    # pass 2: find zarr directories
    for path in dirs:
        if path.suffix == ".zarr":
            if str(path) not in discovered:
                discovered.add(str(path))
                yield _create_dataset(path, "zarr", pipelines)

    # pass 3: collect tiff files by directory and group scanimage sequences
    tiff_files = [f for f in files if f.suffix.lower() in (".tif", ".tiff")]
    tiff_by_dir = {}
    for f in tiff_files:
        parent = str(f.parent)
        if parent not in tiff_by_dir:
            tiff_by_dir[parent] = []
        tiff_by_dir[parent].append(f)

    for dir_path, dir_tiffs in tiff_by_dir.items():
        # skip if inside already-discovered dataset
        if any(dir_path.startswith(d + os.sep) or dir_path == d for d in discovered):
            continue

        # check if these are scanimage raw files
        first_tiff = dir_tiffs[0]
        try:
            is_raw = is_raw_scanimage(first_tiff)
        except Exception:
            is_raw = False

        if is_raw:
            # group by base acquisition name
            groups = _group_scanimage_files(dir_tiffs)
            for (parent_dir, base_name), group_files in groups.items():
                group_files = sorted(group_files)
                # use the directory as the dataset path, with base_name in the name
                dataset_path = Path(parent_dir)
                dataset_key = f"{parent_dir}:{base_name}"
                if dataset_key not in discovered:
                    discovered.add(dataset_key)
                    yield _create_dataset_from_files(
                        group_files, base_name, "scanimage_raw", pipelines
                    )
        else:
            # non-raw tiffs: check for plane structure or treat as generic tiff
            plane_tiffs = [f for f in dir_tiffs if re.search(r"plane\d+", f.stem, re.I)]
            if plane_tiffs:
                # this is a tiff volume directory
                if dir_path not in discovered:
                    discovered.add(dir_path)
                    yield _create_dataset(Path(dir_path), "tiff", pipelines)
            else:
                # generic tiff files - create one dataset per file
                for tf in dir_tiffs:
                    if str(tf) not in discovered:
                        # skip small tiffs (thumbnails, etc)
                        try:
                            if tf.stat().st_size < 10 * 1024 * 1024:
                                continue
                        except OSError:
                            continue
                        discovered.add(str(tf))
                        yield _create_dataset(tf, "tiff", pipelines)

    # pass 4: find binary files (suite2p outputs or standalone)
    bin_files = [f for f in files if f.suffix.lower() == ".bin"]
    for bf in bin_files:
        parent = bf.parent
        parent_str = str(parent)
        # skip if parent already discovered (suite2p dir)
        if any(parent_str.startswith(d) or d.startswith(parent_str) for d in discovered):
            continue
        # check for ops.npy in same directory
        ops_file = parent / "ops.npy"
        if ops_file.exists():
            # already handled in pass 1
            continue
        # standalone binary file
        if str(bf) not in discovered:
            discovered.add(str(bf))
            yield _create_dataset(bf, "binary", pipelines)

    # pass 5: find other files (h5, npy, etc)
    for f in files:
        if str(f) in discovered:
            continue
        # skip files inside already-discovered directories
        if _is_inside_discovered(f, discovered):
            continue
        ext = f.suffix.lower().lstrip(".")
        if ext in ("h5", "hdf5"):
            discovered.add(str(f))
            yield _create_dataset(f, "h5", pipelines)
        elif ext == "npy":
            # skip suite2p output files
            if f.name in SUITE2P_OUTPUT_FILES:
                continue
            # skip small npy files
            try:
                if f.stat().st_size < 1 * 1024 * 1024:  # <1MB
                    continue
            except OSError:
                continue
            discovered.add(str(f))
            yield _create_dataset(f, "numpy", pipelines)


def _create_dataset_from_files(
    files: list[Path],
    name: str,
    pipeline_name: str,
    pipelines: dict,
) -> Dataset:
    """create a Dataset object from a group of files (e.g. scanimage sequence)."""
    info = pipelines.get(pipeline_name)

    # use parent directory as path, files for size calculation
    path = files[0].parent
    size_bytes = _get_files_size(files)

    # get modification time from most recent file
    modified_at = None
    try:
        mtimes = [f.stat().st_mtime for f in files]
        modified_at = datetime.fromtimestamp(max(mtimes))
    except (OSError, PermissionError):
        pass

    # extract metadata from first file
    metadata = _extract_metadata(files[0], pipeline_name)

    # update metadata with file count info
    metadata["num_files"] = len(files)

    return Dataset(
        path=str(path),
        name=name,
        size_bytes=size_bytes,
        modified_at=modified_at,
        scanned_at=datetime.now(),
        pipeline=pipeline_name,
        category=info.category if info else "",
        status=metadata.get("status", DatasetStatus.RAW),
        num_frames=metadata.get("num_frames"),
        num_zplanes=metadata.get("num_zplanes"),
        num_rois=metadata.get("num_rois"),
        shape=metadata.get("shape", ""),
        dtype=metadata.get("dtype", ""),
        dx=metadata.get("dx"),
        dy=metadata.get("dy"),
        dz=metadata.get("dz"),
        fs=metadata.get("fs"),
    )


def _create_dataset(path: Path, pipeline_name: str, pipelines: dict) -> Dataset:
    """create a Dataset object from a path."""
    info = pipelines.get(pipeline_name)

    # get file stats
    try:
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        size_bytes = stat.st_size if path.is_file() else _get_directory_size(path)
    except (OSError, PermissionError):
        modified_at = None
        size_bytes = 0

    # extract metadata
    metadata = _extract_metadata(path, pipeline_name)

    return Dataset(
        path=str(path),
        name=path.name,
        size_bytes=size_bytes,
        modified_at=modified_at,
        scanned_at=datetime.now(),
        pipeline=pipeline_name,
        category=info.category if info else "",
        status=metadata.get("status", DatasetStatus.UNKNOWN),
        num_frames=metadata.get("num_frames"),
        num_zplanes=metadata.get("num_zplanes"),
        num_rois=metadata.get("num_rois"),
        shape=metadata.get("shape", ""),
        dtype=metadata.get("dtype", ""),
        dx=metadata.get("dx"),
        dy=metadata.get("dy"),
        dz=metadata.get("dz"),
        fs=metadata.get("fs"),
    )
