"""
directory scanner for discovering datasets.

uses pipeline_registry patterns to identify datasets.
"""

import os
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator

from mbo_utilities import log
from mbo_utilities.db.models import Dataset, DatasetStatus

logger = log.get("db.scanner")


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
                    metadata["num_frames"] = meta.get("num_frames")
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

    # ensure all pipelines are registered
    register_all_pipelines()
    pipelines = get_all_pipelines()

    root = Path(root).resolve()
    if not root.exists():
        logger.error(f"path does not exist: {root}")
        return

    # collect all marker files and patterns
    marker_patterns = {}  # pattern -> pipeline_name
    for name, info in pipelines.items():
        for marker in info.marker_files:
            marker_patterns[marker] = name
        for pattern in info.input_patterns:
            # only use specific patterns, not generic ones like **/*.tif
            if "?" in pattern or pattern.count("*") > 2:
                marker_patterns[pattern] = name

    # track discovered paths to avoid duplicates
    discovered = set()

    # first pass: find marker files (fast)
    logger.info(f"scanning {root} for datasets...")

    if recursive:
        all_files = list(root.rglob("*"))
    else:
        all_files = list(root.iterdir())

    total_files = len(all_files)

    for idx, path in enumerate(all_files):
        if progress_callback:
            progress_callback(idx, total_files, path)

        # check marker files
        if path.is_file():
            for marker, pipeline_name in marker_patterns.items():
                if path.name == marker or _match_pattern(path, marker):
                    # found a marker - the dataset is the parent directory
                    dataset_path = path.parent
                    if str(dataset_path) not in discovered:
                        discovered.add(str(dataset_path))
                        yield _create_dataset(dataset_path, pipeline_name, pipelines)
                    break

        # check directory patterns (e.g. *.zarr)
        elif path.is_dir():
            for name, info in pipelines.items():
                for pattern in info.input_patterns:
                    if pattern.endswith("/") or pattern.endswith(".zarr"):
                        if _match_pattern(path, pattern.rstrip("/")):
                            if str(path) not in discovered:
                                discovered.add(str(path))
                                yield _create_dataset(path, name, pipelines)
                            break

    # second pass: find files by extension (for files without markers)
    ext_to_pipeline = {}
    for name, info in pipelines.items():
        for ext in info.input_extensions:
            if ext not in ext_to_pipeline:
                ext_to_pipeline[ext] = name

    for idx, path in enumerate(all_files):
        if path.is_file() and str(path) not in discovered:
            ext = path.suffix.lstrip(".").lower()
            if ext in ext_to_pipeline:
                # skip if this is inside an already-discovered dataset
                parent_discovered = any(
                    str(path).startswith(d + os.sep) for d in discovered
                )
                if not parent_discovered:
                    pipeline_name = ext_to_pipeline[ext]
                    # skip generic tiff files unless they're large (likely raw data)
                    if ext in ("tif", "tiff") and path.stat().st_size < 10 * 1024 * 1024:
                        continue
                    discovered.add(str(path))
                    yield _create_dataset(path, pipeline_name, pipelines)


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
