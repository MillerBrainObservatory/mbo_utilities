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

    metadata : dict
        Metadata dictionary containing at least 'shape', 'pixel_resolution',
        'frame_rate'.
    raw_filename : str or Path
        Path to the raw data file. The ops file will be written in the same directory.
    kwargs : dict
        Additional keyword arguments to include in the ops file.
    """
    raw_filename = Path(raw_filename)
    ops_file = raw_filename.parent / "ops.npy"

    if not metadata:
        logger.warning(
            "No metadata provided to write_ops. Skipping ops file creation."
        )
        return

    # Basic required fields
    required = ["shape", "pixel_resolution", "frame_rate"]
    for req in required:
        if req not in metadata:
            raise ValueError(f"Metadata must contain '{req}' key.")

    nframes, Ly, Lx = metadata["shape"]
    pixel_resolution = metadata.get("pixel_resolution")
    frame_rate = metadata.get("frame_rate")

    if pixel_resolution and len(pixel_resolution) >= 2:
        dx, dy = pixel_resolution[:2]
    else:
        dx, dy = 1.0, 1.0

    ops = {
        "nframes": int(nframes),
        "Ly": int(Ly),
        "Lx": int(Lx),
        "fs": float(frame_rate) if frame_rate else 30.0,
        "dx": float(dx),
        "dy": float(dy),
        "umPerPix": float(dx) if dx else 1.0,
        "tau": 1.5,
        "data_path": [str(raw_filename.parent.resolve())],
        "save_path": str(raw_filename.parent.resolve()),
        "save_path0": str(raw_filename.parent.resolve()),
        "fast_disk": str(raw_filename.parent.resolve()),
        "do_registration": 1,
        "two_step_registration": False,
        "keep_movie_raw": False,
        "nplanes": 1,
        "nchannels": 1,
        "functional_chan": 1,
        "tau": 1.5,
        "look_one_level_down": False,
        "baseline": "maximin",
        "win_baseline": 60.0,
        "sig_baseline": 10.0,
        "prctile_baseline": 8.0,
        "neucoeff": 0.7,
        "xrange": [0, 0],
        "yrange": [0, 0],
    }

    # Add any additional metadata from kwargs
    for key, value in kwargs.items():
        ops[key] = value

    # Add additional metadata that may be relevant
    for key in [
        "plane",
        "num_frames",
        "nframes",
        "apply_shift",
        "register_z",
        "s3d-job",
    ]:
        if key in metadata:
            ops[key] = metadata[key]

    np.save(ops_file, ops)
    logger.debug(f"Wrote ops file to {ops_file}")


def get_files(path: Path | str, pattern: str = "*") -> list:
    """
    Get all files matching a pattern in a directory.

    Parameters
    ----------
    path : Path or str
        Directory to search.
    pattern : str, optional
        Glob pattern to match files (default: "*").

    Returns
    -------
    list
        List of Path objects for matching files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    # Handle extensions
    if not pattern.startswith("*"):
        pattern = f"*{pattern}"

    files = list(path.glob(pattern))
    return sorted([f for f in files if f.is_file()])


def find_in_dir(path: str | Path, pattern: str, recursive: bool = False):
    """
    Find files or directories matching a pattern.

    Parameters
    ----------
    path : str or Path
        Directory to search.
    pattern : str
        Glob pattern to match.
    recursive : bool, optional
        If True, search recursively (default: False).

    Returns
    -------
    list
        List of Path objects matching the pattern.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if recursive:
        matches = path.rglob(pattern)
    else:
        matches = path.glob(pattern)

    return sorted(list(matches))


def print_tree(path: Path | str, max_depth: int = 3, prefix: str = ""):
    """
    Print a directory tree structure.

    Parameters
    ----------
    path : Path or str
        Root directory to display.
    max_depth : int, optional
        Maximum depth to recurse (default: 3).
    prefix : str, optional
        Prefix for tree formatting (default: "").
    """
    path = Path(path)
    if not path.exists():
        print(f"Path does not exist: {path}")
        return

    if prefix == "":
        print(f"{path}/")

    if max_depth <= 0:
        return

    entries = sorted([p for p in path.iterdir() if p.is_dir()])
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name + "/")

        if max_depth > 1:
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, max_depth=max_depth - 1, prefix=prefix + extension)


def merge_zarr_zplanes(
    zarr_paths: list[str | Path],
    output_path: str | Path,
    *,
    suite2p_dirs: list[str | Path] | None = None,
    metadata: dict | None = None,
    overwrite: bool = True,
    compression_level: int = 1,
) -> Path:
    """
    Merge multiple single z-plane Zarr files into a single OME-Zarr volume.

    Creates an OME-NGFF v0.5 compliant Zarr store with shape (T, Z, Y, X) by
    stacking individual z-plane Zarr files. Optionally includes Suite2p segmentation
    masks as OME-Zarr labels.

    Parameters
    ----------
    zarr_paths : list of str or Path
        List of paths to single-plane Zarr stores. Should be ordered by z-plane.
        Each Zarr should have shape (T, Y, X).
    output_path : str or Path
        Path for the output merged Zarr store.
    suite2p_dirs : list of str or Path, optional
        List of Suite2p output directories corresponding to each z-plane.
        If provided, ROI masks will be added as OME-Zarr labels.
        Must match length of zarr_paths.
    metadata : dict, optional
        Additional metadata to include in the OME-Zarr attributes.
        Keys like 'pixel_resolution', 'frame_rate', 'dz' will be used
        for coordinate transformations.
    overwrite : bool, default=True
        If True, overwrite existing output Zarr store.
    compression_level : int, default=1
        Gzip compression level (0-9). Higher = better compression, slower.

    Returns
    -------
    Path
        Path to the created OME-Zarr store.

    Raises
    ------
    ValueError
        If zarr_paths is empty or shapes are incompatible.
    FileNotFoundError
        If any input Zarr or Suite2p directory doesn't exist.

    Examples
    --------
    Merge z-plane Zarr files into a volume:

    >>> zarr_files = [
    ...     "session1/plane01.zarr",
    ...     "session1/plane02.zarr",
    ...     "session1/plane03.zarr",
    ... ]
    >>> merge_zarr_zplanes(zarr_files, "session1/volume.zarr")

    Include Suite2p segmentation masks:

    >>> s2p_dirs = [
    ...     "session1/plane01_suite2p",
    ...     "session1/plane02_suite2p",
    ...     "session1/plane03_suite2p",
    ... ]
    >>> merge_zarr_zplanes(
    ...     zarr_files,
    ...     "session1/volume.zarr",
    ...     suite2p_dirs=s2p_dirs,
    ...     metadata={"pixel_resolution": (0.5, 0.5), "frame_rate": 30.0, "dz": 5.0}
    ... )

    See Also
    --------
    imwrite : Write imaging data to various formats including OME-Zarr
    """
    if not HAS_ZARR:
        raise ImportError("zarr package required. Install with: pip install zarr")

    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    zarr_paths = [Path(p) for p in zarr_paths]
    output_path = Path(output_path)

    if not zarr_paths:
        raise ValueError("zarr_paths cannot be empty")

    # Validate all input Zarrs exist
    for zp in zarr_paths:
        if not zp.exists():
            raise FileNotFoundError(f"Zarr store not found: {zp}")

    # Validate suite2p_dirs if provided
    if suite2p_dirs is not None:
        suite2p_dirs = [Path(p) for p in suite2p_dirs]
        if len(suite2p_dirs) != len(zarr_paths):
            raise ValueError(
                f"suite2p_dirs length ({len(suite2p_dirs)}) must match "
                f"zarr_paths length ({len(zarr_paths)})"
            )
        for s2p_dir in suite2p_dirs:
            if not s2p_dir.exists():
                raise FileNotFoundError(f"Suite2p directory not found: {s2p_dir}")

    # Read first Zarr to get dimensions
    z0 = zarr.open(str(zarr_paths[0]), mode="r")
    if hasattr(z0, "shape"):
        # Direct array
        T, Y, X = z0.shape
        dtype = z0.dtype
    else:
        # Group - look for "0" array (OME-Zarr)
        if "0" in z0:
            arr = z0["0"]
            T, Y, X = arr.shape
            dtype = arr.dtype
        else:
            raise ValueError(
                f"Cannot determine shape of {zarr_paths[0]}. "
                f"Expected direct array or group with '0' subarray."
            )

    Z = len(zarr_paths)
    logger.info(f"Creating merged Zarr volume with shape (T={T}, Z={Z}, Y={Y}, X={X})")

    # Remove existing if overwrite
    if output_path.exists() and overwrite:
        import shutil

        shutil.rmtree(output_path)

    # Create OME-Zarr group (Zarr v3)
    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)

    # Create main image array
    image_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    image = zarr.create(
        store=root.store,
        path="0",
        shape=(T, Z, Y, X),
        chunks=(1, 1, Y, X),  # Chunk by frame and z-plane
        dtype=dtype,
        codecs=image_codecs,
        overwrite=True,
    )

    # Copy data from each z-plane
    logger.info("Copying z-plane data...")
    for zi, zpath in enumerate(zarr_paths):
        z_arr = zarr.open(str(zpath), mode="r")

        # Handle both direct arrays and OME-Zarr groups
        if hasattr(z_arr, "shape"):
            plane_data = z_arr[:]
        elif "0" in z_arr:
            plane_data = z_arr["0"][:]
        else:
            raise ValueError(f"Cannot read data from {zpath}")

        if plane_data.shape != (T, Y, X):
            raise ValueError(
                f"Shape mismatch at z={zi}: expected {(T, Y, X)}, got {plane_data.shape}"
            )

        image[:, zi, :, :] = plane_data
        logger.debug(f"Copied z-plane {zi + 1}/{Z} from {zpath.name}")

    # Add Suite2p labels if provided
    if suite2p_dirs is not None:
        logger.info("Adding Suite2p segmentation masks as labels...")
        _add_suite2p_labels(
            root, suite2p_dirs, T, Z, Y, X, dtype, compression_level
        )

    # Build OME metadata
    metadata = metadata or {}
    pixel_resolution = metadata.get("pixel_resolution", [1.0, 1.0])
    frame_rate = metadata.get("frame_rate", metadata.get("fs", 1.0))
    dz = metadata.get("dz", metadata.get("z_step", 1.0))

    if isinstance(pixel_resolution, (list, tuple)) and len(pixel_resolution) >= 2:
        pixel_x, pixel_y = pixel_resolution[0], pixel_resolution[1]
    else:
        pixel_x = pixel_y = 1.0

    time_scale = 1.0 / float(frame_rate) if frame_rate else 1.0

    # Build OME-NGFF v0.5 metadata
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    scale_values = [time_scale, float(dz), float(pixel_y), float(pixel_x)]

    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": scale_values}],
        }
    ]

    multiscales = [
        {
            "version": "0.5",
            "name": metadata.get("name", "merged_volume"),
            "axes": axes,
            "datasets": datasets,
        }
    ]

    # Set OME metadata on the group
    root.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}

    # Add optional metadata
    for key, value in metadata.items():
        if key not in [
            "pixel_resolution",
            "frame_rate",
            "fs",
            "dz",
            "z_step",
            "name",
        ]:
            try:
                import json

                json.dumps(value)
                root.attrs[key] = value
            except (TypeError, ValueError):
                pass

    logger.info(f"Successfully created merged OME-Zarr at {output_path}")
    return output_path


def _add_suite2p_labels(
    root_group,
    suite2p_dirs: list[Path],
    T: int,
    Z: int,
    Y: int,
    X: int,
    dtype,
    compression_level: int,
):
    """
    Add Suite2p segmentation masks as OME-Zarr labels.

    Creates a 'labels' subgroup with ROI masks from Suite2p stat.npy files.
    Follows OME-NGFF v0.5 labels specification.

    Parameters
    ----------
    root_group : zarr.Group
        Root Zarr group to add labels to.
    suite2p_dirs : list of Path
        Suite2p output directories for each z-plane.
    T, Z, Y, X : int
        Dimensions of the volume.
    dtype : np.dtype
        Data type for label array.
    compression_level : int
        Gzip compression level.
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    logger.info("Creating labels array from Suite2p masks...")

    # Create labels subgroup
    labels_group = root_group.create_group("labels", overwrite=True)

    # Create ROI masks array (static across time, just Z, Y, X)
    label_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    masks = zarr.create(
        store=labels_group.store,
        path="labels/0",
        shape=(Z, Y, X),
        chunks=(1, Y, X),
        dtype=np.uint32,  # uint32 for up to 4 billion ROIs
        codecs=label_codecs,
        overwrite=True,
    )

    # Process each z-plane
    roi_id = 1  # Start ROI IDs at 1 (0 = background)

    for zi, s2p_dir in enumerate(suite2p_dirs):
        stat_path = s2p_dir / "stat.npy"
        iscell_path = s2p_dir / "iscell.npy"

        if not stat_path.exists():
            logger.warning(f"stat.npy not found in {s2p_dir}, skipping z={zi}")
            continue

        # Load Suite2p data
        stat = np.load(stat_path, allow_pickle=True)

        # Load iscell if available to filter
        if iscell_path.exists():
            iscell = np.load(iscell_path, allow_pickle=True)[:, 0].astype(bool)
        else:
            iscell = np.ones(len(stat), dtype=bool)

        # Create mask for this z-plane
        plane_mask = np.zeros((Y, X), dtype=np.uint32)

        for roi_idx, (roi_stat, is_cell) in enumerate(zip(stat, iscell)):
            if not is_cell:
                continue

            # Get pixel coordinates for this ROI
            ypix = roi_stat.get("ypix", [])
            xpix = roi_stat.get("xpix", [])

            if len(ypix) == 0 or len(xpix) == 0:
                continue

            # Ensure coordinates are within bounds
            ypix = np.clip(ypix, 0, Y - 1)
            xpix = np.clip(xpix, 0, X - 1)

            # Assign unique ROI ID
            plane_mask[ypix, xpix] = roi_id
            roi_id += 1

        # Write to Zarr
        masks[zi, :, :] = plane_mask
        logger.debug(
            f"Added {(plane_mask > 0).sum()} labeled pixels for z-plane {zi + 1}/{Z}"
        )

    # Add OME-NGFF labels metadata
    labels_metadata = {
        "version": "0.5",
        "labels": ["0"],  # Path to the label array
    }

    # Add metadata for label array
    label_array_meta = {
        "version": "0.5",
        "image-label": {
            "version": "0.5",
            "colors": [],  # Can add color LUT here if desired
            "source": {"image": "../../0"},  # Reference to main image
        },
    }

    labels_group.attrs.update(labels_metadata)
    labels_group["0"].attrs.update(label_array_meta)

    logger.info(f"Added {roi_id - 1} total ROIs across {Z} z-planes")
