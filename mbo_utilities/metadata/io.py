"""
metadata file I/O - reading/writing metadata from TIFF files.

functions for extracting metadata from ScanImage TIFFs and other file formats.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from tqdm.auto import tqdm

import numpy as np
import tifffile
from mbo_utilities import log
from mbo_utilities.file_io import get_files
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities.util import load_npy

# import from sibling modules
from .params import normalize_resolution

__all__ = [
    # file I/O functions
    "has_mbo_metadata",
    "is_raw_scanimage",
    "get_metadata",
    "get_metadata_single",
    "get_metadata_batch",
    "query_tiff_pages",
    "clean_scanimage_metadata",
    "default_ops",
    "_build_ome_metadata",
]

logger = log.get("metadata")


def has_mbo_metadata(file: os.PathLike | str) -> bool:
    """
    Check if a TIFF file has metadata from the Miller Brain Observatory.

    This checks for MBO-specific keys in the shaped_metadata, not just its presence.
    Generic TIFFs written by tifffile may have shaped_metadata with only a 'shape' key,
    which is not sufficient to identify them as MBO files.

    MBO-specific keys include: num_planes, num_rois, frame_rate, pixel_resolution,
    fov, Ly, Lx, etc.

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    bool
        True if the TIFF file has MBO metadata; False otherwise.
    """
    # Keys that indicate MBO origin (beyond just 'shape' which is generic tifffile)
    MBO_MARKER_KEYS = {"num_planes", "num_rois", "frame_rate", "Ly", "Lx", "fov", "pixel_resolution"}

    if not file or not isinstance(file, (str, os.PathLike)):
        raise ValueError(
            "Invalid file path provided: must be a string or os.PathLike object."
            f"Got: {file} of type {type(file)}"
        )
    if Path(file).suffix not in [".tif", ".tiff"]:
        return False
    try:
        with tifffile.TiffFile(file) as tiff_file:
            if not hasattr(tiff_file, "shaped_metadata") or tiff_file.shaped_metadata is None:
                return False
            # shaped_metadata is a tuple of dicts, check the first one
            meta = tiff_file.shaped_metadata[0] if tiff_file.shaped_metadata else {}
            if not isinstance(meta, dict):
                return False
            # Check if any MBO-specific keys are present
            return bool(MBO_MARKER_KEYS & meta.keys())
    except Exception:
        return False


def is_raw_scanimage(file: os.PathLike | str) -> bool:
    """
    Check if a TIFF file is a raw ScanImage TIFF.

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    bool
        True if the TIFF file is a raw ScanImage TIFF; False otherwise.
    """
    if not file or not isinstance(file, (str, os.PathLike)):
        return False
    elif Path(file).suffix not in [".tif", ".tiff"]:
        return False
    try:
        tiff_file = tifffile.TiffFile(file)
        if (
            # TiffFile.shaped_metadata is where we store metadata for processed tifs
            # if this is not empty, we have a processed file
            # otherwise, we have a raw scanimage tiff
            hasattr(tiff_file, "shaped_metadata")
            and tiff_file.shaped_metadata is not None
            and isinstance(tiff_file.shaped_metadata, (list, tuple))
        ):
            logger.info(f"File {file} has shaped_metadata; not a raw ScanImage TIFF.")
            return False
        else:
            if tiff_file.scanimage_metadata is None:
                logger.info(f"No ScanImage metadata found in {file}.")
                return False
            return True
    except Exception:
        return False


def get_metadata(
    file,
    dx: float | None = None,
    dy: float | None = None,
    dz: float | None = None,
    verbose: bool = False,
    # Backward compatibility alias
    z_step: float | None = None,
):
    """
    Extract metadata from a TIFF file or directory of TIFF files produced by ScanImage.

    This function handles single files, lists of files, or directories containing TIFF files.
    When given a directory, it automatically finds and processes all TIFF files in natural
    sort order. For multiple files, it calculates frames per file accounting for z-planes.

    Parameters
    ----------
    file : os.PathLike, str, or list
        - Single file path: processes that file
        - Directory path: processes all TIFF files in the directory
        - List of file paths: processes all files in the list
    dx : float, optional
        X pixel resolution in micrometers. Overrides extracted value.
    dy : float, optional
        Y pixel resolution in micrometers. Overrides extracted value.
    dz : float, optional
        Z step size in micrometers. Overrides extracted value.
        Also available as ``z_step`` for backward compatibility.
    verbose : bool, optional
        If True, returns extended metadata including all ScanImage attributes. Default is False.
    z_step : float, optional
        Alias for ``dz`` (backward compatibility).

    Returns
    -------
    dict
        A dictionary containing extracted metadata with normalized resolution aliases:
        - dx, dy, dz: canonical resolution values in micrometers
        - pixel_resolution: (dx, dy) tuple
        - voxel_size: (dx, dy, dz) tuple
        - umPerPixX, umPerPixY, umPerPixZ: legacy format
        - PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ: OME format

        For multiple files, also includes:
        - 'frames_per_file': list of frame counts per file (accounting for z-planes)
        - 'total_frames': total frames across all files
        - 'file_paths': list of processed file paths
        - 'tiff_pages_per_file': raw TIFF page counts per file

    Raises
    ------
    ValueError
        If no recognizable metadata is found or no TIFF files found in directory.

    Examples
    --------
    >>> # Single file with z-resolution
    >>> meta = get_metadata("path/to/rawscan_00001.tif", dz=5.0)
    >>> print(f"Voxel size: {meta['voxel_size']}")

    >>> # Directory of files
    >>> meta = get_metadata("path/to/scan_directory/")
    >>> print(f"Files processed: {len(meta['file_paths'])}")
    >>> print(f"Frames per file: {meta['frames_per_file']}")

    >>> # List of specific files
    >>> files = ["scan_00001.tif", "scan_00002.tif", "scan_00003.tif"]
    >>> meta = get_metadata(files, dz=5.0)
    """
    # Handle z_step alias for backward compatibility
    if dz is None and z_step is not None:
        dz = z_step

    # Convert input to Path object and handle different input types
    if hasattr(file, "metadata"):
        metadata = dict(file.metadata) if file.metadata else {}
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    if isinstance(file, (list, tuple)):
        # make sure all values in the list are strings or paths
        if not all(isinstance(f, (str, os.PathLike)) for f in file):
            raise ValueError(
                "All items in the list must be of type str or os.PathLike."
                f"Got: {file} of type {type(file)}"
            )
        file_paths = [Path(f) for f in file]
        metadata = get_metadata_batch(file_paths)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    file_path = Path(file)

    if file_path.is_dir():
        # check for .zarr , get_files doesn't work on nested zarr files
        if file_path.suffix in [".zarr"]:
            metadata = get_metadata_single(file_path)
            return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)
        tiff_files = get_files(file_path, "tif", sort_ascending=True)
        if not tiff_files:
            raise ValueError(f"No TIFF files found in directory: {file_path}")
        metadata = get_metadata_batch(tiff_files)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    elif file_path.is_file():
        metadata = get_metadata_single(file_path)
        return normalize_resolution(metadata, dx=dx, dy=dy, dz=dz)

    else:
        raise ValueError(f"Path does not exist or is not accessible: {file_path}")


def get_metadata_single(file: Path):
    """
    Extract metadata from a single TIFF file produced by ScanImage or processed via the save_as function.

    Parameters
    ----------
    file : Path
        The full path to the TIFF file from which metadata is to be extracted.
    verbose : bool, optional
        If True, returns an extended metadata dictionary that includes all available ScanImage attributes.
        Default is False.
    z_step : float, optional
        The z-step size in microns. If provided, it will be included in the returned metadata.

    Returns
    -------
    dict
        A dictionary containing the extracted metadata (e.g., number of planes, frame rate, field-of-view,
        pixel resolution).

    Raises
    ------
    ValueError
        If no recognizable metadata is found in the TIFF file (e.g., the file is not a valid ScanImage TIFF).

    Notes
    -----
    - num_frames represents the number of frames per z-plane

    Examples
    --------
    >>> mdata = get_metadata("path/to/rawscan_00001.tif")
    >>> print(mdata["num_frames"])
    5345
    >>> mdata = get_metadata("path/to/assembled_data.tif")
    >>> print(mdata["shape"])
    (14, 5345, 477, 477)
    >>> meta_verbose = get_metadata("path/to/scanimage_file.tif", verbose=True)
    >>> print(meta_verbose["all"])
    {... Includes all ScanImage FrameData ...}
    """
    if file.suffix in [".zarr", ".h5"]:
        from mbo_utilities import imread

        file = imread(file)
        return file.metadata

    tiff_file = tifffile.TiffFile(file)
    if not is_raw_scanimage(file):
        if (
            not hasattr(tiff_file, "shaped_metadata")
            or tiff_file.shaped_metadata is None
        ):
            raise ValueError(f"No metadata found in {file}.")
        return tiff_file.shaped_metadata[0]

    elif hasattr(tiff_file, "scanimage_metadata"):
        meta = tiff_file.scanimage_metadata
        # if no ScanImage metadata at all → fallback immediately
        if not meta:
            logger.info(f"{file} has no scanimage_metadata, trying ops.npy fallback.")
            for parent in Path(file).parents:
                ops_path = parent / "ops.npy"
                if ops_path.exists():
                    try:
                        ops = load_npy(ops_path).item()
                        return {
                            "num_planes": int(ops.get("nplanes", 1) or 1),
                            "fov": (
                                int(ops.get("Lx", 0) or 0),
                                int(ops.get("Ly", 0) or 0),
                            ),
                            "frame_rate": float(ops.get("fs", 0) or 0),
                            "zoom_factor": ops.get("zoom"),
                            "pixel_resolution": (
                                float(ops.get("umPerPixX", 1) or 1),
                                float(ops.get("umPerPixY", 1) or 1),
                            ),
                            "dtype": "int16",
                            "source": "ops_fallback",
                        }
                    except Exception as e:
                        logger.warning(f"Failed ops.npy fallback for {file}: {e}")
            raise ValueError(f"No metadata found in {file}.")

        si = meta.get("FrameData", {})
        if not si:
            logger.warning(f"No FrameData found in {file}.")
            return None

        pages = tiff_file.pages
        first_page = pages[0]
        shape = first_page.shape

        # Extract ROI and imaging metadata
        roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]
        if isinstance(roi_group, dict):
            num_rois = 1
            roi_group = [roi_group]
        else:
            num_rois = len(roi_group)

        # Handle single channel case where channelSave is int instead of list
        channel_save = si["SI.hChannels.channelSave"]
        if isinstance(channel_save, (int, float)):
            num_planes = 1
        else:
            num_planes = len(channel_save)

        zoom_factor = si["SI.hRoiManager.scanZoomFactor"]
        uniform_sampling = si.get("SI.hScan2D.uniformSampling", "NA")
        objective_resolution = si["SI.objectiveResolution"]
        frame_rate = si["SI.hRoiManager.scanFrameRate"]

        fly_to_time = float(si["SI.hScan2D.flytoTimePerScanfield"])
        line_period = float(si["SI.hRoiManager.linePeriod"])
        num_fly_to_lines = int(round(fly_to_time / line_period))

        sizes = []
        num_pixel_xys = []
        for roi in roi_group:
            scanfields = roi["scanfields"]
            if isinstance(scanfields, list):
                scanfields = scanfields[0]
            sizes.append(scanfields["sizeXY"])
            num_pixel_xys.append(scanfields["pixelResolutionXY"])

        size_xy = sizes[0]
        num_pixel_xy = num_pixel_xys[0]

        fov_x_um = round(objective_resolution * size_xy[0])
        fov_y_um = round(objective_resolution * size_xy[1])
        pixel_resolution = (fov_x_um / num_pixel_xy[0], fov_y_um / num_pixel_xy[1])

        # roi is per-strip dimensions (width, height)
        # fov is total field of view in pixels (num_rois * width, height)
        # fov_um is total field of view in micrometers
        roi_width, roi_height = num_pixel_xy[0], num_pixel_xy[1]

        metadata = {
            "num_planes": num_planes,
            "num_rois": num_rois,
            "roi": (roi_width, roi_height),
            "fov": (num_rois * roi_width, roi_height),
            "fov_um": (fov_x_um, fov_y_um),
            "frame_rate": frame_rate,
            "pixel_resolution": np.round(pixel_resolution, 2),
            "ndim": len(shape),
            "dtype": "int16",
            "size": np.prod(shape),
            "page_height": shape[0],
            "page_width": shape[1],
            "objective_resolution": objective_resolution,
            "zoom_factor": zoom_factor,
            "uniform_sampling": uniform_sampling,
            "num_fly_to_lines": num_fly_to_lines,
            "roi_heights": [px[1] for px in num_pixel_xys],
            "roi_groups": _make_json_serializable(roi_group),
            "si": _make_json_serializable(si),
        }
        return clean_scanimage_metadata(metadata)

    else:
        logger.info(f"No ScanImage metadata found in {file}, trying ops.npy fallback.")
        # fallback: no ScanImage metadata, try nearby ops.npy
        ops_path = Path(file).with_name("ops.npy")
        if not ops_path.exists():
            # climb until you find suite2p or root
            for parent in Path(file).parents:
                ops_path = parent / "ops.npy"
                if ops_path.exists():
                    try:
                        ops = load_npy(ops_path).item()
                        num_planes = int(ops.get("nplanes") or 1)
                        # single-plane suite2p folder → force to 1
                        if "plane0" in str(ops_path.parent).lower():
                            num_planes = 1
                        return {
                            "num_planes": num_planes,
                            "fov": (
                                int(ops.get("Lx") or 0),
                                int(ops.get("Ly") or 0),
                            ),
                            "frame_rate": float(ops.get("fs") or 0),
                            "zoom_factor": ops.get("zoom"),
                            "pixel_resolution": (
                                float(ops.get("umPerPixX") or 1.0),
                                float(ops.get("umPerPixY") or 1.0),
                            ),
                            "dtype": "int16",
                            "source": "ops_fallback",
                        }
                    except Exception as e:
                        logger.warning(f"Failed ops.npy fallback for {file}: {e}")
        if ops_path.exists():
            logger.info(f"Found ops.npy at {ops_path}, attempting to load.")
            try:
                ops = load_npy(ops_path).item()
                return {
                    "num_planes": ops.get("nplanes", 1),
                    "fov": (ops.get("Lx"), ops.get("Ly")),
                    "frame_rate": ops.get("fs"),
                    "zoom_factor": ops.get("zoom", None),
                    "pixel_resolution": (ops.get("umPerPixX"), ops.get("umPerPixY")),
                    "dtype": "int16",
                    "source": "ops_fallback",
                }
            except Exception as e:
                logger.warning(f"Failed ops.npy fallback for {file}: {e}")
        raise ValueError(f"No metadata found in {file}.")


def get_metadata_batch(file_paths: list | tuple):
    """
    Extract and aggregate metadata from a list of TIFF files.

    Parameters
    ----------
    file_paths : list of Path
        List of TIFF file paths.

    Returns
    -------
    dict
        Aggregated metadata with per-file frame information.
    """
    if not file_paths:
        raise ValueError("No files provided")

    # Get metadata from first file only
    metadata = get_metadata_single(file_paths[0])
    n_planes = metadata.get("num_planes", 1)

    # Count frames for all files
    frames_per_file = [
        query_tiff_pages(fp) // n_planes
        for fp in tqdm(file_paths, desc="Counting frames")
    ]

    total_frames = sum(frames_per_file)
    return metadata | {
        "nframes": total_frames,
        "num_frames": total_frames,  # alias for backwards compatibility
        "frames_per_file": frames_per_file,
        "file_paths": [str(fp) for fp in file_paths],
        "num_files": len(file_paths),
    }


def query_tiff_pages(file_path):
    """
    Get page count for TIFF files for both 64-bit (BigTiff) and 32-bit TIFFS.

    Frames per tiff must be uniform.

    1. Reading first TWO IFD offsets only
    2. Calculating page_size = second_offset - first_offset
    3. Estimating: total_pages = file_size / page_size


    Parameters
    ----------
    file_path : str
        Path to ScanImage TIFF file

    Returns
    -------
    int
        Number of pages
    """
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        # Read header (8 bytes)
        header = f.read(8)

        # Detect byte order
        if header[:2] == b"II":
            bo = "<"  # Little-endian
        elif header[:2] == b"MM":
            bo = ">"  # Big-endian
        else:
            raise ValueError("Not a TIFF file")

        # Detect TIFF version
        version = struct.unpack(f"{bo}H", header[2:4])[0]

        if version == 42:
            # Classic TIFF (32-bit offsets)
            offset_fmt = f"{bo}I"
            offset_size = 4
            tag_count_fmt = f"{bo}H"
            tag_count_size = 2
            tag_size = 12
            first_ifd_offset = struct.unpack(offset_fmt, header[4:8])[0]
            header_size = 8

        elif version == 43:
            # BigTIFF (64-bit offsets)
            offset_fmt = f"{bo}Q"
            offset_size = 8
            tag_count_fmt = f"{bo}Q"
            tag_count_size = 8
            tag_size = 20
            f.seek(8)
            first_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]
            header_size = 16

        else:
            raise ValueError(f"Unknown TIFF version: {version}")

        # Go to first IFD
        f.seek(first_ifd_offset)

        # Read tag count
        tag_count = struct.unpack(tag_count_fmt, f.read(tag_count_size))[0]

        # Skip all tags to get to next IFD offset
        f.seek(first_ifd_offset + tag_count_size + (tag_count * tag_size))

        # Read second IFD offset
        second_ifd_offset = struct.unpack(offset_fmt, f.read(offset_size))[0]

        if second_ifd_offset == 0:
            return 1  # Only one page

        # Calculate page size (IFD + image data for one page)
        page_size = second_ifd_offset - first_ifd_offset

        # Calculate total pages
        data_size = file_size - header_size
        num_pages = data_size // page_size

        return int(num_pages)


def clean_scanimage_metadata(meta: dict) -> dict:
    """
    Build a JSON-serializable, nicely nested dict from ScanImage metadata.

    - All non-'si' top-level keys are kept after cleaning
    - All 'SI.*' keys (from anywhere) are nested under 'si' with 'SI.' prefix stripped
    - So SI.hChannels.channelSave -> metadata['si']['hChannels']['channelSave']
    """

    def _clean(x):
        if x is None:
            return None
        if isinstance(x, (np.generic,)):
            x = x.item()
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            x = x.tolist()
        if isinstance(x, float):
            if not np.isfinite(x):
                return None
            return x
        if isinstance(x, (int, bool)):
            return x
        if isinstance(x, str):
            s = x.strip()
            return s if s != "" else None
        if isinstance(x, (list, tuple)):
            out = []
            for v in x:
                cv = _clean(v)
                if cv is not None:
                    out.append(cv)
            return out if out else None
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                cv = _clean(v)
                if cv is not None:
                    out[str(k)] = cv
            return out if out else None
        try:
            json.dumps(x)
            return x
        except Exception:
            return None

    def _prune(d):
        if not isinstance(d, dict):
            return d
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                pv = _prune(v)
                if pv and len(pv) > 0:
                    d[k] = pv
                else:
                    d.pop(k, None)
            elif v in (None, [], ""):
                d.pop(k, None)
        return d

    def _collect_SI_keys(node):
        """Collect all 'SI.*' keys anywhere in the tree."""
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("SI."):
                    out.append((k, v))
                # Recurse into nested dicts/lists
                out.extend(_collect_SI_keys(v))
        elif isinstance(node, (list, tuple)):
            for v in node:
                out.extend(_collect_SI_keys(v))
        return out

    def _nest_into_si(root_dict, dotted_key, value):
        """Nest 'SI.hChannels.channelSave' into root_dict as ['hChannels']['channelSave']"""
        # Strip 'SI.' prefix
        if dotted_key.startswith("SI."):
            dotted_key = dotted_key[3:]

        parts = dotted_key.split(".")
        cur = root_dict
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        leaf = parts[-1]
        cur[leaf] = _clean(value)

    # 1) Copy all top-level keys EXCEPT 'si'
    result = {}
    for k, v in meta.items():
        if k == "si":
            continue
        cv = _clean(v)
        if cv is not None:
            result[k] = cv

    # 2) Initialize 'si' dict
    result["si"] = {}

    # 3) Add non-SI.* keys from meta['si'] (like RoiGroups, etc.)
    if isinstance(meta.get("si"), dict):
        for k, v in meta["si"].items():
            if not (isinstance(k, str) and k.startswith("SI.")):
                cv = _clean(v)
                if cv is not None:
                    result["si"][k] = cv

    # 4) Collect ALL 'SI.*' keys from entire meta tree and nest under result['si']
    si_pairs = _collect_SI_keys(meta)
    for dotted_key, val in si_pairs:
        _nest_into_si(result["si"], dotted_key, val)

    result = _prune(result)

    # 5) Add derived ScanImage stack detection fields
    from .scanimage import (
        detect_stack_type,
        get_num_color_channels,
        get_num_zplanes,
        get_z_step_size,
        get_frame_rate,
        get_roi_info,
    )

    if result.get("si"):
        stack_type = detect_stack_type(result)
        result["stack_type"] = stack_type
        result["lbm_stack"] = stack_type == "lbm"
        result["piezo_stack"] = stack_type == "piezo"
        result["num_color_channels"] = get_num_color_channels(result)

        # set num_zplanes from stack detection if not already present
        if "num_zplanes" not in result and "num_planes" not in result:
            result["num_zplanes"] = get_num_zplanes(result)

        # add dz if available from ScanImage (but NOT for LBM - user must supply)
        if stack_type != "lbm":
            dz = get_z_step_size(result)
            if dz is not None:
                result["dz"] = dz

        # add frame rate if available
        fs = get_frame_rate(result)
        if fs is not None:
            result["fs"] = fs

        # add ROI/FOV info (don't overwrite existing values)
        roi_info = get_roi_info(result)
        for k, v in roi_info.items():
            if v is not None and k not in result:
                result[k] = v

    # add all standard aliases for backward compatibility
    from .params import normalize_metadata
    normalize_metadata(result)

    return result


def default_ops():
    """default options to run pipeline"""
    return {
        # file input/output settings
        "look_one_level_down": False,  # whether to look in all subfolders when searching for tiffs
        "fast_disk": [],  # used to store temporary binary file, defaults to save_path0
        "delete_bin": False,  # whether to delete binary file after processing
        "mesoscan": False,  # for reading in scanimage mesoscope files
        "bruker": False,  # whether or not single page BRUKER tiffs!
        "bruker_bidirectional": False,  # bidirectional multiplane in bruker: 0, 1, 2, 2, 1, 0 (True) vs 0, 1, 2, 0, 1, 2 (False)
        "h5py": [],  # take h5py as input (deactivates data_path)
        "h5py_key": "data",  # key in h5py where data array is stored
        "nwb_file": "",  # take nwb file as input (deactivates data_path)
        "nwb_driver": "",  # driver for nwb file (nothing if file is local)
        "nwb_series": "",  # TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file
        "save_path0": "",  # pathname where you'd like to store results, defaults to first item in data_path
        "save_folder": [],  # directory you"d like suite2p results to be saved to
        "subfolders": [],  # subfolders you"d like to search through when look_one_level_down is set to True
        "move_bin": False,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk
        # main settings
        "nplanes": 1,  # each tiff has these many planes in sequence
        "nchannels": 1,  # each tiff has these many channels per plane
        "functional_chan": 1,  # this channel is used to extract functional ROIs (1-based)
        "tau": 1.3,  # this is the main parameter for deconvolution
        "fs": 10.0,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        "force_sktiff": False,  # whether or not to use scikit-image for tiff reading
        "frames_include": -1,
        "multiplane_parallel": False,  # whether or not to run on server
        "ignore_flyback": [],
        # output settings
        "preclassify": 0.0,  # apply classifier before signal extraction with probability 0.3
        "save_mat": False,  # whether to save output as matlab files
        "save_NWB": False,  # whether to save output as NWB file
        "combined": True,  # combine multiple planes into a single result /single canvas for GUI
        "aspect": 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # bidirectional phase offset
        "do_bidiphase": False,  # whether or not to compute bidirectional phase offset (applies to 2P recordings only)
        "bidiphase": 0,  # Bidirectional Phase offset from line scanning (set by user). Applied to all frames in recording.
        "bidi_corrected": False,  # Whether to do bidirectional correction during registration
        # registration settings
        "do_registration": True,  # whether to register data (2 forces re-registration)
        "two_step_registration": False,  # whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True.
        "keep_movie_raw": False,  # whether to keep binary file of non-registered frames.
        "nimg_init": 300,  # subsampled frames for finding reference image
        "batch_size": 500,  # number of frames per batch
        "maxregshift": 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        "align_by_chan": 1,  # when multi-channel, you can align by non-functional channel (1-based)
        "reg_tif": False,  # whether to save registered tiffs
        "reg_tif_chan2": False,  # whether to save channel 2 registered tiffs
        "subpixel": 10,  # precision of subpixel registration (1/subpixel steps)
        "smooth_sigma_time": 0,  # gaussian smoothing in time
        "smooth_sigma": 1.15,  # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
        "th_badframes": 1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        "norm_frames": True,  # normalize frames when detecting shifts
        "force_refImg": False,  # if True, use refImg stored in ops if available
        "pad_fft": False,  # if True, pads image during FFT part of registration
        # non rigid registration settings
        "nonrigid": True,  # whether to use nonrigid registration
        "block_size": [
            128,
            128,
        ],  # block size to register (** keep this a multiple of 2 **)
        "snr_thresh": 1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        "maxregshiftNR": 5,  # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        "1Preg": False,  # whether to perform high-pass filtering and tapering
        "spatial_hp_reg": 42,  # window for spatial high-pass filtering before registration
        "pre_smooth": 0,  # whether to smooth before high-pass filtering before registration
        "spatial_taper": 40,  # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops["smooth_sigma"])
        # cell detection settings with suite2p
        "roidetect": True,  # whether or not to run ROI extraction
        "spikedetect": True,  # whether or not to run spike deconvolution
        "sparse_mode": True,  # whether or not to run sparse_mode
        "spatial_scale": 0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        "connected": True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        "nbinned": 5000,  # max number of binned frames for cell detection
        "max_iterations": 20,  # maximum number of iterations to do cell detection
        "threshold_scaling": 1.0,  # adjust the automatically determined threshold by this scalar multiplier
        "max_overlap": 0.75,  # cells with more overlap than this get removed during triage, before refinement
        "high_pass": 100,  # running mean subtraction across bins with a window of size "high_pass" (use low values for 1P)
        "spatial_hp_detect": 25,  # window for spatial high-pass filtering for neuropil subtraction before detection
        "denoise": False,  # denoise binned movie for cell detection in sparse_mode
        # cell detection settings with cellpose (used if anatomical_only > 0)
        "anatomical_only": 0,  # run cellpose to get masks on 1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj
        "diameter": 0,  # use diameter for cellpose, if 0 estimate diameter
        "cellprob_threshold": 0.0,  # cellprob_threshold for cellpose
        "flow_threshold": 1.5,  # flow_threshold for cellpose
        "spatial_hp_cp": 0,  # high-pass image spatially by a multiple of the diameter
        "pretrained_model": "cyto",  # path to pretrained model or model type string in Cellpose (can be user model)
        # classification parameters
        "soma_crop": True,  # crop dendrites for cell classification stats like compactness
        # ROI extraction parameters
        "neuropil_extract": True,  # whether or not to extract neuropil; if False, Fneu is set to zero
        "inner_neuropil_radius": 2,  # number of pixels to keep between ROI and neuropil donut
        "min_neuropil_pixels": 350,  # minimum number of pixels in the neuropil
        "lam_percentile": 50.0,  # percentile of lambda within area to ignore when excluding cell pixels for neuropil extraction
        "allow_overlap": False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        "use_builtin_classifier": False,  # whether or not to use built-in classifier for cell detection (overrides
        # classifier specified in classifier_path if set to True)
        "classifier_path": "",  # path to classifier
        # channel 2 detection settings (stat[n]["chan2"], stat[n]["not_chan2"])
        "chan2_thres": 0.65,  # minimum for detection of brightness on channel 2
        # deconvolution settings
        "baseline": "maximin",  # baselining mode (can also choose "prctile")
        "win_baseline": 60.0,  # window for maximin
        "sig_baseline": 10.0,  # smoothing constant for gaussian filter
        "prctile_baseline": 8.0,  # optional (whether to use a percentile baseline)
        "neucoeff": 0.7,  # neuropil coefficient
    }


def _params_from_metadata_caiman(metadata):
    """
    Generate parameters for CNMF from metadata.

    Based on the pixel resolution and frame rate, the parameters are set to reasonable values.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary resulting from `lcp.get_metadata()`.

    Returns
    -------
    dict
        Dictionary of parameters for lbm_mc.

    """
    params = _default_params_caiman()

    if metadata is None:
        logger.info("No metadata found. Using default parameters.")
        return params

    params["main"]["fr"] = metadata["frame_rate"]
    params["main"]["dxy"] = metadata["pixel_resolution"]

    # typical neuron ~16 microns
    gSig = round(16 / metadata["pixel_resolution"][0]) / 2
    params["main"]["gSig"] = (int(gSig), int(gSig))

    gSiz = (4 * gSig + 1, 4 * gSig + 1)
    params["main"]["gSiz"] = gSiz

    max_shifts = [int(round(10 / px)) for px in metadata["pixel_resolution"]]
    params["main"]["max_shifts"] = max_shifts

    strides = [int(round(64 / px)) for px in metadata["pixel_resolution"]]
    params["main"]["strides"] = strides

    # overlap should be ~neuron diameter
    overlaps = [int(round(gSig / px)) for px in metadata["pixel_resolution"]]
    if overlaps[0] < gSig:
        logger.info("Overlaps too small. Increasing to neuron diameter.")
        overlaps = [int(gSig)] * 2
    params["main"]["overlaps"] = overlaps

    rf_0 = (strides[0] + overlaps[0]) // 2
    rf_1 = (strides[1] + overlaps[1]) // 2
    rf = int(np.mean([rf_0, rf_1]))

    stride = int(np.mean([overlaps[0], overlaps[1]]))

    params["main"]["rf"] = rf
    params["main"]["stride"] = stride

    return params


def _default_params_caiman():
    """
    Default parameters for both registration and CNMF.
    The exception is gSiz being set relative to gSig.

    Returns
    -------
    dict
        Dictionary of default parameter values for registration and segmentation.

    Notes
    -----
    This will likely change as CaImAn is updated.
    """
    gSig = 6
    gSiz = (4 * gSig + 1, 4 * gSig + 1)
    return {
        "main": {
            # Motion correction parameters
            "pw_rigid": True,
            "max_shifts": [6, 6],
            "strides": [64, 64],
            "overlaps": [8, 8],
            "min_mov": None,
            "gSig_filt": [0, 0],
            "max_deviation_rigid": 3,
            "border_nan": "copy",
            "splits_els": 14,
            "upsample_factor_grid": 4,
            "use_cuda": False,
            "num_frames_split": 50,
            "niter_rig": 1,
            "is3D": False,
            "splits_rig": 14,
            "num_splits_to_process_rig": None,
            # CNMF parameters
            "fr": 10,
            "dxy": (1.0, 1.0),
            "decay_time": 0.4,
            "p": 2,
            "nb": 3,
            "K": 20,
            "rf": 64,
            "stride": [8, 8],
            "gSig": gSig,
            "gSiz": gSiz,
            "method_init": "greedy_roi",
            "rolling_sum": True,
            "use_cnn": False,
            "ssub": 1,
            "tsub": 1,
            "merge_thr": 0.7,
            "bas_nonneg": True,
            "min_SNR": 1.4,
            "rval_thr": 0.8,
        },
        "refit": True,
    }


def _build_ome_metadata(
    shape: tuple,
    dtype,
    metadata: dict,
) -> dict:
    """
    Build comprehensive OME-NGFF v0.5 metadata from ScanImage and other metadata.

    Creates OMERO rendering settings, custom metadata fields, and proper
    coordinate transformations based on available metadata.

    Parameters
    ----------
    shape : tuple
        Shape of the array (T, Z, Y, X)
    dtype : np.dtype
        Data type of the array
    metadata : dict
        Metadata dictionary with optional keys

    Returns
    -------
    dict
        Complete OME-NGFF v0.5 metadata attributes
    """
    T, Z, Y, X = shape

    pixel_resolution = metadata.get("pixel_resolution", [1.0, 1.0])
    frame_rate = metadata.get("frame_rate", metadata.get("fs", 1.0))
    dz = metadata.get("dz", metadata.get("z_step", 1.0))

    if isinstance(pixel_resolution, (list, tuple)) and len(pixel_resolution) >= 2:
        pixel_x, pixel_y = float(pixel_resolution[0]), float(pixel_resolution[1])
    else:
        pixel_x = pixel_y = 1.0

    time_scale = 1.0 / float(frame_rate) if frame_rate else 1.0

    # Build OME-NGFF v0.5 multiscales
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    scale_values = [time_scale, float(dz), pixel_y, pixel_x]

    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": scale_values}],
        }
    ]

    multiscales = [
        {
            "version": "0.5",
            "name": metadata.get("name", "volume"),
            "axes": axes,
            "datasets": datasets,
        }
    ]

    # Build OME content
    ome_content = {
        "version": "0.5",
        "multiscales": multiscales,
    }

    # Add OMERO rendering metadata
    omero_metadata = _build_omero_metadata(
        shape=shape,
        dtype=dtype,
        metadata=metadata,
    )
    if omero_metadata:
        ome_content["omero"] = omero_metadata

    result = {"ome": ome_content}

    # Add custom metadata fields (ScanImage, acquisition info, etc.)
    custom_meta = {}

    # Add ScanImage metadata
    if "si" in metadata:
        si = metadata["si"]
        custom_meta["scanimage"] = {
            "version": f"{si.get('VERSION_MAJOR', 'unknown')}.{si.get('VERSION_MINOR', 0)}",
            "imaging_system": si.get("imagingSystem", "unknown"),
            "objective_resolution": si.get(
                "objectiveResolution", metadata.get("objective_resolution")
            ),
            "scan_mode": si.get("hScan2D", {}).get("scanMode", "unknown"),
        }

        # Add beam/laser info
        if "hBeams" in si:
            custom_meta["scanimage"]["laser_power"] = si["hBeams"].get("powers", 0)
            custom_meta["scanimage"]["power_fraction"] = si["hBeams"].get(
                "powerFractions", 0
            )

        # Add ROI info
        if "hRoiManager" in si:
            roi_mgr = si["hRoiManager"]
            custom_meta["scanimage"]["roi"] = {
                "scan_zoom": roi_mgr.get("scanZoomFactor", metadata.get("zoom_factor")),
                "lines_per_frame": roi_mgr.get("linesPerFrame"),
                "pixels_per_line": roi_mgr.get("pixelsPerLine"),
                "line_period": roi_mgr.get("linePeriod"),
                "bidirectional": si.get("hScan2D", {}).get("bidirectional", True),
            }

    # Add ROI groups information
    if "roi_groups" in metadata:
        custom_meta["roi_groups"] = metadata["roi_groups"]

    # Add acquisition metadata
    acq_meta = {}
    for key in ["acquisition_date", "experimenter", "description", "specimen"]:
        if key in metadata:
            acq_meta[key] = metadata[key]

    if acq_meta:
        custom_meta["acquisition"] = acq_meta

    # Add microscope metadata
    microscope_meta = {}
    for key in [
        "objective",
        "emission_wavelength",
        "excitation_wavelength",
        "numerical_aperture",
    ]:
        if key in metadata:
            microscope_meta[key] = metadata[key]

    if microscope_meta:
        custom_meta["microscope"] = microscope_meta

    # Add processing metadata
    processing_meta = {}
    for key in ["fix_phase", "phasecorr_method", "use_fft", "register_z"]:
        if key in metadata:
            processing_meta[key] = metadata[key]

    if processing_meta:
        custom_meta["processing"] = processing_meta

    # Add file info
    if "file_paths" in metadata or "num_files" in metadata:
        custom_meta["source_files"] = {
            "num_files": metadata.get("num_files"),
            "num_frames": metadata.get("num_frames"),
            "frames_per_file": metadata.get("frames_per_file"),
        }

    # Add all serializable custom metadata
    for key, value in custom_meta.items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            logger.debug(f"Skipping non-serializable metadata key: {key}")

    # Add any other simple metadata fields
    for key, value in metadata.items():
        if (
            key
            not in [
                "pixel_resolution",
                "frame_rate",
                "fs",
                "dz",
                "z_step",
                "name",
                "si",
                "roi_groups",
                "acquisition_date",
                "experimenter",
                "description",
                "specimen",
                "objective",
                "emission_wavelength",
                "excitation_wavelength",
                "numerical_aperture",
                "fix_phase",
                "phasecorr_method",
                "use_fft",
                "register_z",
                "file_paths",
                "num_files",
                "num_frames",
                "frames_per_file",
            ]
            and key not in result
        ):
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                pass

    return result


def _build_omero_metadata(shape: tuple, dtype, metadata: dict) -> dict:
    """
    Build OMERO rendering metadata for OME-NGFF.

    Parameters
    ----------
    shape : tuple
        Shape of the array (T, Z, Y, X)
    dtype : np.dtype
        Data type of the array
    metadata : dict
        Metadata dictionary

    Returns
    -------
    dict
        OMERO metadata or empty dict if not enough info
    """
    import numpy as np

    T, Z, Y, X = shape

    # Determine data range for window settings
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        data_min, data_max = info.min, info.max
    else:
        data_min, data_max = 0.0, 1.0

    # Build channel metadata
    channels = []

    # Get channel names from metadata
    channel_names = metadata.get("channel_names")
    num_channels = metadata.get("num_planes", 1)

    if channel_names is None:
        # Generate default channel names
        if num_channels == 1:
            channel_names = ["Channel 1"]
        else:
            channel_names = [f"Z-plane {i + 1}" for i in range(num_channels)]

    # Default colors (cycle through common microscopy colors)
    default_colors = [
        "00FF00",  # Green
        "FF0000",  # Red
        "0000FF",  # Blue
        "FFFF00",  # Yellow
        "FF00FF",  # Magenta
        "00FFFF",  # Cyan
        "FFFFFF",  # White
    ]

    for i, name in enumerate(channel_names[:num_channels]):
        channel = {
            "active": True,
            "coefficient": 1.0,
            "color": default_colors[i % len(default_colors)],
            "family": "linear",
            "inverted": False,
            "label": name,
            "window": {
                "end": float(data_max),
                "max": float(data_max),
                "min": float(data_min),
                "start": float(data_min),
            },
        }
        channels.append(channel)

    if not channels:
        return {}

    omero = {
        "channels": channels,
        "rdefs": {
            "defaultT": 0,
            "defaultZ": Z // 2,  # Middle z-plane
            "model": "greyscale",
        },
        "version": "0.5",
    }

    return omero
