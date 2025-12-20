"""
scanimage.py

Functions to detect acquisition parameters from ScanImage metadata,
including stack type, color channels, and timepoints.
"""
from __future__ import annotations

from typing import Literal

StackType = Literal["lbm", "piezo", "single_plane"]


def detect_stack_type(metadata: dict) -> StackType:
    """
    Detect the type of stack from ScanImage metadata.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key with ScanImage data.

    Returns
    -------
    StackType
        One of: "lbm", "piezo", "single_plane"

    Notes
    -----
    Detection logic:
    - LBM: len(si.hChannels.channelSave) > 2
    - Piezo: si.hStackManager.enable == True
    - Single plane: neither of the above
    """
    si = metadata.get("si", {})
    if not si:
        return "single_plane"

    # check for LBM (channels used as z-planes)
    hch = si.get("hChannels", {})
    channel_save = hch.get("channelSave", 1)
    if isinstance(channel_save, list) and len(channel_save) > 2:
        return "lbm"

    # check for piezo stack
    stack_mgr = si.get("hStackManager", {})
    if stack_mgr.get("enable", False):
        return "piezo"

    return "single_plane"


def is_lbm_stack(metadata: dict) -> bool:
    """Check if metadata indicates an LBM stack."""
    return detect_stack_type(metadata) == "lbm"


def is_piezo_stack(metadata: dict) -> bool:
    """Check if metadata indicates a piezo stack."""
    return detect_stack_type(metadata) == "piezo"


def get_lbm_ai_sources(metadata: dict) -> dict[str, list[int]]:
    """
    Extract unique AI sources from LBM virtualChannelSettings.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    dict[str, list[int]]
        Mapping of AI source name to list of channel indices.
        e.g., {"AI0": [1,2,3...14], "AI1": [15,16,17]}

    Notes
    -----
    AI0 only = single color channel
    AI0 + AI1 = dual color channel
    """
    si = metadata.get("si", {})
    scan2d = si.get("hScan2D", {})
    sources: dict[str, list[int]] = {}

    for key, val in scan2d.items():
        if key.startswith("virtualChannelSettings__") and isinstance(val, dict):
            src = val.get("source")
            if src:
                if src not in sources:
                    sources[src] = []
                try:
                    ch_idx = int(key.split("__")[1])
                    sources[src].append(ch_idx)
                except (ValueError, IndexError):
                    pass

    return sources


def get_num_color_channels(metadata: dict) -> int:
    """
    Detect number of color channels.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int
        Number of color channels (1 or 2).

    Notes
    -----
    Uses virtualChannelSettings__N.source for both LBM and non-LBM.
    AI0 only = 1 color channel, AI0 + AI1 = 2 color channels.
    Falls back to channelSave if virtualChannelSettings not available.
    """
    # try virtualChannelSettings first (works for both LBM and non-LBM)
    sources = get_lbm_ai_sources(metadata)
    if sources:
        return len(sources)

    # fallback: check channelSave
    si = metadata.get("si", {})
    hch = si.get("hChannels", {})
    channel_save = hch.get("channelSave", 1)

    if isinstance(channel_save, list) and len(channel_save) == 2:
        return 2
    return 1


def get_num_zplanes(metadata: dict) -> int:
    """
    Get number of z-planes.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int
        Number of z-planes.

    Notes
    -----
    For LBM: len(si.hChannels.channelSave)
    For piezo: si.hStackManager.numSlices
    For single plane: 1
    """
    stack_type = detect_stack_type(metadata)
    si = metadata.get("si", {})

    if stack_type == "lbm":
        hch = si.get("hChannels", {})
        channel_save = hch.get("channelSave", [])
        if isinstance(channel_save, list):
            return len(channel_save)
        return 1

    if stack_type == "piezo":
        stack_mgr = si.get("hStackManager", {})
        return stack_mgr.get("numSlices", 1)

    return 1


def get_frames_per_slice(metadata: dict) -> int:
    """
    Get frames per z-slice.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int
        Frames acquired per z-slice.

    Notes
    -----
    IMPORTANT: use si.hStackManager.framesPerSlice, NOT si.hScan2D.logFramesPerSlice
    The latter is often None/unreliable.
    """
    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})
    return stack_mgr.get("framesPerSlice", 1)


def get_log_average_factor(metadata: dict) -> int:
    """
    Get frame averaging factor.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int
        Averaging factor (>1 means frames were averaged before saving).
    """
    si = metadata.get("si", {})
    scan2d = si.get("hScan2D", {})
    return scan2d.get("logAverageFactor", 1)


def get_z_step_size(metadata: dict) -> float | None:
    """
    Get z-step size in microns.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    float or None
        Z-step size in microns, or None if not available.

    Notes
    -----
    For piezo: si.hStackManager.stackZStepSize
    For LBM: user input required (typically ~20µm for LBM_MIMMS)
    """
    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})

    # try actualStackZStepSize first, then stackZStepSize
    dz = stack_mgr.get("actualStackZStepSize")
    if dz is None:
        dz = stack_mgr.get("stackZStepSize")

    return dz


def compute_num_timepoints(total_frames: int, metadata: dict) -> int:
    """
    Compute number of timepoints from total frames and metadata.

    Parameters
    ----------
    total_frames : int
        Total frames counted from TIFF file(s).
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int
        Number of timepoints (volumes).

    Notes
    -----
    For LBM: each TIFF frame is one timepoint (z-planes interleaved as channels)
    For piezo: total_frames // (numSlices * framesPerSlice), adjusted for averaging

    Decision tree:
    - LBM → num_timepoints = total_frames
    - piezo with averaging → frames_per_volume = numSlices (1 saved frame per slice)
    - piezo no averaging → frames_per_volume = numSlices * framesPerSlice
    - single plane → num_timepoints = total_frames
    """
    stack_type = detect_stack_type(metadata)

    if stack_type == "lbm":
        # LBM: each frame in TIFF is one timepoint
        return total_frames

    if stack_type == "single_plane":
        return total_frames

    # piezo stack
    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})
    scan2d = si.get("hScan2D", {})

    num_slices = stack_mgr.get("numSlices", 1)
    frames_per_slice = stack_mgr.get("framesPerSlice", 1)
    log_avg_factor = scan2d.get("logAverageFactor", 1)

    if log_avg_factor > 1:
        # frames were averaged: 1 saved frame per slice
        frames_per_volume = num_slices
    elif frames_per_slice > 1:
        # multiple frames per slice, no averaging
        frames_per_volume = num_slices * frames_per_slice
    else:
        # single frame per slice
        frames_per_volume = num_slices

    if frames_per_volume <= 0:
        return total_frames

    return total_frames // frames_per_volume


def get_roi_info(metadata: dict) -> dict:
    """
    Get ROI and FOV information from ScanImage metadata.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key and optionally 'roi_groups' or 'num_rois'.

    Returns
    -------
    dict
        Dictionary with ROI/FOV parameters:
        - num_mrois: number of mROIs
        - roi: (width, height) in pixels
        - fov: (x, y) total FOV in pixels
    """
    si = metadata.get("si", {})
    roi_mgr = si.get("hRoiManager", {})

    # get lines per frame (height)
    lines_per_frame = roi_mgr.get("linesPerFrame")

    # get pixels per line (width)
    pixels_per_line = roi_mgr.get("pixelsPerLine")

    # number of mROIs - check multiple sources
    # priority: existing num_rois/num_mrois > roi_groups > hRoiManager.roiGroup
    num_mrois = metadata.get("num_rois") or metadata.get("num_mrois")

    if num_mrois is None:
        # check roi_groups (set by get_metadata_single)
        roi_groups = metadata.get("roi_groups")
        if isinstance(roi_groups, list):
            num_mrois = len(roi_groups)

    if num_mrois is None:
        # fallback to hRoiManager.roiGroup
        num_mrois = 1
        mroi_enable = roi_mgr.get("mroiEnable", False)
        if mroi_enable:
            roi_group = roi_mgr.get("roiGroup")
            if isinstance(roi_group, dict):
                rois = roi_group.get("rois")
                if isinstance(rois, list):
                    num_mrois = len(rois)

    result = {"num_mrois": num_mrois}

    # roi as (width, height) tuple
    if pixels_per_line is not None and lines_per_frame is not None:
        result["roi"] = (pixels_per_line, lines_per_frame)

    # fov as (x, y) tuple in pixels
    if pixels_per_line is not None and lines_per_frame is not None:
        result["fov"] = (num_mrois * pixels_per_line, lines_per_frame)

    return result


def get_frame_rate(metadata: dict) -> float | None:
    """
    Get frame/volume rate from ScanImage metadata.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    float or None
        Frame rate in Hz, or None if not available.
    """
    si = metadata.get("si", {})
    roi_mgr = si.get("hRoiManager", {})

    # scanFrameRate is the most reliable source
    fs = roi_mgr.get("scanFrameRate")
    if fs is not None:
        return round(float(fs), 2)

    # fallback to computing from scanFramePeriod
    period = roi_mgr.get("scanFramePeriod")
    if period is not None and period > 0:
        return round(1.0 / float(period), 2)

    return None


def get_stack_info(metadata: dict) -> dict:
    """
    Get comprehensive stack information from metadata.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    dict
        Dictionary with stack parameters:
        - stack_type: "lbm", "piezo", or "single_plane"
        - num_zplanes: number of z-planes
        - num_color_channels: number of color channels
        - frames_per_slice: frames per z-slice
        - log_average_factor: averaging factor
        - dz: z-step size (None if unknown)
        - fs: frame rate in Hz (None if unknown)
        - roi info (num_rois, roi_width, roi_height, fov_x, fov_y)
    """
    info = {
        "stack_type": detect_stack_type(metadata),
        "num_zplanes": get_num_zplanes(metadata),
        "num_color_channels": get_num_color_channels(metadata),
        "frames_per_slice": get_frames_per_slice(metadata),
        "log_average_factor": get_log_average_factor(metadata),
        "dz": get_z_step_size(metadata),
        "fs": get_frame_rate(metadata),
    }

    # add ROI info
    info.update(get_roi_info(metadata))

    return info


def extract_roi_slices(metadata: dict) -> list[dict]:
    """
    Extract detailed ROI slice information for array indexing.

    Computes actual pixel boundaries for each ROI, accounting for
    fly-to lines between strips and any rounding in height distribution.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing roi_groups, page_height, page_width,
        and num_fly_to_lines.

    Returns
    -------
    list[dict]
        List of ROI info dicts, each containing:
        - y_start: starting y pixel (inclusive)
        - y_end: ending y pixel (exclusive)
        - width: ROI width in pixels
        - height: ROI height in pixels
        - x: x offset (always 0 for strip ROIs)
        - slice: slice object for y-axis indexing

    Notes
    -----
    This function consolidates ROI extraction logic that was previously
    duplicated in MboRawArray._extract_roi_info().

    For multi-ROI acquisitions, the page is divided into strips with
    fly-to lines (dead space) between them. This function computes
    the actual boundaries accounting for these gaps.
    """
    roi_groups = metadata.get("roi_groups", [])
    if not roi_groups:
        return []

    if isinstance(roi_groups, dict):
        roi_groups = [roi_groups]

    page_width = metadata.get("page_width")
    page_height = metadata.get("page_height")
    num_fly_to_lines = metadata.get("num_fly_to_lines", 0)

    if page_width is None or page_height is None:
        return []

    # extract heights from scanfield metadata
    heights_from_metadata = []
    for roi_data in roi_groups:
        scanfields = roi_data.get("scanfields")
        if scanfields is None:
            continue
        if isinstance(scanfields, list):
            scanfields = scanfields[0]
        pixel_res = scanfields.get("pixelResolutionXY")
        if pixel_res and len(pixel_res) >= 2:
            heights_from_metadata.append(pixel_res[1])

    if not heights_from_metadata:
        return []

    # compute actual heights accounting for fly-to lines
    total_metadata_height = sum(heights_from_metadata)
    total_available_height = page_height - (len(roi_groups) - 1) * num_fly_to_lines

    actual_heights = []
    remaining_height = total_available_height
    for i, metadata_height in enumerate(heights_from_metadata):
        if i == len(heights_from_metadata) - 1:
            height = remaining_height
        else:
            height = int(
                round(metadata_height * total_available_height / total_metadata_height)
            )
            remaining_height -= height
        actual_heights.append(height)

    # build ROI slice info
    rois = []
    y_offset = 0

    for height in actual_heights:
        roi_info = {
            "y_start": y_offset,
            "y_end": y_offset + height,
            "width": page_width,
            "height": height,
            "x": 0,
            "slice": slice(y_offset, y_offset + height),
        }
        rois.append(roi_info)
        y_offset += height + num_fly_to_lines

    return rois
