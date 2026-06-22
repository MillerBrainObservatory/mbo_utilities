"""
scanimage.py.

Functions to detect acquisition parameters from ScanImage metadata,
including stack type, color channels, and timepoints.
"""
from __future__ import annotations

from typing import Literal

StackType = Literal["lbm", "piezo", "pollen", "single_plane"]


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
        One of: "lbm", "piezo", "pollen", "single_plane"

    Notes
    -----
    Detection logic:
    - Pollen: LBM + piezo enabled (calibration acquisition)
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
    is_lbm = isinstance(channel_save, list) and len(channel_save) > 2

    # check for piezo stack
    stack_mgr = si.get("hStackManager", {})
    is_piezo = stack_mgr.get("enable", False)

    # pollen calibration: LBM system with piezo z-scanning
    if is_lbm and is_piezo:
        return "pollen"

    if is_lbm:
        return "lbm"

    if is_piezo:
        return "piezo"

    return "single_plane"


def is_lbm_stack(metadata: dict) -> bool:
    """Check if metadata indicates an LBM stack (includes pollen)."""
    return detect_stack_type(metadata) in ("lbm", "pollen")


def is_piezo_stack(metadata: dict) -> bool:
    """Check if metadata indicates a piezo stack (includes pollen)."""
    return detect_stack_type(metadata) in ("piezo", "pollen")


def get_saved_channel_ports(metadata: dict) -> dict[str, list[int]]:
    """
    Map AI port -> sorted list of saved channel indices on that port.

    Restricted to channels present in si.hChannels.channelSave; channels that
    are configured in virtualChannelSettings but not saved to the TIFF are
    excluded. If channelSave is empty/missing, no filtering is applied.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    dict[str, list[int]]
        Mapping of AI port (e.g. "AI0") to its saved channel indices.
        Dual-color single-plane: {"AI0": [1], "AI1": [2]}
        Dual-color 30-bead LBM:  {"AI0": [1..15], "AI1": [16..30]}
        Single-color LBM:        {"AI0": [1..30]}
    """
    si = metadata.get("si", {})
    scan2d = si.get("hScan2D", {})

    cs = si.get("hChannels", {}).get("channelSave", None)
    if isinstance(cs, list):
        flat: list = []
        for x in cs:
            if isinstance(x, list):
                if len(x) != 1:
                    raise NotImplementedError(
                        f"channelSave entry {x!r} has multiple elements; "
                        "multi-element inner lists are not supported"
                    )
                flat.append(x[0])
            else:
                flat.append(x)
        saved: set[int] | None = {int(x) for x in flat}
    elif isinstance(cs, (int, float)):
        saved = {int(cs)}
    else:
        saved = None

    sources: dict[str, list[int]] = {}
    for key, val in scan2d.items():
        if not (key.startswith("virtualChannelSettings__") and isinstance(val, dict)):
            continue
        src = val.get("source")
        if not src:
            continue
        try:
            ch_idx = int(key.split("__")[1])
        except (ValueError, IndexError):
            continue
        if saved is not None and ch_idx not in saved:
            continue
        sources.setdefault(src, []).append(ch_idx)

    for port in sources:
        sources[port].sort()
    return sources


def get_color_channel_ports(metadata: dict) -> list[str]:
    """Sorted list of AI ports represented in the saved TIFF (e.g. ['AI0', 'AI1'])."""
    return sorted(get_saved_channel_ports(metadata))


def get_beamlets_per_port(metadata: dict) -> dict[str, int]:
    """Count of saved channels per AI port."""
    return {p: len(ch) for p, ch in get_saved_channel_ports(metadata).items()}


def get_num_color_channels(metadata: dict) -> int:
    """
    Number of color channels (unique AI ports) in the saved TIFF.

    Counts unique virtualChannelSettings sources restricted to
    si.hChannels.channelSave. Falls back to channelSave length when
    virtualChannelSettings is not present.
    """
    ports = get_color_channel_ports(metadata)
    if ports:
        return len(ports)

    channel_save = metadata.get("si", {}).get("hChannels", {}).get("channelSave", 1)
    if isinstance(channel_save, list) and len(channel_save) == 2:
        return 2
    return 1


def get_num_zplanes(metadata: dict) -> int:
    """
    Get number of z-planes.

    For LBM/pollen: number of saved beamlets per AI port.
    For piezo: si.hStackManager.numSlices.
    For single plane: 1.
    """
    stack_type = detect_stack_type(metadata)
    si = metadata.get("si", {})

    if stack_type in ("lbm", "pollen"):
        cs = si.get("hChannels", {}).get("channelSave", [])
        cs_len = len(cs) if isinstance(cs, list) else (1 if isinstance(cs, (int, float)) else 0)
        per = get_beamlets_per_port(metadata)
        # use VCS only if it covers all saved channels; otherwise fall back
        # to len(channelSave) // num_color_channels for sparse/missing VCS.
        if per and sum(per.values()) >= cs_len > 0:
            return max(per.values())
        if cs_len:
            return max(cs_len // max(get_num_color_channels(metadata), 1), 1)
        return 1

    if stack_type == "piezo":
        return si.get("hStackManager", {}).get("numSlices", 1)

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


def get_num_volumes(metadata: dict) -> int | None:
    """
    Get number of volumes from ScanImage hStackManager.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int or None
        Number of volumes requested, or None if not a piezo stack.

    Notes
    -----
    Uses si.hStackManager.numVolumes for piezo/pollen stacks.
    For LBM, volumes are the same as timepoints (each frame is a volume).
    """
    stack_type = detect_stack_type(metadata)

    if stack_type == "lbm":
        # LBM: not applicable, use num_timepoints instead
        return None

    if stack_type == "single_plane":
        return None

    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})

    # prefer actualNumVolumes over numVolumes
    num_vol = stack_mgr.get("actualNumVolumes")
    if num_vol is None:
        num_vol = stack_mgr.get("numVolumes")

    return num_vol


def get_num_slices(metadata: dict) -> int | None:
    """
    Get number of z-slices per volume from ScanImage hStackManager.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int or None
        Number of z-slices per volume, or None if not a piezo stack.

    Notes
    -----
    Uses si.hStackManager.numSlices for piezo/pollen stacks.
    For LBM, slices are represented as channels (beamlets).
    """
    stack_type = detect_stack_type(metadata)

    if stack_type == "lbm":
        # LBM: slices are channels
        return None

    if stack_type == "single_plane":
        return None

    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})

    # prefer actualNumSlices over numSlices
    num_slices = stack_mgr.get("actualNumSlices")
    if num_slices is None:
        num_slices = stack_mgr.get("numSlices")

    return num_slices


def get_frames_per_volume(metadata: dict) -> int | None:
    """
    Get total frames per volume from ScanImage hStackManager.

    Parameters
    ----------
    metadata : dict
        Metadata dict containing 'si' key.

    Returns
    -------
    int or None
        Frames per volume (numSlices * framesPerSlice if not averaged),
        or None if not a piezo stack.

    Notes
    -----
    Uses si.hStackManager.numFramesPerVolume for piezo/pollen stacks.
    """
    stack_type = detect_stack_type(metadata)

    if stack_type in ("lbm", "single_plane"):
        return None

    si = metadata.get("si", {})
    stack_mgr = si.get("hStackManager", {})

    return stack_mgr.get("numFramesPerVolume")


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
    duplicated in ScanImageArray._extract_roi_info().

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
            height = round(metadata_height * total_available_height / total_metadata_height)
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
