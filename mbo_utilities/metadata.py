from __future__ import annotations

import re
import json
import os
from pathlib import Path

import numpy as np
import tifffile
from mbo_utilities import get_files


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
        print("No metadata found. Using default parameters.")
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
        print("Overlaps too small. Increasing to neuron diameter.")
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


def _params_from_metadata_suite2p(metadata, ops):
    """
    Tau is 0.7 for GCaMP6f, 1.0 for GCaMP6m, 1.25-1.5 for GCaMP6s
    """
    if metadata is None:
        print("No metadata found. Using default parameters.")
        return ops

    # typical neuron ~16 microns
    ops["fs"] = metadata["frame_rate"]
    ops["nplanes"] = 1
    ops["nchannels"] = 1
    ops["do_bidiphase"] = 0
    ops["do_regmetrics"] = True

    # suite2p iterates each plane and takes ops['dxy'][i] where i is the plane index
    ops["dx"] = [metadata["pixel_resolution"][0]]
    ops["dy"] = [metadata["pixel_resolution"][1]]

    return ops


def report_missing_metadata(file: os.PathLike | str):
    tiff_file = tifffile.TiffFile(file)
    if not tiff_file.software == "SI":
        print(f"Missing SI software tag.")
    if not tiff_file.description[:6] == "state.":
        print(f"Missing 'state' software tag.")
    if not "scanimage.SI" in tiff_file.description[-256:]:
        print(f"Missing 'scanimage.SI' in description tag.")


def has_mbo_metadata(file: os.PathLike | str) -> bool:
    """
    Check if a TIFF file has metadata from the Miller Brain Observatory.

    Specifically, this checks for tiff_file.shaped_metadata, which is used to store system and user
    supplied metadata.

    Parameters
    ----------
    file: os.PathLike
        Path to the TIFF file.

    Returns
    -------
    bool
        True if the TIFF file has MBO metadata; False otherwise.
    """
    if not file or not isinstance(file, (str, os.PathLike)):
        raise ValueError(
            "Invalid file path provided: must be a string or os.PathLike object."
            f"Got: {file} of type {type(file)}"
        )
    # Tiffs
    if Path(file).suffix in [".tif", ".tiff"]:
        try:
            tiff_file = tifffile.TiffFile(file)
            if (
                hasattr(tiff_file, "shaped_metadata")
                and tiff_file.shaped_metadata is not None
            ):
                return True
            else:
                return False
        except Exception:
            return False
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
            return False
        else:
            if tiff_file.scanimage_metadata is None:
                print(f"No ScanImage metadata found in {file}.")
                return False
            return True
    except Exception:
        return False


def get_metadata(file, z_step=None, verbose=False):
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
    z_step : float, optional
        The z-step size in microns. If provided, it will be included in the returned metadata.
    verbose : bool, optional
        If True, returns extended metadata including all ScanImage attributes. Default is False.

    Returns
    -------
    dict
        A dictionary containing extracted metadata. For multiple files, includes:
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
    >>> # Single file
    >>> meta = get_metadata("path/to/rawscan_00001.tif")
    >>> print(f"Frames: {meta['num_frames']}")

    >>> # Directory of files
    >>> meta = get_metadata("path/to/scan_directory/")
    >>> print(f"Files processed: {len(meta['file_paths'])}")
    >>> print(f"Frames per file: {meta['frames_per_file']}")

    >>> # List of specific files
    >>> files = ["scan_00001.tif", "scan_00002.tif", "scan_00003.tif"]
    >>> meta = get_metadata(files)
    """
    # Convert input to Path object and handle different input types
    if isinstance(file, (list, tuple)):
        # make sure all values in the list are strings or paths
        if not all(isinstance(f, (str, os.PathLike)) for f in file):
            raise ValueError(
                "All items in the list must be of type str or os.PathLike."
                f"Got: {file} of type {type(file)}"
            )
        file_paths = [Path(f) for f in file]
        return get_metadata_batch(file_paths, z_step=z_step, verbose=verbose)

    file_path = Path(file)

    if file_path.is_dir():
        tiff_files = get_files(file_path, "tif", sort_ascending=True)
        if not tiff_files:
            raise ValueError(f"No TIFF files found in directory: {file_path}")
        return get_metadata_batch(tiff_files, z_step=z_step, verbose=verbose)

    elif file_path.is_file():
        return get_metadata_single(file_path, z_step=z_step, verbose=verbose)

    else:
        raise ValueError(f"Path does not exist or is not accessible: {file_path}")


def get_metadata_single(file: os.PathLike | str, z_step=None, verbose=False):
    """
    Extract metadata from a TIFF file produced by ScanImage or processed via the save_as function.

    This function opens the given TIFF file and retrieves critical imaging parameters and acquisition details.
    It supports both raw ScanImage TIFFs and those modified by downstream processing. If the file contains
    raw ScanImage metadata, the function extracts key fields such as channel information, number of frames,
    field-of-view, pixel resolution, and ROI details. When verbose output is enabled, the complete metadata
    document is returned in addition to the parsed key values.

    Parameters
    ----------
    file : os.PathLike or str
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
        pixel resolution). When verbose is True, the dictionary also includes a key "all" with the full metadata
        from the TIFF header.

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
        if meta is None:
            return None

        si = meta.get("FrameData", {})
        if not si:
            print(f"No FrameData found in {file}.")
            return None
        series = tiff_file.series[0]

        # Extract ROI and imaging metadata
        roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]

        if isinstance(roi_group, dict):
            num_rois = 1
            roi_group = [roi_group]
        else:
            num_rois = len(roi_group)

        num_planes = len(si["SI.hChannels.channelSave"])
        zoom_factor = si["SI.hRoiManager.scanZoomFactor"]
        uniform_sampling = si["SI.hScan2D.uniformSampling"]

        if num_rois > 1:
            try:
                sizes = [
                    roi_group[i]["scanfields"][i]["sizeXY"] for i in range(num_rois)
                ]
                num_pixel_xys = [
                    roi_group[i]["scanfields"][i]["pixelResolutionXY"]
                    for i in range(num_rois)
                ]
            except KeyError:
                sizes = [roi_group[i]["scanfields"]["sizeXY"] for i in range(num_rois)]
                num_pixel_xys = [
                    roi_group[i]["scanfields"]["pixelResolutionXY"]
                    for i in range(num_rois)
                ]

            size_xy = sizes[0]
            num_pixel_xy = num_pixel_xys[0]
        else:
            size_xy = [roi_group[0]["scanfields"]["sizeXY"]][0]
            num_pixel_xy = [roi_group[0]["scanfields"]["pixelResolutionXY"]][0]

        # TIFF header-derived metadata
        objective_resolution = si["SI.objectiveResolution"]
        frame_rate = si["SI.hRoiManager.scanFrameRate"]

        # Field-of-view calculations
        # TODO: We may want an FOV measure that takes into account contiguous ROIs
        # As of now, this is for a single ROI
        fov_x_um = round(objective_resolution * size_xy[0])  # in microns
        fov_y_um = round(objective_resolution * size_xy[1])  # in microns
        fov_roi_um = (fov_x_um, fov_y_um)  # in microns

        pixel_resolution = (fov_x_um / num_pixel_xy[0], fov_y_um / num_pixel_xy[1])
        metadata = {
            "num_planes": num_planes,
            "num_rois": num_rois,
            "fov": fov_roi_um,  # in microns
            "fov_px": tuple(num_pixel_xy),
            "frame_rate": frame_rate,
            "pixel_resolution": np.round(pixel_resolution, 2),
            "ndim": series.ndim,
            "dtype": "int16",
            "size": series.size,
            "roi_width_px": num_pixel_xy[0],
            "roi_height_px": num_pixel_xy[1],
            "objective_resolution": objective_resolution,
            "zoom_factor": zoom_factor,
            "uniform_sampling": uniform_sampling,
        }

        if z_step is not None:
            metadata["z_step"] = z_step

        if verbose:
            metadata["all"] = meta
            return metadata
        else:
            return metadata
    else:
        raise ValueError(f"No metadata found in {file}.")


def get_metadata_batch(file_paths: list | tuple, z_step=None, verbose=False):
    """
    Extract and aggregate metadata from a list of TIFF files.

    Parameters
    ----------
    file_paths : list of Path
        List of TIFF file paths.
    z_step : float, optional
        Z-step in microns.
    verbose : bool, optional
        Include full metadata from first file if True.

    Returns
    -------
    dict
        Aggregated metadata with per-file frame information.
    """
    if not file_paths:
        raise ValueError("No files provided")

    tiff_pages_per_file = []
    frames_per_file = []
    file_path_strings = []
    first_meta = None

    print(f"Processing {len(file_paths)} files...")

    for i, file_path in enumerate(file_paths):
        try:
            file_meta = get_metadata_single(file_path, z_step=z_step, verbose=verbose)
            if file_meta is None:
                print(f"Warning: No metadata found in {file_path}. Skipping.")
                continue

            if i == 0:
                first_meta = file_meta.copy()
            n_pages = len(tifffile.TiffFile(file_path).pages)
            tiff_pages_per_file.append(n_pages)
            frames_per_file.append(int(n_pages / file_meta.get("num_planes")))
            file_path_strings.append(str(file_path))

        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue

    total_frames = sum(frames_per_file)

    first_meta.update(
        {
            "num_frames": total_frames,
            "frames_per_file": frames_per_file,
            "tiff_pages_per_file": tiff_pages_per_file,
            "file_paths": file_path_strings,
            "num_files": len(file_paths),
        }
    )
    print(f"Total: {total_frames} frames across {len(file_paths)} files")

    return first_meta


def params_from_metadata(metadata, base_ops, pipeline="suite2p"):
    """
    Use metadata to get sensible default pipeline parameters.

    If ops are not provided, uses suite2p.default_ops(). Sets framerate, pixel resolution, and do_metrics=True.

    Parameters
    ----------
    metadata : dict
        Result of mbo.get_metadata()
    base_ops : dict
        Ops dict to use as a base.
    pipeline : str, optional
        The pipeline to use. Default is "suite2p".
    """
    if pipeline.lower() == "caiman":
        print("Warning: CaImAn is not stable, proceed at your own risk.")
        return _params_from_metadata_caiman(metadata)
    elif pipeline.lower() == "suite2p":
        print("Setting pipeline to suite2p")
        return _params_from_metadata_suite2p(metadata, base_ops)
    else:
        raise ValueError(
            f"Pipeline {pipeline} not recognized. Use 'caiman' or 'suite2'"
        )


def find_scanimage_metadata(path):
    with tifffile.TiffFile(path) as tif:
        if hasattr(tif, "scanimage_metadata"):
            return tif.scanimage_metadata
        p = tif.pages[0]
        cand = []
        for tag in ("ImageDescription", "Software"):
            if tag in p.tags:
                cand.append(p.tags[tag].value)
        if getattr(p, "description", None):
            cand.append(p.description)
        cand.extend(str(tif.__dict__.get(k, "")) for k in tif.__dict__)
        for s in cand:
            if isinstance(s, bytes):
                s = s.decode(errors="ignore")
            m = re.search(r"{.*ScanImage.*}", s, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return m.group(0)
    return None


def diff_metadata(g1: dict, g2: dict) -> dict:
    diff = {}
    keys = set(g1) | set(g2)
    for key in keys:
        v1 = g1.get(key)
        v2 = g2.get(key)
        if isinstance(v1, dict) and isinstance(v2, dict):
            subdiff = diff_metadata(v1, v2)
            if subdiff:
                diff[key] = subdiff
        elif v1 != v2:
            diff[key] = {"g1": v1, "g2": v2}
    return diff


def clean_scanimage_metadata(meta: dict) -> dict:
    """
    Build a JSON-serializable, nicely nested dict from a ScanImage-style metadata dict.

    - Keeps ALL top-level (non-'si') keys after cleaning (remove empties, NaNs/inf, empty arrays/strings).
    - Includes the full `si` subtree (cleaned), preserving e.g. 'RoiGroups'.
    - Additionally nests any flat 'SI.*' keys found anywhere under the 'si' subtree:
        * If they are inside meta['si']['FrameData'], they become si['FrameData']['SI'][...nested...]
        * If 'SI.*' keys appear elsewhere in the tree, they are nested under a top-level 'SI' root.
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

    def _collect_SI_pairs(node, path=()):
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("SI."):
                    out.append((".".join((k,)), v, path))  # store key, value, and where we found it
                out.extend(_collect_SI_pairs(v, path + (k,)))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                out.extend(_collect_SI_pairs(v, path + (f"[{i}]",)))
        return out

    def _nest_into(root_dict, dotted_key, value, keep_root_SI=True):
        parts = dotted_key.split(".")
        # parts[0] should be "SI"
        start = 0 if keep_root_SI else 1
        cur = root_dict
        for p in parts[start:-1]:
            p = p.split("[")[0]
            cur = cur.setdefault(p, {})
        leaf = parts[-1].split("[")[0]
        cur[leaf] = _clean(value)

    # 1) Start with cleaned copy of ALL top-level non-'si' keys.
    result = {}
    for k, v in meta.items():
        if k == "si":
            continue
        cv = _clean(v)
        if cv is not None:
            result[k] = cv

    # 2) Bring in the 'si' subtree (cleaned) as-is (so 'RoiGroups' is preserved).
    si_clean = None
    if isinstance(meta.get("si"), dict):
        si_clean = _clean(meta["si"])
        if si_clean is not None:
            result["si"] = si_clean
        else:
            result["si"] = {}

    # 3) Find *all* 'SI.*' keys anywhere, and nest them.
    #    If they are under meta['si']['FrameData'], put them into result['si']['FrameData']['SI'][...]
    #    Otherwise, collect them under a top-level 'SI' in the result.
    si_pairs = _collect_SI_pairs(meta)
    if si_pairs:
        # Ensure containers exist where needed
        if "si" not in result or not isinstance(result["si"], dict):
            result["si"] = {}
        si_framedata = result["si"].setdefault("FrameData", {})
        si_framedata_SI = si_framedata.setdefault("SI", {})
        top_SI = result.setdefault("SI", {})

        for dotted_key, val, where in si_pairs:
            # Check if this came from under ['si']['FrameData']
            if len(where) >= 2 and where[0] == "si" and where[1] == "FrameData":
                _nest_into(si_framedata_SI, dotted_key, val, keep_root_SI=False)
            else:
                _nest_into(top_SI, dotted_key, val, keep_root_SI=True)

        # If the top-level 'SI' ended up empty after cleaning/pruning, drop it.
        if not _prune(top_SI):
            result.pop("SI", None)

    # 4) Final prune pass
    return _prune(result)


def save_metadata_html(
        meta: dict,
        out_html: str | Path,
        title: str = "ScanImage Metadata",
        inline_max_chars: int = 200
):
    """
    Clean + render metadata to a collapsible HTML tree with search and expand/collapse all.
    Lists/tuples are shown inline (compact), truncated past `inline_max_chars` with a tooltip.

    This is the most absurd code ever produced by AI.

    """
    cleaned = clean_scanimage_metadata(meta)  # relies on your function being defined

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _inline_repr(val) -> str:
        """Compact, inline representation for leaves (lists/tuples/values)."""
        # Tuples: render with parentheses
        if isinstance(val, tuple):
            try:
                body = ", ".join(_inline_repr(x) if isinstance(x, (list, tuple)) else json.dumps(x)
                                 for x in val)
            except TypeError:
                body = json.dumps(list(val), separators=(',', ': '))
            s = f"({body})"
        # Lists and everything else JSON-able: compact JSON
        elif isinstance(val, list):
            s = json.dumps(val, separators=(',', ': '))
        elif isinstance(val, (dict,)):  # shouldn’t hit here for leaf handler, but safe
            s = json.dumps(val, separators=(',', ': '))
        else:
            try:
                s = json.dumps(val)
            except Exception:
                s = str(val)

        # Truncate for very long strings but keep full in title
        if len(s) > inline_max_chars:
            return f"<span class='inline' title='{esc(s)}'>{esc(s[:inline_max_chars])}…</span>"
        return f"<span class='inline'>{esc(s)}</span>"

    def render_dict(d: dict, level: int = 0):
        keys = sorted(d.keys(), key=lambda k: (not isinstance(k, str), str(k)))
        html = []
        for k in keys:
            v = d[k]
            if isinstance(v, dict):
                html.append(
                    f"""
<details class="node" data-key="{esc(k)}" {'' if level else 'open'}>
  <summary><span class="k">{esc(k)}</span> <span class="badge">dict</span></summary>
  <div class="child">
    {render_dict(v, level+1)}
  </div>
</details>
"""
                )
            else:
                html.append(
                    f"""
<div class="leaf-row" data-key="{esc(k)}">
  <span class="k">{esc(k)}</span>
  <span class="sep">:</span>
  {_inline_repr(v)}
</div>
"""
                )
        return "\n".join(html)

    tree_html = render_dict(cleaned)

    # ---- Assemble page -------------------------------------------------------
    css = """
:root {
  --bg:#0e1116; --fg:#e6edf3; --muted:#9aa5b1; --accent:#4aa3ff; --badge:#293241;
  --mono:#e6edf3; --panel:#151a22; --border:#222a35; --hl:#0b6bcb33;
}
* { box-sizing:border-box; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
body { margin:0; background:var(--bg); color:var(--fg); }
header { position:sticky; top:0; background:linear-gradient(180deg, rgba(14,17,22,.98), rgba(14,17,22,.94)); border-bottom:1px solid var(--border); padding:12px 16px; z-index:10; }
h1 { margin:0 0 8px 0; font-size:16px; font-weight:700; }
.controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
input[type="search"] { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:8px 10px; width:360px; }
button { background:var(--panel); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:8px 10px; cursor:pointer; }
button:hover { border-color:var(--accent); }
main { padding:16px; }
summary { cursor:pointer; padding:6px 8px; border-radius:8px; }
summary:hover { background:var(--hl); }
.node { border-left:2px solid var(--border); margin:4px 0 4px 8px; padding-left:10px; }
.child { margin-left:4px; padding-left:6px; border-left:1px dotted var(--border); }
.k { color:var(--fg); font-weight:700; }
.badge { font-size:11px; background:var(--badge); color:var(--muted); border:1px solid var(--border); padding:2px 6px; border-radius:999px; margin-left:8px; }
.leaf-row { padding:2px 6px; display:flex; gap:6px; align-items:flex-start; border-left:2px solid transparent; }
.leaf-row:hover { background:var(--hl); border-left-color:var(--accent); border-radius:6px; }
.inline { color:var(--mono); white-space:pre-wrap; word-break:break-word; }
summary::-webkit-details-marker { display:none; }
mark { background:#ffe08a44; color:inherit; padding:0 2px; border-radius:3px; }
footer { color:var(--muted); font-size:12px; padding:12px 16px; border-top:1px solid var(--border); }
"""

    js = """
(function(){
  const q = document.getElementById('search');
  const btnExpand = document.getElementById('expandAll');
  const btnCollapse = document.getElementById('collapseAll');

  function normalize(s){ return (s||'').toLowerCase(); }

  function highlight(text, term){
    if(!term) return text;
    const esc = (s)=>s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
    const re = new RegExp(esc(term), 'ig');
    return text.replace(re, m => `<mark>${m}</mark>`);
  }

  function filter(term){
    const t = normalize(term);
    document.querySelectorAll('.leaf-row').forEach(row=>{
      const key = normalize(row.getAttribute('data-key'));
      const val = row.querySelector('.inline')?.textContent || '';
      const hit = key.includes(t) || normalize(val).includes(t);
      row.style.display = hit ? '' : 'none';
      const k = row.querySelector('.k');
      if(k){ k.innerHTML = highlight(k.textContent, term); }
    });
    document.querySelectorAll('details.node').forEach(node=>{
      const rows = node.querySelectorAll('.leaf-row');
      const kids = node.querySelectorAll('details.node');
      let anyVisible = false;
      rows.forEach(r => { if(r.style.display !== 'none') anyVisible = true; });
      kids.forEach(k => { if(k.style.display !== 'none') anyVisible = true; });
      node.style.display = anyVisible ? '' : 'none';
      const sum = node.querySelector('summary .k');
      if(sum){ sum.innerHTML = highlight(sum.textContent, term); }
      if(t && anyVisible) node.open = true;
    });
  }

  q.addEventListener('input', (e)=>filter(e.target.value));
  document.getElementById('expandAll').addEventListener('click', ()=>{
    document.querySelectorAll('details').forEach(d=> d.open = true);
  });
  document.getElementById('collapseAll').addEventListener('click', ()=>{
    document.querySelectorAll('details').forEach(d=> d.open = false);
  });
})();
"""

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>{esc(title)}</h1>
    <div class="controls">
      <input id="search" type="search" placeholder="Search keys/values..." />
      <button id="expandAll">Expand all</button>
      <button id="collapseAll">Collapse all</button>
    </div>
  </header>
  <main>
    {tree_html}
  </main>
  <footer>
    Saved from Python — clean & render for convenient browsing.
  </footer>
  <script>{js}</script>
</body>
</html>"""

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote metadata HTML to: {out_html}")
