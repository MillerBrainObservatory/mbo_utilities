from __future__ import annotations

import os

import numpy as np
import tifffile


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

    split_frames = params["main"]["num_frames_split"]
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

    # suite2p iterates each plane and takes ops['dxy'][i] where i is the plane index
    ops["dx"] = [metadata["pixel_resolution"][0]]
    ops["dy"] = [metadata["pixel_resolution"][1]]

    return ops


def is_raw_scanimage(file: os.PathLike | str):
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
    if not file:
        return False

    tiff_file = tifffile.TiffFile(file)
    # TiffFile.shaped_metadata is where we store metadata for processed tifs
    # if this is not empty, we have a processed file
    # otherwise, we have a raw scanimage tiff
    if (
        hasattr(tiff_file, "shaped_metadata")
        and tiff_file.shaped_metadata is not None
        and isinstance(tiff_file.shaped_metadata, (list, tuple))
        and tiff_file.shaped_metadata[0] not in ([], (), None)
    ):
        return False
    else:
        return True


def get_metadata(file: os.PathLike | str, verbose=False):
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

    Examples
    --------
    >>> meta = get_metadata("path/to/rawscan_00001.tif")
    >>> print(meta["num_frames"])
    5345
    >>> meta = get_metadata("path/to/assembled_data.tif")
    >>> print(meta["shape"])
    (14, 5345, 477, 477)
    >>> meta_verbose = get_metadata("path/to/scanimage_file.tif", verbose=True)
    >>> print(meta_verbose["all"])
    {... Includes all ScanImage FrameData ...}
    """
    tiff_file = tifffile.TiffFile(file)
    # previously processed files
    if not is_raw_scanimage(file):
        return tiff_file.shaped_metadata[0]["image"]
    elif hasattr(tiff_file, "scanimage_metadata"):
        meta = tiff_file.scanimage_metadata
        if meta is None:
            return None

        si = meta.get("FrameData", {})
        if not si:
            print(f"No FrameData found in {file}.")
            return None
        print("Reading tiff series data...")
        series = tiff_file.series[0]
        print("Reading tiff pages...")
        pages = tiff_file.pages
        print("Raw tiff fully read.")

        # Extract ROI and imaging metadata
        roi_group = meta["RoiGroups"]["imagingRoiGroup"]["rois"]

        if isinstance(roi_group, dict):
            num_rois = 1
            roi_group = [roi_group]
        else:
            num_rois = len(roi_group)

        num_planes = len(si["SI.hChannels.channelSave"])

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

            # see if each item in sizes is the same
            assert all([sizes[0] == size for size in sizes]), (
                "ROIs have different sizes"
            )
            assert all(
                [num_pixel_xys[0] == num_pixel_xy for num_pixel_xy in num_pixel_xys]
            ), "ROIs have different pixel resolutions"
            size_xy = sizes[0]
            num_pixel_xy = num_pixel_xys[0]
        else:
            size_xy = [roi_group[0]["scanfields"]["sizeXY"]][0]
            num_pixel_xy = [roi_group[0]["scanfields"]["pixelResolutionXY"]][0]

        # TIFF header-derived metadata
        sample_format = pages[0].dtype.name
        objective_resolution = si["SI.objectiveResolution"]
        frame_rate = si["SI.hRoiManager.scanFrameRate"]
        try:
            z_step_pollen = si["hStackManager.stackZStepSize"]
        except KeyError:
            z_step_pollen = None

        # Field-of-view calculations
        # TODO: We may want an FOV measure that takes into account contiguous ROIs
        # As of now, this is for a single ROI
        fov_x_um = round(objective_resolution * size_xy[0])  # in microns
        fov_y_um = round(objective_resolution * size_xy[1])  # in microns
        fov_roi_um = (fov_x_um, fov_y_um)  # in microns

        pixel_resolution = (fov_x_um / num_pixel_xy[0], fov_y_um / num_pixel_xy[1])

        metadata = {
            "num_planes": num_planes,
            "num_frames": int(len(pages) / num_planes),
            "fov": fov_roi_um,  # in microns
            "fov_px": fov_roi_um,
            "num_rois": num_rois,
            "frame_rate": frame_rate,
            "pixel_resolution": np.round(pixel_resolution, 2),
            "ndim": series.ndim,
            "dtype": "uint16",
            "size": series.size,
            "raw_height": pages[0].shape[0],
            "raw_width": pages[0].shape[1],
            "tiff_pages": len(pages),
            "roi_width_px": num_pixel_xy[0],
            "roi_height_px": num_pixel_xy[1],
            "sample_format": sample_format,
            "objective_resolution": objective_resolution,
            "z_step_pollen": z_step_pollen,
        }
        if verbose:
            metadata["all"] = meta
            return metadata
        else:
            return metadata
    else:
        raise ValueError(f"No metadata found in {file}.")


def params_from_metadata(metadata, ops=None):
    """
    Use metadata to get sensible default pipeline parameters.

    Parameters
    ----------
    metadata : dict
        Result of mbo.get_metadata()
    ops : dict, optional
        If provided, will return suite2p ops
    """
    if ops:
        print("Ops provided. Setting pipeline to suite2p")
        return _params_from_metadata_suite2p(metadata, ops)
    else:
        return _params_from_metadata_caiman(metadata)
