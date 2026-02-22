"""
Fiber activity detection and ROI extraction from timelapse fluorescence images.

This module segments neurite-like structures from a mean projection image,
tiles the segmented mask into small ROIs, extracts temporal fluorescence
traces, and filters ROIs by an activity threshold (peak dF above baseline).

Can be used as a library (import and call ``run``) or from the command line::

    python -m mbo_utilities.analysis.fiber_activity image.tif --frame-rate 17.58
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FiberActivityResult:
    """Container for fiber activity analysis outputs.

    Parameters
    ----------
    traces : NDArray[np.float64]
        Temporal traces, shape ``(n_frames, n_rois)``.
    locations : NDArray[np.int64]
        ROI centre coordinates, shape ``(n_rois, 2)`` with columns ``[x, y]``.
    mean_image : NDArray[np.float64]
        Normalised mean projection of the input stack.
    enhanced_image : NDArray[np.float64]
        Fibre-enhanced image used for segmentation.
    binary_mask : NDArray[np.bool_]
        Binary segmentation mask.
    frame_rate : float
        Acquisition frame rate in Hz.
    """

    traces: NDArray[np.float64]
    locations: NDArray[np.int64]
    mean_image: NDArray[np.float64]
    enhanced_image: NDArray[np.float64]
    binary_mask: NDArray[np.bool_]
    frame_rate: float = 1.0
    _metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------


def _to_3d(data) -> NDArray:
    """Ensure *data* is a concrete ``(T, Y, X)`` float64 array.

    Accepts paths, mbo lazy arrays, or plain ndarrays.  Subtracts the global
    minimum so the background sits at zero (matches MATLAB
    ``OrigImage - min(OrigImage(:))``).
    """
    if isinstance(data, (str, Path)):
        import tifffile

        data = tifffile.imread(str(data))

    stack = np.asarray(data, dtype=np.float64)

    if stack.ndim != 3:
        raise ValueError(
            f"Expected a 3-D (T, Y, X) array, got {stack.ndim}-D {stack.shape}. "
            "For 4-D data use run() which iterates over z-planes automatically."
        )

    # Subtract global minimum
    stack = stack - stack.min()
    return stack


def compute_mean_image(stack: NDArray) -> NDArray[np.float64]:
    """Compute a normalised temporal mean projection.

    Parameters
    ----------
    stack : NDArray
        3-D array ``(T, Y, X)``.

    Returns
    -------
    mean_image : NDArray[np.float64]
        2-D mean image normalised to ``[0, 1]``.
    """
    # Average over time (axis 0) to get 2-D spatial image
    mean_img = np.mean(stack, axis=0)
    return mean_img / mean_img.max()


def enhance_fibers(
    mean_image: NDArray[np.float64],
    fiber_thickness: int = 3,
) -> NDArray[np.float64]:
    """Enhance fibre-like structures using oriented gradient filters.

    Applies horizontal and vertical matched filters whose width matches the
    expected neurite thickness, rectifies negative responses, and computes the
    gradient magnitude.

    Parameters
    ----------
    mean_image : NDArray[np.float64]
        2-D normalised mean image.
    fiber_thickness : int, optional
        Approximate neurite thickness in pixels.  Controls the width of the
        matched filter kernels.  Default is ``3``.

    Returns
    -------
    enhanced : NDArray[np.float64]
        2-D enhanced image normalised to ``[0, 1]``.
    """
    from scipy.ndimage import convolve

    n = fiber_thickness
    # Matched filter: [-1 block | +2 block | -1 block] for horizontal/vertical
    gx = np.hstack([-np.ones((n, n)), 2 * np.ones((n, n)), -np.ones((n, n))])
    gy = gx.T

    # Filter and rectify negative responses
    ix = convolve(mean_image, gx, mode="reflect")
    iy = convolve(mean_image, gy, mode="reflect")
    ix[ix < 0] = 0
    iy[iy < 0] = 0

    # Gradient magnitude
    mag = np.sqrt(ix**2 + iy**2)
    return mag / mag.max()


def segment_fibers(
    enhanced_image: NDArray[np.float64],
    percentile_threshold: float = 95.0,
    min_object_size: int = 4,
) -> NDArray[np.bool_]:
    """Binarise the enhanced image using a global percentile threshold.

    Parameters
    ----------
    enhanced_image : NDArray[np.float64]
        2-D fibre-enhanced image (output of :func:`enhance_fibers`).
    percentile_threshold : float, optional
        Intensity percentile used as the binarisation cutoff.  Pixels above
        this percentile are considered foreground.  Default is ``95.0``.
    min_object_size : int, optional
        Connected components smaller than this (in pixels) are removed.
        Default is ``4``.

    Returns
    -------
    mask : NDArray[np.bool_]
        Binary segmentation mask.
    """
    from scipy.ndimage import label

    # Threshold at the given percentile (e.g. top 5% of pixels)
    threshold = np.percentile(enhanced_image, percentile_threshold)
    mask = enhanced_image > threshold

    # Remove small connected components (equivalent to bwareaopen)
    labeled, n_labels = label(mask)
    for i in range(1, n_labels + 1):
        if np.sum(labeled == i) < min_object_size:
            mask[labeled == i] = False

    return mask


def extract_roi_traces(
    stack: NDArray,
    mask: NDArray[np.bool_],
    roi_size: int = 3,
    activity_threshold: float = 0.2,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Tile the segmented mask into square ROIs and extract temporal traces.

    The mask is scanned with a stride equal to ``roi_size``.  At each grid
    position whose centre pixel lies on the mask, the spatial mean of the
    ``roi_size x roi_size`` block is computed across all frames to produce a
    1-D temporal trace.  ROIs whose peak activity (after light smoothing) does
    not exceed the baseline mean by at least ``activity_threshold`` are
    discarded.

    Parameters
    ----------
    stack : NDArray
        3-D timelapse array ``(T, Y, X)``.
    mask : NDArray[np.bool_]
        Binary segmentation mask ``(Y, X)``.
    roi_size : int, optional
        Side length of each square ROI in pixels.  Also used as the tiling
        stride.  Default is ``3``.
    activity_threshold : float, optional
        Minimum ``(max - mean) / mean`` ratio for an ROI to be retained.
        Default is ``0.2``.

    Returns
    -------
    traces : NDArray[np.float64]
        Temporal traces, shape ``(n_frames, n_rois)``.
    locations : NDArray[np.int64]
        Centre coordinates of kept ROIs, shape ``(n_rois, 2)`` with columns
        ``[x, y]``.
    """
    from scipy.ndimage import uniform_filter1d

    n_frames, img_h, img_w = stack.shape
    half = roi_size // 2

    trace_list = []
    loc_list = []

    # Iterate with stride = roi_size to prevent overlapping ROIs
    for y in range(half, img_h - half, roi_size):
        for x in range(half, img_w - half, roi_size):
            # Only analyse if centre pixel falls on the mask
            if not mask[y, x]:
                continue

            # Extract roi_size x roi_size block across all frames
            roi_vol = stack[
                :, y - half : y + half + 1, x - half : x + half + 1
            ].astype(np.float64)

            # Spatial mean → 1-D temporal trace
            trace = np.mean(roi_vol, axis=(1, 2))

            # Activity filtering: smooth to ignore single-frame noise
            mean_f = np.mean(trace)
            smoothed = uniform_filter1d(trace, size=3)
            max_f = np.max(smoothed)

            # Keep ROI if peak exceeds baseline by threshold ratio
            if mean_f > 0 and (max_f - mean_f) / mean_f >= activity_threshold:
                trace_list.append(trace)
                loc_list.append([x, y])

    if not trace_list:
        return np.empty((n_frames, 0), dtype=np.float64), np.empty(
            (0, 2), dtype=np.int64
        )

    traces = np.column_stack(trace_list)
    locations = np.array(loc_list, dtype=np.int64)
    return traces, locations


def save_results(
    result: FiberActivityResult,
    output_dir: str | Path,
    prefix: str = "FiberActivity",
) -> dict[str, Path]:
    """Persist traces and ROI locations to disk.

    Writes two files into *output_dir*:

    * ``{prefix}_TimeSeries.csv`` — frame number followed by one column per
      ROI.
    * ``{prefix}_Locations.csv`` — ROI ID, X centre, and Y centre.

    Parameters
    ----------
    result : FiberActivityResult
        Analysis output from :func:`run`.
    output_dir : str or Path
        Directory for output files (created if it does not exist).
    prefix : str, optional
        Filename prefix.  Default is ``"FiberActivity"``.

    Returns
    -------
    paths : dict[str, Path]
        Mapping of ``{"traces": <path>, "locations": <path>}``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_frames, n_rois = result.traces.shape

    # Time series CSV: FrameNumber, ROI_1, ROI_2, ...
    ts_path = out / f"{prefix}_TimeSeries.csv"
    header = ",".join(["FrameNumber"] + [f"ROI_{i + 1}" for i in range(n_rois)])
    frames_col = np.arange(1, n_frames + 1).reshape(-1, 1)
    ts_data = np.hstack([frames_col, result.traces])
    np.savetxt(ts_path, ts_data, delimiter=",", header=header, comments="")

    # Locations CSV: ROI_ID, X_Center, Y_Center
    loc_path = out / f"{prefix}_Locations.csv"
    ids = np.arange(1, n_rois + 1).reshape(-1, 1)
    loc_data = np.hstack([ids, result.locations])
    np.savetxt(
        loc_path,
        loc_data,
        delimiter=",",
        header="ROI_ID,X_Center,Y_Center",
        comments="",
        fmt="%d",
    )

    return {"traces": ts_path, "locations": loc_path}


# ---------------------------------------------------------------------------
# Single-plane runner
# ---------------------------------------------------------------------------


def _run_single(
    stack: NDArray,
    *,
    fiber_thickness: int = 3,
    percentile_threshold: float = 95.0,
    min_object_size: int = 4,
    activity_threshold: float = 0.2,
    frame_rate: float = 1.0,
) -> FiberActivityResult:
    """Process a single 3-D ``(T, Y, X)`` plane."""
    stack = stack.astype(np.float64)
    stack = stack - stack.min()

    # Mean image & enhance
    mean_img = compute_mean_image(stack)
    enhanced = enhance_fibers(mean_img, fiber_thickness=fiber_thickness)

    # Segment
    mask = segment_fibers(
        enhanced,
        percentile_threshold=percentile_threshold,
        min_object_size=min_object_size,
    )

    # Extract activity and filter ROIs
    traces, locations = extract_roi_traces(
        stack, mask, roi_size=fiber_thickness, activity_threshold=activity_threshold
    )

    return FiberActivityResult(
        traces=traces,
        locations=locations,
        mean_image=mean_img,
        enhanced_image=enhanced,
        binary_mask=mask,
        frame_rate=frame_rate,
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def run(
    data: str | Path | NDArray,
    *,
    fiber_thickness: int = 3,
    percentile_threshold: float = 95.0,
    min_object_size: int = 4,
    activity_threshold: float = 0.2,
    frame_rate: float = 1.0,
    output_dir: str | Path | None = None,
    prefix: str = "FiberActivity",
) -> FiberActivityResult | list[FiberActivityResult]:
    """Run the full fibre-activity detection pipeline.

    This is the main entry-point.  It chains together all processing steps:

    1. Load / materialise the timelapse stack.
    2. Compute a mean projection (:func:`compute_mean_image`).
    3. Enhance fibre structures (:func:`enhance_fibers`).
    4. Segment the enhanced image (:func:`segment_fibers`).
    5. Extract and filter ROI traces (:func:`extract_roi_traces`).
    6. Optionally save results to disk (:func:`save_results`).

    For 4-D inputs ``(T, Z, Y, X)`` each z-plane is processed independently
    and a list of :class:`FiberActivityResult` is returned (one per plane).

    Parameters
    ----------
    data : str, Path, or array-like
        Path to a TIFF file **or** an array (e.g. from ``mbo.imread``).
        Accepted shapes: ``(T, Y, X)`` for a single plane, or
        ``(T, Z, Y, X)`` for a volume (processed per z-plane).
    fiber_thickness : int, optional
        Approximate neurite thickness in pixels.  Default is ``3``.
    percentile_threshold : float, optional
        Percentile for global segmentation threshold.  Default is ``95.0``.
    min_object_size : int, optional
        Minimum connected-component size to retain.  Default is ``4``.
    activity_threshold : float, optional
        ``(peak - mean) / mean`` ratio for ROI filtering.  Default is ``0.2``.
    frame_rate : float, optional
        Acquisition frame rate in Hz (used for time-axis labelling only).
        Default is ``1.0``.
    output_dir : str, Path, or None, optional
        If given, results are written to this directory.
    prefix : str, optional
        Filename prefix for saved files.  Default is ``"FiberActivity"``.

    Returns
    -------
    result : FiberActivityResult or list[FiberActivityResult]
        Single result for 3-D input, list of per-plane results for 4-D input.

    Examples
    --------
    >>> from mbo_utilities.analysis.fiber_activity import run
    >>> result = run("timelapse.tif", frame_rate=17.58)
    >>> result.traces.shape
    (693, 42)

    >>> import mbo_utilities as mbo
    >>> data = mbo.imread("volume.tif")  # (T, Z, Y, X)
    >>> results = run(data, frame_rate=17.58)  # list, one per z-plane
    """
    # Materialise lazy arrays / load from path
    if isinstance(data, (str, Path)):
        import tifffile

        arr = tifffile.imread(str(data)).astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)

    kwargs = dict(
        fiber_thickness=fiber_thickness,
        percentile_threshold=percentile_threshold,
        min_object_size=min_object_size,
        activity_threshold=activity_threshold,
        frame_rate=frame_rate,
    )

    # --- 4-D: iterate over z-planes ---
    if arr.ndim == 4:
        n_planes = arr.shape[1]
        results = []
        for z in range(n_planes):
            plane = arr[:, z, :, :]
            res = _run_single(plane, **kwargs)
            res._metadata["z_plane"] = z

            if output_dir is not None:
                save_results(res, output_dir, prefix=f"{prefix}_z{z:02d}")

            results.append(res)
        return results

    # --- 3-D: single plane ---
    if arr.ndim == 3:
        result = _run_single(arr, **kwargs)
        if output_dir is not None:
            save_results(result, output_dir, prefix=prefix)
        return result

    raise ValueError(
        f"Expected 3-D (T, Y, X) or 4-D (T, Z, Y, X), got {arr.ndim}-D {arr.shape}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for command-line invocation."""
    parser = argparse.ArgumentParser(
        description="Detect active fibre ROIs in a timelapse fluorescence stack.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a single-plane timelapse TIFF file.",
    )
    parser.add_argument(
        "--fiber-thickness",
        type=int,
        default=3,
        help="Approximate neurite thickness in pixels (default: 3).",
    )
    parser.add_argument(
        "--percentile-threshold",
        type=float,
        default=95.0,
        help="Percentile for segmentation threshold (default: 95.0).",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=4,
        help="Minimum connected-component size in pixels (default: 4).",
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.2,
        help="(peak-mean)/mean ratio for ROI filtering (default: 0.2).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=1.0,
        help="Acquisition frame rate in Hz (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSV files (default: same as input).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="FiberActivity",
        help="Filename prefix for outputs (default: FiberActivity).",
    )
    return parser


def main(argv: list[str] | None = None) -> FiberActivityResult | list[FiberActivityResult]:
    """CLI entry-point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    result : FiberActivityResult or list[FiberActivityResult]
        The analysis result (also saved to disk when ``--output-dir`` is
        provided).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.path.parent

    return run(
        data=args.path,
        fiber_thickness=args.fiber_thickness,
        percentile_threshold=args.percentile_threshold,
        min_object_size=args.min_object_size,
        activity_threshold=args.activity_threshold,
        frame_rate=args.frame_rate,
        output_dir=output_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
