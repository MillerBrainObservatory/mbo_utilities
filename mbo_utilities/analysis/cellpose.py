"""
cellpose utilities for saving, loading, and GUI interaction.

compatible with cellpose gui and lbm-suite2p-python pipeline.
"""

from pathlib import Path
from typing import Union
import numpy as np

from mbo_utilities._parsing import _make_json_serializable


def save_results(
    save_path: Union[str, Path],
    masks: np.ndarray,
    image: np.ndarray,
    flows: tuple = None,
    styles: np.ndarray = None,
    diameter: float = None,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    name: str = None,
) -> Path:
    """
    save cellpose results in gui-compatible format.

    creates:
    - {name}_seg.npy: cellpose gui format (can be loaded directly)
    - {name}_masks.tif: label image viewable in imagej/napari
    - {name}_stat.npy: suite2p-compatible roi statistics

    parameters
    ----------
    save_path : str or Path
        directory to save results.
    masks : ndarray
        labeled mask array from cellpose (0=background, 1,2,...=roi ids).
    image : ndarray
        image used for segmentation (projection).
    flows : tuple, optional
        flow outputs from cellpose model.eval().
    styles : ndarray, optional
        style vector from cellpose.
    diameter : float, optional
        cell diameter used for segmentation.
    cellprob_threshold : float
        cellprob threshold used.
    flow_threshold : float
        flow threshold used.
    name : str, optional
        base name for output files. defaults to 'cellpose'.

    returns
    -------
    Path
        path to the _seg.npy file (for gui loading).
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    name = name or "cellpose"

    n_rois = int(masks.max())

    # cellpose gui format: _seg.npy
    seg_data = {
        "img": image.astype(np.float32),
        "masks": masks.astype(np.uint32),
        "outlines": _masks_to_outlines(masks),
        "chan_choose": [0, 0],
        "ismanual": np.zeros(n_rois, dtype=bool),
        "filename": str(save_path / f"{name}.tif"),
        "flows": flows,
        "est_diam": diameter,
        "cellprob_threshold": cellprob_threshold,
        "flow_threshold": flow_threshold,
    }
    seg_file = save_path / f"{name}_seg.npy"
    np.save(seg_file, seg_data, allow_pickle=True)

    # save image as tiff for reference
    try:
        import tifffile
        tifffile.imwrite(
            save_path / f"{name}.tif",
            image.astype(np.float32),
            compression="zlib",
        )
        tifffile.imwrite(
            save_path / f"{name}_masks.tif",
            masks.astype(np.uint16),
            compression="zlib",
        )
    except ImportError:
        pass

    # suite2p-compatible stat
    stat = masks_to_stat(masks, image)
    np.save(save_path / f"{name}_stat.npy", stat, allow_pickle=True)

    # iscell array (all accepted)
    iscell = np.ones((n_rois, 2), dtype=np.float32)
    np.save(save_path / f"{name}_iscell.npy", iscell)

    print(f"saved {n_rois} rois to {save_path}")
    return seg_file


def load_results(seg_path: Union[str, Path]) -> dict:
    """
    load cellpose results from _seg.npy file.

    parameters
    ----------
    seg_path : str or Path
        path to _seg.npy file or directory containing it.

    returns
    -------
    dict
        dictionary with 'masks', 'img', 'flows', 'outlines', etc.
    """
    seg_path = Path(seg_path)

    # if directory, find _seg.npy file
    if seg_path.is_dir():
        seg_files = list(seg_path.glob("*_seg.npy"))
        if not seg_files:
            raise FileNotFoundError(f"no _seg.npy files in {seg_path}")
        seg_path = seg_files[0]

    data = np.load(seg_path, allow_pickle=True).item()
    return data


def open_in_gui(
    seg_path: Union[str, Path] = None,
    image: np.ndarray = None,
    masks: np.ndarray = None,
):
    """
    open cellpose gui with results or image.

    parameters
    ----------
    seg_path : str or Path, optional
        path to _seg.npy file to load in gui.
    image : ndarray, optional
        image to open directly (without loading from file).
    masks : ndarray, optional
        masks to overlay (requires image).

    notes
    -----
    requires cellpose to be installed with gui dependencies.
    """
    from cellpose.gui import gui

    if seg_path is not None:
        seg_path = Path(seg_path)
        if seg_path.is_dir():
            seg_files = list(seg_path.glob("*_seg.npy"))
            if seg_files:
                seg_path = seg_files[0]

        # gui expects the image file, it will load _seg.npy automatically
        data = load_results(seg_path)
        img_file = data.get("filename")
        if img_file and Path(img_file).exists():
            gui.run(image=str(img_file))
        else:
            # fallback: save temp image and open
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
                import tifffile
                tifffile.imwrite(f.name, data["img"].astype(np.float32))
                gui.run(image=f.name)
    elif image is not None:
        # save to temp and open
        import tempfile
        import tifffile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tifffile.imwrite(f.name, image.astype(np.float32))
            if masks is not None:
                # save masks as _seg.npy
                seg_data = {
                    "img": image.astype(np.float32),
                    "masks": masks.astype(np.uint32),
                    "outlines": _masks_to_outlines(masks),
                    "chan_choose": [0, 0],
                    "ismanual": np.zeros(int(masks.max()), dtype=bool),
                    "filename": f.name,
                    "flows": None,
                }
                seg_file = f.name.replace(".tif", "_seg.npy")
                np.save(seg_file, seg_data, allow_pickle=True)
            gui.run(image=f.name)
    else:
        # open empty gui
        gui.run()


def masks_to_stat(masks: np.ndarray, image: np.ndarray = None) -> np.ndarray:
    """
    convert cellpose masks to suite2p stat array.

    parameters
    ----------
    masks : ndarray
        2d or 3d label image (0=background, 1,2,...=roi ids).
    image : ndarray, optional
        original image for intensity statistics.

    returns
    -------
    ndarray
        array of stat dictionaries compatible with suite2p.
    """
    stat = []
    n_rois = int(masks.max())

    for roi_id in range(1, n_rois + 1):
        roi_mask = masks == roi_id
        if not roi_mask.any():
            continue

        if masks.ndim == 2:
            ypix, xpix = np.where(roi_mask)
            zpix = None
        else:
            zpix, ypix, xpix = np.where(roi_mask)

        npix = len(xpix)
        med_y = np.median(ypix)
        med_x = np.median(xpix)

        y_range = ypix.max() - ypix.min() + 1
        x_range = xpix.max() - xpix.min() + 1
        aspect = max(y_range, x_range) / max(1, min(y_range, x_range))
        radius = np.sqrt(npix / np.pi)

        roi_stat = {
            "ypix": ypix.astype(np.int32),
            "xpix": xpix.astype(np.int32),
            "npix": npix,
            "med": [med_y, med_x],
            "radius": radius,
            "aspect_ratio": aspect,
            "compact": npix / (np.pi * radius**2) if radius > 0 else 0,
        }

        if zpix is not None:
            roi_stat["zpix"] = zpix.astype(np.int32)
            roi_stat["med_z"] = np.median(zpix)

        if image is not None:
            if image.ndim == 2:
                vals = image[ypix, xpix]
            else:
                vals = image[zpix, ypix, xpix] if zpix is not None else image[ypix, xpix]
            roi_stat["mean_intensity"] = float(np.mean(vals))
            roi_stat["max_intensity"] = float(np.max(vals))

        stat.append(roi_stat)

    return np.array(stat, dtype=object)


def stat_to_masks(stat: np.ndarray, shape: tuple) -> np.ndarray:
    """
    convert suite2p stat array back to label mask.

    parameters
    ----------
    stat : ndarray
        array of stat dictionaries from suite2p.
    shape : tuple
        output shape (Y, X) or (Z, Y, X).

    returns
    -------
    ndarray
        label mask (0=background, 1,2,...=roi ids).
    """
    masks = np.zeros(shape, dtype=np.uint32)

    for roi_id, s in enumerate(stat, start=1):
        ypix = s["ypix"]
        xpix = s["xpix"]
        if "zpix" in s and len(shape) == 3:
            zpix = s["zpix"]
            masks[zpix, ypix, xpix] = roi_id
        else:
            masks[ypix, xpix] = roi_id

    return masks


def _masks_to_outlines(masks: np.ndarray) -> np.ndarray:
    """extract outlines from label mask."""
    from scipy import ndimage

    outlines = np.zeros_like(masks, dtype=bool)

    for roi_id in range(1, masks.max() + 1):
        roi_mask = masks == roi_id
        # dilate and subtract to get boundary
        dilated = ndimage.binary_dilation(roi_mask)
        boundary = dilated & ~roi_mask
        outlines |= boundary

    return outlines


def save_comparison(
    save_path: Union[str, Path],
    results: dict,
    base_name: str = "comparison",
):
    """
    save multiple cellpose results for comparison.

    parameters
    ----------
    save_path : str or Path
        directory to save results.
    results : dict
        dictionary mapping method names to dicts with 'masks', 'proj', 'n_cells'.
        format from the max_projection_optimization notebook.
    base_name : str
        base name for output files.

    example
    -------
    >>> save_comparison(
    ...     "output/",
    ...     {
    ...         "max": {"masks": masks1, "proj": proj1, "n_cells": 100},
    ...         "p99": {"masks": masks2, "proj": proj2, "n_cells": 120},
    ...     }
    ... )
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    summary = []

    for method_name, data in results.items():
        masks = data["masks"]
        proj = data["proj"]
        n_cells = data.get("n_cells", int(masks.max()))

        # sanitize method name for filename
        safe_name = method_name.replace(" ", "_").replace("+", "_")

        # save in gui format
        save_results(
            save_path,
            masks=masks,
            image=proj,
            name=f"{base_name}_{safe_name}",
        )

        summary.append({
            "method": method_name,
            "n_cells": n_cells,
            "file": f"{base_name}_{safe_name}_seg.npy",
        })

    # save summary
    import json
    with open(save_path / f"{base_name}_summary.json", "w") as f:
        json.dump(_make_json_serializable(summary), f, indent=2)

    print(f"saved {len(results)} comparisons to {save_path}")
    return save_path
