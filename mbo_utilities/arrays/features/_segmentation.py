"""
Segmentation features for arrays.

Provides compatibility between Cellpose (dense masks) and Suite2p (sparse stats).
"""

from __future__ import annotations

import numpy as np


def stat_to_masks(stat, Ly, Lx, label_map=None):
    """
    Convert Suite2p stat array to label image.

    Delegates to lbm_suite2p_python if available.

    Parameters
    ----------
    stat : list[dict]
        Suite2p statistics.
    Ly : int
        Y dimension.
    Lx : int
        X dimension.
    label_map : dict, optional
        Mapping from original labels to new labels.
        Note: label_map only supported in fallback implementation.

    Returns
    -------
    masks : np.ndarray (uint32)
        Label image.
    """
    # Delegate to lbm_suite2p_python if available (no label_map support there)
    if label_map is None:
        try:
            from lbm_suite2p_python.cellpose import stat_to_masks as lsp_stat_to_masks

            return lsp_stat_to_masks(stat, (Ly, Lx))
        except ImportError:
            pass

    # Fallback implementation (supports label_map)
    masks = np.zeros((Ly, Lx), dtype=np.uint32)
    for k, s in enumerate(stat):
        id_ = k + 1
        if label_map:
            if k not in label_map:
                continue
            id_ = label_map[k]

        ypix = s.get("ypix")
        xpix = s.get("xpix")

        if ypix is None or xpix is None:
            continue

        # Bounds check
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        masks[ypix[valid], xpix[valid]] = id_

    return masks


def masks_to_stat(masks: np.ndarray, img: np.ndarray = None) -> list[dict]:
    """
    Convert dense label image (masks) to Suite2p sparse 'stat' list.

    Delegates to lbm_suite2p_python if available (more comprehensive).

    Parameters
    ----------
    masks : np.ndarray
        2D or 3D label image.
    img : np.ndarray, optional
        Original image for computing intensity statistics.

    Returns
    -------
    list[dict]
        List of stat dicts compatible with Suite2p.
    """
    # Delegate to lbm_suite2p_python (more comprehensive implementation)
    try:
        from lbm_suite2p_python.cellpose import masks_to_stat as lsp_masks_to_stat

        return lsp_masks_to_stat(masks, img)
    except ImportError:
        pass

    # Fallback implementation (2D only, basic fields)
    if masks.ndim != 2:
        raise ValueError(
            "Fallback masks_to_stat only supports 2D. Install lbm_suite2p_python for 3D."
        )

    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]

    stat = []
    for label in unique_labels:
        ypix, xpix = np.nonzero(masks == label)

        if len(ypix) == 0:
            continue

        center_y = np.mean(ypix)
        center_x = np.mean(xpix)
        lam = np.ones(len(ypix), dtype=np.float32)

        roi = {
            "ypix": ypix,
            "xpix": xpix,
            "lam": lam,
            "med": [center_y, center_x],
            "npix": len(ypix),
        }
        stat.append(roi)

    return stat
