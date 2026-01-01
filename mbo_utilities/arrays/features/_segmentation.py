"""
Segmentation features for arrays.

Provides compatibility between Cellpose (dense masks) and Suite2p (sparse stats).
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mbo_utilities import log

logger = log.get("arrays.features.segmentation")


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


class SegmentationMixin:
    """
    Mixin adding segmentation capabilities (Cellpose/Suite2p).

    Bridge between dense masks (Cellpose style) and sparse stats (Suite2p style).
    """

    def detect_cellpose(
        self,
        diameter: float | None = None,
        model_type: str = "cyto3",
        channels: list[int] = [0, 0],
        flow_threshold: float = 0.4,
        batch_size: int = 8,
        time_index: int | None = 0,
        z_index: int | None = None,
        do_3d: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, list, list, list]:
        """
        Run Cellpose detection.

        Parameters
        ----------
        diameter : float
            Expected diameter. If None, auto-detected.
        model_type : str
            Cellpose model type ('cyto', 'cyto3', 'nuclei', etc).
        channels : list
            [cytoplasm, nucleus] channels. 0=grayscale.
        flow_threshold : float
            Flow error threshold.
        time_index : int
             Time point to use for detection (if T axis exists).
             If None, uses mean/max projection? (Not impl yet, user must process input first usually)
             Here we default to frame 0 or user specified.
        z_index : int
             Z plane index (if Z axis exists).

        Returns
        -------
        masks, flows, styles, diams
        """
        try:
            from cellpose import models
        except ImportError:
            raise ImportError(
                "cellpose not installed. Install with `uv pip install cellpose`"
            )

        # Prepare input image
        # We need a 2D or 3D image (Ly, Lx) or (Z, Ly, Lx)
        # Handle slicing info from self.dims

        sl = [slice(None)] * self.ndim

        if self.has_dim("T"):
            t_idx = self.dim_index("T")
            sl[t_idx] = time_index if time_index is not None else 0  # Default T=0

        if self.has_dim("Z") and not do_3d:
            # If we are doing 2D segmentation on specific plane
            if z_index is not None:
                z_idx = self.dim_index("Z")
                sl[z_idx] = z_index
            else:
                # If z_index not specified but we have Z, maybe we want to run on all?
                # But this method usually returns one result.
                # Let's assume user wants specific plane or it's a 3D Z-stack?
                # If do_3d=False and Z exists, warning or handle?
                pass

        img = self[tuple(sl)]

        # Ensure array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Squeeze singleton dims
        img = np.squeeze(img)

        model = models.Cellpose(model_type=model_type, gpu=True)
        masks, flows, styles, diams = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            batch_size=batch_size,
            do_3d=do_3d,
            **kwargs,
        )

        return masks, flows, styles, diams

    def detect_suite2p(
        self,
        ops: dict,
        time_index: int | None = 0,
    ) -> list[dict]:
        """
        Run Suite2p detection (extract_rois).

        Parameters
        ----------
        ops : dict
            Suite2p ops. Must contain 'diameter' or 'diam_prctl' usually?
            Actually 'diameter' in ops.

        Returns
        -------
        stat : list[dict]
            Sparse ROIs.
        """
        try:
            from suite2p.detection import detection
            from suite2p import default_ops
        except ImportError:
            raise ImportError("suite2p not installed")

        # Suite2p expects a binary file or raw frames?
        # Actually detection.detection() runs on a binary file usually via OPS?
        # Or can we pass an image?
        # detection.detection puts results in ops['stat']?
        # Wait, suite2p.detection.detect() might be lower level.
        # Let's look at how typical run_s2p handles it. It calls `detection_wrapper`.

        # Simplest path: Use the image as "Reference" and detect spots?
        # Or do we want full pipeline detection?
        # If user wants "functional inputs/outputs", likely passing a mean img or max img?
        # Cellpose interface is: Image -> Masks.
        # Suite2p interface for just detection: Image -> Stat?

        # suite2p.detection.sourcery class or similar?
        # Actually suite2p detection is complex and tied to Data.
        # If we just want "anatomical" detection using suite2p's algo on a summary image:
        # `suite2p.detection.detect(img=...)`?

        # Let's assume for now we provide the conversion utilities primarily.
        # Implementing full Suite2p detection on arbitrary array is complex.
        # But we can provide the `stat_to_masks` and inputs.

        logger.warning(
            "Suite2p detection from array not fully implemented. Use full run_s2p pipeline usually."
        )
        return []

    def save_segmentation(
        self,
        output_path: str,
        masks: np.ndarray | None = None,
        stat: list[dict] | None = None,
        name: str = "0",
        overwrite: bool = True,
    ) -> str:
        """
        Save segmentation to OME-Zarr labels.

        Parameters
        ----------
        output_path : str
            Path to Zarr group (root).
        masks : np.ndarray
            Dense labels.
        stat : list[dict]
            Sparse labels (will be converted).
        name : str
            Name of label dataset (default '0').
        """
        import zarr
        from zarr.codecs import BytesCodec, GzipCodec

        # Resolve masks
        if masks is None and stat is not None:
            # Need dims. Use self params?
            # Assuming stat matches THIS array's spatial dims.
            y_idx = self.dim_index("Y")
            x_idx = self.dim_index("X")
            Ly, Lx = self.shape[y_idx], self.shape[x_idx]
            masks = stat_to_masks(stat, Ly=Ly, Lx=Lx)

        if masks is None:
            raise ValueError("Must provide masks or stat")

        # Ensure appropriate shape/type
        masks = masks.astype(np.uint32)

        # Open root
        logging.info(f"Saving segmentation to {output_path} group 'labels/{name}'")

        # We assume output_path is a Zarr store/group
        if isinstance(output_path, (str, Path)):
            root = zarr.open(str(output_path), mode="a")  # Append/Edit
        else:
            root = output_path

        labels_grp = root.require_group("labels")
        print(f"DEBUG: Zarr version: {zarr.__version__}")

        # Resolve store
        if hasattr(root, "store"):
            store = root.store
        elif hasattr(labels_grp, "store"):
            store = labels_grp.store
        else:
            # Fallback for V3 if store is hidden or we have path
            store = output_path

        full_path = f"labels/{name}"

        # Handle overwrite manually if needed (zarr.create supports it usually)
        if name in labels_grp and overwrite:
            del labels_grp[name]

        if hasattr(zarr, "open_array"):
            ds = zarr.open_array(
                store=store,
                path=full_path,
                mode="w",
                shape=masks.shape,
                dtype=masks.dtype,
                chunks=masks.shape,
            )
        else:
            # Fallback for older zarr or if open_array is missing
            ds = zarr.create(
                store=store,
                path=full_path,
                shape=masks.shape,
                dtype=masks.dtype,
                chunks=masks.shape,
                overwrite=True,
            )

        ds[:] = masks

        # Metadata
        labels_grp.attrs["labels"] = list(
            set(labels_grp.attrs.get("labels", []) + [name])
        )

        ds.attrs["image-label"] = {
            "version": "0.4",
            "source": {"image": "../../"},  # Relative path to image
        }

        return str(output_path)

    def export_to_cellpose(
        self,
        output_dir: str,
        name: str = "image",
        masks: np.ndarray | None = None,
        stat: list[dict] | None = None,
        time_index: int = 0,
        z_index: int | None = None,
    ) -> tuple[str, str]:
        """
        Export image and masks to Cellpose GUI compliant files (TIFF/NPY).

        Creates:
        - {output_dir}/{name}.tif (Image)
        - {output_dir}/{name}_masks.tif (Labels)

        Parameters
        ----------
        output_dir : str
            Directory to save files.
        name : str
            Base filename.
        masks : np.ndarray
            Masks to save. If None, tries to convert stat.
        stat : list[dict]
            Suite2p stats to convert if masks is None.
        time_index : int
            Time index for image export.
        z_index : int
            Z index for image export.

        Returns
        -------
        img_path, mask_path
        """
        import tifffile
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # 1. Prepare Image
        # Similar logic to detect_cellpose slicing
        sl = [slice(None)] * self.ndim
        if self.has_dim("T"):
            sl[self.dim_index("T")] = time_index
        if self.has_dim("Z") and self.dim_index("Z") < self.ndim:
            # Check if we should slice Z?
            # For 2D export, yes.
            if z_index is not None:
                sl[self.dim_index("Z")] = z_index

        img = self[tuple(sl)]
        if isinstance(img, np.ndarray) is False:
            img = np.array(img)
        img = np.squeeze(img)

        img_path = output_dir / f"{name}.tif"
        tifffile.imwrite(img_path, img)
        msg_img = f"Saved image to {img_path}"

        # 2. Prepare Masks
        if masks is None and stat is not None:
            Ly, Lx = img.shape[-2], img.shape[-1]
            masks = stat_to_masks(stat, Ly, Lx)

        mask_path = None
        if masks is not None:
            # Cellpose expects {name}_masks.tif
            mask_path = output_dir / f"{name}_masks.tif"
            tifffile.imwrite(
                mask_path, masks.astype(np.uint16)
            )  # Cellpose likes uint16 usually

            # Also optionally _seg.npy but masks.tif is simpler for "File -> Load Masks"

        logger.info(f"Exported for Cellpose: {img_path} {mask_path}")
        return str(img_path), str(mask_path)
