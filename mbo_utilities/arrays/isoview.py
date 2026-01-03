"""Lazy array loader for isoview lightsheet microscopy data."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline


logger = logging.getLogger(__name__)

# register isoview pipeline info
_ISOVIEW_INFO = PipelineInfo(
    name="isoview",
    description="Isoview lightsheet microscopy data",
    input_patterns=[
        "**/data_TM??????_SPM??.zarr",
        "**/SPM??_TM??????_CM??_CHN??.zarr",
        "**/TM??????/",
    ],
    output_patterns=[],
    input_extensions=["zarr"],
    output_extensions=[],
    marker_files=[],
    category="reader",
)
register_pipeline(_ISOVIEW_INFO)


class IsoviewArray:
    """
    Lazy loader for isoview lightsheet microscopy data.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities imread/imwrite
    and downstream processing pipelines.

    Supports both structures:
    - Consolidated (new): data_TM000000_SPM00.zarr/camera_0/0/
    - Separate (old): SPM00_TM000000_CM00_CHN01.zarr

    Shape:
    - Multi-timepoint: (T, Z, Views, Y, X) - 5D
    - Single timepoint: (Z, Views, Y, X) - 4D

    Views are (camera, channel) combinations that exist in the data.

    Parameters
    ----------
    path : str or Path
        Path to output directory containing TM* folders or single TM folder.

    Examples
    --------
    >>> arr = IsoviewArray("path/to/output")
    >>> arr.shape
    (10, 543, 4, 2048, 2048)  # (T, Z, Views, Y, X)
    >>> arr.views
    [(0, 0), (1, 0), (2, 1), (3, 1)]
    >>> frame = arr[0, 100, 0]  # timepoint 0, z=100, view 0
    """

    def __init__(self, path: str | Path):
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr>=3.0 required: pip install zarr")

        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # Detect if single TM or multi-TM
        if self.base_path.name.startswith("TM"):
            zarr_files_in_path = list(self.base_path.glob("*.zarr"))
            if zarr_files_in_path:
                self._single_timepoint = True
                self.tm_folders = [self.base_path]
            else:
                raise ValueError(f"TM folder {self.base_path} contains no .zarr files")
        else:
            self.tm_folders = sorted(
                [d for d in self.base_path.iterdir()
                 if d.is_dir() and d.name.startswith("TM")],
                key=lambda x: int(x.name[2:])
            )
            self._single_timepoint = False

            if not self.tm_folders:
                raise ValueError(f"No TM* folders found in {self.base_path}")

        # Detect structure type and discover views
        first_tm = self.tm_folders[0]
        self._detect_structure(first_tm)

        # Cache for opened zarr arrays: (t_idx, view_idx) -> zarr array
        self._zarr_cache = {}

    def _detect_structure(self, tm_folder: Path):
        """Detect consolidated vs separate structure and discover views."""
        import zarr

        # Check for consolidated .zarr files (contain camera_N subdirs)
        consolidated_zarrs = []
        for zf in tm_folder.glob("*.zarr"):
            try:
                z = zarr.open(zf, mode="r")
                if isinstance(z, zarr.Group):
                    if any(k.startswith("camera_") for k in z.group_keys()):
                        consolidated_zarrs.append(zf)
            except:
                continue

        if consolidated_zarrs:
            self._structure = "consolidated"
            self._discover_consolidated(tm_folder, consolidated_zarrs[0])
        else:
            self._structure = "separate"
            self._discover_separate(tm_folder)

        logger.info(
            f"IsoviewArray: structure={self._structure}, "
            f"timepoints={len(self.tm_folders)}, views={len(self._views)}"
        )

    def _discover_separate(self, tm_folder: Path):
        """Parse SPM00_TM000000_CM00_CHN01.zarr filenames."""
        import zarr

        zarr_files = sorted(tm_folder.glob("*.zarr"))
        if not zarr_files:
            raise ValueError(f"No .zarr files in {tm_folder}")

        self._views = []  # [(camera, channel), ...]

        for zf in zarr_files:
            name = zf.stem

            # Skip mask files
            if any(x in name for x in ["Mask", "mask", "coords"]):
                continue

            parts = name.split("_")
            cm_idx = chn_idx = None

            for part in parts:
                if part.startswith("CM"):
                    # Extract only digits after CM
                    cm_str = part[2:]
                    if cm_str.isdigit():
                        cm_idx = int(cm_str)
                elif part.startswith("CHN"):
                    # Extract only digits after CHN
                    chn_str = part[3:]
                    if chn_str.isdigit():
                        chn_idx = int(chn_str)

            if cm_idx is not None and chn_idx is not None:
                self._views.append((cm_idx, chn_idx))

        if not self._views:
            raise ValueError(f"No valid camera/channel combinations in {tm_folder}")

        # Open first valid (non-mask) file to get shape/dtype/metadata
        first_valid = None
        for zf in zarr_files:
            if not any(x in zf.stem for x in ["Mask", "mask", "coords"]):
                first_valid = zf
                break

        if first_valid is None:
            raise ValueError(f"No valid data files in {tm_folder}")

        first_z = zarr.open(first_valid, mode="r")
        if isinstance(first_z, zarr.Group):
            if "0" in first_z:
                first_arr = first_z["0"]
            else:
                raise ValueError(f"OME-Zarr group missing '0' array: {zarr_files[0]}")
            self._zarr_attrs = dict(first_z.attrs)
        else:
            first_arr = first_z
            self._zarr_attrs = dict(first_arr.attrs) if hasattr(first_arr, "attrs") else {}

        self._single_shape = first_arr.shape  # (Z, Y, X)
        self._dtype = first_arr.dtype
        self._consolidated_path = None

    def _discover_consolidated(self, tm_folder: Path, consolidated_zarr: Path):
        """Parse camera_N subgroups from consolidated zarr."""
        import zarr

        self._consolidated_path = consolidated_zarr
        z = zarr.open(consolidated_zarr, mode="r")

        # Find all camera_N groups
        camera_groups = sorted(
            [k for k in z.group_keys() if k.startswith("camera_")],
            key=lambda x: int(x.split("_")[1])
        )

        if not camera_groups:
            raise ValueError(f"No camera_N groups in {consolidated_zarr}")

        # For now, assume each camera has one channel (channel 0)
        # TODO: detect multiple channels per camera from metadata
        self._views = []
        for cam_group in camera_groups:
            cam_idx = int(cam_group.split("_")[1])
            self._views.append((cam_idx, 0))

        # Get shape/dtype from first camera
        first_arr = z[f"{camera_groups[0]}/0"]
        self._single_shape = first_arr.shape  # (Z, Y, X)
        self._dtype = first_arr.dtype

        # Store root attrs as metadata
        self._zarr_attrs = dict(z.attrs)

    def _get_zarr(self, t_idx: int, view_idx: int):
        """Get or open a zarr array for timepoint and view index."""
        import zarr

        cache_key = (t_idx, view_idx)
        if cache_key in self._zarr_cache:
            return self._zarr_cache[cache_key]

        camera, channel = self._views[view_idx]
        tm_folder = self.tm_folders[t_idx]

        if self._structure == "consolidated":
            # Find consolidated zarr in this TM folder
            zarr_files = []
            for zf in tm_folder.glob("*.zarr"):
                try:
                    z = zarr.open(zf, mode="r")
                    if isinstance(z, zarr.Group):
                        if any(k.startswith("camera_") for k in z.group_keys()):
                            zarr_files.append(zf)
                except:
                    continue

            if not zarr_files:
                raise FileNotFoundError(f"No consolidated zarr in {tm_folder}")

            z = zarr.open(zarr_files[0], mode="r")
            arr = z[f"camera_{camera}/0"]

        else:  # separate
            pattern = f"*_CM{camera:02d}_CHN{channel:02d}.zarr"
            matches = list(tm_folder.glob(pattern))

            if not matches:
                raise FileNotFoundError(f"No {pattern} in {tm_folder}")

            z = zarr.open(matches[0], mode="r")
            if isinstance(z, zarr.Group):
                if "0" in z:
                    arr = z["0"]
                else:
                    raise ValueError(f"OME-Zarr group missing '0' array: {matches[0]}")
            else:
                arr = z

        self._zarr_cache[cache_key] = arr
        return arr

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape.

        Returns
        -------
        - Single TM: (Z, Views, Y, X) - 4D
        - Multi TM: (T, Z, Views, Y, X) - 5D
        """
        z, y, x = self._single_shape
        if self._single_timepoint:
            return (z, len(self._views), y, x)
        return (len(self.tm_folders), z, len(self._views), y, x)

    @property
    def dtype(self):
        """Array data type."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return 4 if self._single_timepoint else 5

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    @property
    def min(self) -> float:
        """
        Minimum value across all data.

        For consolidated structure, uses min_intensity from camera metadata.
        Otherwise computes lazily from first view.
        """
        if self._structure == "consolidated" and hasattr(self, "_consolidated_path"):
            import zarr
            z = zarr.open(self._consolidated_path, mode="r")
            # Get min from first camera metadata
            for cam_group in z.group_keys():
                if cam_group.startswith("camera_"):
                    cam_attrs = dict(z[cam_group].attrs)
                    if "min_intensity" in cam_attrs:
                        return float(cam_attrs["min_intensity"])

        # Fallback: compute from first view
        first_arr = self._get_zarr(0, 0)
        return float(np.min(first_arr))

    @property
    def max(self) -> float:
        """
        Maximum value across all data.

        Computed lazily from first view.
        """
        first_arr = self._get_zarr(0, 0)
        return float(np.max(first_arr))

    @property
    def metadata(self) -> dict:
        """
        Return metadata as dict. Always returns dict, never None.

        Contains standard keys for Suite2p compatibility:
        - nframes: number of frames (timepoints)
        - num_frames: alias for nframes
        - Ly: height in pixels
        - Lx: width in pixels
        - nplanes: number of z-planes
        - dx, dy, dz: voxel size in micrometers
        - fs: frame rate in Hz

        Plus isoview-specific fields:
        - num_timepoints, views, shape, structure
        - cameras: per-camera metadata dict
        """
        meta = dict(self._zarr_attrs) if self._zarr_attrs else {}

        # map isoview keys to canonical metadata keys
        # pixel resolution: pixel_resolution_um -> dx, dy
        px_res = meta.get("pixel_resolution_um")
        if px_res is not None:
            meta["dx"] = float(px_res)
            meta["dy"] = float(px_res)

        # z step: z_step -> dz
        z_step = meta.get("z_step")
        if z_step is not None:
            meta["dz"] = float(z_step)

        # frame rate: fps -> fs
        fps = meta.get("fps")
        if fps is not None:
            meta["fs"] = float(fps)

        # LazyArrayProtocol required fields
        meta["num_timepoints"] = len(self.tm_folders)
        meta["nframes"] = len(self.tm_folders)  # suite2p alias
        meta["num_frames"] = len(self.tm_folders)  # legacy alias
        meta["Ly"] = self._single_shape[1]
        meta["Lx"] = self._single_shape[2]

        # z-planes from shape (not timepoints!)
        meta["nplanes"] = self._single_shape[0]
        meta["num_planes"] = self._single_shape[0]

        # isoview-specific fields
        meta["views"] = self._views
        meta["shape"] = self.shape
        meta["structure"] = self._structure
        meta["single_timepoint"] = self._single_timepoint

        # add per-camera metadata
        cam_meta = self.camera_metadata
        if cam_meta:
            meta["cameras"] = cam_meta
            # aggregate per-camera values for display
            for key in ["zplanes", "min_intensity", "illumination_arms", "vps"]:
                values = [cm.get(key) for cm in cam_meta.values() if cm.get(key) is not None]
                if values:
                    # if all same, use single value, otherwise use list
                    if len({str(v) for v in values}) == 1:
                        meta[key] = values[0]
                    else:
                        meta[key] = values

        return meta

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._zarr_attrs.update(value)

    @property
    def camera_metadata(self) -> dict[int, dict]:
        """
        Per-camera metadata (only for consolidated structure).

        Returns dict mapping camera index to camera-specific metadata.
        """
        if self._structure != "consolidated":
            return {}

        import zarr
        z = zarr.open(self._consolidated_path, mode="r")
        cam_meta = {}

        for cam_group in z.group_keys():
            if cam_group.startswith("camera_"):
                cam_idx = int(cam_group.split("_")[1])
                cam_meta[cam_idx] = dict(z[cam_group].attrs)

        return cam_meta

    @property
    def views(self) -> list[tuple[int, int]]:
        """List of (camera, channel) tuples for each view index."""
        return self._views

    @property
    def num_views(self) -> int:
        """Number of camera/channel views."""
        return len(self._views)

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints."""
        return len(self.tm_folders)

    def __len__(self) -> int:
        """Length is first dimension (T or Z depending on structure)."""
        return self.shape[0]

    def view_index(self, camera: int, channel: int) -> int:
        """Get view index for a specific camera/channel combination."""
        try:
            return self._views.index((camera, channel))
        except ValueError:
            raise ValueError(
                f"No view for camera={camera}, channel={channel}. "
                f"Available views: {self._views}"
            )

    def __getitem__(self, key):
        """
        Index the array.

        Shape:
        - Single TM: (Z, Views, Y, X) - 4D
        - Multi TM: (T, Z, Views, Y, X) - 5D

        Supports integers, slices, lists, and combinations.
        """
        if not isinstance(key, tuple):
            key = (key,)

        def to_indices(k, max_val):
            """Convert key to list of indices."""
            if isinstance(k, int):
                if k < 0:
                    k = max_val + k
                return [k]
            if isinstance(k, slice):
                return list(range(*k.indices(max_val)))
            if isinstance(k, (list, np.ndarray)):
                return list(k)
            return list(range(max_val))

        if self._single_timepoint:
            # 4D indexing: (Z, Views, Y, X)
            key = key + (slice(None),) * (4 - len(key))
            z_key, view_key, y_key, x_key = key
            t_indices = [0]
            t_key = 0
        else:
            # 5D indexing: (T, Z, Views, Y, X)
            key = key + (slice(None),) * (5 - len(key))
            t_key, z_key, view_key, y_key, x_key = key
            t_indices = to_indices(t_key, len(self.tm_folders))

        z_indices = to_indices(z_key, self._single_shape[0])
        view_indices = to_indices(view_key, len(self._views))

        # Build output array (always 5D internally)
        out_shape = (
            len(t_indices),
            len(z_indices),
            len(view_indices),
            *self._single_shape[1:],  # Y, X
        )

        # Handle Y, X slicing
        if isinstance(y_key, int):
            out_shape = (*out_shape[:3], 1, *out_shape[4:])
        elif isinstance(y_key, slice):
            y_size = len(range(*y_key.indices(self._single_shape[1])))
            out_shape = (*out_shape[:3], y_size, *out_shape[4:])

        if isinstance(x_key, int):
            out_shape = (*out_shape[:4], 1)
        elif isinstance(x_key, slice):
            x_size = len(range(*x_key.indices(self._single_shape[2])))
            out_shape = (*out_shape[:4], x_size)

        result = np.empty(out_shape, dtype=self._dtype)

        for ti, t_idx in enumerate(t_indices):
            for vi, view_idx in enumerate(view_indices):
                zarr_arr = self._get_zarr(t_idx, view_idx)

                # Index the zarr array (Z, Y, X)
                data = zarr_arr[z_key, y_key, x_key]

                # Handle dimension reduction from integer indexing
                if isinstance(z_key, int):
                    data = data[np.newaxis, ...]
                if isinstance(y_key, int):
                    data = data[:, np.newaxis, :]
                if isinstance(x_key, int):
                    data = data[:, :, np.newaxis]

                result[ti, :, vi, ...] = data

        # Squeeze out singleton dimensions from integer indexing
        if self._single_timepoint:
            # Always squeeze out T dimension
            result = np.squeeze(result, axis=0)
            int_indexed = [
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
            # Squeeze in reverse order
            for ax in range(3, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)
        else:
            int_indexed = [
                isinstance(t_key, int),
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
            # Squeeze in reverse order
            for ax in range(4, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)

        return result

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory."""
        return self[:]

    def get_labels(self, timepoint: int, camera: int,
                   label_type: str = "segmentation") -> np.ndarray:
        """
        Access labels from consolidated structure.

        Args:
            timepoint: Timepoint index
            camera: Camera index
            label_type: 'segmentation', 'xy_coords', 'xz_coords'

        Returns
        -------
            Label array (Z, Y, X)
        """
        if self._structure != "consolidated":
            raise NotImplementedError("Labels only available in consolidated structure")

        import zarr

        tm_folder = self.tm_folders[timepoint]
        zarr_files = []
        for zf in tm_folder.glob("*.zarr"):
            try:
                z = zarr.open(zf, mode="r")
                if isinstance(z, zarr.Group):
                    if any(k.startswith("camera_") for k in z.group_keys()):
                        zarr_files.append(zf)
            except:
                continue

        if not zarr_files:
            raise FileNotFoundError(f"No consolidated zarr in {tm_folder}")

        z = zarr.open(zarr_files[0], mode="r")
        return z[f"camera_{camera}/labels/{label_type}/0"][:]

    def get_projection(self, timepoint: int, camera: int,
                      proj_type: str = "xy") -> np.ndarray:
        """
        Access projections from consolidated structure.

        Args:
            timepoint: Timepoint index
            camera: Camera index
            proj_type: 'xy', 'xz', 'yz'

        Returns
        -------
            Projection array
        """
        if self._structure != "consolidated":
            raise NotImplementedError("Projections only in consolidated structure")

        import zarr

        tm_folder = self.tm_folders[timepoint]
        zarr_files = []
        for zf in tm_folder.glob("*.zarr"):
            try:
                z = zarr.open(zf, mode="r")
                if isinstance(z, zarr.Group):
                    if any(k.startswith("camera_") for k in z.group_keys()):
                        zarr_files.append(zf)
            except:
                continue

        if not zarr_files:
            raise FileNotFoundError(f"No consolidated zarr in {tm_folder}")

        z = zarr.open(zarr_files[0], mode="r")
        return z[f"camera_{camera}/projections/{proj_type}/0"][:]

    @property
    def filenames(self) -> list[Path]:
        """
        Source file paths for LazyArrayProtocol.

        Returns
        -------
        list[Path]
            List of TM folder paths.
        """
        return list(self.tm_folders)

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Dimension labels for LazyArrayProtocol.

        Returns
        -------
        tuple[str, ...]
            ('Z', 'V', 'Y', 'X') for single timepoint
            ('T', 'Z', 'V', 'Y', 'X') for multi-timepoint
            V = Views (camera/channel combinations)
        """
        if self._single_timepoint:
            return ("Z", "V", "Y", "X")
        return ("T", "Z", "V", "Y", "X")

    @property
    def num_planes(self) -> int:
        """
        Number of Z-planes for LazyArrayProtocol.

        Returns
        -------
        int
            Number of Z slices in each volume.
        """
        return self._single_shape[0]

    def close(self) -> None:
        """Release resources (clear zarr cache)."""
        self._zarr_cache.clear()

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite: bool = False,
        target_chunk_mb: int = 50,
        ext: str = ".tiff",
        progress_callback=None,
        debug: bool = False,
        planes: list[int] | int | None = None,
        **kwargs,
    ):
        """(WIP) Write IsoviewArray to disk."""
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"IsoviewArray(shape={self.shape}, dtype={self.dtype}, "
            f"views={self._views}, structure={self._structure})"
        )
