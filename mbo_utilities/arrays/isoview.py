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
        "**/SPC??_TM?????_ANG???_CM?_CHN??_PH?.stack",
    ],
    output_patterns=[],
    input_extensions=["zarr", "stack"],
    output_extensions=[],
    marker_files=["ch00_spec00.xml", "ch0.xml"],
    category="reader",
)
register_pipeline(_ISOVIEW_INFO)


def _parse_isoview_xml(xml_path: Path) -> dict:
    """Parse isoview XML metadata file for dimensions and camera info."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {}

    # find info elements
    for info in root.iter("info"):
        attribs = info.attrib

        if "dimensions" in attribs:
            # format: "1848x768x38" or "1848x768x38, 1848x768x38" for multi-camera
            dims_str = attribs["dimensions"]
            camera_dims = []
            for cam_dims in dims_str.split(","):
                parts = [int(x) for x in cam_dims.strip().split("x")]
                if len(parts) == 3:
                    # xml is (height, width, depth) -> we want (height, width, depth)
                    camera_dims.append(tuple(parts))
            metadata["dimensions"] = camera_dims

        if "z_step" in attribs:
            metadata["z_step"] = float(attribs["z_step"])

        if "exposure_time" in attribs:
            metadata["exposure_time"] = float(attribs["exposure_time"])

        if "detection_objective" in attribs:
            metadata["detection_objective"] = attribs["detection_objective"]

        if "specimen_name" in attribs:
            metadata["specimen_name"] = attribs["specimen_name"]

        if "timestamp" in attribs:
            metadata["timestamp"] = attribs["timestamp"]

    return metadata


def _find_isoview_xml(base_path: Path) -> Path | None:
    """Find XML metadata file in isoview raw data directory."""
    # try common patterns
    patterns = ["ch00_spec00.xml", "ch0.xml", "ch*.xml", "*.xml"]
    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            return matches[0]
    return None


class IsoviewArray:
    """
    Lazy loader for isoview lightsheet microscopy data.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities imread/imwrite
    and downstream processing pipelines.

    Supports three structures:
    - Raw (input): SPC00_TM00000_ANG000_CM0_CHN01_PH0.stack (binary uint16)
    - Consolidated (new): data_TM000000_SPM00.zarr/camera_0/0/
    - Separate (old): SPM00_TM000000_CM00_CHN01.zarr

    Shape:
    - Multi-timepoint: (T, Z, Views, Y, X) - 5D
    - Single timepoint: (Z, Views, Y, X) - 4D

    Views are (camera, channel) combinations that exist in the data.

    Parameters
    ----------
    path : str or Path
        Path to directory containing .stack files or TM* folders with .zarr files.

    Examples
    --------
    >>> arr = IsoviewArray("path/to/raw_data")  # with .stack files
    >>> arr.shape
    (61, 38, 4, 1848, 768)  # (t, z, cm, y, x)
    >>> arr.dims
    ('t', 'z', 'cm', 'y', 'x')
    >>> arr.views
    [(0, 1), (1, 1), (2, 0), (3, 0)]  # (camera, channel) per view
    >>> frame = arr[0, 10, 0]  # t=0, z=10, cm=0
    """

    def __init__(self, path: str | Path):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # check for raw .stack files first
        stack_files = list(self.base_path.glob("*.stack"))
        if stack_files:
            self._structure = "raw"
            self._discover_raw(self.base_path)
            self._zarr_cache = {}
            return

        # otherwise check for zarr-based structures
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr>=3.0 required for zarr files: pip install zarr")

        # detect if single TM or multi-TM
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
                raise ValueError(f"No TM* folders or .stack files found in {self.base_path}")

        # detect structure type and discover views
        first_tm = self.tm_folders[0]
        self._detect_structure(first_tm)

        # cache for opened zarr arrays: (t_idx, view_idx) -> zarr array
        self._zarr_cache = {}

    def _discover_raw(self, base_path: Path):
        """Discover raw .stack files and parse metadata from XML.

        Filename pattern: SPC{specimen}_TM{time}_ANG{angle}_CM{camera}_CHN{channel}_PH{phase}.stack
        """
        import re

        # find xml for dimensions
        xml_path = _find_isoview_xml(base_path)
        if xml_path is None:
            raise ValueError(f"No XML metadata file found in {base_path}")

        xml_meta = _parse_isoview_xml(xml_path)
        self._zarr_attrs = xml_meta

        if "dimensions" not in xml_meta or not xml_meta["dimensions"]:
            raise ValueError(f"No dimensions found in XML: {xml_path}")

        # dimensions from xml: [width, height, depth] = [768, 1848, 38]
        # width = X = 768, height = Y = 1848
        xml_width, xml_height, _ = xml_meta["dimensions"][0]
        self._xml_width = xml_width  # X = 768
        self._xml_height = xml_height  # Y = 1848

        # parse all .stack files
        stack_files = sorted(base_path.glob("*.stack"))
        if not stack_files:
            raise ValueError(f"No .stack files in {base_path}")

        # pattern: SPC00_TM00000_ANG000_CM0_CHN01_PH0.stack
        pattern = re.compile(
            r"SPC(\d+)_TM(\d+)_ANG(\d+)_CM(\d+)_CHN(\d+)_PH(\d+)\.stack"
        )

        # organize by timepoint and (camera, channel)
        self._stack_files = {}  # {timepoint: {(camera, channel): Path}}
        timepoints = set()
        views = set()

        for sf in stack_files:
            match = pattern.match(sf.name)
            if not match:
                logger.warning(f"Skipping unrecognized file: {sf.name}")
                continue

            specimen, timepoint, angle, camera, channel, phase = map(int, match.groups())
            timepoints.add(timepoint)
            views.add((camera, channel))

            if timepoint not in self._stack_files:
                self._stack_files[timepoint] = {}
            self._stack_files[timepoint][(camera, channel)] = sf

        # sort timepoints and views
        self._timepoints = sorted(timepoints)
        self._views = sorted(views)

        if not self._views:
            raise ValueError(f"No valid camera/channel combinations in {base_path}")

        # calculate depth from file size
        first_file = next(iter(self._stack_files[self._timepoints[0]].values()))
        file_size = first_file.stat().st_size
        total_pixels = file_size // 2  # uint16 = 2 bytes
        depth = total_pixels // (xml_width * xml_height)

        # store shape as (Z, Y, X) where Y=height=1848, X=width=768
        self._single_shape = (depth, xml_height, xml_width)  # (Z=38, Y=1848, X=768)
        self._dtype = np.dtype("uint16")
        self._single_timepoint = len(self._timepoints) == 1
        self._consolidated_path = None

        # create tm_folders-like list for compatibility
        self.tm_folders = [base_path] * len(self._timepoints)

        logger.info(
            f"IsoviewArray: structure=raw, "
            f"timepoints={len(self._timepoints)}, views={len(self._views)}, "
            f"shape={self._single_shape}"
        )

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

    def _read_stack_file(self, stack_path: Path) -> np.ndarray:
        """Read raw binary .stack file as numpy array.

        Format: BSQ (band sequential), little-endian uint16

        For XML dims [768, 1848, 38] = [width, height, depth]:
        - reshape to (Z, Y, X) = (38, 1848, 768)
        """
        depth, height, width = self._single_shape  # (Z=38, Y=1848, X=768)
        volume = np.fromfile(stack_path, dtype="<u2")  # little-endian uint16
        volume = volume.reshape((depth, height, width))  # (Z, Y, X) = (38, 1848, 768)
        return volume

    def _get_zarr(self, t_idx: int, view_idx: int):
        """Get or open array data for timepoint and view index.

        For raw structure: reads .stack file into memory
        For zarr structures: returns lazy zarr array
        """
        cache_key = (t_idx, view_idx)
        if cache_key in self._zarr_cache:
            return self._zarr_cache[cache_key]

        camera, channel = self._views[view_idx]

        if self._structure == "raw":
            # get timepoint from sorted list
            timepoint = self._timepoints[t_idx]
            if timepoint not in self._stack_files:
                raise FileNotFoundError(f"No data for timepoint {timepoint}")
            if (camera, channel) not in self._stack_files[timepoint]:
                raise FileNotFoundError(
                    f"No data for camera={camera}, channel={channel} at timepoint {timepoint}"
                )

            stack_path = self._stack_files[timepoint][(camera, channel)]
            arr = self._read_stack_file(stack_path)
            self._zarr_cache[cache_key] = arr
            return arr

        # zarr-based structures
        import zarr
        tm_folder = self.tm_folders[t_idx]

        if self._structure == "consolidated":
            # find consolidated zarr in this TM folder
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
        # for raw structure, use _timepoints length
        num_t = len(self._timepoints) if self._structure == "raw" else len(self.tm_folders)
        return (num_t, z, len(self._views), y, x)

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
        num_t = self.num_timepoints
        meta["num_timepoints"] = num_t
        meta["nframes"] = num_t  # suite2p alias
        meta["num_frames"] = num_t  # legacy alias
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
        if self._structure == "raw":
            return len(self._timepoints)
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
            For raw: list of .stack file paths
            For zarr: list of TM folder paths
        """
        if self._structure == "raw":
            # return all stack files
            files = []
            for t in self._timepoints:
                for (cam, chn), path in self._stack_files[t].items():
                    files.append(path)
            return files
        return list(self.tm_folders)

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Dimension labels for LazyArrayProtocol.

        Returns
        -------
        tuple[str, ...]
            ('z', 'cm', 'y', 'x') for single timepoint
            ('t', 'z', 'cm', 'y', 'x') for multi-timepoint
        """
        if self._single_timepoint:
            return ("z", "cm", "Y", "X")
        return ("t", "z", "cm", "Y", "X")

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


# patterns to exclude (masks, auxiliary files)
EXCLUDE_PATTERNS = [
    "Mask",
    "mask",
    "minIntensity",
    "coords",
]


class IsoViewOutputArray:
    """
    Generic lazy array for IsoView pipeline outputs.

    Handles TM folder structure with various file formats (.tif, .klb, .zarr)
    and view dimensions (cameras, channels, or custom groupings).

    Shape: (T, Z, Views, Y, X) or (Z, Views, Y, X) for single timepoint

    Parameters
    ----------
    path : str or Path
        Path to directory containing TM* folders or a single TM folder
    view_dim : str, optional
        Name for view dimension in dims tuple. Default auto-detects:
        - 'cm' if CM## pattern found in filenames
        - 'ch' if only CHN## pattern (no CM) found
        - 'view' otherwise

    Examples
    --------
    >>> arr = IsoViewOutputArray("path/to/output")
    >>> arr.shape
    (4, 38, 4, 1848, 768)  # (t, z, views, y, x)
    >>> arr.dims
    ('t', 'z', 'cm', 'y', 'x')
    >>> arr.views
    [0, 1, 2, 3]  # camera indices
    >>> frame = arr[0, 10, 0]  # t=0, z=10, view=0
    """

    def __init__(self, path: str | Path, view_dim: str | None = None):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # discover structure
        self._discover_structure()

        # auto-detect or use provided view dimension name
        self._view_dim = view_dim or self._detect_view_dim()

        # cache for loaded volumes
        self._cache = {}
        self._metadata = {}

    def _discover_structure(self):
        """Find TM folders, file type, and views."""
        import re

        # detect if single TM or multi-TM
        if self.base_path.name.startswith("TM"):
            self._single_timepoint = True
            self.tm_folders = [self.base_path]
        else:
            self.tm_folders = sorted(
                [d for d in self.base_path.iterdir()
                 if d.is_dir() and d.name.startswith("TM")],
                key=lambda x: int(x.name[2:])
            )
            self._single_timepoint = len(self.tm_folders) == 1

            if not self.tm_folders:
                raise ValueError(f"No TM* folders found in {self.base_path}")

        # detect file type from first TM folder
        first_tm = self.tm_folders[0]
        self._detect_file_type(first_tm)

        # discover views
        self._discover_views(first_tm)

    def _detect_file_type(self, tm_folder: Path):
        """Detect file type (.tif, .klb, or .zarr) in TM folder."""
        # check for each type, excluding mask files
        tif_files = [f for f in tm_folder.glob("*.tif")
                     if not any(x in f.name for x in EXCLUDE_PATTERNS)]
        klb_files = [f for f in tm_folder.glob("*.klb")
                     if not any(x in f.name for x in EXCLUDE_PATTERNS)]
        zarr_files = [f for f in tm_folder.glob("*.zarr")
                      if not any(x in f.name for x in EXCLUDE_PATTERNS)]

        if tif_files:
            self._file_ext = ".tif"
            self._data_files = tif_files
        elif klb_files:
            self._file_ext = ".klb"
            self._data_files = klb_files
        elif zarr_files:
            self._file_ext = ".zarr"
            self._data_files = zarr_files
        else:
            raise ValueError(
                f"No supported data files (.tif, .klb, .zarr) in {tm_folder}"
            )

        logger.info(f"IsoViewOutputArray: detected {self._file_ext} files")

    def _discover_views(self, tm_folder: Path):
        """Parse filenames to find views and determine shape."""
        import re

        # patterns for extracting view info
        # SPM00_TM000000_CM00_CHN01.tif -> camera=0, channel=1
        # SPM00_TM000000_CM00_CM01_CHN01.tif -> fused cameras 0+1, channel=1
        # SPM00_TM000000_CHN00.tif -> channel=0 (no camera)
        pattern_cm = re.compile(
            r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)" + re.escape(self._file_ext)
        )
        pattern_fused_cm = re.compile(
            r"SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_CHN(\d+)" + re.escape(self._file_ext)
        )
        pattern_chn = re.compile(
            r"SPM(\d+)_TM(\d+)_CHN(\d+)" + re.escape(self._file_ext)
        )

        self._views = []
        self._view_type = None  # 'cm', 'fused', or 'ch'
        self._specimen = None
        self._file_map = {}  # view_idx -> filename pattern parts

        for f in sorted(self._data_files):
            # try fused camera pattern first (CM##_CM##)
            match = pattern_fused_cm.match(f.name)
            if match:
                specimen, timepoint, cam0, cam1, channel = map(int, match.groups())
                self._specimen = specimen
                self._view_type = "fused"
                view_key = (cam0, cam1)
                if view_key not in self._views:
                    self._views.append(view_key)
                    self._file_map[view_key] = {"cam0": cam0, "cam1": cam1, "channel": channel}
                continue

            # try single CM pattern
            match = pattern_cm.match(f.name)
            if match:
                specimen, timepoint, camera, channel = map(int, match.groups())
                self._specimen = specimen
                self._view_type = "cm"
                if camera not in self._views:
                    self._views.append(camera)
                    self._file_map[camera] = {"camera": camera, "channel": channel}
                continue

            # try CHN-only pattern
            match = pattern_chn.match(f.name)
            if match:
                specimen, timepoint, channel = map(int, match.groups())
                self._specimen = specimen
                self._view_type = "ch"
                if channel not in self._views:
                    self._views.append(channel)
                    self._file_map[channel] = {"channel": channel}
                continue

        if not self._views:
            raise ValueError(f"No valid data files found in {tm_folder}")

        # sort views
        self._views = sorted(self._views)

        # get shape from first file
        first_file = self._get_file_path(0, 0)
        self._read_shape(first_file)

        logger.info(
            f"IsoViewOutputArray: views={self._views}, shape={self._single_shape}, "
            f"view_type={self._view_type}"
        )

    def _read_shape(self, file_path: Path):
        """Read shape and dtype from first file."""
        if self._file_ext == ".tif":
            import tifffile
            with tifffile.TiffFile(str(file_path)) as tif:
                # get shape from first series
                shape = tif.series[0].shape
                self._dtype = tif.series[0].dtype
        elif self._file_ext == ".klb":
            import pyklb
            header = pyklb.readheader(str(file_path))
            dims = header['imagesize_tczyx']
            spatial_dims = [d for d in dims if d > 1]
            if len(spatial_dims) >= 3:
                # klb header stores as (Z, Y, X) but Y/X are swapped relative to raw isoview
                # raw .stack files: Y=1848 (long axis), X=768 (short axis)
                # klb header reports: Y=768, X=1848 - need to swap back
                z, y, x = spatial_dims[-3:]
                shape = (z, x, y)  # swap Y and X to match raw convention
            else:
                shape = (1,) * (3 - len(spatial_dims)) + tuple(spatial_dims)
            self._dtype = np.dtype(header['datatype'])
        elif self._file_ext == ".zarr":
            import zarr
            z = zarr.open(file_path, mode="r")
            if isinstance(z, zarr.Group):
                arr = z["0"] if "0" in z else list(z.values())[0]
            else:
                arr = z
            shape = arr.shape
            self._dtype = arr.dtype
        else:
            raise ValueError(f"Unsupported file type: {self._file_ext}")

        # shape should be (Z, Y, X)
        if len(shape) == 3:
            self._single_shape = shape
        elif len(shape) == 2:
            self._single_shape = (1,) + shape
        else:
            # take last 3 dims
            self._single_shape = shape[-3:]

    def _get_file_path(self, t_idx: int, view_idx: int) -> Path:
        """Get file path for timepoint and view index."""
        tm_folder = self.tm_folders[t_idx]
        tp = int(tm_folder.name[2:])  # extract timepoint from folder name
        sp = self._specimen if self._specimen is not None else 0
        view = self._views[view_idx]

        if self._view_type == "fused":
            info = self._file_map[view]
            filename = f"SPM{sp:02d}_TM{tp:06d}_CM{info['cam0']:02d}_CM{info['cam1']:02d}_CHN{info['channel']:02d}{self._file_ext}"
        elif self._view_type == "cm":
            info = self._file_map[view]
            filename = f"SPM{sp:02d}_TM{tp:06d}_CM{info['camera']:02d}_CHN{info['channel']:02d}{self._file_ext}"
        else:  # ch
            filename = f"SPM{sp:02d}_TM{tp:06d}_CHN{view:02d}{self._file_ext}"

        return tm_folder / filename

    def _read_volume(self, t_idx: int, view_idx: int) -> np.ndarray:
        """Read 3D volume for timepoint and view."""
        cache_key = (t_idx, view_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._get_file_path(t_idx, view_idx)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if self._file_ext == ".tif":
            import tifffile
            data = tifffile.imread(str(path))
        elif self._file_ext == ".klb":
            import pyklb
            data = pyklb.readfull(str(path))
            # klb data has Y/X swapped relative to raw isoview - transpose to match
            if data.ndim == 3:
                data = data.transpose(0, 2, 1)  # (Z, Y, X) -> (Z, X, Y) -> becomes (Z, Y, X) after swap
        elif self._file_ext == ".zarr":
            import zarr
            z = zarr.open(path, mode="r")
            if isinstance(z, zarr.Group):
                arr = z["0"] if "0" in z else list(z.values())[0]
            else:
                arr = z
            data = arr[:]
        else:
            raise ValueError(f"Unsupported file type: {self._file_ext}")

        # ensure 3d
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._cache[cache_key] = data
        return data

    def _detect_view_dim(self) -> str:
        """Auto-detect view dimension name from file pattern."""
        if self._view_type == "cm":
            return "cm"
        elif self._view_type == "ch":
            return "ch"
        return "view"

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape: (T, Z, Views, Y, X) or (Z, Views, Y, X)."""
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
    def dims(self) -> tuple[str, ...]:
        """Dimension labels for sliders.

        Spatial dims (Y, X) are uppercase per mbo_utilities convention.
        Slider dims (t, z, cm/ch) are lowercase for fastplotlib.
        """
        if self._single_timepoint:
            return ("z", self._view_dim, "Y", "X")
        return ("t", "z", self._view_dim, "Y", "X")

    @property
    def views(self) -> list[int]:
        """List of view indices (camera or channel numbers)."""
        return self._views

    @property
    def num_views(self) -> int:
        """Number of views."""
        return len(self._views)

    @property
    def num_timepoints(self) -> int:
        """Number of timepoints."""
        return len(self.tm_folders)

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return self._single_shape[0]

    @property
    def metadata(self) -> dict:
        """Return metadata as dict."""
        meta = dict(self._metadata)

        meta["num_timepoints"] = self.num_timepoints
        meta["nframes"] = self.num_timepoints
        meta["num_frames"] = self.num_timepoints
        meta["Ly"] = self._single_shape[1]
        meta["Lx"] = self._single_shape[2]
        meta["nplanes"] = self._single_shape[0]
        meta["num_planes"] = self._single_shape[0]
        meta["views"] = self._views
        meta["view_dim"] = self._view_dim
        meta["shape"] = self.shape
        meta["file_type"] = self._file_ext
        meta["single_timepoint"] = self._single_timepoint
        meta["specimen"] = self._specimen

        return meta

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    def __len__(self) -> int:
        """Length is first dimension (T or Z)."""
        return self.shape[0]

    def __getitem__(self, key):
        """Index the array: (T, Z, Views, Y, X) or (Z, Views, Y, X)."""
        if not isinstance(key, tuple):
            key = (key,)

        def to_indices(k, max_val):
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
            key = key + (slice(None),) * (4 - len(key))
            z_key, view_key, y_key, x_key = key
            t_indices = [0]
            t_key = 0
        else:
            key = key + (slice(None),) * (5 - len(key))
            t_key, z_key, view_key, y_key, x_key = key
            t_indices = to_indices(t_key, len(self.tm_folders))

        z_indices = to_indices(z_key, self._single_shape[0])
        view_indices = to_indices(view_key, len(self._views))

        out_shape = (
            len(t_indices),
            len(z_indices),
            len(view_indices),
            *self._single_shape[1:],
        )

        if isinstance(y_key, int):
            out_shape = (*out_shape[:3], 1, out_shape[4])
        elif isinstance(y_key, slice):
            y_size = len(range(*y_key.indices(self._single_shape[1])))
            out_shape = (*out_shape[:3], y_size, out_shape[4])

        if isinstance(x_key, int):
            out_shape = (*out_shape[:4], 1)
        elif isinstance(x_key, slice):
            x_size = len(range(*x_key.indices(self._single_shape[2])))
            out_shape = (*out_shape[:4], x_size)

        result = np.empty(out_shape, dtype=self._dtype)

        for ti, t_idx in enumerate(t_indices):
            for vi, view_idx in enumerate(view_indices):
                data = self._read_volume(t_idx, view_idx)
                sliced = data[z_key, y_key, x_key]

                if isinstance(z_key, int):
                    sliced = sliced[np.newaxis, ...]
                if isinstance(y_key, int):
                    sliced = sliced[:, np.newaxis, :]
                if isinstance(x_key, int):
                    sliced = sliced[:, :, np.newaxis]

                result[ti, :, vi, ...] = sliced

        # squeeze singleton dimensions
        if self._single_timepoint:
            result = np.squeeze(result, axis=0)
            int_indexed = [
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
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
            for ax in range(4, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)

        return result

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory."""
        return self[:]

    @property
    def filenames(self) -> list[Path]:
        """Source file paths."""
        files = []
        for ti in range(len(self.tm_folders)):
            for vi in range(len(self._views)):
                path = self._get_file_path(ti, vi)
                if path.exists():
                    files.append(path)
        return files

    def close(self) -> None:
        """Release resources (clear cache)."""
        self._cache.clear()

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
        """Write array to disk."""
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
            f"IsoViewOutputArray(shape={self.shape}, dtype={self.dtype}, "
            f"views={self._views}, dims={self.dims}, file_type='{self._file_ext}')"
        )


# register clusterpt pipeline info
_CLUSTERPT_INFO = PipelineInfo(
    name="clusterpt",
    description="IsoView-Processing clusterPT corrected data (KLB)",
    input_patterns=[
        "**/TM??????/SPM??_TM??????_CM??_CHN??.klb",
        "**/SPM??/*.corrected/*/TM??????/",
    ],
    output_patterns=[],
    input_extensions=["klb"],
    output_extensions=[],
    marker_files=["*.configuration.mat"],
    category="reader",
)
register_pipeline(_CLUSTERPT_INFO)


def _check_pyklb():
    """Check if pyklb is available."""
    try:
        import pyklb
        return pyklb
    except ImportError:
        raise ImportError(
            "pyklb is required for KLB files. Install with: "
            "pip install mbo_utilities[isoview]"
        )


class ClusterPTArray:
    """
    Lazy loader for clusterPT (IsoView-Processing) KLB output files.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities imread/imwrite.

    clusterPT output structure:
        outputFolder/
            TM000000/
                SPM00_TM000000_CM00_CHN00.klb
                SPM00_TM000000_CM01_CHN00.klb
                SPM00_TM000000.configuration.mat
                SPM00_TM000000_CHN00.xml
            TM000001/
                ...

    Shape: (T, Z, Views, Y, X) for multi-timepoint, (Z, Views, Y, X) for single.
    Views are (camera, channel) combinations.

    Parameters
    ----------
    path : str or Path
        Path to directory containing TM* folders with .klb files.

    Examples
    --------
    >>> arr = ClusterPTArray("path/to/corrected")
    >>> arr.shape
    (100, 38, 2, 1848, 768)  # (t, z, views, y, x)
    >>> arr.dims
    ('t', 'z', 'cm', 'y', 'x')
    >>> arr.views
    [(0, 0), (1, 0)]  # (camera, channel) per view
    >>> frame = arr[0, 10, 0]  # t=0, z=10, view=0
    """

    def __init__(self, path: str | Path):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        # check for pyklb
        _check_pyklb()

        # detect if single TM or multi-TM
        if self.base_path.name.startswith("TM"):
            klb_files = list(self.base_path.glob("*.klb"))
            if klb_files:
                self._single_timepoint = True
                self.tm_folders = [self.base_path]
            else:
                raise ValueError(f"TM folder {self.base_path} contains no .klb files")
        else:
            self.tm_folders = sorted(
                [d for d in self.base_path.iterdir()
                 if d.is_dir() and d.name.startswith("TM")],
                key=lambda x: int(x.name[2:])
            )
            self._single_timepoint = len(self.tm_folders) == 1

            if not self.tm_folders:
                raise ValueError(f"No TM* folders found in {self.base_path}")

        # discover views from first timepoint
        self._discover_views(self.tm_folders[0])

        # cache for klb data: (t_idx, view_idx) -> ndarray
        self._cache = {}
        self._metadata = {}

    def _discover_views(self, tm_folder: Path):
        """Parse KLB files to find camera/channel combinations and shape."""
        import re
        import pyklb

        # pattern: SPM00_TM000000_CM00_CHN00.klb
        pattern = re.compile(
            r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)\.klb"
        )

        klb_files = sorted(tm_folder.glob("*.klb"))
        if not klb_files:
            raise ValueError(f"No .klb files in {tm_folder}")

        self._views = []  # [(camera, channel), ...]
        self._specimen = None

        for kf in klb_files:
            match = pattern.match(kf.name)
            if not match:
                # skip mask files, projections, etc
                continue

            specimen, timepoint, camera, channel = map(int, match.groups())
            self._specimen = specimen

            view = (camera, channel)
            if view not in self._views:
                self._views.append(view)

        if not self._views:
            raise ValueError(f"No valid KLB data files in {tm_folder}")

        # sort views by (camera, channel)
        self._views = sorted(self._views)

        # read first file to get shape/dtype
        first_view = self._views[0]
        first_file = self._get_klb_path(tm_folder, first_view)
        header = pyklb.readheader(str(first_file))

        # klb stores as (x, y, z) so we need to reverse
        # header['imagesize_tczyx'] gives size in t,c,z,y,x order
        # but for our 3d volumes it's typically just spatial dims
        dims = header['imagesize_tczyx']
        # filter out singleton dimensions at front
        spatial_dims = [d for d in dims if d > 1]
        if len(spatial_dims) >= 3:
            self._single_shape = tuple(spatial_dims[-3:])  # (z, y, x)
        else:
            # might be 2d, pad with 1
            self._single_shape = (1,) * (3 - len(spatial_dims)) + tuple(spatial_dims)

        self._dtype = np.dtype(header['datatype'])

        logger.info(
            f"ClusterPTArray: timepoints={len(self.tm_folders)}, views={len(self._views)}, "
            f"shape={self._single_shape}, dtype={self._dtype}"
        )

    def _get_klb_path(self, tm_folder: Path, view: tuple[int, int]) -> Path:
        """Get path to KLB file for a timepoint folder and view."""
        camera, channel = view
        # extract timepoint from folder name
        tp = int(tm_folder.name[2:])
        sp = self._specimen if self._specimen is not None else 0

        filename = f"SPM{sp:02d}_TM{tp:06d}_CM{camera:02d}_CHN{channel:02d}.klb"
        return tm_folder / filename

    def _read_klb(self, t_idx: int, view_idx: int) -> np.ndarray:
        """Read KLB file for timepoint and view index."""
        cache_key = (t_idx, view_idx)
        if cache_key in self._cache:
            return self._cache[cache_key]

        import pyklb

        view = self._views[view_idx]
        tm_folder = self.tm_folders[t_idx]
        klb_path = self._get_klb_path(tm_folder, view)

        if not klb_path.exists():
            raise FileNotFoundError(f"KLB file not found: {klb_path}")

        # pyklb returns array in (z, y, x) order
        data = pyklb.readfull(str(klb_path))

        # ensure 3d
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._cache[cache_key] = data
        return data

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
    def metadata(self) -> dict:
        """Return metadata as dict."""
        meta = dict(self._metadata)

        # standard keys
        num_t = self.num_timepoints
        meta["num_timepoints"] = num_t
        meta["nframes"] = num_t
        meta["num_frames"] = num_t
        meta["Ly"] = self._single_shape[1]
        meta["Lx"] = self._single_shape[2]
        meta["nplanes"] = self._single_shape[0]
        meta["num_planes"] = self._single_shape[0]
        meta["views"] = self._views
        meta["shape"] = self.shape
        meta["structure"] = "clusterpt"
        meta["single_timepoint"] = self._single_timepoint
        meta["specimen"] = self._specimen

        return meta

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

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
        """Length is first dimension (T or Z)."""
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

        # build output array (5D internally)
        out_shape = (
            len(t_indices),
            len(z_indices),
            len(view_indices),
            *self._single_shape[1:],  # Y, X
        )

        # handle Y, X slicing
        if isinstance(y_key, int):
            out_shape = (*out_shape[:3], 1, out_shape[4])
        elif isinstance(y_key, slice):
            y_size = len(range(*y_key.indices(self._single_shape[1])))
            out_shape = (*out_shape[:3], y_size, out_shape[4])

        if isinstance(x_key, int):
            out_shape = (*out_shape[:4], 1)
        elif isinstance(x_key, slice):
            x_size = len(range(*x_key.indices(self._single_shape[2])))
            out_shape = (*out_shape[:4], x_size)

        result = np.empty(out_shape, dtype=self._dtype)

        for ti, t_idx in enumerate(t_indices):
            for vi, view_idx in enumerate(view_indices):
                data = self._read_klb(t_idx, view_idx)

                # index (Z, Y, X)
                sliced = data[z_key, y_key, x_key]

                # handle dimension reduction
                if isinstance(z_key, int):
                    sliced = sliced[np.newaxis, ...]
                if isinstance(y_key, int):
                    sliced = sliced[:, np.newaxis, :]
                if isinstance(x_key, int):
                    sliced = sliced[:, :, np.newaxis]

                result[ti, :, vi, ...] = sliced

        # squeeze singleton dimensions from integer indexing
        if self._single_timepoint:
            result = np.squeeze(result, axis=0)
            int_indexed = [
                isinstance(z_key, int),
                isinstance(view_key, int),
                isinstance(y_key, int),
                isinstance(x_key, int),
            ]
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
            for ax in range(4, -1, -1):
                if int_indexed[ax] and ax < result.ndim and result.shape[ax] == 1:
                    result = np.squeeze(result, axis=ax)

        return result

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory."""
        return self[:]

    @property
    def filenames(self) -> list[Path]:
        """Source file paths for LazyArrayProtocol."""
        files = []
        for tm in self.tm_folders:
            for view in self._views:
                path = self._get_klb_path(tm, view)
                if path.exists():
                    files.append(path)
        return files

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Dimension labels for LazyArrayProtocol.

        Returns
        -------
        tuple[str, ...]
            ('z', 'cm', 'y', 'x') for single timepoint
            ('t', 'z', 'cm', 'y', 'x') for multi-timepoint
        """
        if self._single_timepoint:
            return ("z", "cm", "Y", "X")
        return ("t", "z", "cm", "Y", "X")

    @property
    def num_planes(self) -> int:
        """Number of Z-planes."""
        return self._single_shape[0]

    def close(self) -> None:
        """Release resources (clear cache)."""
        self._cache.clear()

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
        """Write ClusterPTArray to disk."""
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
            f"ClusterPTArray(shape={self.shape}, dtype={self.dtype}, "
            f"views={self._views}, timepoints={len(self.tm_folders)})"
        )
