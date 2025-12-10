"""
Suite2p binary array reader.

This module provides Suite2pArray for reading Suite2p binary output files
(data.bin, data_raw.bin) with their associated ops.npy metadata.

Also provides Suite2pVolumeArray for reading directories containing multiple
Suite2p plane outputs as a 4D volume (T, Z, H, W).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin
from mbo_utilities.util import load_npy

logger = log.get("arrays.suite2p")


@dataclass
class Suite2pArray(ReductionMixin):
    """
    Lazy array reader for Suite2p binary output files.

    Reads memory-mapped binary files (data.bin or data_raw.bin) alongside
    their ops.npy metadata. Supports switching between raw and registered
    data channels.

    Parameters
    ----------
    filename : str or Path
        Path to ops.npy or a .bin file in a Suite2p output directory.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Shape as (nframes, Ly, Lx).
    dtype : np.dtype
        Data type (always np.int16 for Suite2p).
    metadata : dict
        Contents of ops.npy.
    active_file : Path
        Currently active binary file.
    raw_file : Path
        Path to data_raw.bin (unregistered).
    reg_file : Path
        Path to data.bin (registered).

    Examples
    --------
    >>> arr = Suite2pArray("suite2p_output/ops.npy")
    >>> arr.shape
    (10000, 512, 512)
    >>> frame = arr[0]  # Get first frame
    >>> arr.switch_channel(use_raw=True)  # Switch to raw data
    """

    filename: str | Path
    metadata: dict = field(init=False)
    active_file: Path = field(init=False)
    raw_file: Path = field(default=None)
    reg_file: Path = field(default=None)

    def __post_init__(self):
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".npy" and path.stem == "ops":
            ops_path = path
        elif path.suffix == ".bin":
            ops_path = path.with_name("ops.npy")
            if not ops_path.exists():
                raise FileNotFoundError(f"Missing ops.npy near {path}")
        else:
            raise ValueError(f"Unsupported input: {path}")

        self.metadata = load_npy(ops_path).item()
        self.num_rois = self.metadata.get("num_rois", 1)

        # resolve both possible bins - always look in the same directory as ops.npy
        # (metadata paths may be stale if data was moved)
        ops_dir = ops_path.parent
        self.raw_file = ops_dir / "data_raw.bin"
        self.reg_file = ops_dir / "data.bin"

        # choose which one to use
        if path.suffix == ".bin":
            # User clicked directly on a .bin file - use that specific file
            self.active_file = path
            if not self.active_file.exists():
                raise FileNotFoundError(
                    f"Binary file not found: {self.active_file}\n"
                    f"Available files in {ops_dir}:\n"
                    f"  - data.bin: {'exists' if self.reg_file.exists() else 'missing'}\n"
                    f"  - data_raw.bin: {'exists' if self.raw_file.exists() else 'missing'}"
                )
        else:
            # User clicked on directory/ops.npy - choose best available file
            # Prefer registered (data.bin) over raw (data_raw.bin)
            if self.reg_file.exists():
                self.active_file = self.reg_file
            elif self.raw_file.exists():
                self.active_file = self.raw_file
            else:
                raise FileNotFoundError(
                    f"No binary files found in {ops_dir}\n"
                    f"Expected either:\n"
                    f"  - {self.reg_file} (registered)\n"
                    f"  - {self.raw_file} (raw)\n"
                    f"Please check that Suite2p processing completed successfully."
                )

        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        self.shape = (self.nframes, self.Ly, self.Lx)
        self._dtype = np.int16
        self._target_dtype = None

        # Validate file size matches expected shape
        expected_bytes = int(np.prod(self.shape)) * np.dtype(self._dtype).itemsize
        actual_bytes = self.active_file.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Binary file {self.active_file.name} is too small!\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"File may be corrupted or ops.npy metadata may be incorrect."
            )
        elif actual_bytes > expected_bytes:
            warnings.warn(
                f"Binary file {self.active_file.name} is larger than expected.\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"Extra data will be ignored.",
                UserWarning,
            )

        self._file = np.memmap(
            self.active_file, mode="r", dtype=self.dtype, shape=self.shape
        )
        self.filenames = [self.active_file]

    def switch_channel(self, use_raw=False):
        """Switch between raw and registered data channels."""
        new_file = self.raw_file if use_raw else self.reg_file
        if not new_file.exists():
            raise FileNotFoundError(new_file)
        self._file = np.memmap(new_file, mode="r", dtype=self.dtype, shape=self.shape)
        self.active_file = new_file

    def __getitem__(self, key):
        out = self._file[key]
        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        n = min(10, self.nframes) if self.nframes >= 10 else self.nframes
        return np.stack([self._file[i] for i in range(n)], axis=0)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    def close(self):
        """Close the memory-mapped file."""
        self._file._mmap.close()  # type: ignore

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write Suite2pArray to disk in various formats."""
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

    def imshow(self, **kwargs):
        """Display array using fastplotlib ImageWidget."""
        arrays = []
        names = []

        # Try to load both files if they exist
        if self.raw_file.exists():
            try:
                raw = Suite2pArray(self.raw_file)
                arrays.append(raw)
                names.append("raw")
            except Exception as e:
                logger.warning(f"Could not open raw file {self.raw_file}: {e}")

        if self.reg_file.exists():
            try:
                reg = Suite2pArray(self.reg_file)
                arrays.append(reg)
                names.append("registered")
            except Exception as e:
                logger.warning(f"Could not open registered file {self.reg_file}: {e}")

        # If neither file could be loaded, show the currently active file
        if not arrays:
            arrays.append(self)
            if self.active_file == self.raw_file:
                names.append("raw")
            elif self.active_file == self.reg_file:
                names.append("registered")
            else:
                names.append(self.active_file.name)

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs", None)

        import fastplotlib as fpl

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=(1, len(arrays)),
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )


def _extract_plane_number(name: str) -> int | None:
    """Extract plane number from directory name like 'plane01_stitched' or 'plane14'."""
    match = re.search(r"plane(\d+)", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_suite2p_plane_dirs(directory: Path) -> list[Path]:
    """
    Find Suite2p plane directories in a parent directory.

    Looks for subdirectories containing ops.npy files, sorted by plane number.

    Parameters
    ----------
    directory : Path
        Parent directory to search.

    Returns
    -------
    list[Path]
        List of plane directories sorted by plane number.
    """
    plane_dirs = []
    for subdir in directory.iterdir():
        if subdir.is_dir():
            ops_file = subdir / "ops.npy"
            if ops_file.exists():
                plane_dirs.append(subdir)

    # Sort by plane number extracted from directory name
    def sort_key(p):
        num = _extract_plane_number(p.name)
        return num if num is not None else float("inf")

    return sorted(plane_dirs, key=sort_key)


class Suite2pVolumeArray(ReductionMixin):
    """
    Reader for Suite2p output directories containing multiple planes.

    Presents data as (T, Z, H, W) by stacking individual plane arrays along
    the Z dimension. Each plane is loaded lazily via Suite2pArray.

    Parameters
    ----------
    directory : str or Path
        Path to directory containing plane subdirectories (e.g., plane01_stitched/).
    plane_dirs : list[Path], optional
        Explicit list of plane directories to use. If not provided, auto-detected.
    use_raw : bool, optional
        If True, prefer data_raw.bin over data.bin. Default False.

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (T, Z, H, W).
    dtype : np.dtype
        Data type (int16 for Suite2p).
    planes : list[Suite2pArray]
        Individual plane arrays.

    Examples
    --------
    >>> arr = Suite2pVolumeArray("suite2p_output/")
    >>> arr.shape
    (10000, 14, 512, 512)
    >>> frame = arr[0]  # Get first frame across all planes
    >>> plane7 = arr[:, 6]  # Get all frames from plane 7 (0-indexed)
    """

    def __init__(
        self,
        directory: str | Path,
        plane_dirs: Sequence[Path] | None = None,
        use_raw: bool = False,
    ):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        # Find plane directories
        if plane_dirs is None:
            plane_dirs = find_suite2p_plane_dirs(self.directory)

        if not plane_dirs:
            raise ValueError(
                f"No Suite2p plane directories found in {self.directory}. "
                "Expected subdirectories containing ops.npy files."
            )

        # Load each plane as Suite2pArray
        self.planes: list[Suite2pArray] = []
        self.filenames = []
        for pdir in plane_dirs:
            ops_file = pdir / "ops.npy"
            arr = Suite2pArray(ops_file)
            if use_raw and arr.raw_file.exists():
                arr.switch_channel(use_raw=True)
            self.planes.append(arr)
            self.filenames.append(arr.active_file)

        # Validate consistent shapes across planes
        shapes = [(p.shape[1], p.shape[2]) for p in self.planes]  # (Ly, Lx)
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent spatial shapes across planes: {shapes}")

        nframes = [p.shape[0] for p in self.planes]
        if len(set(nframes)) != 1:
            logger.warning(
                f"Inconsistent frame counts across planes: {nframes}. "
                f"Using minimum: {min(nframes)}"
            )

        self._nframes = min(nframes)
        self._nz = len(self.planes)
        self._ly, self._lx = shapes[0]
        self._dtype = self.planes[0]._dtype
        self._target_dtype = None

        # Aggregate metadata from first plane
        self._metadata = dict(self.planes[0].metadata)
        self._metadata["num_planes"] = self._nz
        self._metadata["plane_dirs"] = [str(p) for p in plane_dirs]

        logger.info(
            f"Loaded Suite2p volume: {self._nframes} frames, {self._nz} planes, "
            f"{self._ly}x{self._lx} px"
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._nframes, self._nz, self._ly, self._lx)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata = value

    @property
    def ndim(self) -> int:
        return 4

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def num_planes(self) -> int:
        return self._nz

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = np.asarray(self[0, 0])
            self._cached_vmin = float(frame.min())
            self._cached_vmax = float(frame.max())

    @property
    def vmin(self) -> float:
        """Min from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmin

    @property
    def vmax(self) -> float:
        """Max from first frame for display (avoids full data read)."""
        self._compute_frame_vminmax()
        return self._cached_vmax

    def __len__(self) -> int:
        return self._nframes

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        # Normalize t_key to respect _nframes limit
        if isinstance(t_key, slice):
            # Clamp slice to valid frame range
            start, stop, step = t_key.indices(self._nframes)
            t_key = slice(start, stop, step)
        elif isinstance(t_key, int):
            if t_key < 0:
                t_key = self._nframes + t_key
            if t_key >= self._nframes:
                raise IndexError(
                    f"Time index {t_key} out of bounds for {self._nframes} frames"
                )

        # Handle single z index
        if isinstance(z_key, int):
            if z_key < 0:
                z_key = self._nz + z_key
            if z_key < 0 or z_key >= self._nz:
                raise IndexError(f"Z index {z_key} out of bounds for {self._nz} planes")
            out = self.planes[z_key][t_key, y_key, x_key]
        else:
            # Handle z slice or full z
            if isinstance(z_key, slice):
                z_indices = range(self._nz)[z_key]
            elif isinstance(z_key, (list, np.ndarray)):
                z_indices = z_key
            else:
                z_indices = range(self._nz)

            # Stack data from selected planes
            arrs = [self.planes[i][t_key, y_key, x_key] for i in z_indices]
            out = np.stack(arrs, axis=1)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

    def __array__(self) -> np.ndarray:
        """Materialize full array into memory: (T, Z, H, W)."""
        arrs = [p[: self._nframes] for p in self.planes]
        return np.stack(arrs, axis=1)

    def switch_channel(self, use_raw: bool = False):
        """Switch all planes between raw and registered data."""
        for plane in self.planes:
            plane.switch_channel(use_raw=use_raw)
        self.filenames = [p.active_file for p in self.planes]

    def close(self):
        """Close all memory-mapped files."""
        for plane in self.planes:
            plane.close()

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
        """Write Suite2pVolumeArray to disk in various formats."""
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


def _add_suite2p_labels(
    root_group,
    suite2p_dirs: list[Path],
    T: int,
    Z: int,
    Y: int,
    X: int,
    dtype,
    compression_level: int,
):
    """
    Add Suite2p segmentation masks as OME-Zarr labels.

    Creates a 'labels' subgroup with ROI masks from Suite2p stat.npy files.
    Follows OME-NGFF v0.5 labels specification.

    Parameters
    ----------
    root_group : zarr.Group
        Root Zarr group to add labels to.
    suite2p_dirs : list of Path
        Suite2p output directories for each z-plane.
    T, Z, Y, X : int
        Dimensions of the volume.
    dtype : np.dtype
        Data type for label array.
    compression_level : int
        Gzip compression level.
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    logger.info("Creating labels array from Suite2p masks...")

    # Create labels subgroup
    labels_group = root_group.create_group("labels", overwrite=True)

    # Create ROI masks array (static across time, just Z, Y, X)
    label_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    masks = zarr.create(
        store=labels_group.store,
        path="labels/0",
        shape=(Z, Y, X),
        chunks=(1, Y, X),
        dtype=np.uint32,  # uint32 for up to 4 billion ROIs
        codecs=label_codecs,
        overwrite=True,
    )

    # Process each z-plane
    roi_id = 1  # Start ROI IDs at 1 (0 = background)

    for zi, s2p_dir in enumerate(suite2p_dirs):
        stat_path = s2p_dir / "stat.npy"
        iscell_path = s2p_dir / "iscell.npy"

        if not stat_path.exists():
            logger.warning(f"stat.npy not found in {s2p_dir}, skipping z={zi}")
            continue

        # Load Suite2p data
        stat = load_npy(stat_path)

        # Load iscell if available to filter
        if iscell_path.exists():
            iscell = load_npy(iscell_path)[:, 0].astype(bool)
        else:
            iscell = np.ones(len(stat), dtype=bool)

        # Create mask for this z-plane
        plane_mask = np.zeros((Y, X), dtype=np.uint32)

        for roi_idx, (roi_stat, is_cell) in enumerate(zip(stat, iscell)):
            if not is_cell:
                continue

            # Get pixel coordinates for this ROI
            ypix = roi_stat.get("ypix", [])
            xpix = roi_stat.get("xpix", [])

            if len(ypix) == 0 or len(xpix) == 0:
                continue

            # Ensure coordinates are within bounds
            ypix = np.clip(ypix, 0, Y - 1)
            xpix = np.clip(xpix, 0, X - 1)

            # Assign unique ROI ID
            plane_mask[ypix, xpix] = roi_id
            roi_id += 1

        # Write to Zarr
        masks[zi, :, :] = plane_mask
        logger.debug(
            f"Added {(plane_mask > 0).sum()} labeled pixels for z-plane {zi + 1}/{Z}"
        )

    # Add OME-NGFF labels metadata
    labels_metadata = {
        "version": "0.5",
        "labels": ["0"],  # Path to the label array
    }

    # Add metadata for label array
    label_array_meta = {
        "version": "0.5",
        "image-label": {
            "version": "0.5",
            "colors": [],  # Can add color LUT here if desired
            "source": {"image": "../../0"},  # Reference to main image
        },
    }

    labels_group.attrs.update(labels_metadata)
    labels_group["0"].attrs.update(label_array_meta)

    logger.info(f"Added {roi_id - 1} total ROIs across {Z} z-planes")


def load_ops(ops_input: str | Path | list[str | Path]):
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return load_npy(ops_input).item()
    elif isinstance(ops_input, dict):
        return ops_input
    logger.warning("No valid ops file provided, returning empty dict.")
    return {}


# Re-export write_ops for backward compatibility (moved to _writers.py to avoid circular imports)
from mbo_utilities._writers import write_ops
