"""
Zarr array reader.

This module provides ZarrArray for reading Zarr v3 stores, including OME-Zarr.
Presents data in TZYX format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin
from mbo_utilities.file_io import HAS_ZARR, logger
from mbo_utilities.arrays.suite2p import _add_suite2p_labels
from mbo_utilities.metadata import _build_ome_metadata, get_param, get_voxel_size

logger = log.get("arrays.zarr")


class ZarrArray(ReductionMixin):
    """
    Reader for Zarr stores (including OME-Zarr).

    Presents data as (T, Z, H, W) with Z=1..nz. Supports both standard
    zarr arrays and OME-Zarr groups with "0" arrays.

    Parameters
    ----------
    filenames : str, Path, or sequence
        Path(s) to zarr store(s).
    compressor : str, optional
        Compressor name (not currently used for reading).
    rois : list[int] or int, optional
        ROI filter (not currently used).

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Shape as (T, Z, H, W).
    dtype : np.dtype
        Data type.
    zs : list
        List of zarr arrays.

    Examples
    --------
    >>> arr = ZarrArray("data.zarr")
    >>> arr.shape
    (10000, 1, 512, 512)
    >>> frame = arr[0, 0]  # Get first frame of first z-plane
    """

    def __init__(
        self,
        filenames: str | Path | Sequence[str | Path],
        compressor: str | None = "default",
        rois: list[int] | int | None = None,
    ):
        try:
            import zarr
        except ImportError:
            logger.error(
                "zarr is not installed. Install with `uv pip install zarr>=3.1.3`."
            )
            raise

        if isinstance(filenames, (str, Path)):
            filenames = [filenames]

        self.filenames = [Path(p).with_suffix(".zarr") for p in filenames]
        self.rois = rois
        for p in self.filenames:
            if not p.exists():
                raise FileNotFoundError(f"No zarr store at {p}")

        # Open zarr stores - handle both standard arrays and OME-Zarr groups
        opened = [zarr.open(p, mode="r") for p in self.filenames]

        # If we opened a Group (OME-Zarr structure), get the "0" array
        self.zs = []
        self._groups = []
        for i, z in enumerate(opened):
            if isinstance(z, zarr.Group):
                if "0" not in z:
                    # get store path for error message (zarr v3 uses .root)
                    store_path = getattr(z.store, 'root', getattr(z.store, 'path', self.filenames[i]))
                    raise ValueError(
                        f"OME-Zarr group missing '0' array in {store_path}"
                    )
                self.zs.append(z["0"])
                self._groups.append(z)
            else:
                self.zs.append(z)
                self._groups.append(None)

        shapes = [z.shape for z in self.zs]
        if len(set(shapes)) != 1:
            raise ValueError(f"Inconsistent shapes across zarr stores: {shapes}")

        # For OME-Zarr, metadata is on the group; for standard zarr, on the array
        self._metadata = []
        for i, z in enumerate(self.zs):
            if self._groups[i] is not None:
                self._metadata.append(dict(self._groups[i].attrs))
            else:
                self._metadata.append(dict(z.attrs))
        self.compressor = compressor
        self._target_dtype = None

    @property
    def metadata(self):
        """Return metadata as a dict."""
        if not self._metadata:
            md = {}
        else:
            md = self._metadata[0].copy()

        # Ensure critical keys are present
        md["dtype"] = self.dtype
        if "num_frames" not in md and "nframes" not in md:
            if self.zs:
                md["num_frames"] = int(self.zs[0].shape[0])

        return md

    @property
    def zstats(self) -> dict | None:
        """
        Return pre-computed z-statistics from metadata if available.

        Returns
        -------
        dict | None
            Dictionary with keys 'mean', 'std', 'snr' (each a list of floats),
            or None if not available.
        """
        md = self.metadata
        if "zstats" in md:
            return md["zstats"]
        return None

    @zstats.setter
    def zstats(self, value: dict):
        """Store z-statistics in metadata for persistence."""
        if not isinstance(value, dict):
            raise TypeError(f"zstats must be a dict, got {type(value)}")
        if not all(k in value for k in ("mean", "std", "snr")):
            raise ValueError("zstats must contain 'mean', 'std', and 'snr' keys")

        if not self._metadata:
            self._metadata = [{}]
        self._metadata[0]["zstats"] = value

    @metadata.setter
    def metadata(self, value: dict):
        """Set metadata. Updates the first zarr file's metadata."""
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")

        if not self._metadata:
            self._metadata = [value]
        else:
            self._metadata[0] = value

    @property
    def shape(self) -> tuple[int, int, int, int]:
        first_shape = self.zs[0].shape
        if len(first_shape) == 4:
            return first_shape
        elif len(first_shape) == 3:
            t, h, w = first_shape
            return t, len(self.zs), h, w
        else:
            raise ValueError(
                f"Unexpected zarr shape: {first_shape}. "
                f"Expected 3D (T, H, W) or 4D (T, Z, H, W)"
            )

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self.zs[0].dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def _compute_frame_vminmax(self):
        """Compute vmin/vmax from first frame (frame 0, plane 0)."""
        if not hasattr(self, '_cached_vmin'):
            frame = self[0, 0] if self.ndim == 4 else self[0]
            frame = np.asarray(frame)
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

    @property
    def size(self):
        return np.prod(self.shape)

    def __array__(self):
        """Materialize full array into memory: (T, Z, H, W)."""
        if len(self.zs) == 1 and len(self.zs[0].shape) == 4:
            return np.asarray(self.zs[0][:])
        arrs = [z[:] for z in self.zs]
        return np.stack(arrs, axis=1)


    @property
    def ndim(self):
        return 4

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        t_key, z_key, y_key, x_key = key

        def normalize(idx):
            if isinstance(idx, range):
                if len(idx) == 0:
                    return slice(0, 0)
                return slice(idx.start, idx.stop, idx.step)
            if isinstance(idx, list) and len(idx) > 0:
                if all(idx[i] + 1 == idx[i + 1] for i in range(len(idx) - 1)):
                    return slice(idx[0], idx[-1] + 1)
                else:
                    return np.array(idx)
            return idx

        t_key = normalize(t_key)
        y_key = normalize(y_key)
        x_key = normalize(x_key)
        z_key = normalize(z_key)

        is_single_4d = len(self.zs) == 1 and len(self.zs[0].shape) == 4

        if is_single_4d:
            out = self.zs[0][t_key, z_key, y_key, x_key]
        elif len(self.zs) == 1:
            if isinstance(z_key, int):
                if z_key != 0:
                    raise IndexError("Z dimension has size 1, only index 0 is valid")
                out = self.zs[0][t_key, y_key, x_key]
            elif isinstance(z_key, slice):
                data = self.zs[0][t_key, y_key, x_key]
                out = data[:, np.newaxis, ...]
            else:
                out = self.zs[0][t_key, y_key, x_key]
        elif isinstance(z_key, int):
            out = self.zs[z_key][t_key, y_key, x_key]
        else:
            if isinstance(z_key, slice):
                z_indices = range(len(self.zs))[z_key]
            elif isinstance(z_key, np.ndarray) or isinstance(z_key, list):
                z_indices = z_key
            else:
                z_indices = range(len(self.zs))
            arrs = [self.zs[i][t_key, y_key, x_key] for i in z_indices]
            out = np.stack(arrs, axis=1)

        if self._target_dtype is not None:
            out = out.astype(self._target_dtype)
        return out

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
        """Write ZarrArray to disk in various formats."""
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


def merge_zarr_zplanes(
    zarr_paths: list[str | Path],
    output_path: str | Path,
    *,
    suite2p_dirs: list[str | Path] | None = None,
    metadata: dict | None = None,
    overwrite: bool = True,
    compression_level: int = 1,
) -> Path:
    """
    Merge multiple single z-plane Zarr files into a single OME-Zarr volume.

    Creates an OME-NGFF v0.5 compliant Zarr store with shape (T, Z, Y, X) by
    stacking individual z-plane Zarr files. Optionally includes Suite2p segmentation
    masks as OME-Zarr labels.

    Parameters
    ----------
    zarr_paths : list of str or Path
        List of paths to single-plane Zarr stores. Should be ordered by z-plane.
        Each Zarr should have shape (T, Y, X).
    output_path : str or Path
        Path for the output merged Zarr store.
    suite2p_dirs : list of str or Path, optional
        List of Suite2p output directories corresponding to each z-plane.
        If provided, ROI masks will be added as OME-Zarr labels.
        Must match length of zarr_paths.
    metadata : dict, optional
        Comprehensive metadata dictionary. Coordinate-related keys are used for
        OME-NGFF transformations, while additional keys are preserved as custom
        metadata. Supported keys:

        **Coordinate transformations:**
        - pixel_resolution : tuple (x, y) in micrometers
        - frame_rate : float, Hz (or 'fs')
        - dz : float, z-step in micrometers (or 'z_step')
        - name : str, volume name

        **ScanImage metadata:**
        - si : dict, complete ScanImage metadata structure
        - roi_groups : list, ROI definitions with scanfield info
        - objective_resolution : float, objective NA
        - zoom_factor : float

        **Acquisition metadata:**
        - acquisition_date : str, ISO format
        - experimenter : str
        - description : str
        - specimen : str

        **Microscope metadata:**
        - objective : str, objective name
        - emission_wavelength : float, nm
        - excitation_wavelength : float, nm
        - numerical_aperture : float

        **Processing metadata:**
        - fix_phase : bool
        - phasecorr_method : str
        - use_fft : bool
        - register_z : bool

        **OMERO rendering:**
        - channel_names : list of str
        - num_planes : int, number of channels/planes

        All metadata is organized into structured groups (scanimage, acquisition,
        microscope, processing) in the output OME-Zarr attributes.
    overwrite : bool, default=True
        If True, overwrite existing output Zarr store.
    compression_level : int, default=1
        Gzip compression level (0-9). Higher = better compression, slower.

    Returns
    -------
    Path
        Path to the created OME-Zarr store.

    Raises
    ------
    ValueError
        If zarr_paths is empty or shapes are incompatible.
    FileNotFoundError
        If any input Zarr or Suite2p directory doesn't exist.

    Examples
    --------
    Merge z-plane Zarr files into a volume:

    >>> zarr_files = [
    ...     "session1/plane01.zarr",
    ...     "session1/plane02.zarr",
    ...     "session1/plane03.zarr",
    ... ]
    >>> merge_zarr_zplanes(zarr_files, "session1/volume.zarr")

    Include Suite2p segmentation masks:

    >>> s2p_dirs = [
    ...     "session1/plane01_suite2p",
    ...     "session1/plane02_suite2p",
    ...     "session1/plane03_suite2p",
    ... ]
    >>> merge_zarr_zplanes(
    ...     zarr_files,
    ...     "session1/volume.zarr",
    ...     suite2p_dirs=s2p_dirs,
    ...     metadata={"pixel_resolution": (0.5, 0.5), "frame_rate": 30.0, "dz": 5.0}
    ... )

    See Also
    --------
    imwrite : Write imaging data to various formats including OME-Zarr
    """
    if not HAS_ZARR:
        raise ImportError("zarr package required. Install with: pip install zarr")

    import zarr
    from zarr.codecs import BytesCodec, GzipCodec

    zarr_paths = [Path(p) for p in zarr_paths]
    output_path = Path(output_path)

    if not zarr_paths:
        raise ValueError("zarr_paths cannot be empty")

    # Validate all input Zarrs exist
    for zp in zarr_paths:
        if not zp.exists():
            raise FileNotFoundError(f"Zarr store not found: {zp}")

    # Validate suite2p_dirs if provided
    if suite2p_dirs is not None:
        suite2p_dirs = [Path(p) for p in suite2p_dirs]
        if len(suite2p_dirs) != len(zarr_paths):
            raise ValueError(
                f"suite2p_dirs length ({len(suite2p_dirs)}) must match "
                f"zarr_paths length ({len(zarr_paths)})"
            )
        for s2p_dir in suite2p_dirs:
            if not s2p_dir.exists():
                raise FileNotFoundError(f"Suite2p directory not found: {s2p_dir}")

    # Read first Zarr to get dimensions
    logger.info(f"Reading first Zarr to determine dimensions: {zarr_paths[0]}")
    z0 = zarr.open(str(zarr_paths[0]), mode="r")
    logger.debug(f"Zarr type: {type(z0)}")

    if hasattr(z0, "shape"):
        # Direct array
        T, Y, X = z0.shape
        dtype = z0.dtype
        logger.debug(f"Detected direct array with shape {(T, Y, X)}, dtype {dtype}")
    else:
        # Group - look for "0" array (OME-Zarr)
        logger.debug(f"Detected group with keys: {list(z0.keys())}")
        if "0" in z0:
            arr = z0["0"]
            T, Y, X = arr.shape
            dtype = arr.dtype
            logger.debug(f"Using '0' subarray with shape {(T, Y, X)}, dtype {dtype}")
        else:
            raise ValueError(
                f"Cannot determine shape of {zarr_paths[0]}. "
                f"Expected direct array or group with '0' subarray. "
                f"Got group with keys: {list(z0.keys())}"
            )

    Z = len(zarr_paths)
    logger.info(f"Creating merged Zarr volume with shape (T={T}, Z={Z}, Y={Y}, X={X})")

    if output_path.exists() and overwrite:
        import shutil

        shutil.rmtree(output_path)

    root = zarr.open_group(str(output_path), mode="w", zarr_format=3)
    image_codecs = [BytesCodec(), GzipCodec(level=compression_level)]
    image = zarr.create(
        store=root.store,
        path="0",
        shape=(T, Z, Y, X),
        chunks=(1, 1, Y, X),  # Chunk by frame and z-plane
        dtype=dtype,
        codecs=image_codecs,
        overwrite=True,
    )

    logger.info("Copying z-plane data...")
    for zi, zpath in enumerate(zarr_paths):
        logger.debug(f"Reading z-plane {zi + 1}/{Z} from {zpath}")
        z_arr = zarr.open(str(zpath), mode="r")

        # Handle both direct arrays and OME-Zarr groups
        if hasattr(z_arr, "shape"):
            plane_data = z_arr[:]
            logger.debug(f"  Read direct array with shape {plane_data.shape}")
        elif "0" in z_arr:
            plane_data = z_arr["0"][:]
            logger.debug(f"  Read '0' subarray with shape {plane_data.shape}")
        else:
            raise ValueError(
                f"Cannot read data from {zpath}. "
                f"Got group with keys: {list(z_arr.keys()) if hasattr(z_arr, 'keys') else 'N/A'}"
            )

        if plane_data.shape != (T, Y, X):
            raise ValueError(
                f"Shape mismatch at z={zi} (file: {zpath.name}): "
                f"expected {(T, Y, X)}, got {plane_data.shape}"
            )

        logger.debug(f"  Writing to output volume at z={zi}")
        image[:, zi, :, :] = plane_data
        logger.info(f"Copied z-plane {zi + 1}/{Z} from {zpath.name}")

    # Add Suite2p labels if provided
    if suite2p_dirs is not None:
        logger.info("Adding Suite2p segmentation masks as labels...")
        _add_suite2p_labels(root, suite2p_dirs, T, Z, Y, X, dtype, compression_level)

    metadata = metadata or {}
    ome_attrs = _build_ome_metadata(
        shape=(T, Z, Y, X),
        dtype=dtype,
        metadata=metadata,
    )

    for key, value in ome_attrs.items():
        root.attrs[key] = value

    # add napari-specific scale metadata using centralized voxel size extraction
    vs = get_voxel_size(metadata)
    frame_rate = get_param(metadata, "fs", default=1.0)
    time_scale = 1.0 / float(frame_rate) if frame_rate else 1.0

    # napari reads scale from array attributes for volumetric viewing
    # Scale order: (T, Z, Y, X) in physical units
    image.attrs["scale"] = [time_scale, vs.dz, vs.dy, vs.dx]

    logger.info(f"Successfully created merged OME-Zarr at {output_path}")
    logger.info(f"Napari scale (t,z,y,x): {image.attrs['scale']}")
    return output_path
