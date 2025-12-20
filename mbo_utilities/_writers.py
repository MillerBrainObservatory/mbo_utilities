import functools
import math
import warnings
from datetime import datetime
from typing import Any

import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter, imwrite
import h5py

from . import log
from ._parsing import _make_json_serializable, _convert_paths_to_strings
from .util import load_npy

from tqdm.auto import tqdm

logger = log.get("writers")

warnings.filterwarnings("ignore")

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]
CHUNKS = {0: "auto", 1: -1, 2: -1}


def _get_mbo_version():
    """Get mbo_utilities version string."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def add_processing_step(
    metadata: dict,
    step_name: str,
    input_files: list | str | None = None,
    output_files: list | str | None = None,
    duration_seconds: float | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Add a processing step to metadata["processing_history"].

    Each step is appended to the history list, preserving previous runs.
    This allows tracking of re-runs and incremental processing across
    both mbo_utilities and downstream tools like lbm_suite2p_python.

    Parameters
    ----------
    metadata : dict
        The metadata dictionary to update.
    step_name : str
        Name of the processing step (e.g., "imwrite", "scan_phase_correction",
        "format_conversion", "z_registration").
    input_files : list of str or str, optional
        List of input file paths for this step.
    output_files : list of str or str, optional
        List of output file paths for this step.
    duration_seconds : float, optional
        How long this step took.
    extra : dict, optional
        Additional metadata for this step (e.g., scan-phase parameters,
        output format, compression settings).

    Returns
    -------
    dict
        The updated metadata dictionary.

    Examples
    --------
    >>> metadata = {}
    >>> add_processing_step(
    ...     metadata,
    ...     "imwrite",
    ...     input_files=["raw.tif"],
    ...     output_files=["output.zarr"],
    ...     extra={"output_format": ".zarr", "fix_phase": True, "use_fft": True}
    ... )
    """
    if "processing_history" not in metadata:
        metadata["processing_history"] = []

    step_record = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "mbo_utilities_version": _get_mbo_version(),
    }

    if input_files is not None:
        if isinstance(input_files, (str, Path)):
            step_record["input_files"] = [str(input_files)]
        else:
            step_record["input_files"] = [str(f) for f in input_files]

    if output_files is not None:
        if isinstance(output_files, (str, Path)):
            step_record["output_files"] = [str(output_files)]
        else:
            step_record["output_files"] = [str(f) for f in output_files]

    if duration_seconds is not None:
        step_record["duration_seconds"] = round(duration_seconds, 2)

    if extra is not None:
        step_record.update(extra)

    metadata["processing_history"].append(step_record)
    return metadata



def _close_specific_bin_writer(filepath):
    """Close a specific binary writer by filepath (thread-safe)."""
    if hasattr(_write_bin, "_writers"):
        key = str(Path(filepath))
        if key in _write_bin._writers:
            _write_bin._writers[key].close()
            _write_bin._writers.pop(key, None)
            _write_bin._offsets.pop(key, None)


def _close_specific_tiff_writer(filepath):
    """Close a specific TIFF writer by filepath (thread-safe)."""
    if hasattr(_write_tiff, "_writers"):
        # Key must match the type used in _write_tiff (Path object, not string)
        key = Path(filepath).with_suffix(".tif")
        if key in _write_tiff._writers:
            _write_tiff._writers[key].close()
            _write_tiff._writers.pop(key, None)
            if hasattr(_write_tiff, "_first_write"):
                _write_tiff._first_write.pop(key, None)


def _close_all_tiff_writers():
    """Close all open TIFF writers (for testing/cleanup)."""
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()
        if hasattr(_write_tiff, "_first_write"):
            _write_tiff._first_write.clear()


def _close_specific_npy_writer(filepath):
    """Close a specific .npy memory-mapped writer by filepath (thread-safe).

    Packages the data with metadata into a single .npy file using np.savez format.
    """
    if hasattr(_write_npy, "_arrays"):
        key = str(Path(filepath).with_suffix(".npy"))
        if key in _write_npy._arrays:
            mmap = _write_npy._arrays[key]
            metadata = _write_npy._metadata.get(key, {})

            # Read data from memmap before closing
            data = np.array(mmap)

            # Flush and close the memmap
            if hasattr(mmap, "flush"):
                mmap.flush()
            if hasattr(mmap, "_mmap") and mmap._mmap is not None:
                mmap._mmap.close()

            # Remove temp file
            temp_path = Path(key).with_suffix(".npy.tmp")
            if temp_path.exists():
                temp_path.unlink()

            # Save as npz with data and metadata, but use .npy extension
            final_path = Path(key)
            # np.savez saves as .npz, so we save to .npz then rename
            npz_path = final_path.with_suffix(".npz")
            np.savez(npz_path, data=data, metadata=np.array(metadata, dtype=object))

            # Rename .npz to .npy (unconventional but works with np.load)
            if final_path.exists():
                final_path.unlink()
            npz_path.rename(final_path)

            _write_npy._arrays.pop(key, None)
            _write_npy._offsets.pop(key, None)
            _write_npy._metadata.pop(key, None)


def compute_pad_from_shifts(plane_shifts):
    shifts = np.asarray(plane_shifts, dtype=int)
    dy_min, dx_min = shifts.min(axis=0)
    dy_max, dx_max = shifts.max(axis=0)
    pad_top = max(0, -dy_min)
    pad_bottom = max(0, dy_max)
    pad_left = max(0, -dx_min)
    pad_right = max(0, dx_max)
    return pad_top, pad_bottom, pad_left, pad_right


def _write_plane(
    data: np.ndarray | Any,
    filename: Path,
    *,
    overwrite=False,
    metadata=None,
    target_chunk_mb=20,
    progress_callback=None,
    debug=False,
    dshape=None,
    plane_index=None,
    shift_vector=None,
    **kwargs,
):
    if dshape is None:
        dshape = data.shape

    metadata = metadata or {}

    if plane_index is not None:
        if not isinstance(plane_index, int):
            raise TypeError(f"plane_index must be an integer, got {type(plane_index)}")
        metadata["plane"] = plane_index + 1

    # Get target frame count (nframes is primary, num_frames is alias)
    nframes_target = (
        kwargs.get("nframes")
        or kwargs.get("num_frames")
        or metadata.get("nframes")
        or metadata.get("num_frames")
    )

    if nframes_target is None or nframes_target == 0:
        nframes_target = data.shape[0]

    nframes_target = int(nframes_target)
    metadata["nframes"] = nframes_target
    metadata["num_frames"] = nframes_target  # alias for backwards compatibility

    # Update dshape to use the target frame count, not the original array shape
    # This ensures metadata["shape"] matches metadata["nframes"]
    dshape = (nframes_target, *dshape[1:])
    metadata["shape"] = dshape

    H0, W0 = data.shape[-2], data.shape[-1]
    fname = filename
    writer = _get_file_writer(fname.suffix, overwrite=overwrite)

    # get chunk size via bytes per timepoint
    itemsize = np.dtype(data.dtype).itemsize
    ntime = int(nframes_target)  # T

    # Handle empty data
    if ntime == 0:
        raise ValueError(f"Cannot write file with 0 frames. Data shape: {data.shape}, nframes_target: {nframes_target}")

    bytes_per_t = int(np.prod(dshape[1:], dtype=np.int64)) * int(itemsize)
    chunk_size = int(target_chunk_mb) * 1024 * 1024

    if chunk_size <= 0:
        chunk_size = 20 * 1024 * 1024

    total_bytes = int(ntime) * int(bytes_per_t)  # keep in int64 range
    nchunks = max(1, math.ceil(total_bytes / chunk_size))

    # don't create more chunks than timepoints
    nchunks = min(nchunks, ntime)

    # distribute frames across chunks as evenly as possible
    base = ntime // nchunks
    extra = ntime % nchunks

    if not debug:
        pbar = tqdm(total=nchunks, desc=f"Saving {fname.name}")
    else:
        pbar = None

    shift_applied = False

    apply_shift = metadata.get("apply_shift", False)
    summary = metadata.get("summary", "")
    s3d_job_dir = metadata.get("s3d-job", "")

    if fname.name == "data_raw.bin":
        # if saving suite2p intermediate
        apply_shift = False

    if shift_vector is not None:
        logger.debug(
            f"Using provided shift_vector of type {type(shift_vector)} length {len(shift_vector)}"
        )
        apply_shift = True
        if plane_index is not None:
            iy, ix = map(int, shift_vector)
            pt, pb, pl, pr = compute_pad_from_shifts([shift_vector])
            H_out = H0 + pt + pb
            W_out = W0 + pl + pr
            yy = slice(pt + iy, pt + iy + H0)
            xx = slice(pl + ix, pl + ix + W0)
            out_shape = (ntime, H_out, W_out)
            shift_applied = True
            metadata[f"plane{plane_index}_shift"] = (iy, ix)
        else:
            raise ValueError("plane_index must be provided when using shift_vector")

    if apply_shift and not shift_applied:
        if summary:
            summary_path = Path(summary).joinpath("summary.npy")
        else:
            summary_path = Path(s3d_job_dir).joinpath("summary/summary.npy")

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Summary file not found in s3d-job directory.\n"
                f"Expected: {summary_path}\n"
                f"s3d_job_dir: {s3d_job_dir}\n"
                f"This usually means Suite3D registration failed or is incomplete."
            )

        try:
            summary = load_npy(summary_path).item()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load summary file: {summary_path}\nError: {e}"
            )

        if not isinstance(summary, dict):
            raise ValueError(
                f"Summary file is not a dict: {type(summary)}\nPath: {summary_path}"
            )

        if "plane_shifts" not in summary:
            raise KeyError(
                f"Summary file is missing 'plane_shifts' key.\n"
                f"Available keys: {list(summary.keys())}\n"
                f"Path: {summary_path}"
            )

        plane_shifts = summary["plane_shifts"]

        if not isinstance(plane_shifts, (list, np.ndarray)):
            raise TypeError(
                f"plane_shifts has invalid type: {type(plane_shifts)}\n"
                f"Expected list or ndarray"
            )

        plane_shifts = np.asarray(plane_shifts)

        if plane_shifts.ndim != 2 or plane_shifts.shape[1] != 2:
            raise ValueError(
                f"plane_shifts has invalid shape: {plane_shifts.shape}\n"
                f"Expected (n_planes, 2)"
            )

        if plane_index is None:
            raise ValueError("plane_index must be provided when using shifts")

        if plane_index >= len(plane_shifts):
            raise IndexError(
                f"plane_index {plane_index} is out of range for plane_shifts "
                f"with length {len(plane_shifts)}"
            )

        pt, pb, pl, pr = compute_pad_from_shifts(plane_shifts)
        H_out = H0 + pt + pb
        W_out = W0 + pl + pr

        iy, ix = map(int, plane_shifts[plane_index])
        yy = slice(pt + iy, pt + iy + H0)
        xx = slice(pl + ix, pl + ix + W0)
        out_shape = (ntime, H_out, W_out)
        shift_applied = True
        metadata[f"plane{plane_index}_shift"] = (iy, ix)
        logger.debug(f"Applying shift for plane {plane_index}: y={iy}, x={ix}")

    if not shift_applied:
        out_shape = (ntime, H0, W0)

    start = 0
    for i in range(nchunks):
        end = start + base + (1 if i < extra else 0)

        # Extract chunk - handle plane_index for z-plane selection
        # NOTE: Use len(data.shape) instead of data.ndim for MboRawArray compatibility
        # (MboRawArray.ndim returns metadata ndim, not actual dimensions)
        if plane_index is not None and len(data.shape) >= 4:
            # For 4D data with plane_index, extract the specific z-plane
            # Index both time and z dimensions in one operation
            chunk = data[start:end, plane_index, :, :]
        elif plane_index is not None:
            # For 3D or 2D data, plane_index is just metadata
            chunk = data[start:end]
        else:
            # No plane_index: standard slicing
            chunk = data[start:end]

        # Convert lazy/disk-backed arrays to contiguous numpy arrays
        # This is critical for performance - memmap slices pass isinstance(np.ndarray)
        # but are extremely slow when passed to writers (220x+ slower)
        if hasattr(chunk, 'compute'):
            # Dask arrays - can hang during implicit compute in writers
            chunk = chunk.compute()
        elif isinstance(chunk, np.memmap):
            # Memmap slices are disk-backed - force copy to memory for fast writes
            chunk = np.array(chunk)
        elif not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)

        # Ensure chunk is 3D (T, Y, X) - squeeze any remaining singleton dimensions
        # This handles cases where plane_index is None but Z dimension is singleton
        if len(chunk.shape) == 4 and chunk.shape[1] == 1:
            chunk = chunk.squeeze(axis=1)

        if shift_applied:
            if chunk.shape[-2:] != (H0, W0):
                if chunk.shape[-2:] == (W0, H0):
                    chunk = np.swapaxes(chunk, -1, -2)
                else:
                    raise ValueError(
                        f"Unexpected chunk shape {chunk.shape[-2:]}, expected {(H0, W0)}"
                    )

            buf = np.zeros(
                (chunk.shape[0], out_shape[1], out_shape[2]), dtype=chunk.dtype
            )
            # if chunk is 4D with singleton second dim, squeeze it
            buf[:, yy, xx] = chunk
            metadata["padded_shape"] = buf.shape

            writer(fname, buf, metadata=metadata, **kwargs)
        else:
            writer(fname, chunk, metadata=metadata, **kwargs)

        if pbar:
            pbar.update(1)
        if progress_callback:
            progress_callback(pbar.n / pbar.total, current_plane=plane_index)
        start = end
    if pbar:
        pbar.close()

    # Close only the specific writer for this file (thread-safe)
    if fname.suffix in [".tiff", ".tif"]:
        _close_specific_tiff_writer(fname)
    elif fname.suffix in [".bin"]:
        _close_specific_bin_writer(fname)
    elif fname.suffix in [".npy"]:
        _close_specific_npy_writer(fname)



def _get_file_writer(ext, overwrite):
    if ext.startswith("."):
        ext = ext.lstrip(".")
    if ext in ["tif", "tiff"]:
        return functools.partial(
            _write_tiff,
            overwrite=overwrite,
        )
    elif ext in ["h5", "hdf5"]:
        return functools.partial(
            _write_h5,
            overwrite=overwrite,
        )
    elif ext in ["zarr"]:
        return functools.partial(
            _write_zarr,
            overwrite=overwrite,
        )
    elif ext == "bin":
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
        )
    elif ext == "npy":
        return functools.partial(
            _write_npy,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
    # Import here to avoid circular import
    from .array_types import BinArray

    if metadata is None:
        metadata = {}

    if not hasattr(_write_bin, "_writers"):
        _write_bin._writers, _write_bin._offsets = {}, {}

    fname = Path(path)
    fname.parent.mkdir(exist_ok=True)

    key = str(fname)
    first_write = False

    # drop cached writer if file was deleted externally
    if key in _write_bin._writers and not Path(key).exists():
        _write_bin._writers.pop(key, None)
        _write_bin._offsets.pop(key, None)

    # Only overwrite if this is a brand new write session (file doesn't exist in cache)
    # Don't delete during active chunked writing
    if overwrite and key not in _write_bin._writers and fname.exists():
        fname.unlink()

    if key not in _write_bin._writers:
        Ly, Lx = data.shape[-2], data.shape[-1]
        nframes = metadata.get("nframes", None)
        if nframes is None:
            nframes = metadata.get("num_frames", None)
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'num_frames'.")

        _write_bin._writers[key] = BinArray(
            filename=key,
            shape=(nframes, Ly, Lx),
            dtype=np.int16,
        )
        _write_bin._offsets[key] = 0
        first_write = True

    bf = _write_bin._writers[key]
    off = _write_bin._offsets[key]

    # Squeeze singleton Z dimension if present (but only Z, not time)
    # NOTE: Use len(data.shape) instead of data.ndim for MboRawArray compatibility
    if len(data.shape) == 4 and data.shape[1] == 1:
        data = data.squeeze(axis=1)

    bf[off : off + data.shape[0]] = data
    bf.flush()
    _write_bin._offsets[key] = off + data.shape[0]

    if first_write:
        write_ops(metadata, fname, **kwargs)


def _write_npy(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
    """
    Write data to a .npy file with chunked/streaming support.

    Uses memory-mapped file for efficient chunked writing.
    Metadata is embedded in the file using np.savez format (stored as .npy).
    """
    if metadata is None:
        metadata = {}

    if not hasattr(_write_npy, "_arrays"):
        _write_npy._arrays = {}
        _write_npy._offsets = {}
        _write_npy._metadata = {}

    fname = Path(path).with_suffix(".npy")
    fname.parent.mkdir(parents=True, exist_ok=True)

    key = str(fname)

    # Drop cached array if file was deleted externally
    if key in _write_npy._arrays and not fname.exists():
        _write_npy._arrays.pop(key, None)
        _write_npy._offsets.pop(key, None)
        _write_npy._metadata.pop(key, None)

    # Only overwrite if this is a brand new write session
    if overwrite and key not in _write_npy._arrays and fname.exists():
        fname.unlink()

    if key not in _write_npy._arrays:
        # Get target shape from metadata
        nframes = metadata.get("nframes") or metadata.get("num_frames")
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'num_frames'.")

        h, w = data.shape[-2], data.shape[-1]
        shape = (int(nframes), h, w)

        # Use a temporary file for chunked writing, then package with metadata at close
        temp_fname = fname.with_suffix(".npy.tmp")

        # Create memory-mapped array for chunked writing
        mmap = np.lib.format.open_memmap(
            temp_fname,
            mode="w+",
            dtype=data.dtype,
            shape=shape,
        )
        _write_npy._arrays[key] = mmap
        _write_npy._offsets[key] = 0
        _write_npy._metadata[key] = _make_json_serializable(metadata)

    mmap = _write_npy._arrays[key]
    off = _write_npy._offsets[key]

    # Squeeze singleton Z dimension if present
    if len(data.shape) == 4 and data.shape[1] == 1:
        data = data.squeeze(axis=1)

    # Write chunk
    mmap[off : off + data.shape[0]] = data
    mmap.flush()
    _write_npy._offsets[key] = off + data.shape[0]


def _close_npy_writers():
    """Close all open .npy memory-mapped writers."""
    if hasattr(_write_npy, "_arrays"):
        # Close each writer properly to package data with metadata
        keys = list(_write_npy._arrays.keys())
        for key in keys:
            _close_specific_npy_writer(key)


def _write_h5(path, data, *, overwrite=True, metadata=None, **kwargs):
    if metadata is None:
        metadata = {}

    filename = Path(path).with_suffix(".h5")

    if not hasattr(_write_h5, "_initialized"):
        _write_h5._initialized = {}
        _write_h5._offsets = {}

    if filename not in _write_h5._initialized:
        nframes = metadata.get("num_frames", None)
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'nun_frames'.")
        h, w = data.shape[-2:]
        with h5py.File(filename, "w" if overwrite else "a") as f:
            f.create_dataset(
                "mov",
                shape=(nframes, h, w),
                maxshape=(None, h, w),
                chunks=(1, h, w),
                dtype=data.dtype,
                compression=None,
            )
            if metadata:
                for k, v in metadata.items():
                    f.attrs[k] = v if np.isscalar(v) else str(v)

        _write_h5._initialized[filename] = True
        _write_h5._offsets[filename] = 0

    offset = _write_h5._offsets[filename]

    with h5py.File(filename, "a") as f:
        f["mov"][offset : offset + data.shape[0]] = data

    _write_h5._offsets[filename] = offset + data.shape[0]


def _write_tiff(path, data, overwrite=True, metadata=None, **kwargs):
    if metadata is None:
        metadata = {}

    filename = Path(path).with_suffix(".tif")

    if not hasattr(_write_tiff, "_writers"):
        _write_tiff._writers = {}
    if not hasattr(_write_tiff, "_first_write"):
        _write_tiff._first_write = {}

    # Check if we're starting a new write session (no writer exists yet)
    is_new_session = filename not in _write_tiff._writers

    # Handle overwrite logic ONLY at the start of a new write session
    if is_new_session:
        if filename.exists() and not overwrite:
            logger.warning(
                f"File {filename} already exists and overwrite=False. Skipping write."
            )
            return

        if filename.exists() and overwrite:
            # Delete existing file before creating new writer
            # On Windows, retry if file is locked by another process
            import time
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    filename.unlink()
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        import gc
                        gc.collect()
                    else:
                        raise

        # Create new writer
        _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True)
        _write_tiff._first_write[filename] = True

    writer = _write_tiff._writers[filename]
    is_first = _write_tiff._first_write.get(filename, True)

    for frame in data:
        writer.write(
            frame,
            contiguous=True,
            photometric="minisblack",
            metadata=_make_json_serializable(metadata) if is_first else {},
        )
    _write_tiff._first_write[filename] = False


def _build_ome_metadata(shape: tuple, metadata: dict) -> dict:
    """
    Build OME-Zarr NGFF v0.5 compliant metadata.

    Parameters
    ----------
    shape : tuple
        Shape of the array (T, Y, X) or (T, Z, Y, X)
    metadata : dict
        Metadata dict containing optional keys:
        - pixel_resolution : tuple (x, y) pixel size in micrometers
        - frame_rate : float, sampling rate in Hz
        - fs : float, alias for frame_rate
        - dx, dy : float, pixel sizes in micrometers
        - dz : float, z-step in micrometers (for 4D data)
        - z_step : float, alias for dz
        - voxel_size : tuple (dx, dy, dz) in micrometers

    Returns
    -------
    dict
        OME-Zarr NGFF v0.5 metadata dictionary ready for zarr.attrs.update()
    """
    from mbo_utilities.metadata import get_voxel_size, get_param

    ndim = len(shape)

    # extract spatial scales using standardized resolution getter
    vs = get_voxel_size(metadata)
    pixel_x = vs.dx
    pixel_y = vs.dy
    z_scale = vs.dz

    # extract temporal scale using canonical parameter access
    frame_rate = get_param(metadata, "fs")
    if frame_rate:
        time_scale = 1.0 / float(frame_rate)
    else:
        time_scale = 1.0

    # Build axes definition
    # Order: time (if present) -> channel (if present) -> spatial (z, y, x)
    axes = []

    if ndim == 3:
        # Shape is (T, Y, X)
        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        scale_values = [time_scale, pixel_y, pixel_x]

    elif ndim == 4:
        # Shape is (T, Z, Y, X)
        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        scale_values = [time_scale, z_scale, pixel_y, pixel_x]

    else:
        # Fallback for unexpected dimensions
        logger.warning(
            f"Unexpected dimensionality {ndim} for OME-Zarr. "
            f"OME-Zarr expects 3D (TYX) or 4D (TZYX) data."
        )
        axes = [{"name": f"dim_{i}", "type": "space"} for i in range(ndim)]
        scale_values = [1.0] * ndim

    # Build OME-NGFF v0.5 metadata
    # coordinateTransformations in each dataset
    coordinate_transforms = [{"type": "scale", "scale": scale_values}]
    datasets = [{"path": "0", "coordinateTransformations": coordinate_transforms}]

    multiscales = [
        {
            "version": "0.5",
            "name": metadata.get("name", ""),
            "axes": axes,
            "datasets": datasets,
        }
    ]

    # v0.5 uses "ome" namespace
    ome_content = {
        "version": "0.5",
        "multiscales": multiscales,
    }

    # Add optional OMERO rendering metadata if present
    omero_metadata = {}
    if "channel_names" in metadata:
        omero_metadata["channels"] = metadata["channel_names"]
    if omero_metadata:
        ome_content["omero"] = omero_metadata

    result = {"ome": ome_content}

    serializable_metadata = _make_json_serializable(metadata)
    for k, v in serializable_metadata.items():
        if k == "channel_names":
            # Already encoded in OME omero section
            continue
        result[k] = v

    return result


def _write_zarr(
    path,
    data,
    *,
    overwrite=True,
    metadata=None,
    **kwargs,
):
    sharded = kwargs.get("sharded", True)
    ome = kwargs.get("ome", True)
    level = kwargs.get("level", 1)

    if metadata is None:
        metadata = {}

    filename = Path(path)
    if not hasattr(_write_zarr, "_arrays"):
        _write_zarr._arrays = {}
        _write_zarr._offsets = {}
        _write_zarr._groups = {}

    # Only overwrite if this is a brand new write session (file doesn't exist in cache)
    # Don't delete during active chunked writing
    if overwrite and filename not in _write_zarr._arrays and filename.exists():
        shutil.rmtree(filename)

    if filename not in _write_zarr._arrays:

        import zarr
        from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec, Crc32cCodec

        nframes = int(metadata["num_frames"])
        h, w = data.shape[-2:]

        if sharded:
            outer = (min(nframes, 100), h, w)  # 100-frame shards
            inner = (1, h, w)
            codec = ShardingCodec(
                chunk_shape=inner,
                codecs=[BytesCodec(), GzipCodec(level=level)],
                index_codecs=[BytesCodec(), Crc32cCodec()],
            )
            codecs = [codec]
            chunks = outer
        else:
            codecs = None
            chunks = (1, h, w)

        if ome:
            # Create OME-Zarr using NGFF v0.5 with Zarr v3
            # Structure: my_image.zarr/ (group) -> 0/ (array)

            # Create Zarr v3 group
            root = zarr.open_group(str(filename), mode="w", zarr_format=3)

            # Prepare codecs for v3
            if sharded:
                outer = (min(nframes, 100), h, w)  # 100-frame shards
                inner = (1, h, w)
                codec = ShardingCodec(
                    chunk_shape=inner,
                    codecs=[BytesCodec(), GzipCodec(level=level)],
                    index_codecs=[BytesCodec(), Crc32cCodec()],
                )
                array_codecs = [codec]
                array_chunks = outer
            else:
                # Use default v3 codecs (no compression)
                array_codecs = None
                array_chunks = chunks

            # Create the array as "0" (full resolution level)
            z = zarr.create(
                store=root.store,
                path="0",
                shape=(nframes, h, w),
                chunks=array_chunks,
                dtype=data.dtype,
                codecs=array_codecs,
                overwrite=True,
            )

            # Build and set OME metadata on the GROUP
            ome_metadata = _build_ome_metadata(
                shape=(nframes, h, w),
                metadata=metadata or {},
            )

            # Set metadata on the group
            for key, value in ome_metadata.items():
                root.attrs[key] = value

            _write_zarr._groups[filename] = root
        else:
            # Standard non-OME zarr (backward compatible)
            z = zarr.create(
                store=str(filename),
                shape=(nframes, h, w),
                chunks=chunks,
                dtype=data.dtype,
                codecs=codecs,
                overwrite=True,
            )

            # Standard metadata (backward compatible)
            # Ensure metadata is JSON-serializable for Zarr
            serializable_metadata = _make_json_serializable(metadata or {})
            for k, v in serializable_metadata.items():
                z.attrs[k] = v

        _write_zarr._arrays[filename] = z
        _write_zarr._offsets[filename] = 0

    z = _write_zarr._arrays[filename]
    offset = _write_zarr._offsets[filename]

    z[offset : offset + data.shape[0]] = data
    _write_zarr._offsets[filename] = offset + data.shape[0]


def _try_generic_writers(
    data: Any,
    outpath: str | Path,
    overwrite: bool = True,
    metadata: dict | None = None,
):
    import shutil
    import gc
    import time

    if metadata is None:
        metadata = {}

    outpath = Path(outpath)
    if outpath.exists():
        if not overwrite:
            raise FileExistsError(f"{outpath} already exists and overwrite=False")
        # Remove existing file or directory to allow overwrite
        if outpath.is_dir():
            shutil.rmtree(outpath)
        else:
            # Force garbage collection to release any open file handles
            gc.collect()
            try:
                outpath.unlink()
            except PermissionError:
                # Windows file locking - wait briefly and retry
                time.sleep(0.1)
                gc.collect()
                outpath.unlink()

    if outpath.suffix.lower() in {".npy", ".npz"}:
        if metadata is None:
            np.save(outpath, data)
        else:
            # Convert Path objects to strings for cross-platform compatibility
            np.savez(outpath, data=data, metadata=_convert_paths_to_strings(metadata))
    elif outpath.suffix.lower() in {".tif", ".tiff"}:
        imwrite(
            outpath,
            data,
            metadata=_make_json_serializable(metadata),
            photometric="minisblack",
            contiguous=True,
        )
    elif outpath.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(outpath, "w" if overwrite else "a") as f:
            f.create_dataset("data", data=data)
            if metadata:
                for k, v in metadata.items():
                    f.attrs[k] = v if np.isscalar(v) else str(v)
    elif outpath.suffix.lower() == ".bin":
        # Suite2p binary format - write data + ops.npy
        arr = np.asarray(data)
        if arr.dtype != np.int16:
            arr = arr.astype(np.int16)

        # Write binary data
        with open(outpath, "wb") as f:
            arr.tofile(f)

        # Write ops.npy alongside
        if metadata:
            ops = metadata.copy()
            ops["Ly"] = arr.shape[-2] if arr.ndim >= 2 else 1
            ops["Lx"] = arr.shape[-1] if arr.ndim >= 1 else 1
            ops["nframes"] = arr.shape[0] if arr.ndim >= 1 else 1
            # Convert Path objects to strings for cross-platform compatibility
            np.save(outpath.parent / "ops.npy", _convert_paths_to_strings(ops))
    elif outpath.suffix.lower() == ".zarr":
        # Zarr v3 format for numpy arrays
        import zarr
        from zarr.codecs import BytesCodec, GzipCodec

        arr = np.asarray(data)

        # Compute chunks: (1, ..., H, W) for time-series data
        if arr.ndim >= 3:
            chunks = (1,) * (arr.ndim - 2) + arr.shape[-2:]
        else:
            chunks = arr.shape

        # Create zarr array with compression
        z = zarr.create(
            store=str(outpath),
            shape=arr.shape,
            chunks=chunks,
            dtype=arr.dtype,
            codecs=[BytesCodec(), GzipCodec(level=5)],
            overwrite=True,
            zarr_format=3,
        )
        z[:] = arr

        # Add metadata as attributes if provided
        if metadata:
            for k, v in metadata.items():
                try:
                    z.attrs[k] = v if np.isscalar(v) or isinstance(v, (list, dict, str)) else str(v)
                except Exception:
                    z.attrs[k] = str(v)
    else:
        raise ValueError(f"Unsupported file extension: {outpath.suffix}")


def write_ops(metadata, raw_filename, **kwargs):
    """
    Write metadata to an ops file alongside the given filename.

    This creates a Suite2p-compatible ops.npy file from the provided metadata.
    The ops file is used by Suite2p for processing configuration.

    Parameters
    ----------
    metadata : dict
        Must contain 'shape' key with (T, Y, X) dimensions.
        Optional keys: 'pixel_resolution', 'frame_rate', 'fs', 'dx', 'dy', 'dz'.
    raw_filename : str or Path
        Path to the data file (e.g., data_raw.bin). The ops.npy will be
        written to the same directory.
    **kwargs
        Additional arguments. 'structural=True' indicates channel 2 data.
    """
    logger.debug(f"Writing ops file for {raw_filename} with metadata: {metadata}")
    if not isinstance(raw_filename, (str, Path)):
        raise TypeError(f"raw_filename must be str or Path, got {type(raw_filename)}")
    filename = Path(raw_filename).expanduser().resolve()

    structural = kwargs.get("structural", False)
    chan = 2 if structural or "data_chan2.bin" in str(filename) else 1
    logger.debug(f"Detected channel {chan}")

    # Always use parent directory - raw_filename should be a file path like data_raw.bin
    # The old check `filename.is_file()` failed when file was just created but not yet flushed
    if filename.suffix:
        # Has a file extension, use parent as root
        root = filename.parent
    else:
        # No extension, assume it's a directory path (backward compatibility)
        root = filename if filename.is_dir() else filename.parent
    ops_path = root / "ops.npy"
    logger.info(f"Writing ops file to {ops_path}")

    shape = metadata["shape"]
    nt, Ly, Lx = shape[0], shape[-2], shape[-1]  # shape is (T, Y, X), so [-2]=Ly, [-1]=Lx

    # Check if num_frames was explicitly set (takes precedence over shape)
    if "num_frames" in metadata:
        nt_metadata = int(metadata["num_frames"])
        if nt_metadata != shape[0]:
            raise ValueError(
                f"Inconsistent frame count in metadata!\n"
                f"metadata['num_frames'] = {nt_metadata}\n"
                f"metadata['shape'][0] = {shape[0]}\n"
                f"These must match. Check your data and metadata."
            )
        nt = nt_metadata
        logger.debug(f"Using explicit num_frames={nt} from metadata")
    elif "nframes" in metadata:
        nt_metadata = int(metadata["nframes"])
        if nt_metadata != shape[0]:
            raise ValueError(
                f"Inconsistent frame count in metadata!\n"
                f"metadata['nframes'] = {nt_metadata}\n"
                f"metadata['shape'][0] = {shape[0]}\n"
                f"These must match. Check your data and metadata."
            )
        nt = nt_metadata
        logger.debug(f"Using explicit nframes={nt} from metadata")

    if "fs" not in metadata:
        if "frame_rate" in metadata:
            metadata["fs"] = metadata["frame_rate"]
        elif "framerate" in metadata:
            metadata["fs"] = metadata["framerate"]
        else:
            logger.warning("No frame rate found; defaulting fs=10")
            metadata["fs"] = 10

    # Use get_voxel_size for consistent resolution handling across all aliases
    from mbo_utilities.metadata import get_voxel_size
    voxel_size = get_voxel_size(metadata)
    dx, dy, dz = voxel_size.dx, voxel_size.dy, voxel_size.dz

    # Load or initialize ops
    if ops_path.exists():
        ops = load_npy(ops_path).item()
    else:
        from mbo_utilities.metadata import default_ops
        ops = default_ops()

    # Update shared core fields - ensure all resolution aliases are consistent
    ops.update(
        {
            "Ly": Ly,
            "Lx": Lx,
            "fs": metadata["fs"],
            # Canonical resolution keys
            "dx": dx,
            "dy": dy,
            "dz": dz,
            # Suite2p aliases (must match canonical)
            "umPerPixX": dx,
            "umPerPixY": dy,
            "umPerPixZ": dz,
            # legacy compatibility
            "pixel_resolution": (dx, dy),
            "z_step": dz,
            "ops_path": str(ops_path),
        }
    )

    # Channel-specific entries
    # Use the potentially overridden nt (from num_frames or nframes)
    if chan == 1:
        ops["nframes_chan1"] = nt
        ops["raw_file"] = str(filename)
    else:
        ops["nframes_chan2"] = nt
        ops["chan2_file"] = str(filename)

    ops["align_by_chan"] = chan

    # Set top-level nframes to match the written channel
    # This ensures consistency between nframes and nframes_chan1/chan2
    ops["nframes"] = nt

    # Merge extra metadata, but DON'T overwrite fields we've already set consistently
    # This prevents inconsistency between resolution aliases and frame counts
    protected_keys = {
        # Frame count fields
        "nframes", "nframes_chan1", "nframes_chan2", "num_frames",
        # Resolution fields (we've already set these consistently)
        "dx", "dy", "dz", "umPerPixX", "umPerPixY", "umPerPixZ",
        "pixel_resolution", "z_step",
    }
    for key, value in metadata.items():
        if key not in protected_keys:
            ops[key] = value

    # Convert Path objects to strings for cross-platform compatibility
    np.save(ops_path, _convert_paths_to_strings(ops))
    logger.debug(
        f"Ops file written to {ops_path} with nframes={ops['nframes']}, nframes_chan1={ops.get('nframes_chan1')}"
    )


def to_video(
    data,
    output_path,
    fps: int = 30,
    speed_factor: float = 1.0,
    plane: int = None,
    vmin: float = None,
    vmax: float = None,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
    temporal_smooth: int = 0,
    spatial_smooth: float = 0,
    gamma: float = 1.0,
    cmap: str = None,
    quality: int = 9,
    codec: str = "libx264",
    max_frames: int = None,
):
    """
    Export array data to video file (mp4/avi).

    Works with 3D (T, Y, X) or 4D (T, Z, Y, X) arrays, including lazy arrays.
    Optimized for high-quality output suitable for presentations and websites.

    Parameters
    ----------
    data : array-like
        3D array (T, Y, X) or 4D array (T, Z, Y, X). Supports lazy arrays.
    output_path : str or Path
        Output video path. Extension determines format (.mp4, .avi, .mov).
    fps : int, default 30
        Base frame rate of the recording.
    speed_factor : float, default 1.0
        Playback speed multiplier. speed_factor=10 plays 10x faster (all frames
        included, just faster playback). Use this to show cell stability quickly.
    plane : int, optional
        For 4D arrays, which z-plane to export (0-indexed). If None, exports plane 0.
    vmin : float, optional
        Min value for intensity scaling. If None, uses vmin_percentile.
    vmax : float, optional
        Max value for intensity scaling. If None, uses vmax_percentile.
    vmin_percentile : float, default 1.0
        Percentile for auto vmin calculation. Lower = darker blacks.
    vmax_percentile : float, default 99.5
        Percentile for auto vmax calculation. Lower = brighter highlights.
    temporal_smooth : int, default 0
        Rolling average window size (frames). Reduces flicker/noise.
        0 = disabled, 3-7 = subtle smoothing, 10+ = heavy smoothing.
    spatial_smooth : float, default 0
        Gaussian blur sigma (pixels). Reduces pixel noise.
        0 = disabled, 0.5-1.0 = subtle, 2+ = heavy blur.
    gamma : float, default 1.0
        Gamma correction. <1 = brighter midtones, >1 = darker midtones.
        0.7-0.8 often looks good for calcium imaging.
    cmap : str, optional
        Matplotlib colormap name (e.g., "viridis", "gray", "hot").
        If None, outputs grayscale.
    quality : int, default 9
        Video quality (1-10, higher is better). 9-10 recommended for web.
    codec : str, default "libx264"
        Video codec. "libx264" for mp4 (best compatibility).
    max_frames : int, optional
        Limit number of frames to export. If None, exports all frames.

    Returns
    -------
    Path
        Path to the created video file.

    Examples
    --------
    >>> from mbo_utilities import imread, to_video
    >>> arr = imread("data.tif")

    >>> # Quick preview at 10x speed (good for checking stability)
    >>> to_video(arr, "preview.mp4", speed_factor=10)

    >>> # High-quality export for website
    >>> to_video(arr, "movie.mp4", fps=30, speed_factor=5,
    ...          temporal_smooth=3, gamma=0.8, quality=10)

    >>> # Export specific z-plane from 4D data
    >>> to_video(arr, "plane3.mp4", plane=3, speed_factor=10)

    >>> # With colormap and custom intensity range
    >>> to_video(arr, "movie.mp4", cmap="viridis", vmin=100, vmax=2000)
    """
    import imageio
    from scipy.ndimage import gaussian_filter

    output_path = Path(output_path)

    # Get array info
    arr = data
    ndim = arr.ndim
    shape = arr.shape

    if ndim == 4:
        # (T, Z, Y, X) - select plane
        plane_idx = plane if plane is not None else 0
        if plane_idx >= shape[1]:
            raise ValueError(f"plane={plane_idx} but array only has {shape[1]} planes")
        n_frames = shape[0]
        height, width = shape[2], shape[3]
        logger.info(f"Exporting 4D array plane {plane_idx}: {n_frames} frames, {height}x{width}")
    elif ndim == 3:
        # (T, Y, X)
        plane_idx = None
        n_frames = shape[0]
        height, width = shape[1], shape[2]
        logger.info(f"Exporting 3D array: {n_frames} frames, {height}x{width}")
    else:
        raise ValueError(f"Expected 3D or 4D array, got {ndim}D")

    # Limit frames if requested
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    # Calculate output fps based on speed factor
    output_fps = int(fps * speed_factor)
    duration = n_frames / output_fps

    logger.info(
        f"Writing {n_frames} frames at {output_fps} fps "
        f"(speed_factor={speed_factor}x, duration={duration:.1f}s)"
    )

    # Determine intensity range from sample frames
    if vmin is None or vmax is None:
        # Sample frames across the video for percentile estimation
        n_samples = min(50, n_frames)
        sample_indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)
        samples = []
        for i in sample_indices:
            if ndim == 4:
                frame = np.asarray(arr[i, plane_idx])
            else:
                frame = np.asarray(arr[i])
            samples.append(frame)
        sample_stack = np.stack(samples)

        if vmin is None:
            vmin = float(np.percentile(sample_stack, vmin_percentile))
        if vmax is None:
            vmax = float(np.percentile(sample_stack, vmax_percentile))

    logger.info(f"Intensity range: [{vmin:.1f}, {vmax:.1f}]")

    # Setup colormap if requested
    if cmap is not None:
        try:
            import matplotlib.pyplot as plt
            colormap = plt.get_cmap(cmap)
        except ImportError:
            logger.warning("matplotlib not available, using grayscale")
            colormap = None
    else:
        colormap = None

    # Map quality (1-10) to crf (28-18, lower crf = better quality)
    crf = int(28 - (quality - 1) * (28 - 18) / 9)

    # Buffer for temporal smoothing
    frame_buffer = [] if temporal_smooth > 0 else None

    # Write video using imageio-ffmpeg
    writer = imageio.get_writer(
        str(output_path),
        fps=output_fps,
        codec=codec,
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],  # yuv420p for browser compatibility
    )

    try:
        for i in tqdm(range(n_frames), desc="Writing video", unit="frames"):
            # Get frame
            if ndim == 4:
                frame = np.asarray(arr[i, plane_idx], dtype=np.float32)
            else:
                frame = np.asarray(arr[i], dtype=np.float32)

            # Temporal smoothing (rolling average)
            if temporal_smooth > 0:
                frame_buffer.append(frame)
                if len(frame_buffer) > temporal_smooth:
                    frame_buffer.pop(0)
                frame = np.mean(frame_buffer, axis=0)

            # Spatial smoothing (Gaussian blur)
            if spatial_smooth > 0:
                frame = gaussian_filter(frame, sigma=spatial_smooth)

            # Normalize to 0-1
            frame = np.clip((frame - vmin) / (vmax - vmin), 0, 1)

            # Gamma correction
            if gamma != 1.0:
                frame = np.power(frame, gamma)

            # Convert to RGB
            if colormap is not None:
                # Apply colormap (returns RGBA)
                frame_rgb = (colormap(frame)[:, :, :3] * 255).astype(np.uint8)
            else:
                # Grayscale -> RGB
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_rgb = np.stack([frame_uint8] * 3, axis=-1)

            writer.append_data(frame_rgb)
    finally:
        writer.close()

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Video saved to {output_path} ({file_size_mb:.1f} MB)")
    return output_path
