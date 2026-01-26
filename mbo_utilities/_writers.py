import functools
import math
import warnings
from datetime import datetime
from typing import Any

import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter, imwrite as tiff_imwrite
import h5py

from . import log
from ._parsing import _make_json_serializable, _convert_paths_to_strings
from .util import load_npy
from .metadata.io import _build_ome_metadata

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
            if hasattr(_write_tiff, "_imagej_mode"):
                _write_tiff._imagej_mode.pop(key, None)


def _close_all_tiff_writers():
    """Close all open TIFF writers (for testing/cleanup)."""
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()
        if hasattr(_write_tiff, "_first_write"):
            _write_tiff._first_write.clear()
        if hasattr(_write_tiff, "_imagej_mode"):
            _write_tiff._imagej_mode.clear()


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
    """compute padding needed to accommodate all plane shifts."""
    shifts = np.asarray(plane_shifts, dtype=int)
    dy_min, dx_min = shifts.min(axis=0)
    dy_max, dx_max = shifts.max(axis=0)
    pad_top = max(0, -dy_min)
    pad_bottom = max(0, dy_max)
    pad_left = max(0, -dx_min)
    pad_right = max(0, dx_max)
    return pad_top, pad_bottom, pad_left, pad_right


def load_registration_shifts(metadata: dict | None, debug: bool = False):
    """
    load z-registration shifts from suite3d job directory.

    parameters
    ----------
    metadata : dict or None
        metadata dict containing 'apply_shift' flag and 's3d-job' path.
    debug : bool
        verbose logging.

    returns
    -------
    tuple
        (apply_shift, plane_shifts, padding) where:
        - apply_shift: bool, whether shifts should be applied
        - plane_shifts: ndarray or None, per-plane [dy, dx] shifts
        - padding: tuple (pt, pb, pl, pr) or (0, 0, 0, 0)
    """
    apply_shift = metadata.get("apply_shift", False) if metadata else False
    plane_shifts = None
    padding = (0, 0, 0, 0)

    if not apply_shift:
        return False, None, padding

    # load plane shifts from s3d-job
    s3d_job_dir = metadata.get("s3d-job", "")
    summary_path = Path(s3d_job_dir).joinpath("summary/summary.npy") if s3d_job_dir else None

    if summary_path and summary_path.is_file():
        try:
            summary = load_npy(summary_path).item()
            if isinstance(summary, dict) and "plane_shifts" in summary:
                plane_shifts = np.asarray(summary["plane_shifts"])
                padding = compute_pad_from_shifts(plane_shifts)
                if debug:
                    pt, pb, pl, pr = padding
                    logger.info(f"Loaded z-registration shifts with padding: top={pt}, bottom={pb}, left={pl}, right={pr}")
                return True, plane_shifts, padding
        except Exception as e:
            logger.warning(f"Failed to load plane shifts: {e}. Proceeding without registration.")

    logger.warning(f"apply_shift=True but no valid s3d-job summary found. Proceeding without registration.")
    return False, None, padding


def apply_shifts_to_chunk(chunk_data, plane_shifts, z_indices, padding, Ly_out, Lx_out):
    """
    apply z-registration shifts to a data chunk.

    parameters
    ----------
    chunk_data : ndarray
        input chunk with shape (T, Z, Y, X).
    plane_shifts : ndarray
        per-plane [dy, dx] shifts.
    z_indices : list
        z-plane indices (0-based) being written.
    padding : tuple
        (pt, pb, pl, pr) padding values.
    Ly_out, Lx_out : int
        output spatial dimensions including padding.

    returns
    -------
    ndarray
        shifted and padded chunk with shape (T, Z, Ly_out, Lx_out).
    """
    pt, pb, pl, pr = padding
    n_t, n_z_chunk, h, w = chunk_data.shape

    shifted_chunk = np.zeros((n_t, n_z_chunk, Ly_out, Lx_out), dtype=chunk_data.dtype)

    for z_local in range(n_z_chunk):
        z_global = z_indices[z_local]
        if z_global < len(plane_shifts):
            iy, ix = map(int, plane_shifts[z_global])
            yy = slice(pt + iy, pt + iy + h)
            xx = slice(pl + ix, pl + ix + w)
            shifted_chunk[:, z_local, yy, xx] = chunk_data[:, z_local, :, :]
        else:
            # no shift for this plane, center it
            shifted_chunk[:, z_local, pt:pt+h, pl:pl+w] = chunk_data[:, z_local, :, :]

    return shifted_chunk


def _write_plane(
    data: np.ndarray | Any,
    filename: Path,
    *,
    overwrite=False,
    metadata=None,
    target_chunk_mb=20,
    progress_callback=None,
    debug=False,
    show_progress=True,
    dshape=None,
    plane_index=None,
    shift_vector=None,
    **kwargs,
):
    if dshape is None:
        dshape = data.shape

    metadata = metadata or {}

    if plane_index is not None:
        if not isinstance(plane_index, (int, np.integer)):
            raise TypeError(f"plane_index must be an integer, got {type(plane_index)}")
        metadata["plane"] = int(plane_index) + 1

    # Get target frame count (nframes is primary, num_frames is alias)
    nframes_target = (
        kwargs.get("nframes")
        or kwargs.get("num_frames")
        or metadata.get("nframes")
        or metadata.get("num_frames")
    )

    if nframes_target is None or nframes_target <= 0:
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

    if show_progress and not debug:
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
        # NOTE: Use len(data.shape) instead of data.ndim for ScanImageArray compatibility
        # (ScanImageArray.ndim returns metadata ndim, not actual dimensions)
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
        if hasattr(chunk, "compute"):
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
    if ext in ["h5", "hdf5"]:
        return functools.partial(
            _write_h5,
            overwrite=overwrite,
        )
    if ext in ["zarr"]:
        return functools.partial(
            _write_zarr,
            overwrite=overwrite,
        )
    if ext == "bin":
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
        )
    if ext == "npy":
        return functools.partial(
            _write_npy,
            overwrite=overwrite,
        )
    raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
    # import here to avoid circular import
    from .arrays.bin import BinArray

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
        nframes = metadata.get("nframes")
        if nframes is None:
            nframes = metadata.get("num_frames")
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
    # NOTE: Use len(data.shape) instead of data.ndim for ScanImageArray compatibility
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
        nframes = metadata.get("num_frames")
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


def _build_imagej_metadata(metadata: dict, shape: tuple) -> tuple[dict, tuple]:
    """
    Build ImageJ-compatible metadata dict and resolution tuple.

    ImageJ expects metadata in a specific format in the ImageDescription tag.
    The key fields are:
    - spacing: z-step size in units
    - unit: physical unit (e.g., 'um')
    - finterval: frame interval in seconds
    - axes: dimension order (e.g., 'TYX', 'TZYX')
    - min/max: display range (optional)
    - loop: animation loop flag (optional)

    The resolution tuple is (pixels_per_unit_x, pixels_per_unit_y), which is
    the inverse of micrometers per pixel.

    Parameters
    ----------
    metadata : dict
        Source metadata dict with imaging parameters.
    shape : tuple
        Array shape (T, Y, X) or (T, Z, Y, X).

    Returns
    -------
    tuple[dict, tuple]
        (imagej_metadata, resolution) ready for tifffile.imwrite(imagej=True).
    """
    from mbo_utilities.metadata import get_voxel_size, get_param

    # get voxel size
    vs = get_voxel_size(metadata)
    dx, dy, dz = vs.dx, vs.dy, vs.dz

    # resolution is pixels per unit (inverse of um/pixel)
    # ImageJ uses these values directly with the 'unit' field
    # so if dx=0.5 um/pixel, resolution should be 2 pixels/um
    res_x = 1.0 / dx if dx and dx > 0 else 1.0
    res_y = 1.0 / dy if dy and dy > 0 else 1.0
    resolution = (res_x, res_y)

    # build imagej metadata dict
    ij_meta = {
        "unit": "um",
        "loop": False,
    }

    # imagej hyperstack dimensions: frames (T), slices (Z), channels (C)
    # we must explicitly set these so imagej doesn't interpret pages as channels
    ndim = len(shape)
    if ndim == 4:
        # TZYX: shape is (T, Z, Y, X)
        ij_meta["frames"] = shape[0]
        ij_meta["slices"] = shape[1]
        ij_meta["channels"] = 1
    elif ndim == 3:
        # TYX: shape is (T, Y, X) - all pages are time frames
        ij_meta["frames"] = shape[0]
        ij_meta["slices"] = 1
        ij_meta["channels"] = 1
    else:
        # YX: single frame
        ij_meta["frames"] = 1
        ij_meta["slices"] = 1
        ij_meta["channels"] = 1

    # z-spacing (for hyperstacks with Z dimension)
    if dz is not None:
        ij_meta["spacing"] = dz

    # frame interval (seconds between frames)
    fs = get_param(metadata, "fs")
    if fs and fs > 0:
        ij_meta["finterval"] = 1.0 / float(fs)

    # optional min/max for display range (if present)
    if "min" in metadata:
        ij_meta["min"] = float(metadata["min"])
    if "max" in metadata:
        ij_meta["max"] = float(metadata["max"])

    return ij_meta, resolution


def _write_tiff(path, data, overwrite=True, metadata=None, imagej=True, **kwargs):
    """
    Write data to TIFF file with optional ImageJ hyperstack compatibility.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : np.ndarray
        Image data to write.
    overwrite : bool
        Whether to overwrite existing file.
    metadata : dict
        Metadata dict containing imaging parameters.
    imagej : bool
        If True (default), write ImageJ-compatible TIFF with proper metadata
        that Fiji/ImageJ can auto-detect (resolution, spacing, frame interval).
        If False, write standard tifffile format with JSON metadata.
    """
    if metadata is None:
        metadata = {}

    filename = Path(path).with_suffix(".tif")

    if not hasattr(_write_tiff, "_writers"):
        _write_tiff._writers = {}
    if not hasattr(_write_tiff, "_first_write"):
        _write_tiff._first_write = {}
    if not hasattr(_write_tiff, "_imagej_mode"):
        _write_tiff._imagej_mode = {}

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

        # Store imagej mode for this file
        _write_tiff._imagej_mode[filename] = imagej

        # Create new writer - use imagej mode if requested
        if imagej:
            _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True, imagej=True)
        else:
            _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True)
        _write_tiff._first_write[filename] = True

    writer = _write_tiff._writers[filename]
    is_first = _write_tiff._first_write.get(filename, True)
    use_imagej = _write_tiff._imagej_mode.get(filename, imagej)

    if use_imagej:
        # imagej mode: reshape data and use imagej-compatible metadata
        ij_meta = None
        resolution = None
        extratags = None

        if is_first:
            # build imagej-compatible metadata only on first write
            target_shape = metadata.get("shape", data.shape)
            ij_meta, resolution = _build_imagej_metadata(metadata, target_shape)

            # store full metadata as JSON in custom TIFF tag 50839
            import json
            json_meta = _make_json_serializable(metadata)
            json_bytes = json.dumps(json_meta).encode("utf-8")
            # extratags format: (code, dtype, count, value, writeonce)
            # dtype 2 = ASCII string
            extratags = [(50839, 2, len(json_bytes), json_bytes, True)]

        # always reshape data to TZCYX so tifffile interprets T as frames
        # 3D (T, Y, X) -> 5D (T, 1, 1, Y, X)
        # 4D (T, Z, Y, X) -> 5D (T, Z, 1, Y, X)
        if data.ndim == 3:
            data_5d = data[:, np.newaxis, np.newaxis, :, :]
        elif data.ndim == 4:
            data_5d = data[:, :, np.newaxis, :, :]
        else:
            data_5d = data

        for frame in data_5d:
            # frame is now (Z, C, Y, X) or (1, 1, Y, X)
            writer.write(
                frame,
                contiguous=True,
                photometric="minisblack",
                resolution=resolution if is_first else None,
                metadata=ij_meta if is_first else None,
                extratags=extratags if is_first else None,
            )
            is_first = False
    else:
        # standard tifffile mode with JSON metadata
        for frame in data:
            writer.write(
                frame,
                contiguous=True,
                photometric="minisblack",
                metadata=_make_json_serializable(metadata) if is_first else {},
            )
            is_first = False

    _write_tiff._first_write[filename] = False


def _write_volumetric_tiff(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
):
    """
    Write volumetric TZYX data as single ImageJ hyperstack tiff.

    parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    overwrite : bool
        overwrite existing files
    target_chunk_mb : int
        chunk size for streaming writes
    progress_callback : callable
        progress callback(fraction, message)
    show_progress : bool
        show tqdm progress bar
    debug : bool
        verbose logging
    """
    from mbo_utilities.arrays.features import (
        OutputFilename,
        ArraySlicing,
        read_chunk,
    )

    if metadata is None:
        metadata = {}

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # build selections dict for ArraySlicing
    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes

    # create slicing state (handles dim normalization, 1-based conversion)
    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    # build output filename from dims
    suffix = output_suffix if output_suffix else "stack"
    output_fn = OutputFilename.from_array(data, planes=planes, frames=frames, suffix=suffix)
    filename = path / output_fn.build(".tif")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename

    if filename.exists():
        filename.unlink()

    # get target shape after selection
    output_shape = slicing.output_shape
    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    # check for z-registration shift application (shared logic)
    if debug:
        logger.info(f"  TIFF metadata: apply_shift={metadata.get('apply_shift')}, s3d-job={metadata.get('s3d-job')}")
    apply_shift, plane_shifts, padding = load_registration_shifts(metadata, debug)
    pt, pb, pl, pr = padding
    if debug:
        logger.info(f"  Registration result: apply_shift={apply_shift}, has_shifts={plane_shifts is not None}")

    # compute padded output dimensions if shifts are applied
    if apply_shift and plane_shifts is not None:
        Ly_out = Ly + pt + pb
        Lx_out = Lx + pl + pr
    else:
        Ly_out, Lx_out = Ly, Lx

    target_shape = (n_frames, n_planes, Ly_out, Lx_out)

    # update metadata for imagej using OutputMetadata for reactive values
    from mbo_utilities.metadata import OutputMetadata
    from mbo_utilities.arrays.features import get_dims

    # get dims from array
    source_dims = get_dims(data)

    # build selections dict with 0-based indices
    output_selections = {}
    if "T" in slicing.selections:
        output_selections["T"] = slicing.selections["T"].indices
    if "Z" in slicing.selections:
        output_selections["Z"] = slicing.selections["Z"].indices

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    # get adjusted metadata dict (now includes reactive Lx, Ly, shape)
    md = out_meta.to_dict()
    # override shape with padded values if shifts applied
    md["shape"] = target_shape
    md["Lx"] = Lx_out
    md["Ly"] = Ly_out
    if apply_shift and plane_shifts is not None:
        md["padded_shape"] = (Ly_out, Lx_out)
        md["original_shape"] = (Ly, Lx)

    # build imagej metadata with adjusted dz and finterval
    ij_meta, resolution = out_meta.to_imagej(target_shape)

    # store full metadata as JSON in ImageJ's Info field
    # use imagej_metadata_tag to create both 50838 and 50839 tags properly
    from tifffile import imagej_metadata_tag
    import json
    json_meta = _make_json_serializable(md)
    json_str = json.dumps(json_meta)
    ij_extratags = imagej_metadata_tag({"Info": json_str}, "<")
    extratags = list(ij_extratags) if ij_extratags else []

    if debug:
        logger.info(f"Writing volumetric tiff: {filename}")
        logger.info(f"  Shape: {target_shape} (TZYX)")
        logger.info(f"  ImageJ meta: frames={ij_meta.get('frames')}, slices={ij_meta.get('slices')}")
        logger.info(f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}")
        if out_meta.z_step_factor > 1:
            logger.info(f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)")

    # open writer
    with TiffWriter(filename, bigtiff=True, imagej=True) as writer:
        first_write = True

        # iterate over chunks using unified slicing
        pbar = None
        if show_progress:
            pbar = tqdm(total=n_frames, desc="Writing TIFF", unit="frames")

        # get z-plane indices being written (0-based)
        z_indices = slicing.selections["Z"].indices if "Z" in slicing.selections else list(range(n_planes))

        for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
            # read chunk using unified reader
            chunk_data = read_chunk(data, chunk_info, slicing.dims)

            # ensure 4D (T, Z, Y, X)
            if chunk_data.ndim == 3:
                chunk_data = chunk_data[:, np.newaxis, :, :]

            # ensure contiguous
            chunk_data = np.ascontiguousarray(chunk_data)

            # apply z-registration shifts if enabled (shared logic)
            if apply_shift and plane_shifts is not None:
                chunk_data = apply_shifts_to_chunk(
                    chunk_data, plane_shifts, z_indices, padding, Ly_out, Lx_out
                )

            # reshape to TZCYX for imagej (add C=1)
            chunk_5d = chunk_data[:, :, np.newaxis, :, :]

            # write each frame (T) with all its Z slices
            for t in range(chunk_5d.shape[0]):
                frame_data = chunk_5d[t]  # (Z, C, Y, X)
                writer.write(
                    frame_data,
                    contiguous=True,
                    photometric="minisblack",
                    resolution=resolution if first_write else None,
                    metadata=ij_meta if first_write else None,
                    extratags=extratags if first_write else None,
                )
                first_write = False

            if pbar:
                frames_in_chunk = len(chunk_info.selections.get("T", [1]))
                pbar.update(frames_in_chunk)

            if progress_callback:
                progress_callback(chunk_info.progress)

        if pbar:
            pbar.close()

    if debug:
        logger.info(f"Wrote {filename} ({filename.stat().st_size / 1e9:.2f} GB)")

    return filename


def _write_volumetric_zarr(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    sharded: bool = True,
    compression_level: int = 1,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "mean",
):
    """
    Write volumetric TZYX data as single OME-NGFF zarr.

    parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    overwrite : bool
        overwrite existing files
    target_chunk_mb : int
        target chunk size in MB for streaming writes
    progress_callback : callable
        progress callback(fraction, message)
    show_progress : bool
        show tqdm progress bar
    debug : bool
        verbose logging
    output_suffix : str | None
        suffix for output filename
    sharded : bool
        use zarr v3 sharding codec
    compression_level : int
        gzip compression level (0=none, 1-9)
    pyramid : bool
        generate multi-resolution pyramid (default False).
        enables napari multiscale viewing and faster navigation.
    pyramid_max_layers : int
        max additional resolution levels (default 4 = levels 0-4).
        only spatial dims (Y, X) are downsampled by 2x per level.
    pyramid_method : str
        downsampling method: "mean" (default), "nearest", "gaussian".
        use "nearest" for label/mask data.
    """
    import zarr
    from zarr.codecs import BytesCodec, GzipCodec, ShardingCodec, Crc32cCodec

    from mbo_utilities.arrays.features import (
        OutputFilename,
        ArraySlicing,
        read_chunk,
        get_dims,
    )
    from mbo_utilities.arrays.features._pyramid import (
        PyramidConfig,
        compute_pyramid_shapes,
        downsample_block,
    )
    from mbo_utilities.metadata import OutputMetadata

    if metadata is None:
        metadata = {}

    # get dimension labels from array (canonical form)
    dims = get_dims(data)

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # build selections dict for ArraySlicing
    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes

    # create slicing state (handles dim normalization, 1-based conversion)
    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    # build output filename from dims
    suffix = output_suffix if output_suffix else "stack"
    output_fn = OutputFilename.from_array(data, planes=planes, frames=frames, suffix=suffix)
    filename = path / output_fn.build(".zarr")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename

    if filename.exists():
        shutil.rmtree(filename)

    # get target shape after selection
    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    # check for z-registration shift application (shared logic)
    apply_shift, plane_shifts, padding = load_registration_shifts(metadata, debug)
    pt, pb, pl, pr = padding

    # compute padded output dimensions if shifts are applied
    if apply_shift and plane_shifts is not None:
        Ly_out = Ly + pt + pb
        Lx_out = Lx + pl + pr
    else:
        Ly_out, Lx_out = Ly, Lx

    target_shape = (n_frames, n_planes, Ly_out, Lx_out)

    # update metadata using OutputMetadata for reactive values
    # get dims from array
    source_dims = get_dims(data)

    # build selections dict from slicing
    output_selections: dict[str, list[int]] = {}
    if "T" in slicing.selections:
        output_selections["T"] = list(slicing.selections["T"].indices)
    if "Z" in slicing.selections:
        output_selections["Z"] = list(slicing.selections["Z"].indices)

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    # get adjusted metadata dict
    md = out_meta.to_dict()
    md["shape"] = target_shape
    if apply_shift and plane_shifts is not None:
        md["padded_shape"] = (Ly_out, Lx_out)
        md["original_shape"] = (Ly, Lx)

    if debug:
        logger.info(f"Writing volumetric zarr: {filename}")
        logger.info(f"  Shape: {target_shape} (TZYX)")
        logger.info(f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}")
        if out_meta.z_step_factor > 1:
            logger.info(f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)")
        if apply_shift:
            logger.info(f"  Z-registration: padding=({pt}, {pb}, {pl}, {pr})")

    # determine chunking based on target_chunk_mb
    # for 4D TZYX, chunk along T dimension
    bytes_per_frame = n_planes * Ly_out * Lx_out * np.dtype(data.dtype).itemsize
    target_bytes = target_chunk_mb * 1024 * 1024
    frames_per_chunk = max(1, int(target_bytes / bytes_per_frame))
    frames_per_chunk = min(frames_per_chunk, n_frames)

    # inner chunk shape: 1 frame at a time for efficient random access
    inner_chunk = (1, n_planes, Ly_out, Lx_out)

    # build codec chain
    if compression_level == 0:
        inner_codecs = [BytesCodec()]
    else:
        inner_codecs = [BytesCodec(), GzipCodec(level=compression_level)]

    if sharded:
        # shard size: multiple frames per shard for efficient sequential reads
        shard_t = min(n_frames, frames_per_chunk)
        shard_shape = (shard_t, n_planes, Ly_out, Lx_out)

        codec = ShardingCodec(
            chunk_shape=inner_chunk,
            codecs=inner_codecs,
            index_codecs=[BytesCodec(), Crc32cCodec()],
        )
        codecs = [codec]
        chunks = shard_shape
    else:
        codecs = inner_codecs
        chunks = inner_chunk

    # create zarr v3 group with OME-NGFF structure
    root = zarr.open_group(str(filename), mode="w", zarr_format=3)

    # compute pyramid levels if enabled
    if pyramid:
        pyramid_config = PyramidConfig(
            max_layers=pyramid_max_layers,
            scale_factors=(1, 1, 2, 2),  # TZYX: only downsample Y, X
            method=pyramid_method,
            min_size=64,
        )
        pyramid_levels = compute_pyramid_shapes(target_shape, pyramid_config)
        if debug:
            logger.info(f"  Pyramid: {len(pyramid_levels)} levels")
            for lvl in pyramid_levels:
                logger.info(f"    Level {lvl.level}: {lvl.shape}")
    else:
        pyramid_levels = None

    # create the array as "0" (full resolution level)
    z = zarr.create(
        store=root.store,
        path="0",
        shape=target_shape,
        chunks=chunks,
        dtype=data.dtype,
        codecs=codecs,
        overwrite=True,
    )

    # build OME-NGFF metadata using OutputMetadata with array's dims
    # for 4D output, use TZYX ordering
    output_dims = ("T", "Z", "Y", "X")
    ome_meta = out_meta.to_ome_ngff(dims=output_dims)
    base_scale = ome_meta["coordinateTransformations"][0]["scale"]

    # store dims in metadata for readers
    md["dims"] = output_dims

    # build full OME-NGFF v0.5 structure with pyramid datasets
    if pyramid_levels:
        datasets = []
        for lvl in pyramid_levels:
            # compute physical scale for this level
            # pyramid scale is relative (e.g., 1, 2, 4), multiply with base
            physical_scale = [
                base_scale[i] * lvl.scale[i] for i in range(len(base_scale))
            ]
            datasets.append({
                "path": lvl.path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": physical_scale}
                ],
            })

        multiscales = [
            {
                "version": "0.5",
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": datasets,
                "type": pyramid_method,
            }
        ]
    else:
        multiscales = [
            {
                "version": "0.5",
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": ome_meta["coordinateTransformations"],
                    }
                ],
            }
        ]

    ome_content = {
        "version": "0.5",
        "multiscales": multiscales,
    }

    # set OME metadata on the group
    root.attrs["ome"] = ome_content

    # also store full metadata as JSON-serializable attrs
    serializable_md = _make_json_serializable(md)
    for k, v in serializable_md.items():
        root.attrs[k] = v

    # set napari-compatible scale on level 0 array
    z.attrs["scale"] = base_scale
    # set dimension_names for NGFF 0.5 compliance (lowercase)
    z.attrs["dimension_names"] = [d.lower() for d in output_dims]

    # write data in chunks
    pbar = None
    total_work = n_frames * (len(pyramid_levels) if pyramid_levels else 1)
    if show_progress:
        pbar = tqdm(total=total_work, desc="Writing Zarr", unit="frames")

    # track output offset (since ChunkInfo doesn't have offset)
    t_offset = 0

    # get z-plane indices being written (0-based)
    z_indices = slicing.selections["Z"].indices if "Z" in slicing.selections else list(range(n_planes))

    # iterate over chunks using unified slicing
    for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
        # read chunk using unified reader
        chunk_data = read_chunk(data, chunk_info, slicing.dims)

        # ensure 4D (T, Z, Y, X)
        if chunk_data.ndim == 3:
            chunk_data = chunk_data[:, np.newaxis, :, :]

        # ensure contiguous for efficient writes
        chunk_data = np.ascontiguousarray(chunk_data)

        # apply z-registration shifts if enabled (shared logic)
        if apply_shift and plane_shifts is not None:
            chunk_data = apply_shifts_to_chunk(
                chunk_data, plane_shifts, z_indices, padding, Ly_out, Lx_out
            )

        # get the output T range for this chunk
        t_start = t_offset
        t_end = t_start + chunk_data.shape[0]

        # write to zarr level 0
        z[t_start:t_end, :, :, :] = chunk_data

        # update offset
        t_offset = t_end

        if pbar:
            pbar.update(chunk_data.shape[0])

        if progress_callback:
            progress_callback(chunk_info.progress / (len(pyramid_levels) if pyramid_levels else 1))

    if pbar:
        pbar.close()

    # generate pyramid levels from level 0 data
    if pyramid_levels and len(pyramid_levels) > 1:
        if show_progress:
            pbar = tqdm(
                total=len(pyramid_levels) - 1,
                desc="Building pyramid",
                unit="levels",
            )

        scale_factors = pyramid_config.get_scale_factors_for_ndim(4)

        for lvl in pyramid_levels[1:]:
            prev_level = lvl.level - 1
            prev_path = str(prev_level)

            # read previous level data
            prev_z = root[prev_path]
            prev_data = prev_z[:]

            # downsample
            level_data = downsample_block(prev_data, scale_factors, pyramid_method)

            # compute chunks for this level
            lvl_shape = level_data.shape
            lvl_chunks = (
                min(chunks[0], lvl_shape[0]),
                min(chunks[1], lvl_shape[1]),
                min(chunks[2], lvl_shape[2]),
                min(chunks[3], lvl_shape[3]),
            )

            # create array for this level (simpler codec for smaller levels)
            if compression_level == 0:
                lvl_codecs = [BytesCodec()]
            else:
                lvl_codecs = [BytesCodec(), GzipCodec(level=compression_level)]

            lvl_z = zarr.create(
                store=root.store,
                path=lvl.path,
                shape=lvl_shape,
                chunks=lvl_chunks,
                dtype=data.dtype,
                codecs=lvl_codecs,
                overwrite=True,
            )

            # write data
            lvl_z[:] = level_data

            # set napari-compatible scale on this level's array
            physical_scale = [
                base_scale[i] * lvl.scale[i] for i in range(len(base_scale))
            ]
            lvl_z.attrs["scale"] = physical_scale

            if debug:
                logger.info(f"  Wrote level {lvl.level}: {lvl_shape}")

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    if debug:
        # estimate size (zarr stores are directories)
        total_size = sum(f.stat().st_size for f in filename.rglob("*") if f.is_file())
        logger.info(f"Wrote {filename} ({total_size / 1e9:.2f} GB)")

    return filename


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
    # chunk configuration: shard_frames is outer (shard) size, chunk_shape is inner
    # chunk_shape can be tuple (t, y, x) or None for default (1, h, w)
    shard_frames = kwargs.get("shard_frames")  # frames per shard
    chunk_shape = kwargs.get("chunk_shape")  # inner chunk shape (t, y, x)

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

        # build codec chain based on compression level
        if level == 0:
            # no compression
            inner_codecs = [BytesCodec()]
        else:
            inner_codecs = [BytesCodec(), GzipCodec(level=level)]

        if sharded:
            # determine inner chunk shape first (needed for shard alignment)
            if chunk_shape is not None:
                inner = chunk_shape
            else:
                inner = (1, h, w)  # default: 1 frame per chunk

            inner_t = inner[0]

            # determine shard size (outer chunks)
            # shard must be divisible by inner chunk time dimension
            if shard_frames is not None:
                shard_t = min(nframes, shard_frames)
            else:
                shard_t = min(nframes, 100)  # default: 100-frame shards

            # ensure shard is divisible by inner chunk (zarr requirement)
            if inner_t > 1 and shard_t % inner_t != 0:
                # round down to nearest multiple of inner_t
                shard_t = (shard_t // inner_t) * inner_t
                if shard_t == 0:
                    shard_t = inner_t  # minimum: one inner chunk per shard

            outer = (shard_t, h, w)

            codec = ShardingCodec(
                chunk_shape=inner,
                codecs=inner_codecs,
                index_codecs=[BytesCodec(), Crc32cCodec()],
            )
            codecs = [codec]
            chunks = outer
        else:
            # non-sharded mode: each chunk is a file
            codecs = inner_codecs
            chunks = chunk_shape if chunk_shape is not None else (1, h, w)

        if ome:
            # Create OME-Zarr using NGFF v0.5 with Zarr v3
            # Structure: my_image.zarr/ (group) -> 0/ (array)

            # Create Zarr v3 group
            root = zarr.open_group(str(filename), mode="w", zarr_format=3)

            # use the codecs/chunks computed above
            array_codecs = codecs
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
                dtype=data.dtype,
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
        # use imagej-compatible format for proper Fiji detection
        target_shape = metadata.get("shape", data.shape)
        ij_meta, resolution = _build_imagej_metadata(metadata, target_shape)
        tiff_imwrite(
            outpath,
            data,
            imagej=True,
            resolution=resolution,
            metadata=ij_meta,
            photometric="minisblack",
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
        chunks = (1,) * (arr.ndim - 2) + arr.shape[-2:] if arr.ndim >= 3 else arr.shape

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

    # use get_param for consistent alias handling
    from mbo_utilities.metadata import get_param, get_voxel_size

    fs = get_param(metadata, "fs")
    if fs is None:
        # check finterval (ImageJ format) and convert to fs
        finterval = get_param(metadata, "finterval")
        if finterval is not None and finterval > 0:
            fs = 1.0 / finterval
        else:
            logger.warning("No frame rate found; defaulting fs=10")
            fs = 10
    metadata["fs"] = fs
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
    plane: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
    temporal_smooth: int = 0,
    spatial_smooth: float = 0,
    gamma: float = 1.0,
    cmap: str | None = None,
    quality: int = 9,
    codec: str = "libx264",
    max_frames: int | None = None,
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
            frame = np.asarray(arr[i, plane_idx]) if ndim == 4 else np.asarray(arr[i])
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
