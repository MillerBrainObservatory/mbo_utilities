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
from .file_io import load_npy
from .metadata.io import _build_ome_metadata

from tqdm.auto import tqdm

logger = log.get("writers")


# metadata serialization helpers (moved from _parsing.py)

def _is_disabled_si_module(value) -> bool:
    """Check if a scanimage module dict has enable=false."""
    if not isinstance(value, dict):
        return False
    enable_val = value.get("enable")
    if enable_val is False:
        return True
    return bool(isinstance(enable_val, str) and enable_val.lower() in ("false", "0"))


def _filter_disabled_modules(metadata: dict, recursive: bool = True) -> dict:
    """Filter out disabled scanimage modules (hXxx with enable=false) from metadata."""
    if not isinstance(metadata, dict):
        return metadata
    result = {}
    for k, v in metadata.items():
        if k.startswith("h") and _is_disabled_si_module(v):
            continue
        if recursive and isinstance(v, dict):
            v = _filter_disabled_modules(v, recursive=True)
        result[k] = v
    return result


# belt-and-suspenders cap. EXPORT_DENYLIST handles known suite2p
# fields, but a future ops version could add a new big array we
# haven't named — drop anything over the cap so output never blows
# up by surprise. 1 MB lets per-frame vectors through (xoff/yoff at
# 10k frames = 80 KB) but stops regPC-class arrays cold.
_MAX_NDARRAY_NBYTES = 1 * 1024 * 1024


def _make_json_serializable(obj, filter_disabled: bool = True):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {k: _make_json_serializable(v, filter_disabled=False) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v, filter_disabled=False) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.nbytes > _MAX_NDARRAY_NBYTES:
            return None
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _convert_paths_to_strings(obj, filter_disabled: bool = True):
    """Recursively convert pathlib.Path objects to strings in a nested structure."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        if filter_disabled:
            obj = _filter_disabled_modules(obj, recursive=True)
        return {k: _convert_paths_to_strings(v, filter_disabled=False) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_strings(v, filter_disabled=False) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj
    return obj

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
    channel_index=None,
    frames=None,
    **kwargs,
):
    if dshape is None:
        dshape = data.shape

    metadata = metadata or {}

    if plane_index is not None:
        if not isinstance(plane_index, (int, np.integer)):
            raise TypeError(f"plane_index must be an integer, got {type(plane_index)}")
        metadata["plane"] = int(plane_index) + 1

    if frames is not None:
        frames_0 = [int(f) - 1 for f in frames]
        nframes_target = len(frames_0)
    else:
        frames_0 = None
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
    metadata["num_frames"] = nframes_target

    dshape = (nframes_target, *dshape[1:])
    metadata["shape"] = dshape

    H0, W0 = data.shape[-2], data.shape[-1]
    fname = filename
    writer = _get_file_writer(fname.suffix, overwrite=overwrite)

    itemsize = np.dtype(data.dtype).itemsize
    ntime = int(nframes_target)

    if ntime == 0:
        raise ValueError(
            f"Cannot write file with 0 frames. Data shape: {data.shape}, nframes_target: {nframes_target}"
        )

    bytes_per_t = int(np.prod(dshape[1:], dtype=np.int64)) * int(itemsize)
    chunk_size = int(target_chunk_mb) * 1024 * 1024

    if chunk_size <= 0:
        chunk_size = 20 * 1024 * 1024

    total_bytes = int(ntime) * int(bytes_per_t)
    nchunks = max(1, math.ceil(total_bytes / chunk_size))
    nchunks = min(nchunks, ntime)

    base = ntime // nchunks
    extra = ntime % nchunks

    if show_progress and not debug:
        pbar = tqdm(total=nchunks, desc=f"Saving {fname.name}")
    else:
        pbar = None

    start = 0
    for i in range(nchunks):
        end = start + base + (1 if i < extra else 0)

        c_idx = channel_index if channel_index is not None else 0
        z_idx = plane_index if plane_index is not None else 0
        if frames_0 is not None:
            sel = frames_0[start:end]
            if sel and sel == list(range(sel[0], sel[-1] + 1)):
                chunk = data[sel[0]:sel[-1] + 1, c_idx, z_idx, :, :]
            else:
                chunk = np.stack(
                    [np.asarray(data[fi, c_idx, z_idx, :, :]) for fi in sel]
                )
        else:
            chunk = data[start:end, c_idx, z_idx, :, :]

        if hasattr(chunk, "compute"):
            chunk = chunk.compute()
        elif isinstance(chunk, np.memmap):
            chunk = np.array(chunk)
        elif not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)

        writer(fname, chunk, metadata=metadata, **kwargs)

        if pbar:
            pbar.update(1)
        if progress_callback:
            progress_callback(pbar.n / pbar.total, current_plane=plane_index)
        start = end
    if pbar:
        pbar.close()

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

        # stamp the ACTUAL chunk dims into metadata so write_ops records
        # what's really on disk. shape5d and the metadata["shape"] from
        # _imwrite_base can differ from the chunk — e.g. ScanImage's
        # shape5d uses ROI metadata (550) but process_rois returns the
        # real stitched height (542). when axial shifts are applied, the
        # chunk already has padded dims, so this is correct for both cases.
        metadata["Ly"] = Ly
        metadata["Lx"] = Lx
        metadata["shape"] = (nframes, Ly, Lx)

        _write_bin._writers[key] = BinArray(
            filename=key,
            shape=(nframes, Ly, Lx),
            dtype=np.int16,
        )
        _write_bin._offsets[key] = 0
        first_write = True

    bf = _write_bin._writers[key]
    off = _write_bin._offsets[key]

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
    # Default to "mov" — matches H5Array auto-detect, suite2p, caiman.
    # Callers can override via the dataset_name kwarg flowing from imwrite.
    dataset_name = kwargs.get("dataset_name") or "mov"

    if not hasattr(_write_h5, "_initialized"):
        _write_h5._initialized = {}
        _write_h5._offsets = {}
        _write_h5._dataset_names = {}

    if filename not in _write_h5._initialized:
        nframes = metadata.get("num_frames")
        if nframes is None:
            raise ValueError("Metadata must contain 'nframes' or 'nun_frames'.")
        h, w = data.shape[-2:]
        with h5py.File(filename, "w" if overwrite else "a") as f:
            f.create_dataset(
                dataset_name,
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
        _write_h5._dataset_names[filename] = dataset_name

    offset = _write_h5._offsets[filename]
    # use the dataset name registered on first write so chunked appends
    # always target the same dataset, even if a later call passes a
    # different (or no) dataset_name kwarg.
    active_name = _write_h5._dataset_names[filename]

    with h5py.File(filename, "a") as f:
        f[active_name][offset : offset + data.shape[0]] = data

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

    # always 5D TCZYX
    ij_meta["frames"] = shape[0]
    ij_meta["slices"] = shape[2]
    ij_meta["channels"] = shape[1]

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
            _write_tiff._writers[filename] = TiffWriter(
                filename, bigtiff=True, imagej=True
            )
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

            # store filtered metadata as JSON in custom TIFF tag 50839.
            # strip_for_export drops suite2p-only fields (regPC, meanImg,
            # xoff/yoff, pipeline settings, etc.) — those belong only in
            # ops.npy. without this filter, regPC alone can balloon the
            # tag to 500+ MB after JSON expansion.
            import json
            from .metadata.base import strip_for_export

            json_meta = _make_json_serializable(strip_for_export(metadata))
            json_bytes = json.dumps(json_meta).encode("utf-8")
            # extratags format: (code, dtype, count, value, writeonce)
            # dtype 2 = ASCII string
            extratags = [(50839, 2, len(json_bytes), json_bytes, True)]

        # data is always 5D TCZYX; transpose to TZCYX for imagej page ordering
        data_5d = np.moveaxis(data, 1, 2)  # TCZYX -> TZCYX

        for frame in data_5d:
            # frame is (Z, C, Y, X) after TZCYX transpose
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
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "mean",
):
    """
    Write volumetric TZYX or TCYX data as single ImageJ hyperstack tiff.

    parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, C, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    channels : list | None
        channel selection (1-based indices). None = all channels.
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
    from mbo_utilities.arrays.features._pyramid import (
        PyramidConfig,
        compute_pyramid_shapes,
        downsample_block,
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
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    # create slicing state (handles dim normalization, 1-based conversion)
    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    # build output filename from dims
    suffix = output_suffix if output_suffix else "stack"
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
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
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    # always 5D TZCYX for imagej page ordering
    target_shape = (n_frames, n_planes, n_channels, Ly, Lx)

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
    if "C" in slicing.selections:
        output_selections["C"] = slicing.selections["C"].indices

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape
    md["Lx"] = Lx
    md["Ly"] = Ly

    # build imagej metadata with adjusted dz and finterval
    ij_meta, resolution = out_meta.to_imagej(target_shape)

    # store filtered metadata as JSON in ImageJ's Info field. strip
    # suite2p-only fields first — regPC/tPC/meanImg/etc. belong in
    # ops.npy, not stamped into every tiff page header.
    from tifffile import imagej_metadata_tag
    import json
    from .metadata.base import strip_for_export

    json_meta = _make_json_serializable(strip_for_export(md))
    json_str = json.dumps(json_meta)
    ij_extratags = imagej_metadata_tag({"Info": json_str}, "<")
    extratags = list(ij_extratags) if ij_extratags else []

    if debug:
        logger.info(f"Writing volumetric tiff: {filename}")
        logger.info(f"  Shape: {target_shape} (TZCYX)")
        logger.info(
            f"  ImageJ meta: frames={ij_meta.get('frames')}, slices={ij_meta.get('slices')}, channels={ij_meta.get('channels')}"
        )
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )
        if out_meta.z_step_factor > 1:
            logger.info(
                f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)"
            )

    # setup pyramids if requested
    if pyramid:
        pyramid_config = PyramidConfig(
            max_layers=pyramid_max_layers,
            scale_factors=(1, 1, 1, 2, 2) if n_channels > 1 else (1, 1, 2, 2),
            method=pyramid_method,
            min_size=64,
        )
        pyramid_levels = compute_pyramid_shapes(target_shape, pyramid_config)
        if debug:
            logger.info(f"  Pyramid: {len(pyramid_levels)} levels")
    else:
        pyramid_levels = None

    # open writer
    with TiffWriter(filename, bigtiff=True, imagej=not pyramid, ome=pyramid) as writer:
        first_write = True

        # iterate over chunks using unified slicing
        pbar = None
        if show_progress:
            total_pages = n_frames * n_planes * n_channels
            pbar = tqdm(total=total_pages, desc="Writing TIFF", unit="pg")

        for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
            chunk_data = read_chunk(data, chunk_info, slicing.dims)

            # always 5D TCZYX input → transpose to TZCYX for ImageJ page order
            chunk_data = chunk_data.transpose(0, 2, 1, 3, 4)
            chunk_data = np.ascontiguousarray(chunk_data)

            chunk_5d = chunk_data  # (T, Z, C, Y, X)

            # write each frame (T) with all its Z slices
            for t in range(chunk_5d.shape[0]):
                frame_data = chunk_5d[t]  # (Z, C, Y, X)

                if pyramid and pyramid_levels and len(pyramid_levels) > 1:
                    frame_pyramid = []
                    current = frame_data
                    scale_factors_frame = (1, 1, 2, 2)  # Z, C, Y, X

                    for _ in range(1, len(pyramid_levels)):
                        current = downsample_block(
                            current, scale_factors_frame, pyramid_method
                        )
                        if current.dtype != frame_data.dtype:
                            current = current.astype(frame_data.dtype)
                        frame_pyramid.append(current)

                    writer.write(
                        frame_data,
                        contiguous=True,
                        photometric="minisblack",
                        resolution=resolution if first_write else None,
                        subifds=len(frame_pyramid),
                    )

                    for downsampled in frame_pyramid:
                        writer.write(
                            downsampled,
                            contiguous=True,
                            photometric="minisblack",
                            subfiletype=1,
                        )
                else:
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
                pbar.update(frames_in_chunk * n_planes * n_channels)

            if progress_callback:
                progress_callback(chunk_info.progress)

        if pbar:
            pbar.close()

    if debug:
        logger.info(f"Wrote {filename} ({filename.stat().st_size / 1e9:.2f} GB)")

    return filename


def _write_volumetric_h5(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    dataset_name: str = "mov",
    compression: str | None = "gzip",
    compression_level: int = 1,
):
    """Write volumetric TZYX data into a single HDF5 file.

    This is the h5 analogue of `_write_volumetric_zarr`. Previously, h5
    output went through the per-plane streaming writer (`_write_plane`),
    which produced one `.h5` file per z-plane (`zplane01_stack.h5`,
    `zplane03_stack.h5`, ...). That mismatched zarr's behaviour, where the
    full volume lives in a single store. This writer puts everything into
    one `.h5` with a 4D `(T, Z, Y, X)` dataset under `dataset_name`,
    matching what mbo's H5Array reader expects on the read side.

    parameters
    ----------
    data : array-like
        Source array. The 5D TCZYX shape is read via `read_chunk`; the
        singleton C dim is dropped before writing (h5 dataset is 4D).
    path : Path
        Output directory. Filename is auto-generated from dim tags.
    metadata : dict | None
        Imaging metadata; serializable values land on the file's
        root attrs and on the dataset attrs.
    planes, frames, channels : list | None
        1-based selections that subset the source on the way out.
    overwrite : bool
        Replace existing file rather than failing.
    target_chunk_mb : int
        Sizing hint for the streaming read loop. Each h5 chunk on disk
        is fixed at one frame `(1, Z, Y, X)` for fast random access.
    dataset_name : str
        H5 dataset name (default 'mov' — same default as `_write_h5`
        and what mbo's H5Array reader auto-detects). Pass via the
        save-as dialog's H5 dataset-name field.
    compression : str | None
        H5 compression filter ('gzip', 'lzf', or None).
    compression_level : int
        Gzip level 0-9 (ignored for lzf/None).
    """
    import h5py
    from mbo_utilities.arrays.features import (
        OutputFilename,
        ArraySlicing,
        read_chunk,
        get_dims,
    )
    from mbo_utilities.metadata import OutputMetadata

    if metadata is None:
        metadata = {}

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # build selections dict for ArraySlicing — same idiom as the zarr writer
    selections = {}
    if frames is not None:
        selections["T"] = [frames] if isinstance(frames, int) else frames
    if planes is not None:
        selections["Z"] = [planes] if isinstance(planes, int) else planes
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    suffix = output_suffix if output_suffix else "stack"
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
    filename = path / output_fn.build(".h5")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename
    if filename.exists():
        filename.unlink()

    # target shape after selection — same n_channels==1 enforcement as zarr
    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1
    if n_channels != 1:
        raise NotImplementedError(
            f"_write_volumetric_h5 writes 4D TZYX and only supports a single "
            f"channel; got n_channels={n_channels}. Select one channel via the "
            f"`channels` kwarg."
        )

    Ly, Lx = slicing.spatial_shape
    Ly_out, Lx_out = int(Ly), int(Lx)
    n_frames = int(n_frames)
    n_planes = int(n_planes)

    target_shape = (n_frames, n_planes, Ly_out, Lx_out)

    # update metadata using OutputMetadata for reactive dz/fs values
    source_dims = get_dims(data)
    output_selections: dict[str, list[int]] = {}
    if "T" in slicing.selections:
        output_selections["T"] = list(slicing.selections["T"].indices)
    if "Z" in slicing.selections:
        output_selections["Z"] = list(slicing.selections["Z"].indices)
    if "C" in slicing.selections:
        output_selections["C"] = list(slicing.selections["C"].indices)

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape
    md["dataset_name"] = dataset_name
    md["dims"] = ("T", "Z", "Y", "X")

    if debug:
        logger.info(f"Writing volumetric h5: {filename}")
        logger.info(f"  Shape: {target_shape} (TZYX)")
        logger.info(f"  Dataset: /{dataset_name}")
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )

    # h5 chunking — one frame per chunk so any (t, z, y, x) read pulls
    # exactly its plane's worth of bytes off disk. matches the zarr
    # `inner_chunk = (1, n_planes, Ly_out, Lx_out)` decision.
    inner_chunk = (1, n_planes, Ly_out, Lx_out)

    # h5 compression knobs
    h5_compression = None
    h5_compression_opts = None
    if compression is not None and compression_level > 0:
        if compression.lower() == "gzip":
            h5_compression = "gzip"
            h5_compression_opts = int(compression_level)
        elif compression.lower() == "lzf":
            h5_compression = "lzf"
        else:
            logger.warning(
                f"Unknown h5 compression '{compression}'; falling back to none."
            )

    pbar = None
    if show_progress:
        pbar = tqdm(total=n_frames, desc="Writing H5", unit="frames")

    try:
        with h5py.File(filename, "w") as f:
            dset = f.create_dataset(
                dataset_name,
                shape=target_shape,
                maxshape=(None, n_planes, Ly_out, Lx_out),
                chunks=inner_chunk,
                dtype=data.dtype,
                compression=h5_compression,
                compression_opts=h5_compression_opts,
            )

            from .metadata.base import strip_for_export
            serializable_md = _make_json_serializable(strip_for_export(md))
            for k, v in serializable_md.items():
                if v is None:
                    continue
                try:
                    f.attrs[k] = v if np.isscalar(v) else str(v)
                except (TypeError, ValueError):
                    f.attrs[k] = str(v)
            dset.attrs["dims"] = ["T", "Z", "Y", "X"]

            t_offset = 0
            for chunk_info in slicing.iter_chunks(
                chunk_dim="T", target_mb=target_chunk_mb
            ):
                chunk_data = read_chunk(data, chunk_info, slicing.dims)
                if chunk_data.ndim == 5:
                    chunk_data = chunk_data[:, 0, :, :, :]
                chunk_data = np.ascontiguousarray(chunk_data)

                t_start = t_offset
                t_end = t_start + chunk_data.shape[0]
                dset[t_start:t_end, :, :, :] = chunk_data
                t_offset = t_end

                if pbar:
                    pbar.update(chunk_data.shape[0])
                if progress_callback:
                    progress_callback(chunk_info.progress)
    finally:
        if pbar:
            pbar.close()

    if debug:
        logger.info(
            f"Wrote {filename} ({filename.stat().st_size / 1e9:.2f} GB)"
        )

    return filename


def _write_volumetric_zarr(
    data,
    path: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    overwrite: bool = True,
    target_chunk_mb: int = 50,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
    output_suffix: str | None = None,
    sharded: bool = True,
    compression_level: int = 1,
    compressor: str = "gzip",
    shuffle: str | None = None,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "mean",
):
    """
    Write volumetric TZYX data as single OME-NGFF zarr.

    parameters
    ----------
    data : array-like
        data with shape (T, Z, Y, X), (T, C, Y, X), (T, Y, X), or (Z, Y, X)
    path : Path
        output directory (filename auto-generated from dims)
    metadata : dict
        imaging metadata for resolution, spacing, etc.
    planes : list | None
        z-plane selection (1-based indices). None = all planes.
    frames : list | None
        timepoint selection (1-based indices). None = all frames.
    channels : list | None
        channel selection (1-based indices). None = all channels.
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
        compression level. meaning depends on `compressor`:
        gzip: 0=none, 1-9. zstd: 1-22. blosc: 1-9. ignored for "none".
    compressor : str
        compression codec. one of "none", "gzip", "zstd", "blosc-lz4",
        "blosc-zstd". default "gzip".
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

    def _build_inner_codecs(name: str, level: int, shuffle: str | None = None) -> list:
        # build the codec chain that sits inside a shard (or replaces the
        # whole chain when sharded=False). bytescodec is always the first
        # stage; compressor is appended unless name == "none".
        # `shuffle` only applies to blosc-* and must be one of
        # "noshuffle", "shuffle", "bitshuffle", or None (= blosc default).
        name = (name or "none").lower()
        if name == "none" or level == 0:
            return [BytesCodec()]
        if name == "gzip":
            return [BytesCodec(), GzipCodec(level=level)]
        if name == "zstd":
            from zarr.codecs import ZstdCodec
            return [BytesCodec(), ZstdCodec(level=level)]
        if name in ("blosc-lz4", "blosc-zstd"):
            from zarr.codecs import BloscCodec
            cname = "lz4" if name == "blosc-lz4" else "zstd"
            blosc_kwargs = {
                "cname": cname,
                "clevel": level,
                "typesize": np.dtype(data.dtype).itemsize,
            }
            if shuffle is not None:
                blosc_kwargs["shuffle"] = shuffle
            return [BytesCodec(), BloscCodec(**blosc_kwargs)]
        raise ValueError(
            f"unknown compressor {name!r}; expected one of "
            "'none', 'gzip', 'zstd', 'blosc-lz4', 'blosc-zstd'"
        )

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
    if channels is not None:
        selections["C"] = [channels] if isinstance(channels, int) else channels

    # create slicing state (handles dim normalization, 1-based conversion)
    slicing = ArraySlicing.from_array(data, selections=selections, one_based=True)

    # build output filename from dims
    suffix = output_suffix if output_suffix else "stack"
    output_fn = OutputFilename.from_array(
        data, planes=planes, frames=frames, channels=channels, suffix=suffix
    )
    filename = path / output_fn.build(".zarr")

    if filename.exists() and not overwrite:
        logger.warning(f"File {filename} exists and overwrite=False. Skipping.")
        return filename

    if filename.exists():
        shutil.rmtree(filename)

    # get target shape after selection
    n_frames = slicing.selections["T"].count if "T" in slicing.selections else 1
    n_planes = slicing.selections["Z"].count if "Z" in slicing.selections else 1
    n_channels = slicing.selections["C"].count if "C" in slicing.selections else 1
    Ly, Lx = slicing.spatial_shape

    # coerce to plain python int — `slicing.spatial_shape` returns np.int
    # from `arr.shape`. zarr v3's `parse_shapelike` rejects np.int64 with
    # a TypeError, so normalize here and use the cleaned values for every
    # downstream tuple.
    Ly_out, Lx_out = int(Ly), int(Lx)
    n_frames = int(n_frames)
    n_planes = int(n_planes)
    n_channels = int(n_channels)

    # 5D output when the source carries a real C axis with >1 entries; otherwise
    # collapse C to keep 4D TZYX so the existing on-disk layout is unchanged.
    output_5d = "C" in slicing.selections and n_channels > 1
    if output_5d:
        target_shape = (n_frames, n_channels, n_planes, Ly_out, Lx_out)
    else:
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
    if "C" in slicing.selections:
        output_selections["C"] = list(slicing.selections["C"].indices)

    out_meta = OutputMetadata(
        source=metadata or {},
        source_shape=data.shape,
        source_dims=source_dims,
        selections=output_selections,
    )

    md = out_meta.to_dict()
    md["shape"] = target_shape

    if debug:
        logger.info(f"Writing volumetric zarr: {filename}")
        logger.info(
            f"  Shape: {target_shape} ({'TCZYX' if output_5d else 'TZYX'})"
        )
        logger.info(
            f"  Output metadata: dz={out_meta.dz}, fs={out_meta.fs}, contiguous={out_meta.is_contiguous}"
        )
        if out_meta.z_step_factor > 1:
            logger.info(
                f"  Z-step factor: {out_meta.z_step_factor}x (saving every {out_meta.z_step_factor} plane)"
            )

    # determine chunking based on target_chunk_mb. chunk along T;
    # bytes-per-T includes C only when the output is 5D.
    itemsize = np.dtype(data.dtype).itemsize
    if output_5d:
        bytes_per_frame = n_channels * n_planes * Ly_out * Lx_out * itemsize
        inner_chunk = (1, 1, 1, Ly_out, Lx_out)
    else:
        bytes_per_frame = n_planes * Ly_out * Lx_out * itemsize
        # inner chunk shape: one (t, z) plane per chunk. (1, 1, Y, X) — picking
        # full Z-stack chunks forced every per-plane read to decompress the
        # whole Z stack, making scrubbing ~10x slower than necessary.
        inner_chunk = (1, 1, Ly_out, Lx_out)

    target_bytes = target_chunk_mb * 1024 * 1024
    frames_per_chunk = max(1, int(target_bytes / bytes_per_frame))
    frames_per_chunk = min(frames_per_chunk, n_frames)

    # build codec chain
    inner_codecs = _build_inner_codecs(compressor, compression_level, shuffle=shuffle)

    if sharded:
        # shard size: multiple frames per shard for efficient sequential reads
        shard_t = min(n_frames, frames_per_chunk)
        if output_5d:
            shard_shape = (shard_t, n_channels, n_planes, Ly_out, Lx_out)
        else:
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

    # compute pyramid levels if enabled. PyramidConfig.get_scale_factors_for_ndim
    # picks (1,1,2,2) for 4D and (1,1,1,2,2) for 5D, so leave scale_factors at
    # the default and let it adapt to target_shape's ndim.
    if pyramid:
        pyramid_config = PyramidConfig(
            max_layers=pyramid_max_layers,
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

    # OME-NGFF axes order: TZYX for 4D, TCZYX for 5D
    output_dims = ("T", "C", "Z", "Y", "X") if output_5d else ("T", "Z", "Y", "X")
    ome_meta = out_meta.to_ome_ngff(dims=output_dims)
    base_scale = ome_meta["coordinateTransformations"][0]["scale"]
    dimension_names = [d.lower() for d in output_dims]

    # store dims in metadata for readers
    md["dims"] = output_dims

    # create the array as "0" (full resolution level). dimension_names is a
    # top-level zarr v3 array metadata field (not an attribute) per the
    # NGFF 0.5 / zarr v3 spec.
    z = zarr.create(
        store=root.store,
        path="0",
        shape=target_shape,
        chunks=chunks,
        dtype=data.dtype,
        codecs=codecs,
        dimension_names=dimension_names,
        overwrite=True,
    )

    # ngff 0.5 multiscale entry. version lives on the parent `ome` object
    # only — having it here too is invalid in 0.5. `metadata` is required by
    # the strict schema; it documents how the pyramid was generated.
    multiscale_metadata = {
        "method": pyramid_method,
        "description": (
            f"Downsampled with mbo_utilities using {pyramid_method} method"
            if pyramid_levels
            else "Single resolution level (no pyramid)"
        ),
    }

    if pyramid_levels:
        datasets = []
        for lvl in pyramid_levels:
            # pyramid scale is relative (e.g., 1, 2, 4), multiply with base
            physical_scale = [
                base_scale[i] * lvl.scale[i] for i in range(len(base_scale))
            ]
            datasets.append(
                {
                    "path": lvl.path,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": physical_scale}
                    ],
                }
            )

        multiscales = [
            {
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": datasets,
                "type": pyramid_method,
                "metadata": multiscale_metadata,
            }
        ]
    else:
        multiscales = [
            {
                "name": metadata.get("name", filename.stem),
                "axes": ome_meta["axes"],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": ome_meta[
                            "coordinateTransformations"
                        ],
                    }
                ],
                "type": "none",
                "metadata": multiscale_metadata,
            }
        ]

    ome_content = {
        "version": "0.5",
        "multiscales": multiscales,
    }

    # 5D output gets an omero.channels block keyed off the source channel_names
    # so napari-ome-zarr / OMERO readers can render each C entry as its own
    # channel (label + window). The 4D path intentionally skips this — the
    # existing _build_omero_metadata treats Z as channels, which is wrong for
    # NGFF and would regress current 4D consumers.
    if output_5d:
        channel_names = metadata.get("channel_names") if metadata else None
        if not channel_names or len(channel_names) < n_channels:
            channel_names = [f"Channel {i + 1}" for i in range(n_channels)]
        else:
            channel_names = list(channel_names[:n_channels])

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            data_min, data_max = float(info.min), float(info.max)
        else:
            data_min, data_max = 0.0, 1.0

        default_colors = [
            "00FF00", "FF0000", "0000FF", "FFFF00",
            "FF00FF", "00FFFF", "FFFFFF",
        ]
        channels_block = [
            {
                "active": True,
                "coefficient": 1.0,
                "color": default_colors[i % len(default_colors)],
                "family": "linear",
                "inverted": False,
                "label": str(channel_names[i]),
                "window": {
                    "end": data_max,
                    "max": data_max,
                    "min": data_min,
                    "start": data_min,
                },
            }
            for i in range(n_channels)
        ]
        ome_content["omero"] = {
            "channels": channels_block,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": n_planes // 2,
                "model": "color" if n_channels > 1 else "greyscale",
            },
            "version": "0.5",
        }

    # set OME metadata on the group
    root.attrs["ome"] = ome_content

    # also store filtered metadata as JSON-serializable attrs. strip
    # suite2p-only fields first — those belong in ops.npy, not in the
    # zarr group attrs where every reader has to load them.
    from .metadata.base import strip_for_export
    serializable_md = _make_json_serializable(strip_for_export(md))
    for k, v in serializable_md.items():
        root.attrs[k] = v

    # napari-compatible scale on level 0 array (dimension_names is set as
    # a top-level zarr v3 field via zarr.create above, not in attrs)
    z.attrs["scale"] = base_scale

    # write data in chunks
    pbar = None
    total_work = n_frames * (len(pyramid_levels) if pyramid_levels else 1)
    if show_progress:
        pbar = tqdm(total=total_work, desc="Writing Zarr", unit="frames")

    t_offset = 0

    try:
        for chunk_info in slicing.iter_chunks(chunk_dim="T", target_mb=target_chunk_mb):
            chunk_data = read_chunk(data, chunk_info, slicing.dims)

            # collapse the singleton C only for 4D TZYX output. read_chunk
            # always returns 5D TCZYX, so the 5D writer keeps it as-is.
            if not output_5d and chunk_data.ndim == 5:
                chunk_data = chunk_data[:, 0, :, :, :]

            chunk_data = np.ascontiguousarray(chunk_data)

            t_start = t_offset
            t_end = t_start + chunk_data.shape[0]

            # write to zarr level 0 — leading-axis slice works for 4D and 5D
            z[t_start:t_end] = chunk_data

            # update offset
            t_offset = t_end

            if pbar:
                pbar.update(chunk_data.shape[0])

            if progress_callback:
                progress_callback(
                    chunk_info.progress / (len(pyramid_levels) if pyramid_levels else 1)
                )
    finally:
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

        scale_factors = pyramid_config.get_scale_factors_for_ndim(len(target_shape))

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
            lvl_chunks = tuple(
                min(chunks[i], lvl_shape[i]) for i in range(len(lvl_shape))
            )

            # create array for this level (simpler codec for smaller levels)
            lvl_codecs = _build_inner_codecs(compressor, compression_level, shuffle=shuffle)

            lvl_z = zarr.create(
                store=root.store,
                path=lvl.path,
                shape=lvl_shape,
                chunks=lvl_chunks,
                dtype=data.dtype,
                codecs=lvl_codecs,
                dimension_names=dimension_names,
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

            # Create the array as "0" (full resolution level). 3D (T, Y, X)
            # for the per-plane writer path; matches the axes/scale produced
            # by _build_ome_metadata for ndim==3.
            z = zarr.create(
                store=root.store,
                path="0",
                shape=(nframes, h, w),
                chunks=array_chunks,
                dtype=data.dtype,
                codecs=array_codecs,
                dimension_names=["t", "y", "x"],
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
    dataset_name: str | None = None,
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
        # Default to "mov" — matches the streaming `_write_h5` path,
        # suite2p / caiman convention, and the auto-detect order in
        # `H5Array.__init__` (mov → data → scan_corrections). Callers
        # who need the legacy name can pass `dataset_name="data"`.
        h5_dataset = dataset_name if dataset_name is not None else "mov"
        with h5py.File(outpath, "w" if overwrite else "a") as f:
            f.create_dataset(h5_dataset, data=data)
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

        # Write ops.npy alongside. All timepoint aliases must be set
        # consistently from the actual chunk size; otherwise downstream
        # readers see stale source values for keys we forgot to update.
        if metadata:
            ops = metadata.copy()
            ops["Ly"] = arr.shape[-2]
            ops["Lx"] = arr.shape[-1]
            nt = arr.shape[0]
            ops["nframes"] = nt
            ops["num_frames"] = nt
            ops["num_timepoints"] = nt
            ops["n_frames"] = nt
            ops["timepoints"] = nt
            ops["T"] = nt
            ops["nt"] = nt
            # Convert Path objects to strings for cross-platform compatibility
            np.save(outpath.parent / "ops.npy", _convert_paths_to_strings(ops))
    elif outpath.suffix.lower() == ".zarr":
        # Zarr v3 format for numpy arrays
        import zarr
        from zarr.codecs import BytesCodec, GzipCodec

        arr = np.asarray(data)

        # chunks: (1, ..., Y, X) - singleton leading dims, full spatial
        chunks = (1,) * (arr.ndim - 2) + arr.shape[-2:]

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
                    z.attrs[k] = (
                        v
                        if np.isscalar(v) or isinstance(v, (list, dict, str))
                        else str(v)
                    )
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
    nt, Ly, Lx = (
        shape[0],
        shape[-2],
        shape[-1],
    )  # shape is (T, Y, X), so [-2]=Ly, [-1]=Lx

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

    # Set top-level frame-count fields consistently from `nt`. We
    # protect these from the metadata-merge below to prevent stale
    # source values from clobbering the truncated count, but that means
    # we MUST set every alias here — otherwise they end up missing /
    # None in ops.npy and downstream readers get inconsistent values.
    ops["nframes"] = nt
    ops["num_frames"] = nt
    ops["num_timepoints"] = nt
    ops["n_frames"] = nt
    ops["timepoints"] = nt
    ops["T"] = nt
    ops["nt"] = nt

    # Merge extra metadata, but DON'T overwrite fields we've already set consistently
    # This prevents inconsistency between resolution aliases and frame counts
    protected_keys = {
        # Frame count fields — every alias is set explicitly above
        "nframes",
        "nframes_chan1",
        "nframes_chan2",
        "num_frames",
        "num_timepoints",
        "n_frames",
        "timepoints",
        "T",
        "nt",
        # Resolution fields (we've already set these consistently)
        "dx",
        "dy",
        "dz",
        "umPerPixX",
        "umPerPixY",
        "umPerPixZ",
        "pixel_resolution",
        "z_step",
    }
    for key, value in metadata.items():
        if key not in protected_keys:
            ops[key] = value

    # Convert Path objects to strings for cross-platform compatibility
    np.save(ops_path, _convert_paths_to_strings(ops))
    logger.debug(
        f"Ops file written to {ops_path} with nframes={ops['nframes']}, nframes_chan1={ops.get('nframes_chan1')}"
    )


VIDEO_QUALITY_PRESETS = ("preview", "high", "visually lossless", "lossless")


def _format_overlay_time(t_seconds: float) -> str:
    if t_seconds < 60:
        return f"{t_seconds:5.1f}s"
    m, s = divmod(t_seconds, 60)
    return f"{int(m)}m {s:4.1f}s"


def _draw_scalebar(frame_rgb: np.ndarray, dx_um: float) -> None:
    """Draw a scalebar in the bottom-left with the label centered below the bar.

    Bar width is exactly 10% of the frame so multiplying the label by 10
    gives the full-frame width. The whole block (bar + gap + label + outline)
    is laid out from the bottom up and clamped to stay inside the frame, so
    the label never gets cut off on the bottom or the left.
    """
    import cv2
    h, w = frame_rgb.shape[:2]
    bar_px = max(2, int(round(w * 0.10)))
    bar_um = bar_px * dx_um

    text = f"{bar_um:.3g} um"
    # smaller font than before — h/900 instead of h/600, capped at 0.6
    font_scale = max(0.3, min(0.6, h / 900.0))
    font_thickness = 1
    (text_w, text_h), text_baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness,
    )

    bar_h = max(2, int(round(h * 0.010)))
    gap = max(2, int(round(h * 0.012)))  # space between bar and label
    pad_x = max(4, int(round(w * 0.025)))
    pad_y = max(4, int(round(h * 0.020)))
    outline_pad = font_thickness + 1  # the +2 outline added per putText call

    # lay out from the bottom up so the label baseline sits inside the frame
    label_baseline_y = h - pad_y - outline_pad
    label_top_y = label_baseline_y - text_h
    bar_bottom_y = label_top_y - gap
    bar_top_y = bar_bottom_y - bar_h

    bar_x0 = pad_x
    bar_x1 = bar_x0 + bar_px

    # text centered on bar, but never off the left edge
    text_x = bar_x0 + (bar_px - text_w) // 2
    text_x = max(pad_x, text_x)
    # also keep right edge inside the frame
    text_x = min(text_x, w - pad_x - text_w)

    # black backing under the bar for contrast on bright cmaps
    cv2.rectangle(
        frame_rgb,
        (bar_x0 - 1, bar_top_y - 1), (bar_x1 + 1, bar_bottom_y + 1),
        (0, 0, 0), -1,
    )
    cv2.rectangle(
        frame_rgb,
        (bar_x0, bar_top_y), (bar_x1, bar_bottom_y),
        (255, 255, 255), -1,
    )

    cv2.putText(
        frame_rgb, text, (text_x, label_baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), font_thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb, text, (text_x, label_baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), font_thickness, cv2.LINE_AA,
    )


def _draw_time_overlay(frame_rgb: np.ndarray, t_seconds: float) -> None:
    """Draw an MM:SS / X.Xs clock on `frame_rgb` in-place (top-left)."""
    import cv2
    h = frame_rgb.shape[0]
    text = _format_overlay_time(t_seconds)
    scale = max(0.4, min(1.5, h / 600.0))
    thickness = max(1, int(round(scale * 1.6)))
    pos = (max(6, int(scale * 12)), max(18, int(scale * 28)))
    cv2.putText(
        frame_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )

# yuv420p is left to imageio's default — adding -pix_fmt here collides with
# imageio's own injected -pix_fmt and triggers a "Multiple -pix_fmt" ffmpeg
# warning. yuv420p is mathematically lossless for grayscale input since R=G=B
# means chroma is identically zero (subsampling zero is still zero) and is the
# only browser-compatible chroma layout.
# veryslow + yuv444p was tried and crashed the bundled imageio-ffmpeg binary
# mid-stream (broken pipe). slow gives ~95% of the compression efficiency at a
# fraction of the memory/time and is rock-solid.
# "lossless" intentionally uses crf=8, not crf=0: -crf 0 enables libx264's
# lossless mode which produces a non-standard High Profile stream that Windows
# Photos / Chrome / native QuickTime refuse to display (file is created but
# won't open). crf=8 stays inside standard High Profile so every player works,
# and for 8-bit output it is mathematically lossless within the LSB.
_X264_PRESET_TABLE = {
    "preview":           {"crf": 23, "preset": "medium", "tune": None},
    "high":              {"crf": 18, "preset": "slow",   "tune": None},
    "visually lossless": {"crf": 14, "preset": "slow",   "tune": None},
    "lossless":          {"crf":  8, "preset": "slow",   "tune": "psnr"},
}

_MPEG4_QSCALE_TABLE = {
    "preview": 8,
    "high": 4,
    "visually lossless": 2,
    "lossless": 1,
}


def _resolve_quality_preset(quality: str | int) -> str:
    """Normalize quality (string preset name or legacy 1-10 int) to a preset key."""
    if isinstance(quality, str):
        key = quality.strip().lower().replace("_", " ")
        if key not in _X264_PRESET_TABLE:
            raise ValueError(
                f"Unknown quality preset {quality!r}. "
                f"Expected one of {VIDEO_QUALITY_PRESETS}."
            )
        return key
    q = int(quality)
    if q <= 3:
        return "preview"
    if q <= 7:
        return "high"
    if q <= 9:
        return "visually lossless"
    return "lossless"


def _build_video_output_params(codec: str, quality: str | int) -> list[str]:
    """Build ffmpeg output_params for (codec, quality preset)."""
    preset = _resolve_quality_preset(quality)
    if codec in ("libx264", "libx265"):
        cfg = _X264_PRESET_TABLE[preset]
        params = ["-crf", str(cfg["crf"]), "-preset", cfg["preset"]]
        if cfg.get("tune"):
            params.extend(["-tune", cfg["tune"]])
        return params
    if codec == "mpeg4":
        return ["-qscale:v", str(_MPEG4_QSCALE_TABLE[preset])]
    if codec == "rawvideo":
        return []
    return ["-q:v", str(_MPEG4_QSCALE_TABLE[preset])]


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
    quality: str | int = "visually lossless",
    codec: str = "libx264",
    max_frames: int | None = None,
    mean_subtract: np.ndarray | None = None,
    temporal_mode: str = "mean",
    time_overlay: bool = False,
    scalebar: bool = False,
    pixel_size_um: float | None = None,
):
    """
    Export array data to video file (mp4 or mov).

    Works with 3D (T, Y, X) or 4D (T, Z, Y, X) arrays, including lazy arrays.
    Optimized for high-quality output suitable for presentations and websites.

    Parameters
    ----------
    data : array-like
        3D array (T, Y, X) or 4D array (T, Z, Y, X). Supports lazy arrays.
    output_path : str or Path
        Output video path. Extension determines format (.mp4 or .mov).
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
        Rolling-window size in frames. Reduces flicker/noise.
        0 = disabled, 3-7 = subtle, 10+ = heavy. The aggregation applied to
        the window is controlled by `temporal_mode`.
    temporal_mode : {"mean", "max", "std"}, default "mean"
        How the rolling window is aggregated. "mean" smooths flicker, "max"
        emphasizes transient activity, "std" highlights variance/noise.
    mean_subtract : ndarray, optional
        2D image (Y, X) subtracted from every frame before contrast scaling.
        Useful for removing static structure to highlight dynamics.
    time_overlay : bool, default False
        Draw a clock in the top-left of every frame showing elapsed *recording*
        time (frame_index / fps). Independent of speed_factor — the overlay
        always reflects the source recording duration, so a sped-up clip ticks
        through real seconds faster on screen.
    scalebar : bool, default False
        Draw a scalebar at exactly 10% of frame width in the bottom-left, with
        a "NN um" label. Multiply the label by 10 to read the full-frame width.
        Requires `pixel_size_um` (taken from `data.dx` automatically when the
        input array exposes the VoxelSize feature).
    pixel_size_um : float, optional
        Pixel size in micrometers (X). Read from `data.dx` if not provided.
        If neither source has a positive value, scalebar is silently skipped
        with a warning.
    spatial_smooth : float, default 0
        Gaussian blur sigma (pixels). Reduces pixel noise.
        0 = disabled, 0.5-1.0 = subtle, 2+ = heavy blur.
    gamma : float, default 1.0
        Gamma correction. <1 = brighter midtones, >1 = darker midtones.
        0.7-0.8 often looks good for calcium imaging.
    cmap : str, optional
        Matplotlib colormap name (e.g., "viridis", "gray", "hot").
        If None, outputs grayscale.
    quality : str or int, default "visually lossless"
        Quality preset. One of:
          - "preview"           crf 23, preset medium, yuv420p (small/fast)
          - "high"              crf 18, preset slow,    yuv420p
          - "visually lossless" crf 14, preset veryslow, yuv444p
          - "lossless"          crf 0,  preset veryslow, yuv444p (math-lossless)
        Ints 1-10 are mapped to presets for backwards compatibility
        (1-3 -> preview, 4-7 -> high, 8-9 -> visually lossless, 10 -> lossless).
    codec : str, default "libx264"
        Video codec. "libx264" for mp4 (best compatibility). For mpeg4 the
        preset is mapped to -qscale:v; lossless mode is not supported there.
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

    ext = output_path.suffix.lower()
    if codec == "rawvideo" and ext != ".mkv":
        raise ValueError(
            f"codec='rawvideo' is not supported in {ext} containers — "
            f"use .mkv (or pick codec='libx264' for .mp4/.mov)."
        )

    # resolve scalebar pixel size before np.asarray strips metadata-bearing wrappers.
    # Try arr.dx, then arr.metadata['dx'], then arr.metadata['pixel_resolution'][0]
    # (ScanImage stores it as a (dx, dy) tuple).
    if scalebar and pixel_size_um is None:
        candidates = []
        candidates.append(getattr(data, "dx", None))
        candidates.append(getattr(data, "dy", None))
        md = getattr(data, "metadata", None)
        if isinstance(md, dict):
            candidates.append(md.get("dx"))
            candidates.append(md.get("dy"))
            pr = md.get("pixel_resolution")
            if isinstance(pr, (tuple, list)) and len(pr) >= 1:
                candidates.append(pr[0])
            elif pr is not None:
                candidates.append(pr)
        for value in candidates:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v > 0:
                pixel_size_um = v
                break
    if scalebar and (pixel_size_um is None or pixel_size_um <= 0):
        logger.warning(
            f"scalebar requested but pixel_size_um is unavailable "
            f"(resolved value: {pixel_size_um!r}); skipping."
        )
        scalebar = False
    elif scalebar:
        logger.info(f"Scalebar enabled with pixel_size_um={pixel_size_um}")

    # normalize to 5D TCZYX (handles raw numpy arrays from external callers)
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, np.newaxis, :, :]
    elif arr.ndim == 4:
        arr = arr[:, np.newaxis, :, :, :]
    shape = arr.shape

    plane_idx = plane if plane is not None else 0
    if plane_idx >= shape[2]:
        raise ValueError(f"plane={plane_idx} but array only has {shape[2]} planes")
    n_frames = shape[0]
    height, width = shape[3], shape[4]
    logger.info(
        f"Exporting plane {plane_idx}: {n_frames} frames, {height}x{width}"
    )

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

    # Determine intensity range from sample frames. When mean_subtract is on,
    # percentile must be computed on subtracted samples so contrast matches
    # what gets rendered.
    if vmin is None or vmax is None:
        n_samples = min(50, n_frames)
        sample_indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)
        samples = []
        for i in sample_indices:
            frame = np.asarray(arr[i, 0, plane_idx], dtype=np.float32)
            if mean_subtract is not None:
                frame = frame - np.asarray(mean_subtract, dtype=np.float32)
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

    output_params = _build_video_output_params(codec, quality)
    logger.info(f"ffmpeg output params: {' '.join(output_params)}")

    _temporal_aggregators = {
        "mean": lambda buf: np.mean(buf, axis=0),
        "max":  lambda buf: np.max(buf, axis=0),
        "std":  lambda buf: np.std(buf, axis=0),
    }
    if temporal_mode not in _temporal_aggregators:
        raise ValueError(
            f"temporal_mode={temporal_mode!r} not in {list(_temporal_aggregators)}"
        )
    aggregate_window = _temporal_aggregators[temporal_mode]
    frame_buffer = [] if temporal_smooth > 0 else None

    mean_sub_2d = None
    if mean_subtract is not None:
        mean_sub_2d = np.asarray(mean_subtract, dtype=np.float32)
        if mean_sub_2d.shape != (height, width):
            raise ValueError(
                f"mean_subtract shape {mean_sub_2d.shape} != frame ({height}, {width})"
            )

    writer = imageio.get_writer(
        str(output_path),
        fps=output_fps,
        codec=codec,
        macro_block_size=1,
        output_params=output_params,
    )

    try:
        for i in tqdm(range(n_frames), desc="Writing video", unit="frames"):
            # get frame from 5D TCZYX (first channel, selected plane)
            frame = np.asarray(arr[i, 0, plane_idx], dtype=np.float32)

            if mean_sub_2d is not None:
                frame = frame - mean_sub_2d

            if temporal_smooth > 0:
                frame_buffer.append(frame)
                if len(frame_buffer) > temporal_smooth:
                    frame_buffer.pop(0)
                frame = aggregate_window(frame_buffer)

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

            if time_overlay or scalebar:
                # cv2 draws in-place; ensure the array is contiguous and writable
                frame_rgb = np.ascontiguousarray(frame_rgb)
                if time_overlay:
                    _draw_time_overlay(frame_rgb, i / fps)
                if scalebar:
                    _draw_scalebar(frame_rgb, pixel_size_um)

            writer.append_data(frame_rgb)
    finally:
        writer.close()

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Video saved to {output_path} ({file_size_mb:.1f} MB)")
    return output_path


def _write_volumetric_video(
    arr,
    outpath: Path,
    metadata: dict | None = None,
    planes: list | None = None,
    frames: list | None = None,
    channels: list | None = None,
    ext: str = "mp4",
    overwrite: bool = False,
    output_suffix: str | None = None,
    progress_callback=None,
    show_progress: bool = True,
    fps: int = 30,
    speed_factor: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
    temporal_smooth: int = 0,
    spatial_smooth: float = 0.0,
    gamma: float = 1.0,
    cmap: str | None = None,
    quality: str | int = "visually lossless",
    codec: str = "libx264",
    mean_subtract_stack: np.ndarray | None = None,
    temporal_mode: str = "mean",
    time_overlay: bool = False,
    scalebar: bool = False,
):
    """
    Write one video file per (z-plane, channel).

    Filename pattern: zplaneNN_tpN-tpN_chNN_<suffix>.<ext> — Z first because
    each file is a single plane; channel suffix is always present.

    `mean_subtract_stack` is an optional (C, Z, Y, X) array of per-channel,
    per-plane mean images; the (c, z) slice is forwarded to to_video for that
    iteration.
    """
    from mbo_utilities.arrays.features import (
        OutputFilename,
        DimensionTag,
        TAG_REGISTRY,
    )

    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    ext_clean = ext.lower().lstrip(".")

    num_planes = arr.shape5d[2]
    num_channels = getattr(arr, "num_color_channels", arr.shape5d[1])
    nframes_total = arr.shape5d[0]

    planes_0idx = [p - 1 for p in planes] if planes else list(range(num_planes))
    channels_0idx = [c - 1 for c in channels] if channels else list(range(num_channels))

    if frames:
        frame_indices_0 = [f - 1 for f in frames]
    else:
        frame_indices_0 = None

    suffix = output_suffix.lstrip("_") if output_suffix else "movie"

    # carry dx through to to_video — np.asarray(sliced) below would strip it.
    # Try several sources because not every array type inherits VoxelSizeMixin:
    #   1. arr.dx (VoxelSizeMixin, e.g. some Suite2pArray/Zarr paths)
    #   2. arr.metadata['dx']                       (suite2p ops style)
    #   3. arr.metadata['pixel_resolution'][0]      (ScanImage style — (dx, dy))
    pixel_size_um = None
    pixel_size_source = "none"
    if scalebar:
        candidates: list[tuple[str, object]] = []
        candidates.append(("arr.dx", getattr(arr, "dx", None)))
        md = getattr(arr, "metadata", None)
        if isinstance(md, dict):
            candidates.append(("metadata['dx']", md.get("dx")))
            pr = md.get("pixel_resolution")
            if isinstance(pr, (tuple, list)) and len(pr) >= 1:
                candidates.append(("metadata['pixel_resolution'][0]", pr[0]))
            elif pr is not None:
                candidates.append(("metadata['pixel_resolution']", pr))
        for source, value in candidates:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v > 0:
                pixel_size_um = v
                pixel_size_source = source
                break
        logger.info(
            f"Scalebar pixel size: candidates={[(s, v) for s, v in candidates]} "
            f"-> pixel_size_um={pixel_size_um!r} (from {pixel_size_source})"
        )

    t_tag = DimensionTag.from_dim_size(TAG_REGISTRY["T"], nframes_total, frames)

    total_files = len(planes_0idx) * len(channels_0idx)
    file_idx = 0

    for plane_idx in planes_0idx:
        for c_idx in channels_0idx:
            z_tag = DimensionTag.from_dim_size(TAG_REGISTRY["Z"], num_planes, [plane_idx + 1])
            c_tag = DimensionTag.from_dim_size(TAG_REGISTRY["C"], num_channels, [c_idx + 1])

            filename = OutputFilename([z_tag, c_tag, t_tag], suffix=suffix).build(f".{ext_clean}")
            target = outpath / filename

            if target.exists() and not overwrite:
                logger.warning(f"File {target} exists. Skipping write.")
                file_idx += 1
                continue

            sliced = arr[:, c_idx, plane_idx]
            if frame_indices_0 is not None:
                sliced = np.asarray(sliced)[frame_indices_0]

            logger.info(
                f"Writing video {file_idx + 1}/{total_files}: {target.name}"
            )
            mean_img = None
            if mean_subtract_stack is not None:
                mean_img = mean_subtract_stack[c_idx, plane_idx]

            to_video(
                sliced,
                target,
                fps=fps,
                speed_factor=speed_factor,
                plane=None,
                vmin=vmin,
                vmax=vmax,
                vmin_percentile=vmin_percentile,
                vmax_percentile=vmax_percentile,
                temporal_smooth=temporal_smooth,
                spatial_smooth=spatial_smooth,
                gamma=gamma,
                cmap=cmap,
                quality=quality,
                codec=codec,
                mean_subtract=mean_img,
                temporal_mode=temporal_mode,
                time_overlay=time_overlay,
                scalebar=scalebar,
                pixel_size_um=pixel_size_um,
            )

            file_idx += 1
            if progress_callback:
                progress_callback(file_idx / total_files, target.name)

    if progress_callback:
        progress_callback(1.0, "Complete")

    return outpath
