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
        if isinstance(k, str) and k.startswith("h") and _is_disabled_si_module(v):
            continue
        if recursive and isinstance(v, dict):
            v = _filter_disabled_modules(v, recursive=True)
        result[k] = v
    return result


# drop ndarrays larger than this from serialized output
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
        # what's really on disk. the array's 5D shape and the
        # metadata["shape"] from _imwrite_base can differ from the chunk —
        # e.g. ScanImage's shape uses ROI metadata (550) but process_rois
        # returns the real stitched height (542). when axial shifts are applied, the
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
    suffix = output_suffix or ""
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
    which produced one `.h5` file per z-plane (`zplane01.h5`,
    `zplane03.h5`, ...). That mismatched zarr's behaviour, where the
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

    suffix = output_suffix or ""
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

    # h5 chunking — one Y×X plane per chunk so a fixed-(t, z) GUI scrub
    # pulls exactly that plane off disk. The previous (1, n_planes, Y, X)
    # spec packed every Z layer into one chunk, forcing the whole volume
    # to decompress for any single-plane read — the same Z-bunching bug
    # the zarr writers fixed.
    inner_chunk = (1, 1, Ly_out, Lx_out)

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
    target_chunk_mb: int = 2048,
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
    pyramid_method: str = "median",
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
        Maximum shard byte budget in MB. The shard layout aims for one shard
        per ``(c, z)`` (i.e. ``shards=(T, 1, 1, Y, X)`` for 5D, ``(T, 1, Y, X)``
        for 4D) so a fixed-``(c, z)`` T-scrub opens a single file. If
        ``T * Y * X * itemsize`` would exceed ``target_chunk_mb``, the shard's
        T-extent is truncated and additional shards are added along T. The
        2 GB default keeps typical multi-thousand-frame recordings to one
        shard per ``(c, z)``.
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
        per-level factors follow the webknossos anisotropic-mag algorithm
        (Z is downsampled at deep levels for anisotropic volumes).
    pyramid_method : str
        downsampling reducer: "median" (default, webknossos intensity),
        "mode" (webknossos labels/masks), "mean", "nearest", "gaussian".
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
    from mbo_utilities.arrays.features._pyramid import downsample_block
    from mbo_utilities.arrays.isoview.consolidate import _compute_anisotropic_mags
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
    suffix = output_suffix or ""
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

    # inner chunk shape: one Y×X plane per chunk. This is the unit fpl/the
    # GUI scrub fetches, so any larger chunk forces zarr to decompress
    # extra data per frame. For 5D the chunk pins (t, c, z); for 4D it
    # pins (t, z). Benchmark (D:/demo/zarr_chunking_benchmark): this chunk
    # shape is ~22x faster than the legacy Z-major (1, 1, Z, 64, 64).
    itemsize = np.dtype(data.dtype).itemsize
    if output_5d:
        inner_chunk = (1, 1, 1, Ly_out, Lx_out)
    else:
        inner_chunk = (1, 1, Ly_out, Lx_out)

    bytes_per_yx = Ly_out * Lx_out * itemsize

    # build codec chain
    inner_codecs = _build_inner_codecs(compressor, compression_level, shuffle=shuffle)

    if sharded:
        # one shard per (c, z) at all T — the GUI's fixed-(c, z) scrub
        # opens exactly one shard file per channel-plane and reads chunks
        # from it sequentially. File count = C * Z * ceil(T / shard_t),
        # which is C*Z when target_chunk_mb is large enough to hold the
        # whole T axis (the 2 GB default covers ~1400 frames at 1848x768
        # uint16). Bump target_chunk_mb to widen the per-(c,z) shard or
        # shrink it to split T across multiple shards.
        target_bytes = target_chunk_mb * 1024 * 1024
        max_shard_t = max(1, target_bytes // max(1, bytes_per_yx))
        shard_t = min(n_frames, max_shard_t)
        if output_5d:
            shard_shape = (shard_t, 1, 1, Ly_out, Lx_out)
        else:
            shard_shape = (shard_t, 1, Ly_out, Lx_out)

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

    # OME-NGFF axes order: TZYX for 4D, TCZYX for 5D
    output_dims = ("T", "C", "Z", "Y", "X") if output_5d else ("T", "Z", "Y", "X")
    ome_meta = out_meta.to_ome_ngff(dims=output_dims)
    base_scale = ome_meta["coordinateTransformations"][0]["scale"]
    dimension_names = [d.lower() for d in output_dims]

    # store dims in metadata for readers
    md["dims"] = output_dims

    # pyramid factors via the webknossos anisotropic-mag algorithm: each step
    # doubles the axis with the smallest physical size (mag * voxel), so
    # anisotropic volumes converge toward isotropy before downsampling. Z is
    # downsampled at deep levels, unlike a fixed YX-only scheme. base_scale's
    # last three entries are the (z, y, x) voxel size for both 4D and 5D.
    if pyramid:
        nz_p, ny_p, nx_p = (int(s) for s in target_shape[-3:])
        voxel_zyx = tuple(float(v) for v in base_scale[-3:])
        pyramid_mags = _compute_anisotropic_mags(
            voxel_zyx, (nz_p, ny_p, nx_p), pyramid_max_layers, min_size=64
        )
        level_spatial = [
            (max(1, nz_p // mz), max(1, ny_p // my), max(1, nx_p // mx))
            for mz, my, mx in pyramid_mags
        ]
        if debug:
            logger.info(f"  Pyramid: {len(pyramid_mags)} levels, mags {pyramid_mags}")
    else:
        pyramid_mags = None
        level_spatial = None

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
            if pyramid_mags
            else "Single resolution level (no pyramid)"
        ),
    }

    if pyramid_mags:
        datasets = []
        for lvl_idx, (mz, my, mx) in enumerate(pyramid_mags):
            # level scale = level-0 voxel size * cumulative mag on (z, y, x)
            physical_scale = list(base_scale)
            physical_scale[-3] = base_scale[-3] * mz
            physical_scale[-2] = base_scale[-2] * my
            physical_scale[-1] = base_scale[-1] * mx
            datasets.append(
                {
                    "path": str(lvl_idx),
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
    total_work = n_frames * (len(pyramid_mags) if pyramid_mags else 1)
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
                    chunk_info.progress / (len(pyramid_mags) if pyramid_mags else 1)
                )
    finally:
        if pbar:
            pbar.close()

    # generate pyramid levels from level 0 data
    if pyramid_mags and len(pyramid_mags) > 1:
        if show_progress:
            pbar = tqdm(
                total=len(pyramid_mags) - 1,
                desc="Building pyramid",
                unit="levels",
            )

        for lvl_idx in range(1, len(pyramid_mags)):
            prev_path = str(lvl_idx - 1)

            # read previous level data
            prev_z = root[prev_path]
            prev_data = prev_z[:]

            # per-level factor from realized parent/child spatial shapes,
            # clamped to >=1 so a singleton axis is never reduced to 0. T
            # (and C) are never downsampled.
            pz, py, px = level_spatial[lvl_idx - 1]
            cz, cy, cx = level_spatial[lvl_idx]
            zf, yf, xf = max(1, pz // cz), max(1, py // cy), max(1, px // cx)
            factor = (1, 1, zf, yf, xf) if output_5d else (1, zf, yf, xf)

            level_data = downsample_block(prev_data, factor, pyramid_method)

            # compute chunk + shard shapes for this level using the SAME
            # recipe as level 0: one Y×X plane per chunk (T, C, Z pinned to
            # 1), all T per (c, z) shard. Y and X come from the downsampled
            # level shape; Z may shrink at deep levels but the chunk still
            # pins one plane. Without this, lower-resolution levels were
            # stored as a single huge chunk, forcing a full-volume decompress
            # to view any frame and breaking T-scrub when a viewer fell back
            # to a pyramid level.
            lvl_shape = level_data.shape
            lvl_Y = lvl_shape[-2]
            lvl_X = lvl_shape[-1]
            if output_5d:
                lvl_inner = (1, 1, 1, lvl_Y, lvl_X)
            else:
                lvl_inner = (1, 1, lvl_Y, lvl_X)

            lvl_inner_codecs = _build_inner_codecs(
                compressor, compression_level, shuffle=shuffle
            )

            if sharded:
                lvl_bytes_per_yx = lvl_Y * lvl_X * itemsize
                lvl_target_bytes = target_chunk_mb * 1024 * 1024
                lvl_max_shard_t = max(
                    1, lvl_target_bytes // max(1, lvl_bytes_per_yx)
                )
                lvl_shard_t = min(lvl_shape[0], lvl_max_shard_t)
                if output_5d:
                    lvl_shard = (lvl_shard_t, 1, 1, lvl_Y, lvl_X)
                else:
                    lvl_shard = (lvl_shard_t, 1, lvl_Y, lvl_X)
                lvl_codec = ShardingCodec(
                    chunk_shape=lvl_inner,
                    codecs=lvl_inner_codecs,
                    index_codecs=[BytesCodec(), Crc32cCodec()],
                )
                lvl_codecs = [lvl_codec]
                lvl_chunks = lvl_shard
            else:
                lvl_codecs = lvl_inner_codecs
                lvl_chunks = lvl_inner

            lvl_z = zarr.create(
                store=root.store,
                path=str(lvl_idx),
                shape=lvl_shape,
                chunks=lvl_chunks,
                dtype=data.dtype,
                codecs=lvl_codecs,
                dimension_names=dimension_names,
                overwrite=True,
            )

            # write data
            lvl_z[:] = level_data

            # napari-compatible scale: level-0 voxel size * cumulative mag
            mz, my, mx = pyramid_mags[lvl_idx]
            physical_scale = list(base_scale)
            physical_scale[-3] = base_scale[-3] * mz
            physical_scale[-2] = base_scale[-2] * my
            physical_scale[-1] = base_scale[-1] * mx
            lvl_z.attrs["scale"] = physical_scale

            if debug:
                logger.info(f"  Wrote level {lvl_idx}: {lvl_shape}")

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
                # default: one shard per file — keeps file count to a small
                # constant even for long recordings. Aligns with the GUI's
                # fixed-(c, z) T-scrub pattern (one shard open, all T inside).
                shard_t = nframes

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
        # Zarr v3 format for numpy arrays. Match the layout produced by
        # `_write_volumetric_zarr` so generic-writer output behaves the
        # same in the GUI (one chunk per Y×X plane, one shard per (c,z)
        # at all T). Without this, raw numpy arrays got T*C*Z chunk
        # files instead of C*Z shard files, and the GUI's T-scrub
        # had no per-frame fast path.
        import zarr
        from zarr.codecs import (
            BytesCodec,
            Crc32cCodec,
            GzipCodec,
            ShardingCodec,
        )

        arr = np.asarray(data)

        if arr.ndim < 3:
            # 2D: one chunk, sharding adds no value
            create_kwargs = dict(
                chunks=arr.shape,
                codecs=[BytesCodec(), GzipCodec(level=5)],
            )
        else:
            # ≥3D: chunks=(1,...,1,Y,X) per Y×X plane;
            # shards=(T,1,...,1,Y,X) so a fixed-(c,z) T-scrub opens
            # exactly one shard file. Compression off by default to
            # match _imwrite_base; callers needing it can compress
            # before calling.
            Y, X = arr.shape[-2], arr.shape[-1]
            inner = (1,) * (arr.ndim - 2) + (Y, X)
            shard = (arr.shape[0],) + (1,) * (arr.ndim - 3) + (Y, X)
            create_kwargs = dict(
                chunks=shard,
                codecs=[
                    ShardingCodec(
                        chunk_shape=inner,
                        codecs=[BytesCodec()],
                        index_codecs=[BytesCodec(), Crc32cCodec()],
                    )
                ],
            )

        z = zarr.create(
            store=str(outpath),
            shape=arr.shape,
            dtype=arr.dtype,
            overwrite=True,
            zarr_format=3,
            **create_kwargs,
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


def to_video(*args, **kwargs):
    """Encode array data to an mp4 video file.

    The implementation moved to ``mbo_utilities.arrays.mp4``; this shim keeps
    the ``mbo_utilities._writers.to_video`` import path stable.
    """
    from mbo_utilities.arrays.mp4 import to_video as _to_video

    return _to_video(*args, **kwargs)
