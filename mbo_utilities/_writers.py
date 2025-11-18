import functools
import math
import warnings
from typing import Any

import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter, imwrite
import h5py

from . import log
from .file_io import write_ops
from ._parsing import _make_json_serializable
from .metadata import save_metadata_html

from ._binary import BinaryFile

from tqdm.auto import tqdm

logger = log.get("writers")

warnings.filterwarnings("ignore")

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]
CHUNKS = {0: "auto", 1: -1, 2: -1}



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
    metadata["shape"] = dshape

    if plane_index is not None:
        if not isinstance(plane_index, int):
            raise TypeError(f"plane_index must be an integer, got {type(plane_index)}")
        metadata["plane"] = plane_index + 1

    # Get target frame count from kwargs, metadata, or data shape (in that priority)
    nframes_target = kwargs.get("num_frames", metadata.get("num_frames"))

    # If nframes_target is 0 or None, use actual data shape
    if nframes_target is None or nframes_target == 0:
        nframes_target = data.shape[0]

    # Update metadata with the actual target
    if nframes_target is not None:
        metadata["num_frames"] = int(nframes_target)
        metadata["nframes"] = int(nframes_target)

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
            summary = np.load(Path(summary_path), allow_pickle=True).item()
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

    if "cleaned_scanimage_metadata" in metadata:
        meta_path = filename.parent.joinpath("metadata.html")
        save_metadata_html(metadata, meta_path)


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
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(path, data, *, overwrite: bool = False, metadata=None, **kwargs):
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

        _write_bin._writers[key] = BinaryFile(
            Ly, Lx, key, n_frames=metadata["num_frames"], dtype=np.int16
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
    bf.file.flush()
    _write_bin._offsets[key] = off + data.shape[0]

    if first_write:
        write_ops(metadata, fname, **kwargs)


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

    Returns
    -------
    dict
        OME-Zarr NGFF v0.5 metadata dictionary ready for zarr.attrs.update()
    """
    ndim = len(shape)

    # Extract spatial scales from metadata
    pixel_resolution = metadata.get("pixel_resolution", None)
    if pixel_resolution is not None:
        if isinstance(pixel_resolution, (list, tuple)) and len(pixel_resolution) >= 2:
            pixel_x = float(pixel_resolution[0])
            pixel_y = float(pixel_resolution[1])
        else:
            pixel_x = pixel_y = 1.0
    else:
        pixel_x = metadata.get("dx", 1.0)
        pixel_y = metadata.get("dy", 1.0)

    # Extract temporal scale
    frame_rate = metadata.get("frame_rate") or metadata.get("fs")
    if frame_rate:
        time_scale = 1.0 / float(frame_rate)  # seconds per frame
    else:
        time_scale = 1.0

    # Extract z-scale (if 4D)
    z_scale = metadata.get("z_step") or metadata.get("dz", 1.0)

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
            np.savez(outpath, data=data, metadata=metadata)
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
    else:
        raise ValueError(f"Unsupported file extension: {outpath.suffix}")
