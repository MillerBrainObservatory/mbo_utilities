import functools
import warnings
from typing import Any

import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter
import h5py

from . import log
from .file_io import write_ops
from ._parsing import _make_json_serializable
from .util import is_running_jupyter

try:
    from suite2p.io import BinaryFile

    HAS_SUITE2P = True
except ImportError:
    HAS_SUITE2P = False
    BinaryFile = None

if is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm

logger = log.get("assembly")

warnings.filterwarnings("ignore")

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]
CHUNKS = {0: "auto", 1: -1, 2: -1}


def _close_tiff_writers():
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()


def _save_data(
    data: Any | list[Any],
    outpath: str | Path,
    planes=None,
    overwrite=False,
    ext=".tiff",
    metadata=None,
    target_chunk_mb=20,
    progress_callback=None,
    debug=False,
):
    metadata = metadata or {}
    if data.ndim == 2:
        data = data[np.newaxis, np.newaxis, ...]  # → (1, 1, Y, X)
    elif data.ndim == 3:
        data = data[np.newaxis, ...]  # → (1, T, Y, X)
    elif data.ndim == 4:
        pass  # already (Z, T, Y, X)
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    nt, nz, ny, nx = data.shape

    if "." in ext:
        ext = ext.split(".")[-1]
    if ext == "tiff":
        ext = "tif"

    outpath = Path(outpath)
    outpath.mkdir(exist_ok=True)

    final_shape = (nt, ny, nx)
    logger.info(f"Final shape: {final_shape} (nt, height, width)")

    metadata["fov"] = [ny, nx]
    metadata["shape"] = (nt, nx, ny)
    metadata["dims"] = ["time", "width", "height"]
    metadata["nframes"] = nt
    metadata["n_frames"] = nt  # alias
    metadata["num_frames"] = nt  # alias

    writer = _get_file_writer(ext, overwrite=overwrite)

    chunk_size = target_chunk_mb * 1024 * 1024
    total_chunks = sum(
        min(
            data.shape[0],
            max(
                1,
                int(
                    np.ceil(
                        data.shape[0] * data.shape[2] * data.shape[3] * 2 / chunk_size
                    )
                ),
            ),
        )
        for _ in planes
    )
    logger.info(
        f"Total chunks to save: {total_chunks} (target chunk size: {chunk_size / 1024 / 1024:.2f} MB)"
    )
    if not debug:
        pbar = tqdm(total=total_chunks, desc="Saving plane ", position=0)
    else:
        pbar = None

    for chan_index in planes:
        if ext == "bin":
            fname = outpath / f"plane{chan_index + 1}" / "data_raw.bin"
        else:
            fname = outpath / f"plane{chan_index + 1}.{ext}"

        if fname.exists() and not overwrite:
            print(
                f"File {fname} already exists with overwrite=False; skipping save.",
                flush=True,
            )
            if pbar:
                # simulate full save for skipped file
                nbytes_chan = data.shape[0] * data.shape[2] * data.shape[3] * 2
                num_chunks = min(
                    data.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size)))
                )
                pbar.update(num_chunks)
                pbar.set_description(f"Skipped plane {chan_index + 1}")
            continue

        if pbar:
            pbar.set_description(f"Saving plane {chan_index + 1}")

        metadata_plane = metadata.copy()

        metadata["save_path"] = str(fname.parent.expanduser().resolve())
        metadata_plane["plane"] = chan_index + 1
        metadata_plane["plane_index"] = chan_index

        nbytes_chan = data.shape[0] * data.shape[2] * data.shape[3] * 2
        num_chunks = min(data.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size))))

        base_frames_per_chunk = data.shape[0] // num_chunks
        extra_frames = data.shape[0] % num_chunks

        start = 0
        for chunk in range(num_chunks):
            frames_in_this_chunk = base_frames_per_chunk + (
                1 if chunk < extra_frames else 0
            )
            end = start + frames_in_this_chunk
            data_chunk = data[start:end, chan_index, :, :]
            logger.debug(
                f"Saving chunk {chunk + 1}/{num_chunks} for plane {chan_index + 1}:"
                f" {data_chunk.shape} (frames, height, width)"
            )
            writer(fname, data_chunk, metadata=metadata_plane)
            start = end
            if pbar:
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(pbar.n / pbar.total, current_plane=chan_index + 1)

    if pbar:
        pbar.close()

    if ext in ["tiff", "tif"]:
        _close_tiff_writers()


def _get_file_writer(ext, overwrite):
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
        if not HAS_SUITE2P:
            raise ValueError(
                "Suite2p needed to write binary files, please install it:\n"
                "pip install suite2p[io]"
            )
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(path, data, *, overwrite: bool = False, metadata=None):
    if not hasattr(_write_bin, "_writers"):
        _write_bin._writers, _write_bin._offsets = {}, {}

    fname = Path(path)
    fname.parent.mkdir(exist_ok=True)

    key = str(fname)
    first_write = False
    if key not in _write_bin._writers:
        if overwrite and fname.exists():
            fname.unlink()

        Ly, Lx = data.shape[1], data.shape[2]
        _write_bin._writers[key] = BinaryFile(
            Ly, Lx, key, n_frames=metadata["nframes"], dtype=np.int16
        )
        _write_bin._offsets[key] = 0
        first_write = True

    bf = _write_bin._writers[key]
    off = _write_bin._offsets[key]
    bf[off : off + data.shape[0]] = data
    bf.file.flush()
    _write_bin._offsets[key] = off + data.shape[0]

    if first_write:
        raw_filename = fname  # points to data_raw.bin
        write_ops(metadata, raw_filename)

    logger.debug(f"Wrote {data.shape[0]} frames to {fname}.")


def _write_h5(path, data, *, overwrite=True, metadata=None):
    filename = Path(path).with_suffix(".h5")

    if not hasattr(_write_h5, "_initialized"):
        _write_h5._initialized = {}
        _write_h5._offsets = {}

    if filename not in _write_h5._initialized:
        nframes = metadata["nframes"]
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


def _write_tiff(path, data, *, overwrite=True, metadata=None):
    filename = Path(path).with_suffix(".tif")

    if not hasattr(_write_tiff, "_writers"):
        _write_tiff._writers = {}

    if filename not in _write_tiff._writers:
        if filename.exists() and overwrite:
            filename.unlink()
        _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True)

        _write_tiff._first_write = {filename: True}
    else:
        _write_tiff._first_write = {filename: False}

    writer = _write_tiff._writers[filename]
    is_first = _write_tiff._first_write.get(filename, False)

    for frame in data:
        writer.write(
            frame,
            contiguous=True,
            photometric="minisblack",
            bitspersample=16,
            metadata=_make_json_serializable(metadata) if is_first else None,
            # extratags=[
            #     (340, "i", 1, (smin,), False),
            #     (341, "i", 1, (smax,), False),
            # ] if is_first else None,
        )
        _write_tiff._first_write[filename] = False


def _write_zarr(path, data, *, overwrite=True, metadata=None):
    import zarr

    filename = Path(path).with_suffix(".zarr")
    if not hasattr(_write_zarr, "_arrays"):
        _write_zarr._arrays, _write_zarr._offsets = {}, {}
    if filename not in _write_zarr._arrays:
        if overwrite and filename.exists():
            shutil.rmtree(filename)
        nframes = metadata["nframes"]
        h, w = data.shape[-2:]
        z = zarr.open(
            store=str(filename),
            mode="w",
            shape=(nframes, h, w),
            chunks=(1, h, w),
            dtype=data.dtype,
        )
        if metadata:
            for k, v in metadata.items():
                try:
                    z.attrs[k] = v if np.isscalar(v) else str(v)
                except Exception:
                    z.attrs[k] = str(v)
        _write_zarr._arrays[filename] = z
        _write_zarr._offsets[filename] = 0
    z = _write_zarr._arrays[filename]
    offset = _write_zarr._offsets[filename]
    z[offset : offset + data.shape[0]] = data
    _write_zarr._offsets[filename] = offset + data.shape[0]


def _write_zarr_v2(path, data, *, overwrite=True, metadata=None):
    filename = Path(path).with_suffix(".zarr")

    if not hasattr(_write_zarr, "_arrays"):
        _write_zarr._arrays = {}
        _write_zarr._offsets = {}

    if filename not in _write_zarr._arrays:
        if filename.exists() and overwrite:
            shutil.rmtree(filename)

        import zarr

        nframes = metadata["nframes"]
        h, w = data.shape[-2:]
        z = zarr.open(
            store=str(filename),
            mode="w",
            shape=(nframes, h, w),
            chunks=(1, h, w),
            dtype=data.dtype,
        )
        if metadata:
            for k, v in metadata.items():
                try:
                    z.attrs[k] = v
                except TypeError:
                    z.attrs[k] = str(v)

        _write_zarr._arrays[filename] = z
        _write_zarr._offsets[filename] = 0

    z = _write_zarr._arrays[filename]
    offset = _write_zarr._offsets[filename]
    z[offset : offset + data.shape[0]] = data
    _write_zarr._offsets[filename] = offset + data.shape[0]
