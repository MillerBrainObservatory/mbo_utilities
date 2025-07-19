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

logger = log.get("writers")

warnings.filterwarnings("ignore")

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]
CHUNKS = {0: "auto", 1: -1, 2: -1}


def _close_bin_writers():
    if hasattr(_write_bin, "_writers"):
        for bf in _write_bin._writers.values():
            bf.close()
        _write_bin._writers.clear()
        _write_bin._offsets.clear()


def _close_tiff_writers():
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()


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
):
    if dshape is None:
        dshape = data.shape

    metadata = metadata or {}
    metadata["shape"] = dshape

    if plane_index:
        assert type(plane_index) is int, "plane_index must be an integer"
        metadata["plane"] = plane_index + 1

    fname = filename
    writer = _get_file_writer(fname.suffix, overwrite=overwrite)

    chunk_size = target_chunk_mb * 1024 * 1024
    nbytes = np.prod(dshape) * np.dtype(data.dtype).itemsize
    nchunks = max(1, int(np.ceil(nbytes / chunk_size)))

    # this works for txy and tzxy
    ntime = data.shape[0]
    base = ntime // nchunks
    extra = ntime % nchunks

    if not debug:
        pbar = tqdm(total=nchunks, desc=f"Saving {fname.name}")
    else:
        pbar = None

    start = 0
    for i in range(nchunks):
        end = start + base + (1 if i < extra else 0)
        chunk = data[start:end, plane_index, :, :] if plane_index is not None else data[start:end, :, :]
        writer(fname, chunk, metadata=metadata)
        if pbar:
            pbar.update(1)
        if progress_callback:
            progress_callback(pbar.n / pbar.total, current_plane=plane_index)
        start = end

    if pbar:
        pbar.close()

    if fname.suffix in [".tiff", ".tif"]:
        _close_tiff_writers()
    elif fname.suffix in [".bin"]:
        _close_bin_writers()

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


def _write_tiff(
    path, data, overwrite=True, metadata=None,
):
    filename = Path(path).with_suffix(".tif")

    if not hasattr(_write_tiff, "_writers"):
        _write_tiff._writers = {}
    if not hasattr(_write_tiff, "_first_write"):
        _write_tiff._first_write = {}

    if filename not in _write_tiff._writers:
        if filename.exists() and overwrite:
            filename.unlink()
        _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True)
        _write_tiff._first_write[filename] = True

    writer = _write_tiff._writers[filename]
    is_first = _write_tiff._first_write.get(filename, True)

    for frame in data:
        writer.write(
            frame,
            contiguous=True,
            photometric="minisblack",
            metadata=_make_json_serializable(metadata) if is_first else None,
        )
        _write_tiff._first_write[filename] = False


def _write_zarr(path, data, *, overwrite=True, metadata=None):
    filename = Path(path).with_suffix(".zarr")

    if not hasattr(_write_zarr, "_arrays"):
        _write_zarr._arrays = {}
        _write_zarr._offsets = {}

    if filename not in _write_zarr._arrays:
        if filename.exists() and overwrite:
            shutil.rmtree(filename)

        import zarr

        nframes = metadata["num_frames"]
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

