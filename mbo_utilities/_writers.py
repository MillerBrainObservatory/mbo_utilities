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
        assert type(plane_index) is int, "plane_index must be an integer"
        metadata["plane"] = plane_index + 1

    nframes_target = kwargs.get("num_frames", metadata.get("num_frames"))
    if nframes_target is not None:
        metadata["num_frames"] = int(nframes_target)
        metadata["nframes"] = int(nframes_target)
    else:
        nframes_target = data.shape[0]

    H0, W0 = data.shape[-2], data.shape[-1]
    fname = filename
    writer = _get_file_writer(fname.suffix, overwrite=overwrite)

    # get chunk size via bytes per timepoint
    itemsize = np.dtype(data.dtype).itemsize
    ntime = int(nframes_target)  # T

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
        logger.debug(f"Using provided shift_vector of type {type(shift_vector)} length {len(shift_vector)}")
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

        if summary_path.is_file():
            summary = np.load(Path(summary_path), allow_pickle=True).item()
        else:
            raise FileNotFoundError(
                f"Summary file not found s3d-job dir: \n "
                f"{s3d_job_dir} \n"
                f"or summary path: {summary_path}"
            )

        plane_shifts = summary["plane_shifts"]

        assert plane_index is not None, "plane_index must be provided when using shifts"

        pt, pb, pl, pr = compute_pad_from_shifts(plane_shifts)
        H_out = H0 + pt + pb
        W_out = W0 + pl + pr

        iy, ix = map(int, plane_shifts[plane_index])
        yy = slice(pt + iy, pt + iy + H0)
        xx = slice(pl + ix, pl + ix + W0)
        out_shape = (ntime, H_out, W_out)
        shift_applied = True
        metadata[f"plane{plane_index}_shift"] = (iy, ix)

    if not shift_applied:
        out_shape = (ntime, H0, W0)

    start = 0
    for i in range(nchunks):
        end = start + base + (1 if i < extra else 0)
        chunk = (
            data[start:end, plane_index, :, :]
            if plane_index is not None
            else data[start:end, :, :]
        )

        if chunk.ndim == 4 and chunk.shape[1] == 1:
            chunk = chunk.squeeze()

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

    if fname.suffix in [".tiff", ".tif"]:
        _close_tiff_writers()
    elif fname.suffix in [".bin"]:
        _close_bin_writers()

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

    if key not in _write_bin._writers:
        if overwrite and fname.exists():
            fname.unlink()

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

    # squeeze 4D arrays to 3D
    if data.ndim == 4 and data.shape[1] == 1:
        data = data.squeeze()
    bf[off : off + data.shape[0]] = data
    bf.file.flush()
    _write_bin._offsets[key] = off + data.shape[0]

    if first_write:
        write_ops(metadata, fname, **kwargs)

    logger.debug(f"Wrote {data.shape[0]} frames to {fname}.")


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

    if filename not in _write_tiff._writers:
        if filename.exists() and not overwrite:
            logger.warning(
                f"File {filename} already exists and overwrite=False. Skipping write."
            )
            return
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
            metadata=_make_json_serializable(metadata) if is_first else {},
        )
    _write_tiff._first_write[filename] = False

def _write_zarr(
    path,
    data,
    *,
    overwrite=True,
    metadata=None,
    level=1,
    **kwargs,
):
    sharded = kwargs.get("sharded", False)

    filename = Path(path)
    if not hasattr(_write_zarr, "_arrays"):
        _write_zarr._arrays = {}
        _write_zarr._offsets = {}

    if overwrite and filename in _write_zarr._arrays:
        del _write_zarr._arrays[filename]
        del _write_zarr._offsets[filename]

    if filename not in _write_zarr._arrays:
        if filename.exists() and overwrite:
            shutil.rmtree(filename)

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

        z = zarr.create(
            store=str(filename),
            shape=(nframes, h, w),
            chunks=chunks,
            dtype=data.dtype,
            codecs=codecs,
            overwrite=True,
        )

        for k, v in (metadata or {}).items():
            z.attrs[k] = v

        _write_zarr._arrays[filename] = z
        _write_zarr._offsets[filename] = 0

    z = _write_zarr._arrays[filename]
    offset = _write_zarr._offsets[filename]
    z[offset : offset + data.shape[0]] = data
    _write_zarr._offsets[filename] = offset + data.shape[0]


def _write_zarr_v2(path, data, *, overwrite=True, metadata=None, **kwargs):
    compressor = None

    filename = Path(path)

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
            filters=compressor,
        )
        metadata = _make_json_serializable(metadata) if metadata else {}
        for k, v in metadata.items():
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
    metadata: dict = {},
):
    outpath = Path(outpath)
    if outpath.exists() and not overwrite:
        raise FileExistsError(f"{outpath} already exists and overwrite=False")

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
