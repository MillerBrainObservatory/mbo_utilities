import copy
import functools
import os
import time
import warnings
import logging
from typing import Callable

import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter
import h5py
from icecream import ic

from .file_io import _make_json_serializable, Scan_MBO
from .metadata import get_metadata
from .util import is_running_jupyter
from .plot_util import save_phase_images_png

try:
    from suite2p.io import BinaryFile

    HAS_SUITE2P = True
except ImportError:
    HAS_SUITE2P = True
    BInaryFIle = None

if is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm

MBO_DEBUG = bool(int(os.getenv("MBO_DEBUG", "0")))  # export MBO_DEV=1 to enable
logging.basicConfig(level=logging.DEBUG if MBO_DEBUG else logging.INFO)

if not MBO_DEBUG:
    ic.disable()

# set a name the gui can use to identify this module
logger = logging.getLogger("mbo.save_as")
logger.setLevel(logging.WARNING)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

CHUNKS = {0: "auto", 1: -1, 2: -1}

warnings.filterwarnings("ignore")


def close_tiff_writers():
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()


def save_as(
    scan: Scan_MBO,
    savedir: str | Path,
    planes: list | tuple = None,
    metadata: dict = None,
    overwrite: bool = True,
    ext: str = ".tiff",
    order: list | tuple = None,
    trim_edge: list | tuple = (0, 0, 0, 0),
    fix_phase: bool = True,
    save_phase_png: bool = False,
    target_chunk_mb: int = 20,
    progress_callback: Callable = None,
    upsample: int = 20,
    debug: bool = False,
):
    """
    Save scan data to the specified directory in the desired format.

    Parameters
    ----------
    scan : scanreader.Scan_MBO
        An object representing scan data. Must have attributes such as `num_channels`,
        `num_frames`, `fields`, and `rois`, and support indexing for retrieving frame data.
    savedir : os.PathLike
        Path to the directory where the data will be saved.
    planes : int, list, or tuple, optional
        Plane indices to save. If `None`, all planes are saved. Default is `None`.
        1 based indexing.
    trim_edge : list, optional
        Number of pixels to trim on each W x H edge. (Left, Right, Top, Bottom). Default is (0,0,0,0).
    metadata : dict, optional
        Additional metadata to update the scan object's metadata. Default is `None`.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is `True`.
    ext : str, optional
        File extension for the saved data. Supported options are .tiff, .zarr and .h5.
        Default is `'.tiff'`.
    order : list or tuple, optional
        A list or tuple specifying the desired order of planes. If provided, the number of
        elements in `order` must match the number of planes. Default is `None`.
    fix_phase : bool, optional
        Whether to fix scan-phase (x/y) alignment. Default is `True`.
    save_phase_png : bool, optional
        If correcting scan-phase, save a directory with pre/post images centered on the most
        active regions of the frame, saved to the save_path. Default is 'False'.
    target_chunk_mb : int, optional
        Chunk size in megabytes for saving data. Increase to help with scan-phase correction.
    progress_callback : callable, optional
        A callback function to emit progress-bar events. It should accept a single float
        argument representing the progress (0.0 to 1.0) and an optional `current_plane` argument.
    debug : bool, optional
        If `True`, enables debugging mode with detailed output. Default is `False`.
    upsample : int, optional
        Upsampling factor for phase correction.
        Value of 1 means no upsampling. Default is `20`.

    Raises
    ------
    ValueError
        If an unsupported file extension is provided.
    """
    # Logging
    if debug:
        ic.enable()
        ic("Debugging mode ON")
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        ic.disable()

    # save path
    savedir = Path(savedir)
    ic(savedir)

    if not savedir.parent.is_dir():
        raise ValueError(f"{savedir} is not inside a valid directory.")
    savedir.mkdir(exist_ok=True)

    # handle channels and planes
    if not hasattr(scan, "num_channels"):
        raise ValueError(
            "Unable to determine the number of planes in this recording from 'scan.num_channels'"
        )

    if isinstance(planes, int):
        planes = [planes - 1]
    elif planes is None:  # DON'T use "if not planes", then 0 will be treated as falsy
        planes = list(range(scan.num_channels))
    else:
        planes = [p - 1 for p in planes]
    ic(planes)

    over_idx = [p for p in planes if p < 0 or p >= scan.num_planes]
    if over_idx:
        raise ValueError(
            f"Invalid plane indices {', '.join(map(str, [p + 1 for p in over_idx]))}; must be in range 1…{scan.num_channels}"
        )

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
            )
        planes = [planes[i] for i in order]

    # handle metadata
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata must be a dictionary, got {type(metadata)} instead."
        )

    # metadata is now either {} or None, so we can safely update it
    metadata = get_metadata(scan.tiff_files[0].filehandle.path)  # from the file

    # keep the scanimage metadata under the "si" key
    metadata.update(
        {"si": _make_json_serializable(scan.tiff_files[0].scanimage_metadata)}
    )
    metadata["save_path"] = str(savedir.resolve())

    # which rois to save
    if scan.roi is None:
        roi_list = [None]  # full‐stack
    elif scan.roi == 0:
        roi_list = list(range(1, scan.num_rois + 1))  # all individual ROIs
    elif isinstance(scan.roi, int):
        roi_list = [scan.roi]  # single ROI
    else:
        roi_list = list(scan.roi)  # list of ROIs

    start_time = time.time()

    # this is a bit confusing. If roi=None, that atttribute is set on the scan object and
    # when it is inddexed e.g. scan[0], it will return a stack with each roi assembled.
    if 0 in roi_list:
        if len(roi_list) > 1:
            roi_list = [r + 1 for r in roi_list if r is not None]
    for r in roi_list:
        ic(r, roi_list)
        subscan = copy.copy(scan)
        subscan.roi = r

        target = savedir if r is None else savedir / f"roi{r}"
        target.mkdir(exist_ok=True)
        meta = (metadata or {}).copy()
        if r is not None:
            meta["roi"] = r
        _save_data(
            subscan,
            target,
            planes=planes,
            overwrite=overwrite,
            ext=ext,
            trim_edge=trim_edge,
            fix_phase=fix_phase,
            save_phase_png=save_phase_png,
            target_chunk_mb=target_chunk_mb,
            metadata=meta,
            progress_callback=progress_callback,
            upsample=upsample,
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Time elapsed: {int(elapsed_time // 60)} minutes {int(elapsed_time % 60)} seconds."
    )


def _save_data(
    scan,
    path,
    planes,
    overwrite,
    ext,
    metadata,
    trim_edge=None,
    fix_phase=True,
    save_phase_png=False,
    target_chunk_mb=20,
    progress_callback=None,
    upsample=20,
    debug=False,
):
    tmp_copy, png_dir = None, None
    if "." in ext:
        ext = ext.split(".")[-1]
    if ext == "tiff":
        ext = "tif"

    path = Path(path)
    path.mkdir(exist_ok=True)

    nt, nz, nx, ny = scan.shape_full
    ic(nt, nz, nx, ny)

    left, right, top, bottom = trim_edge
    left = min(left, nx - 1)
    right = min(right, nx - left)
    top = min(top, ny - 1)
    bottom = min(bottom, ny - top)
    ic(left, right, top, bottom)

    new_height = ny - (top + bottom)
    new_width = nx - (left + right)
    ic(new_height, new_width)

    metadata["fov"] = [new_height, new_width]
    metadata["shape"] = (nt, new_width, new_height)
    metadata["dims"] = ["time", "width", "height"]
    metadata["trimmed"] = [left, right, top, bottom]
    metadata["nframes"] = nt
    metadata["num_frames"] = nt  # alias
    metadata["save_path"] = str(path.expanduser().resolve())

    final_shape = (nt, new_height, new_width)
    ic(final_shape)
    writer = _get_file_writer(
        ext, overwrite=overwrite, metadata=metadata, data_shape=final_shape
    )

    chunk_size = target_chunk_mb * 1024 * 1024
    total_chunks = sum(
        min(
            scan.shape[0],
            max(
                1,
                int(
                    np.ceil(
                        scan.shape[0] * scan.shape[2] * scan.shape[3] * 2 / chunk_size
                    )
                ),
            ),
        )
        for _ in planes
    )
    pbar = tqdm(total=total_chunks, desc="Saving plane ", position=0)

    pre_exists = True
    for chan_index in planes:
        pbar.set_description(f"Saving plane {chan_index + 1}")
        if ext == "bin":
            fname = path / f"plane{chan_index}" / "data_raw.bin"
        else:
            fname = path / f"plane_{chan_index + 1:02d}.{ext}"

        if fname.exists() and not overwrite:
            pbar.update(1)
            ic(fname, overwrite)
            break

        pre_exists = False
        if save_phase_png:
            png_dir = path / f"scan_phase_check_plane_{chan_index + 1:02d}"
            png_dir.mkdir(exist_ok=True)
            ic(f"Saving scan-phase images to {png_dir}")

        nbytes_chan = scan.shape[0] * scan.shape[2] * scan.shape[3] * 2
        num_chunks = min(scan.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size))))

        base_frames_per_chunk = scan.shape[0] // num_chunks
        extra_frames = scan.shape[0] % num_chunks

        start = 0
        for chunk in range(num_chunks):
            frames_in_this_chunk = base_frames_per_chunk + (
                1 if chunk < extra_frames else 0
            )
            end = start + frames_in_this_chunk
            data_chunk = scan[
                start:end, chan_index, top : ny - bottom, left : nx - right
            ]

            # if fix_phase:
            #     if save_phase_png:
            #         tmp_copy = data_chunk.copy()
            #     data_chunk = correct_phase_chunk(data_chunk, upsample=upsample)
            #     if save_phase_png:
            #         save_phase_images_png(tmp_copy, data_chunk, png_dir, chan_index)

            writer(fname, data_chunk, chan_index=chan_index)
            start = end
            pbar.update(1)
            if progress_callback is not None:
                progress_callback(pbar.n / pbar.total, current_plane=chan_index + 1)

    pbar.close()

    if pre_exists and not overwrite:
        print("All output files exist; skipping save.")
        return

    if ext in ["tiff", "tif"]:
        close_tiff_writers()
    elif ext == "bin":
        write_ops(metadata, path, planes)


def write_ops(metadata: dict, base_path: str | Path, planes):
    base_path = Path(base_path).expanduser().resolve()
    if "si" in metadata.keys():
        del metadata["si"]

    if isinstance(planes, int):
        planes = [planes]
    for plane_idx in planes:
        plane_dir = Path(base_path) / f"plane{plane_idx}"

        raw_bin = plane_dir.joinpath("data_raw.bin")
        ops_path = plane_dir.joinpath("ops.npy")

        # TODO: This is not an accurate way to get a metadata value that should not have
        #        to be calculated. We use shape to account for the trimmed pixels
        shape = metadata["shape"]
        nt = shape[0]
        Ly = shape[-2]
        Lx = shape[-1]
        dx, dy = metadata.get("pixel_resolution", [2, 2])
        ops = {
            "Ly": Ly,
            "Lx": Lx,
            "fs": np.round(metadata.get("frame_rate"), 2),
            "nframes": nt,
            "raw_file": str(raw_bin.resolve()),
            "reg_file": str(raw_bin.resolve()),
            "dx": dx,
            "dy": dy,
            "metadata": metadata,
        }
        np.save(ops_path, ops)


def _get_file_writer(ext, overwrite, metadata=None, data_shape=None, **kwargs):
    if ext in ["tif", "tiff"]:
        return functools.partial(
            _write_tiff, overwrite=overwrite, metadata=metadata, data_shape=data_shape
        )
    elif ext in ["h5", "hdf5"]:
        return functools.partial(
            _write_h5, overwrite=overwrite, metadata=metadata, data_shape=data_shape
        )
    elif ext == "bin":
        if not HAS_SUITE2P:
            raise ValueError("Suite2p not installed.")
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
            chan_index=kwargs.get("chan_index", None),
            data_shape=data_shape,
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(
    path,
    data,
    *,
    overwrite: bool = False,
    data_shape=None,
    chan_index=None,
):
    if chan_index is None:
        raise ValueError("chan_index must be provided")

    if not hasattr(_write_bin, "_writers"):
        _write_bin._writers, _write_bin._offsets = {}, {}

    fname = Path(path)
    fname.parent.mkdir(exist_ok=True)

    if overwrite and fname.exists():
        fname.unlink()
        _write_bin._writers.pop(str(fname), None)
        _write_bin._offsets.pop(str(fname), None)

    key = str(fname)
    if key not in _write_bin._writers:
        n_frames = data_shape[0] if data_shape else data.shape[0]
        Ly, Lx = data.shape[1], data.shape[2]
        _write_bin._writers[key] = BinaryFile(
            Ly, Lx, key, n_frames=n_frames, dtype=np.int16
        )
        _write_bin._offsets[key] = 0

    bf = _write_bin._writers[key]
    off = _write_bin._offsets[key]
    bf[off : off + data.shape[0]] = data
    bf.file.flush()
    _write_bin._offsets[key] = off + data.shape[0]


def _write_h5(
    path, data, overwrite=True, metadata=None, data_shape=None, chan_index=None
):
    filename = Path(path).with_suffix(".h5")

    if not hasattr(_write_h5, "_initialized"):
        _write_h5._initialized = {}
        _write_h5._offsets = {}

    if filename not in _write_h5._initialized:
        with h5py.File(filename, "w" if overwrite else "a") as f:
            f.create_dataset(
                "mov", shape=data_shape, dtype=data.dtype, chunks=True, compression=None
            )

            if metadata:
                for k, v in metadata.items():
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        f.attrs[k] = str(v)

        _write_h5._initialized[filename] = True
        _write_h5._offsets[filename] = 0

    offset = _write_h5._offsets[filename]
    with h5py.File(filename, "a") as f:
        f["mov"][offset : offset + data.shape[0]] = data

    _write_h5._offsets[filename] += data.shape[0]


def _write_tiff(
    path, data, overwrite=True, metadata=None, data_shape=None, chan_index=None
):
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
            metadata=metadata if is_first else None,
        )
        _write_tiff._first_write[filename] = False


def _write_zarr(
    path, data, overwrite=True, metadata=None, data_shape=None, chan_index=None
):
    try:
        import zarr
    except ImportError:
        raise ImportError("Please install zarr to use ext='.zarr'")

    # data is assumed to have shape (n, H, W)
    filename = Path(path).with_suffix(".zarr")
    if not hasattr(_write_zarr, "_initialized"):
        _write_zarr._initialized = {}

    if filename not in _write_zarr._initialized:
        if filename.exists() and overwrite:
            shutil.rmtree(filename)
        # Instead of using data.shape as the initial shape,
        # start with zero along the appending axis.
        empty_shape = (0,) + data.shape[1:]
        max_shape = (None,) + data.shape[1:]
        z = zarr.creation.create(
            store=str(filename),
            shape=empty_shape,
            chunks=(1,) + data.shape[1:],  # one slice per chunk
            dtype=data.dtype,
            overwrite=True,
            max_shape=max_shape,
        )
        if metadata:
            for k, v in metadata.items():
                try:
                    z.attrs[k] = v
                except TypeError:
                    z.attrs[k] = str(v)
        _write_zarr._initialized[filename] = 0

    # Open the array in append mode
    z = zarr.open_array(str(filename), mode="a")
    # Append new data along the 0th axis
    z.append(data)
    # Update the count (optional, since append grows the array automatically)
    _write_zarr._initialized[filename] = z.shape[0]
