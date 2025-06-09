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

from . import log
from .file_io import _make_json_serializable, Scan_MBO, write_ops
from .lazy_array import LazyArrayLoader
from .metadata import get_metadata
from .util import is_running_jupyter

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

logger = log.get("assembly")

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
    fix_phase: bool = False,
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
        Whether to fix scan-phase (x/y) alignment. Default is `False`.
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
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True  # send to terminal
    else:
        logger.setLevel(logging.WARNING)
        logger.info("Debug mode disabled; setting log level to WARNING.")
        logger.propagate = False  # don't send to terminal

    # save path
    savedir = Path(savedir)

    if not savedir.parent.is_dir():
        raise ValueError(f"{savedir} is not inside a valid directory.")
    savedir.mkdir(exist_ok=True)

    # handle channels and planes
    if not hasattr(scan, "num_channels"):
        raise ValueError(
            "Unable to determine the number of planes in this recording from 'scan.num_channels'"
        )

    if isinstance(planes, int):
        logger.info(f"Saving only plane {planes}.")
        planes = [planes - 1]
    elif planes is None:  # DON'T use "if not planes", then 0 will be treated as falsy
        logger.info(f"Saving all {scan.num_channels} planes.")
        planes = list(range(scan.num_channels))
    else:
        planes = [p - 1 for p in planes]

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
        logger.info("No metadata provided; using empty dictionary.")
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata must be a dictionary, got {type(metadata)} instead."
        )

    # metadata is now either {} or None, so we can safely update it
    metadata = get_metadata(scan.tiff_files[0].filehandle.path)  # from the file
    logger.info("Using metadata from the first TIFF file in the scan." f" Metadata keys: {list(metadata.keys())}")

    # keep the scanimage metadata under the "si" key
    metadata.update(
        {"si": _make_json_serializable(scan.tiff_files[0].scanimage_metadata)}
    )
    metadata["save_path"] = str(savedir.resolve())

    # which ROIs to save
    if scan.selected_roi is None:
        # None ⇒ full (tiled) stack
        roi_list = [None]
    elif scan.selected_roi == 0:
        # 0 ⇒ split into every individual ROI
        roi_list = list(range(1, scan.num_rois + 1))
    elif isinstance(scan.selected_roi, int):
        # a single 1‐based ROI
        roi_list = [scan.selected_roi]
    else:
        # an explicit list of ROI indices
        roi_list = list(scan.selected_roi)

    logger.info(f"Saving ROIs: {roi_list} (0 means full stack, None means all ROIs).")

    start_time = time.time()

    # this is a bit confusing. If roi=None, that atttribute is set on the scan object and
    # when it is inddexed e.g. scan[0], it will return a stack with each roi assembled.
    if 0 in roi_list:
        if len(roi_list) > 1:
            roi_list = [r + 1 for r in roi_list if r is not None]
    for roi in roi_list:
        logger.info(f"Saving ROI {roi} of {scan.num_rois}.")
        subscan = copy.copy(scan)
        subscan.selected_roi = roi

        target = savedir if roi is None else savedir / f"roi{roi}"
        target.mkdir(exist_ok=True)
        meta = metadata.copy()
        if roi is not None:
            meta["roi"] = roi
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
            debug=debug,
        )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Time elapsed: {int(elapsed_time // 60)} minutes {int(elapsed_time % 60)} seconds."
    )

def save_nonscan(
    data,
    savepath,
    ext,
    metadata=None,
    overwrite=False,
    trim_edge=None,
    target_chunk_mb=20,
):
    if "." in ext:
        ext = ext.split(".")[-1]
    if ext == "tiff":
        ext = "tif"

    array_object = LazyArrayLoader(data)
    data = array_object.load().squeeze()
    fpath = array_object.fpath

    savepath = Path(savepath)
    savepath.mkdir(exist_ok=True)

    if data.ndim == 3:
        nt, ny, nx = data.shape
        use_planes = [0]
    else:
        raise ValueError(f"Unsupported data.ndim={data.ndim}; expected 3.")

    # Trim edges
    left, right, top, bottom = trim_edge or (0, 0, 0, 0)
    left = min(left, nx - 1)
    right = min(right, nx - left)
    top = min(top, ny - 1)
    bottom = min(bottom, ny - top)

    new_height = ny - (top + bottom)
    new_width = nx - (left + right)

    if metadata is None:
        logger.info("No metadata provided; using empty dictionary.")
        metadata = {}

    metadata["fov"] = [new_height, new_width]
    metadata["shape"] = (nt, new_width, new_height)
    metadata["dims"] = ["time", "width", "height"]
    metadata["trimmed"] = [left, right, top, bottom]
    metadata["nframes"] = nt
    metadata["n_frames"] = nt    # alias
    metadata["num_frames"] = nt  # alias

    final_shape = (nt, new_height, new_width)
    logger.info(f"Final shape: {final_shape} (nt, height, width)")
    writer = _get_file_writer(
        ext,
        overwrite=overwrite,
    )

    chunk_size = target_chunk_mb * 1024 * 1024
    total_chunks = sum(
        min(
            nt,
            max(1, int(np.ceil(nt * new_height * new_width * 2 / chunk_size))),
        )
        for _ in use_planes
    )
    logger.info(f"Total chunks to save: {total_chunks} (target chunk size: {chunk_size / 1024 / 1024:.2f} MB)")
    if ext == "bin":
        fname = savepath / "data_raw.bin"
    else:
        if "plane" in metadata:
            logger.info(f"Using 'plane' from metadata: {metadata['plane']}")
            fname = savepath / f"plane{metadata['plane']}.{ext}"
        elif "name" in metadata:
            logger.info(f"Using 'name' from metadata: {metadata['name']}")
        else:
            logger.info("No 'plane' or 'name' in metadata; using default naming.")
            fname = f"data_{final_shape[0]}_{final_shape[1]}_{final_shape[2]}.{ext}"

    metadata["save_path"] = str(fname.expanduser().resolve())

    if fname.exists() and not overwrite:
        print(f"File {fname} exists with overwrite=False; skipping.", flush=True)
        return

    pre_exists = False

    nbytes_chan = nt * new_height * new_width * 2
    num_chunks = min(nt, max(1, int(np.ceil(nbytes_chan / chunk_size))))

    base_frames_per_chunk = nt // num_chunks
    extra_frames = nt % num_chunks

    start = 0
    for chunk in range(num_chunks):
        frames_in_chunk = base_frames_per_chunk + (1 if chunk < extra_frames else 0)
        end = start + frames_in_chunk

        block = data[start:end, top : ny - bottom, left : nx - right]

        logger.info(
            f"Saving chunk {chunk + 1}/{num_chunks}:"
            f" {block.shape} (frames, height, width)"
        )
        writer(fname, block, metadata=metadata)
        start = end

    if pre_exists and not overwrite:
        print("All output files exist; skipping save.")
        return

    if ext in ["tif", "tiff"]:
        close_tiff_writers()
    return fname


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
    if "." in ext:
        ext = ext.split(".")[-1]
    if ext == "tiff":
        ext = "tif"

    if fix_phase:
        scan.fix_phase = True
    if upsample:
        scan.upsample = upsample

    path = Path(path)
    path.mkdir(exist_ok=True)

    nt, nz, nx, ny = scan.shape

    left, right, top, bottom = trim_edge
    left = min(left, nx - 1)
    right = min(right, nx - left)
    top = min(top, ny - 1)
    bottom = min(bottom, ny - top)

    new_height = ny - (top + bottom)
    new_width = nx - (left + right)

    metadata["fov"] = [new_height, new_width]
    metadata["shape"] = (nt, new_width, new_height)
    metadata["dims"] = ["time", "width", "height"]
    metadata["trimmed"] = [left, right, top, bottom]
    metadata["nframes"] = nt
    metadata["n_frames"] = nt    # alias
    metadata["num_frames"] = nt  # alias

    final_shape = (nt, new_height, new_width)
    logger.info(f"Final shape: {final_shape} (nt, height, width)")
    writer = _get_file_writer(
        ext, overwrite=overwrite
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
    logger.info(f"Total chunks to save: {total_chunks} (target chunk size: {chunk_size / 1024 / 1024:.2f} MB)")
    if not debug:
        pbar = tqdm(total=total_chunks, desc="Saving plane ", position=0)
    else:
        pbar=None

    pre_exists = True
    for chan_index in planes:
        pre_exists = False

        if ext == "bin":
            fname = path / f"plane{chan_index + 1}" / "data_raw.bin"
        else:
            fname = path / f"plane{chan_index + 1}.{ext}"

        if fname.exists() and not overwrite:
            print(f"File {fname} already exists with overwrite=False; skipping save.", flush=True)
            if pbar:
                pbar.update(1)
            break

        if pbar:
            pbar.set_description(f"Saving plane {chan_index + 1}")

        if save_phase_png:
            png_dir = path / f"scan_phase_check_plane_{chan_index + 1:02d}"
            # png_dir.mkdir(exist_ok=True)

        metadata_plane = metadata.copy()

        metadata["save_path"] = str(fname.parent.expanduser().resolve())
        metadata_plane["plane"] = chan_index + 1 # 1-based indexing
        metadata_plane["plane_index"] = chan_index

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
            logger.info(
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

    if pre_exists and not overwrite:
        print("All output files exist; skipping save.")
        return

    if ext in ["tiff", "tif"]:
        close_tiff_writers()


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
    elif ext == "bin":
        if not HAS_SUITE2P:
            raise ValueError("Suite2p not installed.")
        return functools.partial(
            _write_bin,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_bin(
    path,
    data,
    overwrite: bool = False,
    metadata=None,
):
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

    logger.info(f"Wrote {data.shape[0]} frames to {fname}.")


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


def _write_tiff(
    path, data, overwrite=True, metadata=None
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
            metadata=_make_json_serializable(metadata) if is_first else None,
        )
        _write_tiff._first_write[filename] = False


def _write_zarr(
    path, data, overwrite=True, metadata=None
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
        z = zarr.create(
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
