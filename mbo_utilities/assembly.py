import argparse
import functools
import os
import time
import warnings
import logging
import numpy as np

import shutil
from pathlib import Path
from tifffile import TiffWriter
import h5py

import mbo_utilities
from .file_io import _make_json_serializable, read_scan
from .metadata import get_metadata
from .util import is_running_jupyter
from .scanreader.utils import listify_index

if is_running_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

CHUNKS = {0: "auto", 1: -1, 2: -1}

warnings.filterwarnings("ignore")

print = functools.partial(print, flush=True)


def close_tiff_writers():
    if hasattr(_write_tiff, "_writers"):
        for writer in _write_tiff._writers.values():
            writer.close()
        _write_tiff._writers.clear()


def process_slice_str(slice_str):
    if not isinstance(slice_str, str):
        raise ValueError(f"Expected a string argument, received: {slice_str}")
    if slice_str.isdigit():
        return int(slice_str)
    else:
        parts = slice_str.split(":")
    return slice(*[int(p) if p else None for p in parts])


def process_slice_objects(slice_str):
    return tuple(map(process_slice_str, slice_str.split(",")))


def print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")


def save_as(
    scan,
    savedir: os.PathLike,
    planes: list | tuple = None,
    metadata: dict = None,
    overwrite: bool = True,
    append_str: str = "",
    ext: str = ".tiff",
    order: list | tuple = None,
    trim_edge: list | tuple = (0, 0, 0, 0),
    fix_phase: bool = True,
    target_chunk_mb: int = 20,
):
    """
    Save scan data to the specified directory in the desired format.

    Parameters
    ----------
    scan : scanreader.ScanMultiROI
        An object representing scan data. Must have attributes such as `num_channels`,
        `num_frames`, `fields`, and `rois`, and support indexing for retrieving frame data.
    savedir : os.PathLike
        Path to the directory where the data will be saved.
    planes : int, list, or tuple, optional
        Plane indices to save. If `None`, all planes are saved. Default is `None`.
    trim_edge : list, optional
        Number of pixels to trim on each W x H edge. (Left, Right, Top, Bottom). Default is (0,0,0,0).
    metadata : dict, optional
        Additional metadata to update the scan object's metadata. Default is `None`.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is `True`.
    append_str : str, optional
        String to append to the file name. Default is `''`.
    ext : str, optional
        File extension for the saved data. Supported options are .tiff, .zarr and .h5.
        Default is `'.tiff'`.
    order : list or tuple, optional
        A list or tuple specifying the desired order of planes. If provided, the number of
        elements in `order` must match the number of planes. Default is `None`.
    fix_phase : bool, optional
        Whether to fix scan-phase (x/y) alignment. Default is `True`.
    target_chunk_mb : int, optional
        Chunk size in megabytes for saving data. Increase to help with scan-phase correction.

    Raises
    ------
    ValueError
        If an unsupported file extension is provided.
    """

    savedir = Path(savedir)
    if not savedir.parent.is_dir():
        raise ValueError(f"{savedir} is not inside a valid directory.")
    savedir.mkdir(exist_ok=True)

    if planes is None:
        planes = list(range(scan.num_channels))
    elif not isinstance(planes, (list, tuple)):
        planes = [planes]

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
            )
        planes = [planes[i] for i in order]

    mdata = {
        "si": _make_json_serializable(scan.tiff_files[0].scanimage_metadata),
        "image": _make_json_serializable(
            get_metadata(scan.tiff_files[0].filehandle.path)
        ),
    }

    if metadata is not None:
        mdata.update(metadata)

    if not savedir.exists():
        logger.debug(f"Creating directory: {savedir}")
        savedir.mkdir(parents=True)
    start_time = time.time()
    _save_data(
        scan,
        savedir,
        planes,
        overwrite,
        ext,
        append_str,
        metadata=mdata,
        trim_edge=trim_edge,
        fix_phase=fix_phase,
        target_chunk_mb=target_chunk_mb,
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
    file_extension,
    append_str,
    metadata,
    trim_edge=None,
    fix_phase=False,
    target_chunk_mb=20,
):
    if "." in file_extension:
        file_extension = file_extension.split(".")[-1]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

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

    final_shape = (nt, new_height, new_width)
    writer = _get_file_writer(
        file_extension, overwrite=overwrite, metadata=metadata, data_shape=final_shape
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
    pbar = tqdm(total=total_chunks, desc="Saving planes")

    for chan_index in planes:
        if append_str:
            fname = path / f"plane_{chan_index + 1:02d}_{append_str}.{file_extension}"
        else:
            fname = path / f"plane_{chan_index + 1:02d}.{file_extension}"

        if fname.exists() and not overwrite:
            logger.warning(f"File already exists: {fname}")
            continue

        nbytes_chan = scan.shape[0] * scan.shape[2] * scan.shape[3] * 2
        num_chunks = min(scan.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size))))
        # nbytes_chan = scan.shape[0] * scan.shape[2] * scan.shape[3] * 2
        # num_chunks = min(scan.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size))))

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

            if fix_phase:
                ofs = mbo_utilities.return_scan_offset(data_chunk)
                if ofs:
                    data_chunk = mbo_utilities.fix_scan_phase(data_chunk, -ofs)

            writer(fname, data_chunk)
            start = end
            pbar.update(1)

    pbar.close()

    if file_extension in ["tiff", ".tiff", "tif", ".tif"]:
        close_tiff_writers()


def _get_file_writer(ext, overwrite, metadata=None, data_shape=None):
    if ext in [".tif", ".tiff", "tif", "tiff"]:
        return functools.partial(
            _write_tiff, overwrite=overwrite, metadata=metadata, data_shape=data_shape
        )
    elif ext in [".zarr", "zarr"]:
        return functools.partial(_write_zarr, overwrite=overwrite, metadata=metadata)
    elif ext in [".h5", "h5", "hdf5", ".hdf5"]:
        return functools.partial(
            _write_h5, overwrite=overwrite, metadata=metadata, data_shape=data_shape
        )
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _write_h5(path, data, overwrite=True, metadata=None, data_shape=None):
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


def _write_tiff(path, data, overwrite=True, metadata=None, data_shape=None):
    filename = Path(path).with_suffix(".tif")

    if not hasattr(_write_tiff, "_writers"):
        _write_tiff._writers = {}

    if filename not in _write_tiff._writers:
        if filename.exists() and overwrite:
            filename.unlink()
        _write_tiff._writers[filename] = TiffWriter(filename, bigtiff=True)
        _write_tiff._first_write = True
    else:
        _write_tiff._first_write = False

    writer = _write_tiff._writers[filename]

    for frame in data:
        writer.write(
            frame,
            contiguous=True,
            photometric="minisblack",
            metadata=metadata if _write_tiff._first_write else None,
        )
        _write_tiff._first_write = False


def _write_zarr(path, data, overwrite=True, metadata=None, single_file=False):
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


def main():
    parser = argparse.ArgumentParser(
        description="CLI for processing ScanImage tiff files."
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the file or directory to process.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=":",  # all frames
        help="Frames to read (0 based). Use slice notation like NumPy arrays ("
        "e.g., :50 gives frames 0 to 50, 5:15:2 gives frames 5 to 15 in steps of 2).",
    )
    parser.add_argument(
        "--planes",
        type=str,
        default=":",  # all planes
        help="Planes to read (0 based). Use slice notation like NumPy arrays (e.g., 1:5 gives planes "
        "2 to 6",
    )
    parser.add_argument(
        "--target_chunk_mb",
        type=int,
        nargs=1,
        default=20,
        help="Target chunk size, in MB",
    )
    parser.add_argument(
        "--trimx",
        type=int,
        nargs=2,
        default=(0, 0),
        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right "
        "edges).",
    )
    parser.add_argument(
        "--trimy",
        type=int,
        nargs=2,
        default=(0, 0),
        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom "
        "edges).",
    )
    # Boolean Flags
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Print a dictionary of scanimage metadata for files at the given path.",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        help="Save each ROI in its own folder, organized like 'zarr/roi_1/plane_1/, without this "
        "arguemnet it would save like 'zarr/plane_1/roi_1'.",
    )

    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        help="Path to save data to. If not provided, the path will be printed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if saving data..",
    )
    parser.add_argument(
        "--tiff", action="store_false", help="Flag to save as .tiff. Default is True"
    )
    parser.add_argument(
        "--zarr", action="store_true", help="Flag to save as .zarr. Default is False"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Output verbose debug information."
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Include min, max, mean, std in metadata per plane.",
    )
    parser.add_argument(
        "--delete_first_frame",
        action="store_false",
        help="Flag to delete the first frame of the scan when saving.",
    )
    # Commands
    args = parser.parse_args()

    # If no arguments are provided, print help and exit
    if len(vars(args)) == 0 or not args.path:
        parser.print_help()
        return None

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    path = Path(args.path).expanduser()
    if path.is_dir():
        files = [str(x) for x in Path(args.path).expanduser().glob("*.tif*")]
    elif path.is_file():
        files = [str(path)]
    else:
        raise FileNotFoundError(f"File or directory not found: {args.path}")

    if len(files) < 1:
        raise ValueError(
            f"Input path given is a non-tiff file: {args.path}.\n"
            f"scanreader is currently limited to scanimage .tiff files."
        )
    else:
        print(f"Found {len(files)} file(s) in {args.path}")

    if args.metadata:
        metadata = get_metadata(files[0])
        print(f"Metadata for {files[0]}:")
        # filter out the verbose scanimage frame/roi metadata
        print_params({k: v for k, v in metadata.items() if k not in ["si", "roi_info"]})

    if args.save:
        savepath = Path(args.save).expanduser()
        logger.info(f"Saving data to {savepath}.")

        t_scan_init = time.time()
        scan = read_scan(files)
        t_scan_init_end = time.time() - t_scan_init
        logger.info(f"--- Scan initialized in {t_scan_init_end:.2f} seconds.")

        frames = listify_index(process_slice_str(args.frames), scan.num_frames)
        zplanes = listify_index(process_slice_str(args.planes), scan.num_channels)

        if args.delete_first_frame:
            frames = frames[1:]
            logger.debug(f"Deleting first frame. New frames: {frames}")

        logger.debug(f"Frames: {len(frames)}")
        logger.debug(f"Z-Planes: {len(zplanes)}")

        if args.zarr:
            ext = ".zarr"
            logger.debug("Saving as .zarr.")
        elif args.tiff:
            ext = ".tiff"
            logger.debug("Saving as .tiff.")
        else:
            raise NotImplementedError("Only .zarr and .tif are supported file formats.")

        t_save = time.time()
        save_as(
            scan,
            savepath,
            # frames=frames, # TODO
            planes=zplanes,
            overwrite=args.overwrite,
            ext=ext,
            target_chunk_mb=args.target_chunk_mb,
        )
        t_save_end = time.time() - t_save
        logger.info(f"--- Processing complete in {t_save_end:.2f} seconds. --")
        return scan
    else:
        print(args.path)


if __name__ == "__main__":
    main()
