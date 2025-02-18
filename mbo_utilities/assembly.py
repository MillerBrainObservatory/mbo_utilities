import argparse
import functools
import os
import time
import warnings
import logging
from pathlib import Path
import numpy as np
import dask.array as da
from tqdm import tqdm

import tifffile

from mbo_utilities.image import extract_center_square
from mbo_utilities.file_io import  make_json_serializable, read_scan, save_mp4
from mbo_utilities.metadata import get_metadata, is_raw_scanimage
from mbo_utilities.util import norm_minmax
from scanreader.utils import listify_index

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

CHUNKS = {0: 'auto', 1: -1, 2: -1}

# suppress warnings
warnings.filterwarnings("ignore")

print = functools.partial(print, flush=True)


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
        planes=None,
        frames=None,
        metadata=None,
        overwrite=True,
        append_str='',
        ext='.tiff',
        order=None,
        image_size=None,
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
    frames : list or tuple, optional
        Frame indices to save. If `None`, all frames are saved. Default is `None`.
    metadata : dict, optional
        Additional metadata to update the scan object's metadata. Default is `None`.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is `True`.
    append_str : str, optional
        String to append to the file name. Default is `''`.
    ext : str, optional
        File extension for the saved data. Supported options are `'.tiff'` and `'.zarr'`.
        Default is `'.tiff'`.
    order : list or tuple, optional
        A list or tuple specifying the desired order of planes. If provided, the number of
        elements in `order` must match the number of planes. Default is `None`.
    image_size : int, optional
        Size of the image to save. Default is 255x255 pixel image. If the image is larger
        than the movie dimensions, it will be cropped to fit. Expected dimensions are square.

    Raises
    ------
    ValueError
        If an unsupported file extension is provided.

    Notes
    -----
    This function creates the specified directory if it does not already exist.
    Data is saved per channel, organized by planes.
    """

    savedir = Path(savedir)
    if planes is None:
        planes = list(range(scan.num_channels))
    elif not isinstance(planes, (list, tuple)):
        planes = [planes]
    if frames is None:
        frames = list(range(scan.num_frames))
    elif not isinstance(frames, (list, tuple)):
        frames = [frames]

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
            )
        planes = [planes[i] for i in order]
    if not metadata:
        metadata = {'si': scan.tiff_files[0].scanimage_metadata,
                    'image': make_json_serializable(get_metadata(scan.tiff_files[0].filehandle.path))}

    if not savedir.exists():
        logger.debug(f"Creating directory: {savedir}")
        savedir.mkdir(parents=True)
    _save_data(scan, savedir, planes, overwrite, ext, append_str, metadata=metadata, image_size=None)


def _save_data(scan, path, planes, overwrite, file_extension, append_str, metadata, image_size=None):
    start = time.time()
    if '.' in file_extension:
        file_extension = file_extension.split('.')[-1]

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for chan in planes:
        print(f"Saving z-plane {chan + 1}...")

        fname = path.joinpath(f"plane_{chan+1:02d}.{file_extension}")

        if fname.exists():
            fname.unlink()

        z_shape = (scan.shape[0], scan.shape[2], scan.shape[3])
        start_z = time.time()
        tifffile.imwrite(fname, da.zeros(shape=z_shape), metadata=metadata)
        end_z = time.time()
        elapsed_time_z = end_z - start_z
        print(f"Time elapsed to write empty tiff: {int(elapsed_time_z // 60)} minutes {int(elapsed_time_z % 60)} seconds.")
        tif = tifffile.memmap(fname)

        chunk_size = 10 * 1024 * 1024
        nbytes_chan = scan.shape[0] * scan.shape[2] * scan.shape[3] * 2
        num_chunks = min(scan.shape[0], max(1, int(np.ceil(nbytes_chan / chunk_size))))

        base_frames_per_chunk = scan.shape[0] // num_chunks
        extra_frames = scan.shape[0] % num_chunks

        if fname.exists() and not overwrite:
            logger.warning(f'File already exists: {filename}. To overwrite, set overwrite=True (--overwrite in command line)')
            return

        with tqdm(total=num_chunks, desc='Saving chunks', position=0, leave=True) as pbar:
            start = 0
            for i, chunk in enumerate(range(num_chunks)):
                frames_in_this_chunk = base_frames_per_chunk + (1 if chunk < extra_frames else 0)
                end = start + frames_in_this_chunk
                s = scan[start:end, chan, :, :]
                tif[start:end, :, :] = s
                start = end
                pbar.update(1)

    elapsed_time = time.time() - start
    print(f"Time elapsed: {int(elapsed_time // 60)} minutes {int(elapsed_time % 60)} seconds.")


def _get_file_writer(ext, overwrite, metadata=None, image_size=None):
    if ext in ['.tif', '.tiff']:
        return functools.partial(_write_tiff, overwrite=overwrite, metadata=metadata, image_size=image_size)
    elif ext == '.zarr':
        return functools.partial(_write_zarr, overwrite=overwrite, metadata=metadata)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')


def _write_tiff(path, name, data, overwrite=True, metadata=None, image_size=None):
    filename = Path(path / f'{name}.tiff')
    fpath = Path(path) / 'summary_images'
    fpath.mkdir(exist_ok=True, parents=True)
    mean_filename = fpath / f'{name}_mean.tiff'
    movie_filename = fpath / f'{name}.mp4'
    if filename.exists() and not overwrite:
        logger.warning(
            f'File already exists: {filename}. To overwrite, set overwrite=True (--overwrite in command line)')
        return

    ####
    print(f"Writing {filename}")
    t_write = time.time()
    tifffile.imwrite(filename, data, metadata=metadata)

    ####
    print(f"Writing {movie_filename}")
    data = norm_minmax(data)
    if image_size:
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]
    else:
        image_size = 255

    # make sure image_size isnt larger than movie dimensions
    if image_size > data.shape[1]:
        image_size = data.shape[1]

    data = extract_center_square(data, image_size)
    save_mp4(str(movie_filename), data)

    ####
    data = np.mean(data, axis=0)
    print(f"Writing {mean_filename}")
    tifffile.imwrite(mean_filename, data, metadata=metadata)
    t_write_end = time.time() - t_write
    print(f"Data written in {t_write_end:.2f} seconds.")


def _write_zarr(path, name, data, metadata=None, overwrite=True):
    raise NotImplementedError("Zarr writing is not yet implemented.")
    # store = zarr.DirectoryStore(path)
    # root = zarr.group(store, overwrite=overwrite)
    # ds = root.create_dataset(name=name, data=data.squeeze(), overwrite=True)
    # if metadata:
    #     ds.attrs['metadata'] = metadata


def main():
    parser = argparse.ArgumentParser(description="CLI for processing ScanImage tiff files.")
    parser.add_argument("path",
                        type=str,
                        nargs='?',
                        default=None,
                        help="Path to the file or directory to process.")
    parser.add_argument("--frames",
                        type=str,
                        default=":",  # all frames
                        help="Frames to read (0 based). Use slice notation like NumPy arrays ("
                             "e.g., :50 gives frames 0 to 50, 5:15:2 gives frames 5 to 15 in steps of 2)."
                        )
    parser.add_argument("--planes",
                        type=str,
                        default=":",  # all planes
                        help="Planes to read (0 based). Use slice notation like NumPy arrays (e.g., 1:5 gives planes "
                             "2 to 6")
    parser.add_argument("--trimx",
                        type=int,
                        nargs=2,
                        default=(0, 0),
                        help="Number of x-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for left and right "
                             "edges).")
    parser.add_argument("--trimy", type=int, nargs=2, default=(0, 0),
                        help="Number of y-pixels to trim from each ROI. Tuple or list (e.g., 4 4 for top and bottom "
                             "edges).")
    # Boolean Flags
    parser.add_argument("--metadata", action="store_true",
                        help="Print a dictionary of scanimage metadata for files at the given path.")
    parser.add_argument("--roi",
                        action='store_true',
                        help="Save each ROI in its own folder, organized like 'zarr/roi_1/plane_1/, without this "
                             "arguemnet it would save like 'zarr/plane_1/roi_1'."
                        )

    parser.add_argument("--save", type=str, nargs='?', help="Path to save data to. If not provided, the path will be "
                                                            "printed.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing files if saving data..")
    parser.add_argument("--tiff", action='store_false', help="Flag to save as .tiff. Default is True")
    parser.add_argument("--zarr", action='store_true', help="Flag to save as .zarr. Default is False")
    parser.add_argument("--assemble", action='store_true', help="Flag to assemble the each ROI into a single image.")
    parser.add_argument("--debug", action='store_true', help="Output verbose debug information.")
    parser.add_argument("--delete_first_frame", action='store_false', help="Flag to delete the first frame of the "
                                                                           "scan when saving.")
    # Commands
    args = parser.parse_args()

    # If no arguments are provided, print help and exit
    if len(vars(args)) == 0 or not args.path:
        parser.print_help()
        return

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    path = Path(args.path).expanduser()
    if path.is_dir():
        files = [str(x) for x in Path(args.path).expanduser().glob('*.tif*')]
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
        print(f'Found {len(files)} file(s) in {args.path}')

    if args.metadata:
        metadata = get_metadata(files[0])
        print(f"Metadata for {files[0]}:")
        # filter out the verbose scanimage frame/roi metadata
        print_params({k: v for k, v in metadata.items() if k not in ['si', 'roi_info']})

    if args.assemble:
        join_contiguous = True
    else:
        join_contiguous = False

    if args.save:
        savepath = Path(args.save).expanduser()
        logger.info(f"Saving data to {savepath}.")

        t_scan_init = time.time()
        scan = read_scan(files, join_contiguous=join_contiguous, )
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
            ext = '.zarr'
            logger.debug("Saving as .zarr.")
        elif args.tiff:
            ext = '.tiff'
            logger.debug("Saving as .tiff.")
        else:
            raise NotImplementedError("Only .zarr and .tif are supported file formats.")

        t_save = time.time()
        save_as(
            scan,
            savepath,
            frames=frames,
            planes=zplanes,
            overwrite=args.overwrite,
            ext=ext,
        )
        t_save_end = time.time() - t_save
        logger.info(f"--- Processing complete in {t_save_end:.2f} seconds. --")
        return scan
    else:
        print(args.path)


if __name__ == '__main__':
    main()
