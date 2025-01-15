import argparse
import functools
import os
import time
import warnings
from pathlib import Path
import numpy as np
import zarr
from scanreader import read_scan
from scanreader.utils import listify_index
from .lcp_io import get_metadata, make_json_serializable

import tifffile
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

ARRAY_METADATA = ["dtype", "shape", "nbytes", "size"]

CHUNKS = {0: 'auto', 1: -1, 2: -1}

# https://brainglobe.info/documentation/brainglobe-atlasapi/adding-a-new-atlas.html
BRAINGLOBE_STRUCTURE_TEMPLATE = {
    "acronym": "VIS",  # shortened name of the region
    "id": 3,  # region id
    "name": "visual cortex",  # full region name
    "structure_id_path": [1, 2, 3],  # path to the structure in the structures hierarchy, up to current id
    "rgb_triplet": [255, 255, 255],
    # default color for visualizing the region, feel free to leave white or randomize it
}

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


def return_scan_offset(image_in, nvals: int = 8):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

    This function calculates the scan offset correction by analyzing the cross-correlation
    between interleaved lines or columns of the input image. The cross-correlation peak
    determines the amount of offset between the lines or columns, which is then used to
    correct for any misalignment in the imaging process.

    Parameters
    ----------
    image_in : ndarray | ndarray-like
        Input image or volume. It can be 2D, 3D, or 4D.

    .. note::

        Dimensions: [height, width], [time, height, width], or [time, plane, height, width].
        The input array must be castable to numpy. e.g. np.shape, np.ravel.

    nvals : int
        Number of pixel-wise shifts to include in the search for best correlation.

    Returns
    -------
    int
        The computed correction value, based on the peak of the cross-correlation.

    Examples
    --------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> return_scan_offset(img, 1)

    Notes
    -----
    This function assumes that the input image contains interleaved lines or columns that
    need to be analyzed for misalignment. The cross-correlation method is sensitive to
    the similarity in pattern between the interleaved lines or columns. Hence, a strong
    and clear peak in the cross-correlation result indicates a good alignment, and the
    corresponding lag value indicates the amount of misalignment.
    """
    from scipy import signal

    image_in = image_in.squeeze()

    if len(image_in.shape) == 3:
        image_in = np.mean(image_in, axis=0)
    elif len(image_in.shape) == 4:
        image_in = np.mean(np.mean(image_in, axis=0), axis=0)

    n = nvals

    in_pre = image_in[::2, :]
    in_post = image_in[1::2, :]

    min_len = min(in_pre.shape[0], in_post.shape[0])
    in_pre = in_pre[:min_len, :]
    in_post = in_post[:min_len, :]

    buffers = np.zeros((in_pre.shape[0], n))

    in_pre = np.hstack((buffers, in_pre, buffers))
    in_post = np.hstack((buffers, in_post, buffers))

    in_pre = in_pre.T.ravel(order="F")
    in_post = in_post.T.ravel(order="F")

    # Zero-center and clip negative values to zero
    # Iv1 = Iv1 - np.mean(Iv1)
    in_pre[in_pre < 0] = 0

    in_post = in_post - np.mean(in_post)
    in_post[in_post < 0] = 0

    in_pre = in_pre[:, np.newaxis]
    in_post = in_post[:, np.newaxis]

    r_full = signal.correlate(in_pre[:, 0], in_post[:, 0], mode="full", method="auto")
    unbiased_scale = len(in_pre) - np.abs(np.arange(-len(in_pre) + 1, len(in_pre)))
    r = r_full / unbiased_scale

    mid_point = len(r) // 2
    lower_bound = mid_point - n
    upper_bound = mid_point + n + 1
    r = r[lower_bound:upper_bound]
    lags = np.arange(-n, n + 1)

    # Step 3: Find the correction value
    correction_index = np.argmax(r)
    return lags[correction_index]


def fix_scan_phase(
        data_in: np.ndarray,
        offset: int,
):
    """
    Corrects the scan phase of the data based on a given offset along a specified dimension.

    Parameters:
    -----------
    dataIn : ndarray
        The input data of shape (sy, sx, sc, sz).
    offset : int
        The amount of offset to correct for.

    Returns:
    --------
    ndarray
        The data with corrected scan phase, of shape (sy, sx, sc, sz).
    """
    dims = data_in.shape
    ndim = len(dims)
    if ndim == 2:
        sy, sx = data_in.shape
        data_out = np.zeros_like(data_in)

        if offset > 0:
            # Shift even df left and odd df right by 'offset'
            data_out[0::2, :sx - offset] = data_in[0::2, offset:]
            data_out[1::2, offset:] = data_in[1::2, :sx - offset]
        elif offset < 0:
            offset = abs(offset)
            # Shift even df right and odd df left by 'offset'
            data_out[0::2, offset:] = data_in[0::2, :sx - offset]
            data_out[1::2, :sx - offset] = data_in[1::2, offset:]
        else:
            print("Phase = 0, no correction applied.")
            return data_in

        return data_out
    if ndim == 4:
        st, sc, sy, sx = data_in.shape
        if offset != 0:
            data_out = np.zeros((st, sc, sy, sx + abs(offset)))
        else:
            print("Phase = 0, no correction applied.")
            return data_in

        if offset > 0:
            data_out[:, :, 0::2, :sx] = data_in[:, :, 0::2, :]
            data_out[:, :, 1::2, offset: offset + sx] = data_in[:, :, 1::2, :]
            data_out = data_out[:, :, :, : sx + offset]
        elif offset < 0:
            offset = abs(offset)
            data_out[:, :, 0::2, offset: offset + sx] = data_in[:, :, 0::2, :]
            data_out[:, :, 1::2, :sx] = data_in[:, :, 1::2, :]
            data_out = data_out[:, :, :, offset:]

        return data_out

    if ndim == 3:
        st, sy, sx = data_in.shape
        if offset != 0:
            # Create output array with appropriate shape adjustment
            data_out = np.zeros((st, sy, sx + abs(offset)))
        else:
            print("Phase = 0, no correction applied.")
            return data_in

        if offset > 0:
            # For positive offset
            data_out[:, 0::2, :sx] = data_in[:, 0::2, :]
            data_out[:, 1::2, offset: offset + sx] = data_in[:, 1::2, :]
            # Trim output by excluding columns that contain only zeros
            data_out = data_out[:, :, : sx + offset]
        elif offset < 0:
            # For negative offset
            offset = abs(offset)
            data_out[:, 0::2, offset: offset + sx] = data_in[:, 0::2, :]
            data_out[:, 1::2, :sx] = data_in[:, 1::2, :]
            # Trim output by excluding the first 'offset' columns
            data_out = data_out[:, :, offset:]

        return data_out

    raise NotImplementedError()


def save_as(
        scan,
        savedir: os.PathLike,
        planes=None,
        frames=None,
        metadata=None,
        overwrite=True,
        ext='.tiff',
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
    ext : str, optional
        File extension for the saved data. Supported options are `'.tiff'` and `'.zarr'`.
        Default is `'.tiff'`.

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
    elif not isinstance(planes, (list, tuple)):
        frames = [frames]
    if not metadata:
        metadata = {'si': scan.tiff_files[0].scanimage_metadata,
                    'image': make_json_serializable(get_metadata(scan.tiff_files[0].filehandle.path))}

    if not savedir.exists():
        logger.debug(f"Creating directory: {savedir}")
        savedir.mkdir(parents=True)
    _save_data(scan, savedir, planes, frames, overwrite, ext, metadata)


def _save_data(scan, path, planes, frames, overwrite, file_extension, metadata):
    path.mkdir(parents=True, exist_ok=True)
    print(f'Planes: {planes}')

    file_writer = _get_file_writer(file_extension, overwrite, metadata)
    if len(scan.fields) > 1:
        for idx, field in enumerate(scan.fields):
            for chan in planes:
                if 'tif' in file_extension:
                    arr = scan[idx, :, :, chan, frames]  # [y,x,T]
                    logger.debug('arr shape:', arr.shape)
                    file_writer(path, f'plane_{chan + 1}_roi_{idx + 1}', arr.T)
    else:
        for chan in planes:
            if 'tif' in file_extension:
                arr = scan[:, :, :, chan, frames]  # [y,x,T]
                logger.debug('arr shape:', arr.shape)
                file_writer(path, f'plane_{chan + 1}', arr.T)


def _get_file_writer(ext, overwrite, metadata=None):
    if ext in ['.tif', '.tiff']:
        return functools.partial(_write_tiff, overwrite=overwrite, metadata=metadata)
    elif ext == '.zarr':
        return functools.partial(_write_zarr, overwrite=overwrite, metadata=metadata)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')


def _write_tiff(path, name, data, overwrite=True, metadata=None):
    filename = Path(path / f'{name}.tiff')
    if filename.exists() and not overwrite:
        logger.warning(
            f'File already exists: {filename}. To overwrite, set overwrite=True (--overwrite in command line)')
        return
    logger.info(f"Writing {filename}")
    t_write = time.time()
    data = np.transpose(data.squeeze(), (0, 2, 1))
    tifffile.imwrite(filename, data, metadata=metadata)
    t_write_end = time.time() - t_write
    logger.info(f"Data written in {t_write_end:.2f} seconds.")


def _write_zarr(path, name, data, metadata=None, overwrite=True):
    store = zarr.DirectoryStore(path)
    root = zarr.group(store, overwrite=overwrite)
    ds = root.create_dataset(name=name, data=data.squeeze(), overwrite=True)
    if metadata:
        ds.attrs['metadata'] = metadata


def main():
    parser = argparse.ArgumentParser(description="CLI for processing ScanImage tiff files.")
    parser.add_argument("path",
                        type=str,
                        nargs='?',  # Change this to make 'path' optional
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
        t_metadata = time.time()
        metadata = get_metadata(files[0])
        t_metadata_end = time.time() - t_metadata
        print(f"Metadata read in {t_metadata_end:.2f} seconds.")
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

