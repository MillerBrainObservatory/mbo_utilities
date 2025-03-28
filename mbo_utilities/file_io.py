from __future__ import annotations

from itertools import product

import numpy as np
import tifffile
from pathlib import Path
import ffmpeg
import re
import dask.array as da

from matplotlib import cm

from mbo_utilities.scanreader import scans
from mbo_utilities.scanreader.core import expand_wildcard
from mbo_utilities.scanreader.multiroi import ROI
from mbo_utilities.util import norm_minmax


def npy_to_dask(files, name="", axis=1, astype=None):
    sample_mov = np.load(files[0], mmap_mode="r")
    file_ts = [np.load(f, mmap_mode="r").shape[axis] for f in files]
    nz, nt_sample, ny, nx = sample_mov.shape

    dtype = sample_mov.dtype
    chunks = [(nz,), (nt_sample,), (ny,), (nx,)]
    chunks[axis] = tuple(file_ts)
    chunks = tuple(chunks)
    name = "from-npy-stack-%s" % name

    keys = list(product([name], *[range(len(c)) for c in chunks]))
    values = [(np.load, files[i], "r") for i in range(len(chunks[axis]))]

    dsk = dict(zip(keys, values))

    arr = da.Array(dsk, name, chunks, dtype)
    if astype is not None:
        arr = arr.astype(astype)

    return arr


def _make_json_serializable(obj):
    """Convert metadata to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def read_scan(pathnames, dtype=np.int16, join_contiguous=False):
    """ Reads a ScanImage scan. """
    # Expand wildcards
    filenames = expand_wildcard(pathnames)
    if len(filenames) == 0:
        error_msg = 'Pathname(s) {} do not match any files in disk.'.format(pathnames)
        raise FileNotFoundError(error_msg)

    scan = ScanMultiROIReordered(join_contiguous=join_contiguous)

    # Read metadata and data (lazy operation)
    scan.read_data(filenames, dtype=dtype)

    return scan


class ScanMultiROIReordered(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
    """

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = tuple(list(k) if isinstance(k, range) else k for k in key)

        # Call the parent class's __getitem__ with the reordered key
        item = super().__getitem__((0, key[2], key[3], key[1], key[0]))
        if item.ndim == 2:
            return item
        elif item.ndim == 3:
            return np.transpose(item, (2, 0, 1))
        elif item.ndim == 4:
            return np.transpose(item, (3, 2, 0, 1))
        else:
            raise ValueError(f"Unexpected number of dimensions: {item.ndim}")

    @property
    def shape(self):
        return self.num_frames, self.num_channels, self.field_heights[0], self.field_widths[0]

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return self.num_frames * self.num_channels * self.field_heights[0] * self.field_widths[0]

    @property
    def scanning_depths(self):
        """
        We override this because LBM should always be at a single scanning depth.
        """
        return [0]


    def _create_rois(self):
        """
        Create scan rois from the configuration file. Override the base method to force
        ROI's that have multiple 'zs' to a single depth.
        """
        try:
            roi_infos = self.tiff_files[0].scanimage_metadata['RoiGroups']['imagingRoiGroup']['rois']
        except KeyError:
            raise RuntimeError('This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata.')
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]
        roi_infos = list(filter(lambda r: isinstance(r['zs'], (int, float, list)), roi_infos))  # discard empty/malformed ROIs
        for roi_info in roi_infos:
            # LBM uses a single depth that is not stored in metadata, so force this to be 0
            roi_info['zs'] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois


def get_files(base_dir, str_contains="", max_depth=1, sort_ascending=True) -> list | Path:
    """
    Recursively searches for files with a specific extension up to a given depth and stores their paths in a pickle file.

    Parameters
    ----------
    base_dir : str or Path
        The base directory to start searching.
    str_contains : str
        The string that the file names should contain.
    max_depth : int
        The maximum depth of subdirectories to search.
    sort_ascending : bool, optional
        Whether to sort files alphanumerically by filename, with digits in ascending order (i.e. 1, 2, 10) (default is False).

    Returns
    -------
    list
        A list of full file paths matching the given extension.
    """
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Directory '{base_path}' does not exist.")
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{base_path}' is not a directory.")
    if max_depth == 0:
        print("Max-depth of 0 is not allowed. Setting to 1.")
        max_depth = 1

    base_depth = len(base_path.parts)
    pattern = f'*{str_contains}*' if str_contains else '*'

    files = [
        file for file in base_path.rglob(pattern)
        if len(file.parts) - base_depth <= max_depth
           and file.is_file()
    ]

    if sort_ascending:
        def numerical_sort_key(path):
            match = re.search(r'\d+', path.name)
            return int(match.group()) if match else float('inf')

        files.sort(key=numerical_sort_key)

    return [str(file) for file in files]


def stack_from_files(files: list, proj="mean"):
    """Stacks a list of TIFF files into a Dask array. Can be 3D Tyx or 4D Tzyx.

    Parameters
    ----------
    files : list
        List of TIFF files to stack.
    proj : str, optional
        Projection to use (mean, max, std). Default is 'mean'.

    Returns
    -------
    dask.array.core.Array
        Dask array of the stacked files.

    """
    lazy_arrays = []
    for file in files:
        if Path(file).suffix not in [".tif", ".tiff"]:
            continue
        arr = tifffile.memmap(file)
        if proj == "mean":
            img = np.mean(arr, axis=0)
        elif proj == "max":
            img = np.max(arr, axis=0)
        elif proj == "std":
            img = np.std(arr, axis=0)
        else:
            raise ValueError(f"Unsupported projection '{proj}'")
        lazy_arrays.append(img)

    return np.stack(lazy_arrays, axis=0)


def save_png(fname, data):
    # TODO: move this to a separate module that imports matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(data)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved data to {fname}")


def save_mp4(fname: str | Path | np.ndarray, images, framerate=60, speedup=1, chunk_size=100, cmap="gray", win=7,
             vcodec='libx264', normalize=True):
    """
    Save a video from a 3D array or TIFF stack to `.mp4`.

    Parameters
    ----------
    fname : str
        Output video file name.
    images : numpy.ndarray or str
        Input 3D array (T x H x W) or a file path to a TIFF stack.
    framerate : int, optional
        Original framerate of the video, by default 60.
    speedup : int, optional
        Factor to increase the playback speed, by default 1 (no speedup).
    chunk_size : int, optional
        Number of frames to process and write in a single chunk, by default 100.
    cmap : str, optional
        Colormap to apply to the video frames, by default "gray".
        Must be a valid Matplotlib colormap name.
    win : int, optional
        Temporal averaging window size. If `win > 1`, frames are averaged over
        the specified window using convolution. By default, 7.
    vcodec : str, optional
        Video codec to use, by default 'libx264'.
    normalize : bool, optional
        Flag to min-max normalize the video frames, by default True.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist when `images` is provided as a file path.
    ValueError
        If `images` is not a valid 3D NumPy array or a file path to a TIFF stack.

    Notes
    -----
    - The input array `images` must have the shape (T, H, W), where T is the number of frames,
      H is the height, and W is the width.
    - The `win` parameter performs temporal smoothing by averaging over adjacent frames.

    Examples
    --------
    Save a video from a 3D NumPy array with a colormap and speedup:

    >>> import numpy as np
    >>> images = np.random.rand(100, 600, 576) * 255
    >>> save_mp4('output.mp4', images, framerate=30, cmap='viridis', speedup=2)

    Save a video with temporal averaging applied over a 5-frame window at 4x speed:

    >>> save_mp4('output_smoothed.mp4', images, framerate=30, speedup=4, cmap='gray', win=5)

    Save a video from a TIFF stack:

    >>> save_mp4('output.mp4', 'path/to/stack.tiff', framerate=60, cmap='gray')
    """
    if isinstance(images, (str, Path)):
        print(f"Loading TIFF stack from {images}")
        if Path(images).is_file():
            images = tifffile.memmap(images)
        else:
            raise FileNotFoundError(f"File not found: {images}")

    T, height, width = images.shape
    colormap = cm.get_cmap(cmap)

    if normalize:
        print("Normalizing mp4 images to [0, 1]")
        images = norm_minmax(images)

    if win and win > 1:
        print(f"Applying temporal averaging with window size {win}")
        kernel = np.ones(win) / win
        images = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=images)

    print(f"Saving {T} frames to {fname}")
    output_framerate = int(framerate * speedup)
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', framerate=output_framerate)
        .output(str(fname), pix_fmt='yuv420p', vcodec=vcodec, r=output_framerate)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = images[start:end]
        colored_chunk = (colormap(chunk)[:, :, :, :3] * 255).astype(np.uint8)
        for frame in colored_chunk:
            process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    print(f"Video saved to {fname}")
