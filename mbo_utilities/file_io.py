from __future__ import annotations

import os
import re
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import dask.array as da
import ffmpeg
import numpy as np
import tifffile
from matplotlib import cm

from .metadata import is_raw_scanimage
from .scanreader import scans
from .scanreader.multiroi import ROI
from .util import norm_minmax

CHUNKS = {0: 1, 1: "auto", 2: -1, 3: -1}


def zarr_to_dask(zarr_parent):
    """Convert directory of zarr arrays into a Z-stack."""
    # search 3 dirs deep for arrays within our zarr group
    files = get_files(zarr_parent, ".zarray", 3)
    return da.stack([da.from_zarr(Path(x).parent) for x in files], axis=1)


def npy_to_dask(files, name="", axis=1, astype=None):
    """
    Creates a Dask array that lazily stacks multiple .npy files along a specified axis without fully loading them into memory.

    Taken from suite3d for convenience
    https://github.com/alihaydaroglu/suite3d/blob/py310/suite3d/utils.py
    To avoid the unnessessary import. Very nice function, thanks Ali!

    Parameters
    ----------
    files : list of str or Path
        A list of file paths pointing to .npy files containing array data. Each file must have the same shape except
        possibly along the concatenation axis.
    name : str, optional
        A string to be appended to a base name ("from-npy-stack-") to label the resulting Dask array. Default is an empty string.
    axis : int, optional
        The axis along which to stack/concatenate the arrays from the provided files. Default is 1.
    astype : numpy.dtype, optional
        If provided, the resulting Dask array will be cast to this data type. Otherwise, the data type is inferred
        from the first file.

    Returns
    -------
    dask.array.Array

    Examples
    --------
    >>> # https://www.fastplotlib.org/
    >>> import fastplotlib as fpl
    >>> import mbo_utilities as mbo
    >>> files = mbo.get_files("path/to/images/", 'fused', 3) # suite3D output
    >>> arr = npy_to_dask(files, name="stack", axis=1)
    >>> print(arr.shape)
    (nz, nt, ny, nx )
    >>> # Optionally, cast the array to float32
    >>> arr = npy_to_dask(files, axis=1, astype=np.float32)
    >>> fpl.ImageWidget(arr.transpose(1, 0, 2, 3)).show()
    """
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

    dsk = dict(zip(keys, values, strict=False))

    arr = da.Array(dsk, name, chunks, dtype)
    if astype is not None:
        arr = arr.astype(astype)

    return arr


def is_escaped_string(path: str) -> bool:
    return bool(re.search(r"\\[a-zA-Z]", path))


def _make_json_serializable(obj):
    """Convert metadata to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def expand_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """
    Expand a path, list of paths, or wildcard pattern into a sorted list of actual files.

    This is a handy wrapper for loading images or data files when you’ve got a folder,
    some wildcards, or a mix of both.

    Parameters
    ----------
    paths : str, Path, or list of (str or Path)
        Can be a single path, a wildcard pattern like '*.tif', a folder, or a list of those.

    Returns
    -------
    list of Path
        Sorted list of full paths to matching files.

    Examples
    --------
    >>> expand_paths("data/*.tif")
    [Path('data/img_000.tif'), Path('data/img_001.tif'), ...]

    >>> expand_paths(Path("data"))
    [Path('data/img_000.tif'), Path('data/img_001.tif'), ...]

    >>> expand_paths(["data/*.tif", Path("more_data")])
    [Path('data/img_000.tif'), Path('more_data/img_050.tif'), ...]
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(f"Expected str, Path, or sequence of them, got {type(paths)}")

    result = []
    for p in paths:
        p = Path(p)
        if "*" in str(p):
            result.extend(p.parent.glob(p.name))
        elif p.is_dir():
            result.extend(p.glob("*"))
        elif p.exists() and p.is_file():
            result.append(p)

    return sorted(p.resolve() for p in result if p.is_file())


def read_scan(pathnames, dtype=np.int16):
    """
    Reads a ScanImage scan from a given file or set of file paths and returns a
    ScanMultiROIReordered object with lazy-loaded data.

    Parameters
    ----------
    pathnames : str, Path, or sequence of str/Path
        A single path, a wildcard pattern (e.g. ``*.tif``), or a list of paths
        specifying the ScanImage TIFF files to read.
    dtype : numpy.dtype, optional
        The data type to use when reading the scan data. Default is np.int16.

    Returns
    -------
    ScanMultiROIReordered
        A scan object with metadata and lazily loaded data. Raises FileNotFoundError
        if no files match the specified path(s).

    Notes
    -----
    If the provided path string appears to include escaped characters (for example,
    unintentional backslashes), a warning message is printed suggesting the use of a
    raw string (r'...') or double backslashes.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> import matplotlib.pyplot as plt
    >>> scan = mbo.read_scan(r"C:\\path\to\\scan\\*.tif")
    >>> plt.imshow(scan[0, 5, 0, 0], cmap='gray')  # First frame of z-plane 6
    """
    if isinstance(pathnames, str) and is_escaped_string(pathnames):
        print(
            "Detected possible escaped characters in the path."
            " Use a raw string (r'...') or double backslashes."
        )
    filenames = expand_paths(pathnames)
    if len(filenames) == 0:
        error_msg = f"Pathname(s) {pathnames} do not match any files in disk."
        raise FileNotFoundError(error_msg)

    scan = ScanMultiROIReordered(join_contiguous=True)
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
        if item.ndim == 3:
            return np.transpose(item, (2, 0, 1))
        if item.ndim == 4:
            return np.transpose(item, (3, 2, 0, 1))
        raise ValueError(f"Unexpected number of dimensions: {item.ndim}")

    @property
    def shape(self):
        return (
            self.num_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return (
            self.num_frames
            * self.num_channels
            * self.field_heights[0]
            * self.field_widths[0]
        )

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
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"][
                "imagingRoiGroup"
            ]["rois"]
        except KeyError:
            raise RuntimeError(
                "This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata."
            )
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]
        roi_infos = list(
            filter(lambda r: isinstance(r["zs"], (int, float, list)), roi_infos)
        )  # discard empty/malformed ROIs
        for roi_info in roi_infos:
            # LBM uses a single depth that is not stored in metadata, so force this to be 0
            roi_info["zs"] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois


def get_files(
    base_dir, str_contains="", max_depth=1, sort_ascending=True, exclude_dirs=None
) -> list | Path:
    """
    Recursively search for files in a specified directory whose names contain a given substring,
    limiting the search to a maximum subdirectory depth. Optionally, the resulting list of file paths
    is sorted in ascending order using numeric parts of the filenames when available.

    Parameters
    ----------
    base_dir : str or Path
        The base directory where the search begins. This path is expanded (e.g., '~' is resolved)
        and converted to an absolute path.
    str_contains : str, optional
        A substring that must be present in a file's name for it to be included in the result.
        If empty, all files are matched.
    max_depth : int, optional
        The maximum number of subdirectory levels (relative to the base directory) to search.
        Defaults to 1. If set to 0, it is automatically reset to 1.
    sort_ascending : bool, optional
        If True (default), the matched file paths are sorted in ascending alphanumeric order.
        The sort key extracts numeric parts from filenames so that, for example, "file2" comes
        before "file10".
    exclude_dirs : iterable of str or Path, optional
        An iterable of directories to exclude from the resulting list of file paths. By default
        will exclude ".venv/", "__pycache__/", ".git" and ".github"].

    Returns
    -------
    list of str
        A list of full file paths (as strings) for files within the base directory (and its
        subdirectories up to the specified depth) that contain the provided substring.

    Raises
    ------
    FileNotFoundError
        If the base directory does not exist.
    NotADirectoryError
        If the specified base_dir is not a directory.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> # Get all files that contain "ops.npy" in their names by searching up to 3 levels deep:
    >>> ops_files = mbo.get_files("path/to/files", "ops.npy", max_depth=3)
    >>> # Get only files containing "tif" in the current directory (max_depth=1):
    >>> tif_files = mbo.get_files("path/to/files", "tif")
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
    pattern = f"*{str_contains}*" if str_contains else "*"

    if exclude_dirs is None:
        exclude_dirs = [".venv", ".git", "__pycache__"]

    def is_excluded(path):
        return any(excl in path.parts for excl in exclude_dirs)

    files = [
        file
        for file in base_path.rglob(pattern)
        if len(file.parts) - base_depth <= max_depth
        and file.is_file()
        and not is_excluded(file)
    ]

    if sort_ascending:
        import re

        def numerical_sort_key(path):
            match = re.search(r"\d+", path.name)
            return int(match.group()) if match else float("inf")

        files.sort(key=numerical_sort_key)

    return [str(file) for file in files]


def zstack_from_files(files: list, proj="mean"):
    """
    Creates a Z-Stack image by applying a projection to each TIFF file in the provided list and stacking the results into a NumPy array.

    Parameters
    ----------
    files : list of str or Path
        A list of file paths to TIFF images. Files whose extensions are not '.tif' or '.tiff' are ignored.
    proj : str, optional
        The type of projection to apply to each TIFF image. Valid options are 'mean', 'max', and 'std'. Default is 'mean'.

    Returns
    -------
    numpy.ndarray
        A stacked array of projected images with the new dimension corresponding to the file order. For example, for N input files,
        the output shape will be (N, height, width).

    Raises
    ------
    ValueError
        If an unsupported projection type is provided.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> files = mbo.get_files("/path/to/files", "tif")
    >>> z_stack = mbo.zstack_from_files(files, proj="max")
    >>> z_stack.shape  # (3, height, width)
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
    """
    Saves a given image array as a PNG file using Matplotlib.

    Parameters
    ----------
    fname : str or Path
        The file name (or full path) where the PNG image will be saved.
    data : array-like
        The image data to be visualized and saved. Can be any 2D or 3D array that Matplotlib can display.

    Examples
    --------
    >>> import mbo_utilities as mbo
    >>> import tifffile
    >>> data = tifffile.memmap("path/to/plane_0.tiff")
    >>> frame = data[0, ...]
    >>> mbo.save_png("plane_0_frame_1.png", frame)
    """
    # TODO: move this to a separate module that imports matplotlib
    import matplotlib.pyplot as plt

    plt.imshow(data)
    plt.axis("tight")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"Saved data to {fname}")


def save_mp4(
    fname: str | Path | np.ndarray,
    images,
    framerate=60,
    speedup=1,
    chunk_size=100,
    cmap="gray",
    win=7,
    vcodec="libx264",
    normalize=True,
):
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
    Save a video from a 3D NumPy array with a gray colormap and 2x speedup:

    >>> import numpy as np
    >>> images = np.random.rand(100, 600, 576) * 255
    >>> save_mp4('output.mp4', images, framerate=17, cmap='gray', speedup=2)

    Save a video with temporal averaging applied over a 5-frame window at 4x speed:

    >>> save_mp4('output_smoothed.mp4', images, framerate=30, speedup=4, cmap='gray', win=5)

    Save a video from a TIFF stack:

    >>> save_mp4('output.mp4', 'path/to/stack.tiff', framerate=60, cmap='gray')
    """
    if not isinstance(fname, (str, Path)):
        raise TypeError(f"Expected fname to be str or Path, got {type(fname)}")
    if isinstance(images, (str, Path)):
        print(f"Loading TIFF stack from {images}")
        if Path(images).is_file():
            try:
                images = tifffile.memmap(images)
            except MemoryError:
                images = tifffile.imread(images)
        else:
            raise FileNotFoundError(
                f"Images given as a string or path, but not a valid file: {images}"
            )
    elif not isinstance(images, np.ndarray):
        raise ValueError(
            f"Expected images to be a numpy array or a file path, got {type(images)}"
        )

    T, height, width = images.shape
    colormap = cm.get_cmap(cmap)

    if normalize:
        print("Normalizing mp4 images to [0, 1]")
        images = norm_minmax(images)

    if win and win > 1:
        print(f"Applying temporal averaging with window size {win}")
        kernel = np.ones(win) / win
        images = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=images
        )

    print(f"Saving {T} frames to {fname}")
    output_framerate = int(framerate * speedup)
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{width}x{height}",
            framerate=output_framerate,
        )
        .output(str(fname), pix_fmt="yuv420p", vcodec=vcodec, r=output_framerate)
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


def _is_arraylike(obj) -> bool:
    """
    Checks if the object is array-like.
    For now just checks if obj has `__getitem__()`
    """
    for attr in ["__getitem__", "shape", "ndim"]:
        if not hasattr(obj, attr):
            return False

    return True


def to_lazy_array(data_in: os.PathLike | np.ndarray | list[os.PathLike | np.ndarray]):
    """
    Convencience function to resolve various data_in variants into lazy arrays.
    """
    if _is_arraylike(data_in):
        return data_in
    if isinstance(data_in, list):
        if is_raw_scanimage(data_in[0]):
            return read_scan(data_in)
        else:
            return zstack_from_files(data_in)
    if isinstance(data_in, (str, Path)):
        data_in = Path(data_in).expanduser().resolve()
        if data_in.is_file():
            # check suffix
            if data_in.suffix in [".tif", ".tiff"]:
                return tifffile.memmap(data_in)
            elif data_in.suffix == ".npy":
                return np.memmap(data_in)
    else:
        raise TypeError(f"Invalid type {type(data_in)}")
