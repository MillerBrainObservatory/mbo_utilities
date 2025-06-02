import re
from collections.abc import Sequence
from itertools import product
from pathlib import Path

from icecream import ic

import dask.array as da
import numpy as np
import tifffile

from . import log
from .metadata import is_raw_scanimage
from .phasecorr import compute_scan_phase_offsets, apply_scan_phase_offsets
from .scanreader import scans, utils
from .scanreader.multiroi import ROI
from .util import subsample_array

# subpixel fft, cross-correlation,
# or first cross-correlation then subpixel
PHASECORR_METHODS = ["subpix", "two_step"]


CHUNKS = {0: 1, 1: "auto", 2: -1, 3: -1}


SAVE_AS_TYPES = [".tiff", ".bin", ".h5"]


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


def read_scan(
        pathnames,
        dtype=np.int16,
        roi=None,
        fix_phase: bool = False,
        phasecorr_method: str = "subpix",
        border: int | tuple[int, int, int, int] = 0,
        upsample: int = 20,
        max_offset: int = 8,
):
    """
    Reads a ScanImage scan from a given file or set of file paths and returns a
    ScanMultiROIReordered object with lazy-loaded data.

    Parameters
    ----------
    pathnames : str, Path, or sequence of str/Path
        A single path to, a wildcard pattern (e.g. ``*.tif``), or a list of paths
        specifying the ScanImage TIFF files to read.
    roi : int, optional
        Specify ROI to export if only exporting a single ROI. 1-based.
        Defaults to None, which exports pre-assembled (tiled) rois.
    fix_phase : bool, optional
        If True, applies phase correction to the scan data. Default is False.
    phasecorr_method : str, optional
        The method to use for phase correction. Options are 'subpix', 'two_step',
    border : int or tuple of int, optional
        The border size to use for phase correction. If an int, applies the same
        border to all sides. If a tuple, specifies (top, bottom, left, right) borders.
    upsample : int, optional
        The for subpixel correction, upsample factor for phase correction.
        A value of 1 clamps to whole-pixel. Default is 10.
    max_offset : int, optional
        The maximum allowed phase offset in pixels. If the computed offset exceeds
        this value, it is clamped to the maximum. Default is 3.
    dtype : numpy.dtype, optional
        The data type to use when reading the scan data. Default is np.int16.

    Returns
    -------
    Scan_MBO
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
    >>> scan = mbo.read_scan(r"D:\\demo\\raw")
    >>> plt.imshow(scan[0, 5, 0, 0], cmap='gray')  # First frame of z-plane 6
    >>> scan = mbo.read_scan(r"D:\\demo\\raw", roi=1) # First ROI
    >>> plt.imshow(scan[0, 5, 0, 0], cmap='gray')  # indexing works the same
    """
    filenames = expand_paths(pathnames)
    if len(filenames) == 0:
        error_msg = f"Pathname(s) {pathnames} do not match any files in disk."
        raise FileNotFoundError(error_msg)
    if not is_raw_scanimage(filenames[0]):
        raise ValueError(
            f"The file {filenames[0]} does not appear to be a raw ScanImage TIFF file."
        )

    scan = Scan_MBO(
        roi=roi,
        fix_phase=fix_phase,
        phasecorr_method=phasecorr_method,
        border=border,
        upsample=upsample,
        max_offset=max_offset,
    )
    scan.read_data(filenames, dtype=dtype)

    return scan


class Scan_MBO(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
    """

    def __init__(
            self,
            roi: int | Sequence[int] | None = None,
            fix_phase: bool = False,
            phasecorr_method: str = "subpix",
            border: int | tuple[int, int, int, int] = 0,
            upsample: int = 20,
            max_offset: int = 8,
    ):
        super().__init__(join_contiguous=True)
        self._selected_roi = roi
        self._fix_phase = fix_phase
        self._phasecorr_method = phasecorr_method
        self.border: int | tuple[int, int, int, int] = border
        self.max_offset: int = max_offset
        self.upsample: int = upsample
        self.pbar = None
        self.show_pbar = False
        self._offset = 0.0

        # Debugging toggles
        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.logger = log.get("scan")
        self.logger.debug(
            f"Initializing MBO Scan with parameters:\n"
            f"roi: {roi}, "
            f"fix_phase: {fix_phase}, "
            f"phasecorr_method: {phasecorr_method}, "
            f"border: {border}, "
            f"upsample: {upsample}, "
            f"max_offset: {max_offset}"
        )
        self.logger.info("MBO Scan initialized.")

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
        """
        Set the phase offset for phase correction.
        If value is a scalar, it applies the same offset to all frames.
        If value is an array, it must match the number of frames.
        """
        if isinstance(value, (int, float)):
            self._offset = float(value)
        elif isinstance(value, np.ndarray):
            self._offset = value
        else:
            raise TypeError("Offset must be a scalar or a 1D numpy array.")

    @property
    def phasecorr_method(self):
        """
        Get the current phase correction method.
        Options are 'two_step', 'subpix', or 'crosscorr'.
        """
        return self._phasecorr_method

    @phasecorr_method.setter
    def phasecorr_method(self, value: str):
        """
        Set the phase correction method.
        Options are 'two_step', 'subpix', or 'crosscorr'.
        """
        if value not in PHASECORR_METHODS:
            raise ValueError(
                f"Unsupported phase correction method: {value}. "
                f"Supported methods are: {PHASECORR_METHODS}"
            )
        if value in ["two_step", "crosscorr"]:
            raise NotImplementedError()
        self._phasecorr_method = value

    @property
    def fix_phase(self):
        """
        Get whether phase correction is applied.
        If True, phase correction is applied to the data.
        """
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool):
        """
        Set whether to apply phase correction.
        If True, phase correction is applied to the data.
        """
        if not isinstance(value, bool):
            raise ValueError("do_phasecorr must be a boolean value.")
        self._fix_phase = value

    @property
    def selected_roi(self):
        """
        Get the current ROI index.
        If roi is None, returns -1 to indicate no specific ROI.
        """
        return self._selected_roi

    @selected_roi.setter
    def selected_roi(self, value):
        """
        Set the current ROI index.
        If value is None, sets roi to -1 to indicate no specific ROI.
        """
        self._selected_roi = value

    def _read_pages(
        self, frames, chans, yslice=slice(None), xslice=slice(None), **kwargs
    ):
        C = self.num_channels
        pages = [f * C + c for f in frames for c in chans]

        H = len(utils.listify_index(yslice, self._page_height))
        W = len(utils.listify_index(xslice, self._page_width))
        buf = np.empty((len(pages), H, W), dtype=self.dtype)

        start = 0
        for tf in self.tiff_files:
            end = start + len(tf.pages)
            idxs = [i for i, p in enumerate(pages) if start <= p < end]
            if idxs:
                frame_idx = [pages[i] - start for i in idxs]
                if self._fix_phase:
                    chunk = tf.asarray(key=frame_idx)[..., yslice, xslice]
                    self.offset = compute_scan_phase_offsets(
                        chunk,
                        upsample=self.upsample,
                        max_offset=self.max_offset,
                        border=self.border,
                    )
                    buf[idxs] = apply_scan_phase_offsets(chunk, self.offset)
                else:
                    buf[idxs] = tf.asarray(key=frame_idx)[..., yslice, xslice]
            start = end
        return buf.reshape(len(frames), len(chans), H, W)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key, z_key, _, _ = tuple(_convert_range_to_slice(k) for k in key) + (
            slice(None),
        ) * (4 - len(key))
        frames = utils.listify_index(t_key, self.num_frames)
        chans = utils.listify_index(z_key, self.num_channels)
        if not frames or not chans:
            return np.empty(0)

        H_out = self.field_heights[0]

        # Return a tuple of all individual ROI slices
        if isinstance(self.selected_roi, list) or self.selected_roi == 0:
            roi_outputs = []
            for roi_idx in range(self.num_rois):
                oxs = self.fields[0].output_xslices[roi_idx]
                oys = self.fields[0].output_yslices[roi_idx]
                xs = self.fields[0].xslices[roi_idx]
                ys = self.fields[0].yslices[roi_idx]

                H_roi = oys.stop - oys.start
                W_roi = oxs.stop - oxs.start
                if W_roi <= 0 or H_roi <= 0:
                    roi_outputs.append(
                        np.empty(
                            (len(frames), len(chans), H_roi, W_roi), dtype=self.dtype
                        )
                    )
                    continue

                data = self._read_pages(frames, chans, yslice=ys, xslice=xs)
                squeeze = []
                if isinstance(t_key, int):
                    squeeze.append(0)
                if isinstance(z_key, int):
                    squeeze.append(1)
                if squeeze:
                    data = data.squeeze(axis=tuple(squeeze))

                roi_outputs.append(data)
            return tuple(roi_outputs)

        elif self.selected_roi is not None and self.selected_roi > 0:
            oxs = self.fields[0].output_xslices[0]
            oys = self.fields[0].output_yslices[self.selected_roi - 1]
            xs = self.fields[0].xslices[self.selected_roi - 1]
            ys = self.fields[0].yslices[self.selected_roi - 1]

            W_out = oxs.stop - oxs.start
            H_out = self.field_heights[0]
            out = np.zeros((len(frames), len(chans), H_out, W_out), dtype=self.dtype)

            data = self._read_pages(frames, chans, yslice=ys, xslice=xs)
            out[:, :, oys, oxs] = data

            squeeze = []
            if isinstance(t_key, int):
                squeeze.append(0)
            if isinstance(z_key, int):
                squeeze.append(1)
            if squeeze:
                out = out.squeeze(axis=tuple(squeeze))

            return out

        else:
            W_out = self.field_widths[0]
            out = np.zeros((len(frames), len(chans), H_out, W_out), dtype=self.dtype)
            for ys, xs, oys, oxs in zip(
                self.fields[0].yslices,
                self.fields[0].xslices,
                self.fields[0].output_yslices,
                self.fields[0].output_xslices,
            ):
                data = self._read_pages(frames, chans, yslice=ys, xslice=xs)
                out[:, :, oys, oxs] = data

            squeeze = []
            if isinstance(t_key, int):
                squeeze.append(0)
            if isinstance(z_key, int):
                squeeze.append(1)
            if squeeze:
                out = out.squeeze(axis=tuple(squeeze))
            return out

    @property
    def total_frames(self):
        return sum(len(tf.pages) // self.num_channels for tf in self.tiff_files)

    def xslices(self):
        return self.fields[0].xslices

    def yslices(self):
        return self.fields[0].yslices

    def output_xslices(self):
        return self.fields[0].output_xslices

    def output_yslices(self):
        return self.fields[0].output_yslices

    @property
    def num_planes(self):
        """LBM alias for num_channels."""
        return self.num_channels

    def min(self):
        """
        Returns the minimum value of the first tiff page.
        """
        # page = self[(0, 0, slice(None), slice(None))]
        page = self.tiff_files[0].pages[0]
        return np.min(page.asarray())

    def max(self):
        """
        Returns the maximum value of the first tiff page.
        """
        page = self.tiff_files[0].pages[0]
        return np.max(page.asarray())

    @property
    def shape(self):
        """Shape is relative to the current ROI."""
        if self.selected_roi is not None:
            if not isinstance(self.selected_roi, (list, tuple)):
                if self.selected_roi > 0:
                    s = self.fields[0].output_xslices[self.selected_roi - 1]
                    width = s.stop - s.start
                    return (
                        self.total_frames,
                        self.num_channels,
                        self.field_heights[0],
                        width,
                    )
        # roi = None, or a list/tuple indicates the shape should be relative to the full FOV
        return (
            self.total_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def shape_full(self):
        return (
            self.total_frames,
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
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        except KeyError:
            raise RuntimeError("This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata.")
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]

        # discard empty/malformed ROIs
        roi_infos = list(filter(lambda r: isinstance(r["zs"], (int, float, list)), roi_infos))

        # LBM uses a single depth that is not stored in metadata,
        # so force this to be 0.
        for roi_info in roi_infos:
            roi_info["zs"] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def __array__(self):
        """
        Convert the scan data to a NumPy array.
        Calculate the size of the scan and subsample to keep under memory limits.
        """
        return subsample_array(self, ignore_dims=[-1, -2, -3])


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
        ic("Max-depth of 0 is not allowed. Setting to 1.")
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


def stack_from_files(files: list, proj="mean"):
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
    >>> z_stack = mbo.stack_from_files(files, proj="max")
    >>> z_stack.shape  # (3, height, width)
    """
    lazy_arrays = []
    for file in files:
        if Path(file).suffix not in [".tif", ".tiff"]:
            continue
        try:
            arr = tifffile.memmap(file)
        except ValueError:
            ic(f"Failed to memmap {file}, trying to read directly.")
            arr = tifffile.imread(file)
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


def _is_arraylike(obj) -> bool:
    """
    Checks if the object is array-like.
    For now just checks if obj has `__getitem__()`
    """
    for attr in ["__getitem__", "shape", "ndim"]:
        if not hasattr(obj, attr):
            return False

    return True


def _get_mbo_project_root() -> Path:
    """Return the root path of the mbo_utilities repository (based on this file)."""
    return Path(__file__).resolve().parent.parent


def _get_mbo_dirs() -> dict:
    """
    Ensure ~/mbo and its subdirectories exist.

    Returns a dict with paths to the root, settings, and cache directories.
    """
    base = Path.home().joinpath("mbo")
    settings = base.joinpath("settings")
    cache = base.joinpath("cache")
    logs = base.joinpath("logs")

    for d in (base, settings, cache, logs):
        d.mkdir(exist_ok=True)

    return {
        "base": base,
        "settings": settings,
        "cache": cache,
        "logs": logs,
    }


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


def _convert_range_to_slice(k):
    return slice(k.start, k.stop, k.step) if isinstance(k, range) else k


def _intersect_slice(user: slice, mask: slice):
    ic(user, mask)
    start = max(user.start or 0, mask.start)
    stop = min(user.stop or mask.stop, mask.stop)
    return slice(start, stop)
