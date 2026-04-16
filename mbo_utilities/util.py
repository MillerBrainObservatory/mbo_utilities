import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike


def time_func(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Time a function call, return (result, elapsed_ms).

    uses time.perf_counter() for high-resolution timing.

    Parameters
    ----------
    func : callable
        function to time
    *args, **kwargs
        arguments to pass to func

    Returns
    -------
    tuple
        (result, elapsed_ms) where result is func's return value
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


@dataclass
class TimingStats:
    """Statistics for a set of timing measurements.

    standardized container for benchmark timing results.
    """

    times_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_times(cls, times_ms: list[float]) -> "TimingStats":
        """Compute stats from raw timing list."""
        arr = np.array(times_ms)
        return cls(
            times_ms=times_ms,
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )


def smooth_data(data, window_size=5):
    """
    Smooth 1D data using a moving average filter.

    Applies a moving average (convolution with a uniform window) to smooth the input data array.

    Parameters
    ----------
    data : numpy.ndarray
        Input one-dimensional array to be smoothed.
    window_size : int, optional
        The size of the moving window. The default value is 5.

    Returns
    -------
    numpy.ndarray
        The smoothed array, which is shorter than the input by window_size-1 elements due to
        the valid convolution mode.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7])
    >>> smooth_data(data, window_size=3)
    array([2., 3., 4., 5., 6.])
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def norm_minmax(images):
    """
    Normalize a NumPy array to the [0, 1] range.

    Scales the values in the input array to be between 0 and 1 based on the array's minimum and maximum values.
    This is often used as a preprocessing step before visualization of multi-scale data.

    Parameters
    ----------
    images : numpy.ndarray
       The input array to be normalized.

    Returns
    -------
    numpy.ndarray
       The normalized array with values scaled between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([10, 20, 30])
    >>> norm_minmax(arr)
    array([0. , 0.5, 1. ])
    """
    return (images - images.min()) / (images.max() - images.min())


def is_qt_installed() -> bool:
    """Returns True if PyQt5 is installed, otherwise False."""
    try:
        import PyQt5

        return True
    except ImportError:
        return False


def is_imgui_installed() -> bool:
    """Returns True if imgui_bundle is installed, otherwise False."""
    try:
        import imgui_bundle

        return True
    except ImportError:
        return False


def is_running_jupyter():
    """Returns true if users environment is running Jupyter."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if (
            shell == "ZMQInteractiveShell"
        ):  # are there other aliases for a jupyter shell
            return True  # jupyterlab
        if shell == "TerminalInteractiveShell":
            return False  # ipython from terminal
        return False
    except NameError:
        return False


def subsample_array(
    arr: ArrayLike, max_size: int = 1e6, ignore_dims: Sequence[int] | None = None
):
    r"""
    Subsamples an input array while preserving its relative dimensional proportions.

    The dimensions (shape) of the array can be represented as:

    .. math::

        [d_1, d_2, \\dots d_n]

    The product of the dimensions can be represented as:

    .. math::

        \\prod_{i=1}^{n} d_i

    To find the factor ``f`` by which to divide the size of each dimension in order to
    get max_size ``s`` we must solve for ``f`` in the following expression:

    .. math::

        \\prod_{i=1}^{n} \\frac{d_i}{\\mathbf{f}} = \\mathbf{s}

    The solution for ``f`` is is simply the nth root of the product of the dims divided by the max_size
    where n is the number of dimensions

    .. math::

        \\mathbf{f} = \\sqrt[n]{\\frac{\\prod_{i=1}^{n} d_i}{\\mathbf{s}}}

    Parameters
    ----------
    arr: np.ndarray
        input array of any dimensionality to be subsampled.

    max_size: int, default 1e6
        maximum number of elements in subsampled array

    ignore_dims: Sequence[int], optional
        List of dimension indices to exclude from subsampling (i.e. retain full resolution).
        For example, `ignore_dims=[0]` will avoid subsampling along the first axis.

    Returns
    -------
    np.ndarray
        subsample of the input array
    """
    if np.prod(arr.shape, dtype=np.int64) <= max_size:
        return arr[:]  # no need to subsample if already below the threshold

    shape = np.array(arr.shape, dtype=np.int64)

    # Determine which dimensions to subsample
    if ignore_dims is not None:
        ignore_set = set(ignore_dims)
        subsample_dims = [i for i in range(arr.ndim) if i not in ignore_set]
    else:
        subsample_dims = list(range(arr.ndim))
        ignore_set = set()

    if not subsample_dims:
        # All dims ignored, return full array
        return np.asarray(arr[:])

    # Calculate product of dimensions we're subsampling
    subsample_shape = np.array([shape[i] for i in subsample_dims])
    ignored_shape = np.array([shape[i] for i in range(arr.ndim) if i in ignore_set])
    ignored_product = np.prod(ignored_shape, dtype=np.int64) if len(ignored_shape) > 0 else 1

    # Target size for subsampled dimensions only
    target_subsample_size = max_size / ignored_product
    current_subsample_size = np.prod(subsample_shape, dtype=np.int64)

    if current_subsample_size <= target_subsample_size:
        return np.asarray(arr[:])

    # Calculate factor for subsampled dimensions only
    n_subsample = len(subsample_dims)
    f = np.power(current_subsample_size / target_subsample_size, 1.0 / n_subsample)

    # Build slices - only subsample non-ignored dimensions
    slices = []
    for i in range(arr.ndim):
        if i in ignore_set:
            slices.append(slice(None))
        else:
            new_size = max(1, int(np.floor(shape[i] / f)))
            step = max(1, int(np.floor(shape[i] / new_size)))
            slices.append(slice(None, None, step))

    return np.asarray(arr[tuple(slices)])


def listify_index(index, dim_size):
    """Generates the list representation of an index for the given dim_size.

    Args:
        index: A single index (integer, slice or list/tuple/array of integers).
        dim_size: Size of the dimension corresponding to the index.

    Returns
    -------
        A list of positive integers. List of indices.

    Raises
    ------
        TypeError: If index is not either integer, slice, or array.
    """
    if np.issubdtype(type(index), np.signedinteger):
        index_as_list = [index] if index >= 0 else [dim_size + index]
    elif isinstance(index, (list, tuple, np.ndarray)):
        index_as_list = [x if x >= 0 else (dim_size + x) for x in index]
    elif isinstance(index, slice):
        start, stop, step = index.indices(
            dim_size
        )  # transforms Nones and negative ints to valid slice
        index_as_list = list(range(start, stop, step))
    else:
        error_msg = (
            f"index {index} is not integer, slice or array/list/tuple of integers"
        )
        raise TypeError(error_msg)

    return index_as_list


def index_length(index, dim_size):
    """
    Compute length of index without creating full list.

    much faster than len(listify_index(...)) for slices.
    """
    if np.issubdtype(type(index), np.signedinteger):
        return 1
    if isinstance(index, (list, tuple, np.ndarray)):
        return len(index)
    if isinstance(index, slice):
        start, stop, step = index.indices(dim_size)
        if step > 0:
            return max(0, (stop - start + step - 1) // step)
        return max(0, (start - stop - step - 1) // (-step))
    raise TypeError(f"index {index} is not integer, slice or array")


def load_npy(path):
    """
    Load .npy file across Windows, Mac, and Linux.

    Handles the case where .npy files containing pickled Path objects
    were created on Linux (PosixPath) but opened on Windows, or vice versa.

    Parameters
    ----------
    path : str or Path
        Path to the .npy file.

    Returns
    -------
    np.ndarray
        Loaded array data.

    Examples
    --------
    >>> ops = load_npy("ops.npy")
    >>> if ops.ndim == 0:
    ...     ops = ops.item()  # Convert 0-d array to dict
    """
    import pathlib
    import sys
    import pickle

    # Create a custom unpickler that handles cross-platform Path objects
    class CrossPlatformUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Redirect PosixPath/WindowsPath to the current platform's Path
            if module == "pathlib":
                if name in ("PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"):
                    # Return Path which auto-selects the right type for current platform
                    return pathlib.Path
            return super().find_class(module, name)

    # Try loading with cross-platform unpickler first
    try:
        with open(path, "rb") as f:
            # Read numpy header to get to the pickle data
            version = np.lib.format.read_magic(f)
            _shape, _fortran_order, dtype = np.lib.format._read_array_header(f, version)

            if dtype.hasobject:
                # Contains Python objects - use our custom unpickler
                # Reset file position and let numpy handle the header again
                f.seek(0)
                np.lib.format.read_magic(f)
                np.lib.format._read_array_header(f, version)

                # Now unpickle the data
                return CrossPlatformUnpickler(f).load()
            # No objects, can use standard numpy load
            f.seek(0)
            return np.load(f, allow_pickle=True)
    except Exception:
        # Fallback: try the old patching method
        _original_posix = getattr(pathlib, "PosixPath", None)
        _original_windows = getattr(pathlib, "WindowsPath", None)

        try:
            if sys.platform == "win32":
                pathlib.PosixPath = pathlib.WindowsPath
            else:
                pathlib.WindowsPath = pathlib.PosixPath

            return np.load(path, allow_pickle=True)
        finally:
            if _original_posix is not None:
                pathlib.PosixPath = _original_posix
            if _original_windows is not None:
                pathlib.WindowsPath = _original_windows


def get_dtype(dtype):
    """
    Ensure input is a valid numpy.dtype object.

    This guards against libraries (like Zarr v3) returning string representations ('int16')
    instead of proper dtype objects, which can break downstream code expecting .name attributes.

    Parameters
    ----------
    dtype : str, type, np.dtype, or any valid numpy dtype input
        The data type to convert.

    Returns
    -------
    np.dtype
        The corresponding numpy dtype object.

    Examples
    --------
    >>> safe_dtype('int16').name
    'int16'
    >>> safe_dtype(np.int16).itemsize
    2
    """
    if isinstance(dtype, np.dtype):
        return dtype
    try:
        return np.dtype(dtype)
    except TypeError:
        # Fallback for some weird cases or ensure string conversion first
        return np.dtype(str(dtype))
