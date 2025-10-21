from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

def check():
    import importlib, pandas as pd

    rows = []

    try:
        torch = importlib.import_module("torch")
        t_ver = getattr(torch, "__version__", "") or ""
        rows.append({"component": "torch-cpu", "ok": True, "version": t_ver, "details": "import ok"})
        try:
            cuda_ver = getattr(getattr(torch, "version", None), "cuda", "") or ""
            cuda_avail = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
            dev_count = getattr(getattr(torch, "cuda", None), "device_count", lambda: 0)() if cuda_avail else 0
            dev_name = ""
            if cuda_avail and dev_count > 0:
                try:
                    dev_name = torch.cuda.get_device_name(0)
                except Exception:
                    dev_name = "GPU detected"
            rows.append(
                {
                    "component": "torch-gpu",
                    "ok": cuda_avail,
                    "version": cuda_ver,
                    "details": f"devices={dev_count}; name={dev_name}",
                }
            )
        except Exception as e:
            rows.append({"component": "torch-gpu", "ok": False, "version": "", "details": f"check error: {e}"})
    except Exception as e:
        rows.append({"component": "torch-cpu", "ok": False, "version": "", "details": str(e)})
        rows.append({"component": "torch-gpu", "ok": False, "version": "", "details": "torch not available"})

    try:
        cupy = importlib.import_module("cupy")
        cu_ver = getattr(cupy, "__version__", "") or ""
        rows.append({"component": "cupy-cpu", "ok": True, "version": cu_ver, "details": "import ok"})
        try:
            get_dc = getattr(getattr(cupy, "cuda", None), "runtime", None)
            get_dc = getattr(get_dc, "getDeviceCount", None)
            n = int(get_dc()) if callable(get_dc) else 0
            rows.append(
                {
                    "component": "cupy-gpu",
                    "ok": n > 0,
                    "version": cu_ver,
                    "details": f"devices={n}",
                }
            )
        except Exception as e:
            rows.append({"component": "cupy-gpu", "ok": False, "version": cu_ver, "details": f"device check error: {e}"})
    except Exception as e:
        rows.append({"component": "cupy-cpu", "ok": False, "version": "", "details": str(e)})
        rows.append({"component": "cupy-gpu", "ok": False, "version": "", "details": "cupy not available"})

    df = pd.DataFrame(rows, columns=["component", "ok", "version", "details"])
    df = df.sort_values("component").reset_index(drop=True)
    return df


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


def norm_percentile(image, low_p=1, high_p=98):
    """
    Normalize an image based on percentile contrast stretching.

    Computes the low and high percentile (e.g., 1st and 98th percentiles) of the pixel
    values, and scales the image so that those percentiles map to 0 and 1 respectively.
    Values outside the range are clipped, improving contrast especially when the data contain outliers.

    Parameters
    ----------
    image : numpy.ndarray
       The input image array to be normalized.
    low_p : float, optional
       The lower percentile for normalization (default is 1).
    high_p : float, optional
       The upper percentile for normalization (default is 98).

    Returns
    -------
    numpy.ndarray
       The normalized image as a float array, with values in the range [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> image = np.array([0, 50, 100, 150, 200, 250])
    >>> norm_percentile(image, low_p=10, high_p=90)
    array([0.  , 0.  , 0.25, 0.75, 1.  , 1.  ])
    """
    p_low, p_high = np.percentile(image, (low_p, high_p))
    return np.clip((image - p_low) / (p_high - p_low), 0, 1)


def match_array_size(arr1, arr2, mode="trim"):
    """
    Adjust two arrays to a common shape by trimming or padding.

    This function accepts two NumPy arrays and modifies them so that both have the same shape.
    In "trim" mode, the arrays are cropped to the smallest common dimensions.
    In "pad" mode, each array is padded with zeros to match the largest dimensions.
    The resulting arrays are stacked along a new first axis.

    Parameters
    ----------
    arr1 : numpy.ndarray
        The first input array.
    arr2 : numpy.ndarray
        The second input array.
    mode : str, optional
        The method to use for resizing the arrays. Options are:
          - "trim": Crop the arrays to the smallest common size (default).
          - "pad": Pad the arrays with zeros to the largest common size.

    Returns
    -------
    numpy.ndarray
        A stacked array of shape (2, ...) containing the resized versions of arr1 and arr2.

    Raises
    ------
    ValueError
        If an invalid mode is provided (i.e., not "trim" or "pad").

    Examples
    --------
    >>> import numpy as np
    >>> arr1 = np.random.rand(5, 7)
    >>> arr2 = np.random.rand(6, 5)
    >>> stacked = match_array_size(arr1, arr2, mode="trim")
    >>> stacked.shape
    (2, 5, 5)
    >>> stacked = match_array_size(arr1, arr2, mode="pad")
    >>> stacked.shape
    (2, 6, 7)
    """
    shape1 = np.array(arr1.shape)
    shape2 = np.array(arr2.shape)

    if mode == "trim":
        min_shape = np.minimum(shape1, shape2)
        arr1 = arr1[tuple(slice(0, s) for s in min_shape)]
        arr2 = arr2[tuple(slice(0, s) for s in min_shape)]

    elif mode == "pad":
        max_shape = np.maximum(shape1, shape2)
        padded1 = np.zeros(max_shape, dtype=arr1.dtype)
        padded2 = np.zeros(max_shape, dtype=arr2.dtype)
        slices1 = tuple(slice(0, s) for s in shape1)
        slices2 = tuple(slice(0, s) for s in shape2)
        padded1[slices1] = arr1
        padded2[slices2] = arr2
        arr1, arr2 = padded1, padded2
    else:
        raise ValueError("Invalid mode. Use 'trim' or 'pad'.")
    return np.stack([arr1, arr2], axis=0)


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
    """
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

    # get factor by which to divide all dims
    f = np.power((np.prod(arr.shape, dtype=np.int64) / max_size), 1.0 / arr.ndim)

    # new shape for subsampled array
    ns = np.floor(np.array(arr.shape, np.int64) / f).clip(min=1)

    # get the step size for the slices
    slices = list(
        slice(None, None, int(s)) for s in np.floor(arr.shape / ns).astype(int)
    )

    # ignore dims e.g. RGB, which we don't want to downsample
    if ignore_dims is not None:
        for dim in ignore_dims:
            slices[dim] = slice(None)

    slices = tuple(slices)

    return np.asarray(arr[slices])


def _process_slice_str(slice_str):
    if not isinstance(slice_str, str):
        raise ValueError(f"Expected a string argument, received: {slice_str}")
    if slice_str.isdigit():
        return int(slice_str)
    else:
        parts = slice_str.split(":")
    return slice(*[int(p) if p else None for p in parts])


def _process_slice_objects(slice_str):
    return tuple(map(_process_slice_str, slice_str.split(",")))


def _print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            _print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")
