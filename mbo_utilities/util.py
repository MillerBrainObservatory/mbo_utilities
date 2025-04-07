import numpy as np


def align_images_zstack(images, mode="trim"):
    """
    Align a list of 2D images to a common shape for creating a Z-stack. Helpful for e.g. suite2p max-images that are cropped
    to different x/y sizes.

    This function takes a list of 2D NumPy arrays and adjusts their sizes so that they all share
    the same dimensions. Two alignment modes are provided:
      - "trim": Crop each image to the smallest height and width among the images.
      - "pad": Pad each image with zeros to the size of the largest height and width among the images.

    Parameters
    ----------
    images : list of numpy.ndarray
        A list of 2D images to be aligned.
    mode : str, optional
        The method used for alignment. Must be either "trim" (default) or "pad".

    Returns
    -------
    numpy.ndarray
        A 3D NumPy array (Z-stack) of shape (N, H, W), where N is the number of images and H and W are
        the common height and width determined by the alignment mode.

    Examples
    --------
    >>> import numpy as np
    >>> img1 = np.random.rand(400, 500)
    >>> img2 = np.random.rand(450, 480)
    >>> zstack = align_images_zstack([img1, img2], mode="trim")
    >>> zstack.shape
    (2, 400, 480)
    """
    shapes = np.array([img.shape for img in images])

    if mode == "trim":
        target_shape = np.min(shapes, axis=0)
        aligned_images = [img[:target_shape[0], :target_shape[1]] for img in images]

    elif mode == "pad":
        target_shape = np.max(shapes, axis=0)
        aligned_images = [np.pad(img,
                                 ((0, target_shape[0] - img.shape[0]),
                                  (0, target_shape[1] - img.shape[1])),
                                 mode='constant') for img in images]
    else:
        raise ValueError("Invalid mode. Choose 'trim' or 'pad'.")
    return np.stack(aligned_images, axis=0)


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
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def norm_minmax(images):
    """ Normalize a NumPy array to the range [0, 1]. """
    return (images - images.min()) / (images.max() - images.min())

def norm_percentile(image, low_p=1, high_p=98):
    """
    Normalizes an image using percentile-based contrast stretching.

    Parameters
    ----------
    image : np.ndarray
        Input image array to be normalized.
    low_p : float, optional
        Lower percentile for normalization (default is 1).
    high_p : float, optional
        Upper percentile for normalization (default is 98).

    Returns
    -------
    np.ndarray
        Normalized image with values scaled between 0 and 1.

    Notes
    -----
    - This method enhances contrast by clipping extreme pixel values.
    - Percentile-based normalization is useful for images with outliers.
    """
    p_low, p_high = np.percentile(image, (low_p, high_p))
    return np.clip((image - p_low) / (p_high - p_low), 0, 1)

def match_array_size(arr1, arr2, mode="trim"):
    """
    Adjusts two arrays to have the same shape by either trimming or padding them.

    Parameters
    ----------
    arr1 : np.ndarray
        First input array.
    arr2 : np.ndarray
        Second input array.
    mode : str, optional
        Method for matching array sizes:
        - "trim" (default): Trims both arrays to the smallest common shape.
        - "pad": Pads both arrays with zeros to the largest common shape.

    Returns
    -------
    np.ndarray
        A stacked array of shape (2, ...) containing the adjusted arrays.

    Raises
    ------
    ValueError
        If an invalid mode is provided.
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


def is_running_jupyter():
    """Returns true if users environment is running Jupyter."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # are there other aliases for a jupyter shell
            return True  # jupyterlab
        elif shell == 'TerminalInteractiveShell':
            return False  # ipython from terminal
        else:
            return False
    except NameError:
        return False

