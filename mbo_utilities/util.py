import numpy as np


def align_images_zstack(images, mode="trim"):
    """
    Aligns images to a common shape for Z-stack viewing.

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D max-projection images.
    mode : str, optional
        "trim" - Crops to the smallest common shape (default).
        "pad"  - Pads to the largest shape with zeros.

    Returns
    -------
    np.ndarray
        3D Z-stack in (Z, X, Y) format.
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

def float2uint8(scan):
    """ Converts an scan (or image) from floats to uint8 (preserving the range)."""
    scan = (scan - scan.min()) / (scan.max() - scan.min())
    scan = (scan * 255).astype(np.uint8, copy=False)
    return scan

def smooth_data(data, window_size=5):
    """Smooth the data using a moving average."""
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

