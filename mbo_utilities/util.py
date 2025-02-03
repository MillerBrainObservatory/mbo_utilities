import numpy as np


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


def extract_center_square(images, size):
    """
    Extract a square crop from the center of the input images.

    Parameters
    ----------
    images : numpy.ndarray
        Input array. Can be 2D (H x W) or 3D (T x H x W), where:
        - H is the height of the image(s).
        - W is the width of the image(s).
        - T is the number of frames (if 3D).
    size : int
        The size of the square crop. The output will have dimensions
        (size x size) for 2D inputs or (T x size x size) for 3D inputs.

    Returns
    -------
    numpy.ndarray
        A square crop from the center of the input images. The returned array
        will have dimensions:
        - (size x size) if the input is 2D.
        - (T x size x size) if the input is 3D.

    Raises
    ------
    ValueError
        If `images` is not a NumPy array.
        If `images` is not 2D or 3D.
        If the specified `size` is larger than the height or width of the input images.

    Notes
    -----
    - For 2D arrays, the function extracts a square crop directly from the center.
    - For 3D arrays, the crop is applied uniformly across all frames (T).
    - If the input dimensions are smaller than the requested `size`, an error will be raised.

    Examples
    --------
    Extract a center square from a 2D image:

    >>> import numpy as np
    >>> image = np.random.rand(600, 576)
    >>> cropped = extract_center_square(image, size=200)
    >>> cropped.shape
    (200, 200)

    Extract a center square from a 3D stack of images:

    >>> stack = np.random.rand(100, 600, 576)
    >>> cropped_stack = extract_center_square(stack, size=200)
    >>> cropped_stack.shape
    (100, 200, 200)
    """
    if not isinstance(images, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if images.ndim == 2:  # 2D array (H x W)
        height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]

    elif images.ndim == 3:  # 3D array (T x H x W)
        T, height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[:,
               center_h - half_size:center_h + half_size,
               center_w - half_size:center_w + half_size]
    else:
        raise ValueError("Input array must be 2D or 3D.")
