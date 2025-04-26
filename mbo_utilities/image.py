import numpy as np


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
        return images[
            center_h - half_size : center_h + half_size,
            center_w - half_size : center_w + half_size,
        ]

    elif images.ndim == 3:  # 3D array (T x H x W)
        T, height, width = images.shape
        center_h, center_w = height // 2, width // 2
        half_size = size // 2
        return images[
            :,
            center_h - half_size : center_h + half_size,
            center_w - half_size : center_w + half_size,
        ]
    else:
        raise ValueError("Input array must be 2D or 3D.")


def return_scan_offset(image_in, nvals: int = 8):
    """
    Compute the scan offset correction between interleaved lines or columns in an image.

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
    >>> from mbo_utilities import return_scan_offset, read_scan
    >>> scan = read_scan(r"data/assembled/plane_05.tif")
    >>> frame = np.mean(scan[:400, 0, :, :], axis=0)  #  400 frames of the first plane
    >>> offset = return_scan_offset(frame, 1)  # on the first axis
    >>> print(f"Computed scan offset: {offset}")


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
        raise AttributeError("Input image must be 2D or 3D.")

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


def _fix_scan_phase_2d(data_in: np.ndarray, offset: int) -> np.ndarray:
    """
    Corrects bidirectional scan phase by shifting only odd rows by the given pixel offset.
    """
    if offset == 0:
        return data_in

    data_out = np.copy(data_in)
    if offset > 0:
        data_out[1::2, :-offset] = data_in[1::2, offset:]
    else:
        offset = abs(offset)
        data_out[1::2, offset:] = data_in[1::2, :-offset]

    return data_out


def fix_scan_phase(data_in: np.ndarray, offset: int):
    """
    Applies scan phase correction to 2D or 3D data.

    If input is 3D, it computes the mean along the first dimension before applying the correction.

    Parameters
    ----------
    data_in : ndarray
        Input data, either 2D (sy, sx) or 3D (st, sy, sx).
    offset : int
        The amount of offset to correct for.

    Returns
    -------
    ndarray
        The corrected array of the same shape as input.
    """
    if data_in.ndim == 2:
        return _fix_scan_phase_2d(data_in, offset)
    elif data_in.ndim == 3:
        return np.stack(
            [_fix_scan_phase_2d(frame, offset) for frame in data_in], axis=0
        )
    else:
        raise ValueError("Unsupported number of dimensions. Expected 2D or 3D.")
