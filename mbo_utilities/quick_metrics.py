"""
quick_metrics.py
"""
import numpy as np
from skimage.restoration import estimate_sigma


def estimate_noise(image):
    """
    Estimate the noise in a 2D array.
    
    Parameters
    ----------
    image : 2D numpy array
        The array for which to estimate the noise.
        
    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.

    Notes
    -----
    This function assumes the noise follows a Gaussian distribution. The
    estimation algorithm is based on the median absolute deviation of the
    wavelet detail coefficients as described in section 4.2 of [1]_.

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> img = img + sigma * rng.standard_normal(img.shape)
    >>> sigma_hat = estimate_noise(img)
    """
    return estimate_sigma(image, multichannel=True, average_sigmas=True)
