import numpy as np


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
