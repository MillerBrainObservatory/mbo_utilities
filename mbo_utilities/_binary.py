# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "fastplotlib",
#     "glfw",
# ]
# ///

import os
import numpy as np
from pathlib import Path
import tifffile

try:
    from suite2p.io.binary import BinaryFile
except ImportError:
    BinaryFile = None


class VolumetricBinaryFile:
    """
    A binary file class that supports saving image data in raw binary format
    for either 3D (T, Y, X) or 4D (Z, T, Y, X) datasets.

    Parameters
    ----------
    shape : tuple
        The full shape of the data to be saved. For a 3D dataset, use (T, Y, X).
        For a 4D dataset, use (Z, T, Y, X).
    filename : str
        The file to create or open.
    dtype : str or np.dtype, optional
        The data type for the binary file. Default is 'int16'.

    Notes
    -----
    If the file does not exist, this class creates a new file in mode "w+".
    If it exists, it is opened in "r+" mode.
    """

    def __init__(self, shape, filename, dtype="int16"):
        self.filename = str(Path(filename))
        self.dtype = np.dtype(dtype)
        self._is_new = not os.path.exists(self.filename)

        # If writing a new file, shape must be provided.
        if self._is_new:
            if shape is None:
                raise ValueError("For a new file you must provide the full shape.")
            self._shape = shape
            mode = "w+"
        else:
            self._shape = shape
            mode = "r+"

        self._file = np.memmap(  # type: ignore # noqa: E501
            self.filename, mode=mode, dtype=self.dtype, shape=self._shape
        )

    @property
    def shape(self):
        """Returns the full shape of the data."""
        return self._file.shape

    @property
    def size(self):
        """Returns the total number of elements."""
        return self._file.size

    @property
    def nbytes(self):
        """Returns the total number of bytes in the file."""
        return os.path.getsize(self.filename)

    def __getitem__(self, idx):
        """Allow NumPy-like slicing."""
        return self._file[idx]

    def __setitem__(self, idx, value):
        """Allow NumPy-like assignment.

        If the data type of the value is not the same as dtype, values will be clipped.
        """
        if np.asarray(value).dtype != self.dtype:
            # Clip values to avoid overflow (assumes int16, for example)
            max_val = np.iinfo(self.dtype).max - 1
            self._file[idx] = np.clip(value, None, max_val).astype(self.dtype)
        else:
            self._file[idx] = value

    def close(self):
        """Closes the memmap."""
        if hasattr(self._file, "_mmap"):
            self._file._mmap.close()  # type: ignore # noqa: E501

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def ndim(self):
        return len(self._shape)


def tiff_to_binary(tiff_path, out_path, dtype="int16"):
    data = tifffile.memmap(tiff_path)
    out_path = Path(out_path).with_suffix(".bin")

    if data.ndim != 3:
        raise ValueError("Must be assembled, 3D (T, Y, X)")

    nframes, x, y = data.shape
    bf = BinaryFile(  # type: ignore # noqa
        Ly=y, Lx=x, filename=str(Path(out_path)), n_frames=nframes, dtype=dtype
    )

    bf[:] = data
    bf.close()

    print(f"Wrote binary file '{out_path}' with {nframes} frames of shape ({x},{y}).")
