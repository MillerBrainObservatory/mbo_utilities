# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "fastplotlib",
#     "glfw",
# ]
# ///
from typing import Optional, Tuple, Sequence
from contextlib import contextmanager
from tifffile import TiffWriter
import numpy as np
import os
from pathlib import Path
import tifffile


class BinaryFile:
    def __init__(
        self,
        Ly: int,
        Lx: int,
        filename: str,
        n_frames: int = None,
        dtype: str = "int16",
    ):
        """
        Creates/Opens a Suite2p BinaryFile for reading and/or writing image data that acts like numpy array

        Parameters
        ----------
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        filename: str
            The filename of the file to read from or write to
        """
        self.Ly = Ly
        self.Lx = Lx
        self.filename = filename
        self.dtype = dtype
        write = not os.path.exists(self.filename)

        if write and n_frames is None:
            raise ValueError(
                "need to provide number of frames n_frames when writing file"
            )
        elif not write:
            n_frames = self.n_frames
        shape = (n_frames, self.Ly, self.Lx)
        mode = "w+" if write else "r+"
        self.file = np.memmap(  # type: ignore # noqa: E501
            self.filename, mode=mode, dtype=self.dtype, shape=shape
        )
        self._index = 0
        self._can_read = True

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(
        from_filename: str, to_filename: str
    ) -> None:
        """
        Works with npz files, pickled npy files, etc.

        Parameters
        ----------
        from_filename: str
            The npy file to convert
        to_filename: str
            The binary file that will be created
        """
        np.load(from_filename).tofile(to_filename)

    @property
    def nbytesread(self):
        """number of bytes per frame (FIXED for given file)"""
        return np.int64(2 * self.Ly * self.Lx)

    @property
    def nbytes(self):
        """total number of bytes in the file."""
        return os.path.getsize(self.filename)

    @property
    def n_frames(self) -> int:
        """total number of frames in the file."""
        return int(self.nbytes // self.nbytesread)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        The dimensions of the data in the file

        Returns
        -------
        n_frames: int
            The number of frames
        Ly: int
            The height of each frame
        Lx: int
            The width of each frame
        """
        return self.n_frames, self.Ly, self.Lx

    @property
    def size(self) -> int:
        """
        Returns the total number of pixels

        Returns
        -------
        size: int
        """
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """
        Closes the file.
        """
        self.file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __setitem__(self, *items):
        indices, data = items
        if data.dtype != "int16":
            self.file[indices] = np.minimum(data, 2**15 - 2).astype("int16")
        else:
            self.file[indices] = data

    def __getitem__(self, *items):
        indices, *crop = items
        return self.file[indices]

    def sampled_mean(self) -> float:
        """
        Returns the sampled mean.
        """
        n_frames = self.n_frames
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
        frames = self.file[inds].astype(np.float32)
        return frames.mean(axis=0)

    @property
    def data(self) -> np.ndarray:
        """
        Returns all the frames in the file.

        Returns
        -------
        frames: n_frames x Ly x Lx
            The frame data
        """
        return self.file[:]

    def bin_movie(
        self,
        bin_size: int,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        bad_frames: Optional[np.ndarray] = None,
        reject_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Returns binned movie that rejects bad_frames (bool array) and crops to (y_range, x_range).

        Parameters
        ----------
        bin_size: int
            The size of each bin
        x_range: int, int
            Crops the data to a minimum and maximum x range.
        y_range: int, int
            Crops the data to a minimum and maximum y range.
        bad_frames: int array
            The indices to *not* include.
        reject_threshold: float

        Returns
        -------
        frames: nImg x Ly x Lx
            The frames
        """

        good_frames = (
            ~bad_frames
            if bad_frames is not None
            else np.ones(self.n_frames, dtype=bool)
        )

        batch_size = min(np.sum(good_frames), 500)
        batches = []
        for k in np.arange(0, self.n_frames, batch_size):
            indices = slice(k, min(k + batch_size, self.n_frames))
            data = self.file[indices]

            if x_range is not None and y_range is not None:
                data = data[:, slice(*y_range), slice(*x_range)]  # crop

            good_indices = good_frames[indices]
            if np.mean(good_indices) > reject_threshold:
                data = data[good_indices]

            if data.shape[0] > bin_size:
                data = binned_mean(mov=data, bin_size=bin_size)
                batches.extend(data)

        mov = np.stack(batches)
        return mov

    def write_tiff(self, fname, range_dict={}):
        "Writes BinaryFile's contents using selected ranges from range_dict into a tiff file."
        n_frames, Ly, Lx = self.shape
        frame_range, y_range, x_range = (0, n_frames), (0, Ly), (0, Lx)
        with TiffWriter(fname, bigtiff=True) as f:
            # Iterate through current data and write each frame to a tiff
            # All ranges should be Tuples(int,int)
            if "frame_range" in range_dict:
                frame_range = range_dict["frame_range"]
            if "x_range" in range_dict:
                x_range = range_dict["x_range"]
            if "y_range" in range_dict:
                y_range = range_dict["y_range"]
            print(
                "Frame Range: {}, y_range: {}, x_range{}".format(
                    frame_range, y_range, x_range
                )
            )
            for i in range(frame_range[0], frame_range[1]):
                curr_frame = np.floor(
                    self.file[i, y_range[0] : y_range[1], x_range[0] : x_range[1]]
                ).astype(np.int16)
                f.write(curr_frame, contiguous=True)
        print("Tiff has been saved to {}".format(fname))


def from_slice(s: slice) -> Optional[np.ndarray]:
    """Creates an np.arange() array from a Python slice object.  Helps provide numpy-like slicing interfaces."""
    return (
        np.arange(s.start, s.stop, s.step) if any([s.start, s.stop, s.step]) else None
    )


def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size "bin_size")."""
    n_frames, Ly, Lx = mov.shape
    mov = mov[: (n_frames // bin_size) * bin_size]
    return mov.reshape(-1, bin_size, Ly, Lx).astype(np.float32).mean(axis=1)


@contextmanager
def temporary_pointer(file):
    """context manager that resets file pointer location to its original place upon exit."""
    orig_pointer = file.tell()
    yield file
    file.seek(orig_pointer)


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
