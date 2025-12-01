"""
NumPy array wrapper.

This module provides NumpyArray for wrapping NumPy arrays and .npy files
as lazy arrays conforming to LazyArrayProtocol.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin

logger = log.get("arrays.numpy")


class NumpyArray(ReductionMixin):
    """
    Lazy array wrapper for NumPy arrays and .npy files.

    Conforms to LazyArrayProtocol for compatibility with mbo_utilities I/O
    and processing pipelines. Supports 2D (image), 3D (time series), and
    4D (volumetric) data.

    Parameters
    ----------
    array : np.ndarray, str, or Path
        Either a numpy array (will be saved to temp file for memory mapping)
        or a path to a .npy file.
    metadata : dict, optional
        Metadata dictionary. If not provided, basic metadata is inferred
        from array shape.

    Examples
    --------
    >>> # From .npy file
    >>> arr = NumpyArray("data.npy")
    >>> arr.shape
    (100, 512, 512)

    >>> # From in-memory array (creates temp file)
    >>> data = np.random.randn(100, 512, 512).astype(np.float32)
    >>> arr = NumpyArray(data)
    >>> arr[0:10]  # Lazy slicing

    >>> # 4D volumetric data
    >>> vol = NumpyArray("volume.npy")  # shape: (T, Z, Y, X)
    >>> vol.ndim
    4
    """

    def __init__(self, array: np.ndarray | str | Path, metadata: dict | None = None):
        if isinstance(array, (str, Path)):
            self.path = Path(array)
            if not self.path.exists():
                raise FileNotFoundError(f"Numpy file not found: {self.path}")

            # Try loading - could be pure .npy or npz with embedded metadata
            loaded = np.load(self.path, mmap_mode="r", allow_pickle=True)

            if isinstance(loaded, np.lib.npyio.NpzFile):
                # NPZ format with embedded data and metadata
                self.data = loaded["data"]
                if "metadata" in loaded.files:
                    # Extract metadata dict from numpy array
                    meta_arr = loaded["metadata"]
                    if meta_arr.ndim == 0:
                        # Scalar array containing dict
                        self._metadata = meta_arr.item()
                    else:
                        self._metadata = {}
                else:
                    self._metadata = {}
                self._npz_file = loaded  # Keep reference to prevent closing
            else:
                # Pure .npy file
                self.data = loaded
                self._metadata = {}
                self._npz_file = None

            self._tempfile = None
        elif isinstance(array, np.ndarray):
            logger.info("Creating temporary .npy file for array.")
            tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
            np.save(tmp, array)  # type: ignore
            tmp.close()
            self.path = Path(tmp.name)
            self.data = np.load(self.path, mmap_mode="r")
            self._tempfile = tmp
            self._npz_file = None
            self._metadata = {}
            logger.debug(f"Temporary file created at {self.path}")
        else:
            raise TypeError(f"Expected np.ndarray or path, got {type(array)}")

        # Override with explicit metadata if provided
        if metadata is not None:
            self._metadata = metadata

        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        """Return length of first dimension (number of frames for 3D/4D)."""
        return self.shape[0]

    def __array__(self):
        return np.asarray(self.data)

    @property
    def filenames(self) -> list[Path]:
        return [self.path]

    @property
    def metadata(self) -> dict:
        # Ensure basic metadata is always present
        md = dict(self._metadata)
        if "nframes" not in md:
            md["nframes"] = self.shape[0] if self.ndim >= 1 else 1
        if "num_frames" not in md:
            md["num_frames"] = md["nframes"]
        if "Ly" not in md and self.ndim >= 2:
            md["Ly"] = self.shape[-2]
        if "Lx" not in md and self.ndim >= 2:
            md["Lx"] = self.shape[-1]
        return md

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        self._metadata = value


    def close(self):
        """Release resources and clean up temporary files."""
        if self._npz_file is not None:
            try:
                self._npz_file.close()
            except Exception:
                pass
            self._npz_file = None
        if self._tempfile:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except Exception:
                pass
            self._tempfile = None

    def __del__(self):
        self.close()

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write NumpyArray to disk in various formats."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )

    def imshow(self, **kwargs):
        """Display array using fastplotlib ImageWidget."""
        import fastplotlib as fpl

        histogram_widget = kwargs.pop("histogram_widget", True)
        figure_kwargs = kwargs.pop("figure_kwargs", {"size": (800, 800)})
        # Get min/max from first frame for contrast scaling
        first_frame = self.data[0] if self.ndim >= 1 else self.data
        graphic_kwargs = kwargs.pop(
            "graphic_kwargs", {"vmin": float(first_frame.min()), "vmax": float(first_frame.max())}
        )

        # Set up slider dimensions based on array dimensionality
        if self.ndim == 4:
            slider_dim_names = ("t", "z")
            window_funcs = kwargs.pop("window_funcs", (np.mean, None))
            window_sizes = kwargs.pop("window_sizes", (1, None))
        elif self.ndim == 3:
            slider_dim_names = ("t",)
            window_funcs = kwargs.pop("window_funcs", (np.mean,))
            window_sizes = kwargs.pop("window_sizes", (1,))
        else:
            slider_dim_names = None
            window_funcs = None
            window_sizes = None

        return fpl.ImageWidget(
            data=self.data,
            slider_dim_names=slider_dim_names,
            window_funcs=window_funcs,
            window_sizes=window_sizes,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            graphic_kwargs=graphic_kwargs,
            **kwargs,
        )
