"""
Suite2p binary array reader.

This module provides Suite2pArray for reading Suite2p binary output files
(data.bin, data_raw.bin) with their associated ops.npy metadata.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin

logger = log.get("arrays.suite2p")


@dataclass
class Suite2pArray(ReductionMixin):
    """
    Lazy array reader for Suite2p binary output files.

    Reads memory-mapped binary files (data.bin or data_raw.bin) alongside
    their ops.npy metadata. Supports switching between raw and registered
    data channels.

    Parameters
    ----------
    filename : str or Path
        Path to ops.npy or a .bin file in a Suite2p output directory.

    Attributes
    ----------
    shape : tuple[int, int, int]
        Shape as (nframes, Ly, Lx).
    dtype : np.dtype
        Data type (always np.int16 for Suite2p).
    metadata : dict
        Contents of ops.npy.
    active_file : Path
        Currently active binary file.
    raw_file : Path
        Path to data_raw.bin (unregistered).
    reg_file : Path
        Path to data.bin (registered).

    Examples
    --------
    >>> arr = Suite2pArray("suite2p_output/ops.npy")
    >>> arr.shape
    (10000, 512, 512)
    >>> frame = arr[0]  # Get first frame
    >>> arr.switch_channel(use_raw=True)  # Switch to raw data
    """

    filename: str | Path
    metadata: dict = field(init=False)
    active_file: Path = field(init=False)
    raw_file: Path = field(default=None)
    reg_file: Path = field(default=None)

    def __post_init__(self):
        path = Path(self.filename)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".npy" and path.stem == "ops":
            ops_path = path
        elif path.suffix == ".bin":
            ops_path = path.with_name("ops.npy")
            if not ops_path.exists():
                raise FileNotFoundError(f"Missing ops.npy near {path}")
        else:
            raise ValueError(f"Unsupported input: {path}")

        self.metadata = np.load(ops_path, allow_pickle=True).item()
        self.num_rois = self.metadata.get("num_rois", 1)

        # resolve both possible bins - always look in the same directory as ops.npy
        # (metadata paths may be stale if data was moved)
        ops_dir = ops_path.parent
        self.raw_file = ops_dir / "data_raw.bin"
        self.reg_file = ops_dir / "data.bin"

        # choose which one to use
        if path.suffix == ".bin":
            # User clicked directly on a .bin file - use that specific file
            self.active_file = path
            if not self.active_file.exists():
                raise FileNotFoundError(
                    f"Binary file not found: {self.active_file}\n"
                    f"Available files in {ops_dir}:\n"
                    f"  - data.bin: {'exists' if self.reg_file.exists() else 'missing'}\n"
                    f"  - data_raw.bin: {'exists' if self.raw_file.exists() else 'missing'}"
                )
        else:
            # User clicked on directory/ops.npy - choose best available file
            # Prefer registered (data.bin) over raw (data_raw.bin)
            if self.reg_file.exists():
                self.active_file = self.reg_file
            elif self.raw_file.exists():
                self.active_file = self.raw_file
            else:
                raise FileNotFoundError(
                    f"No binary files found in {ops_dir}\n"
                    f"Expected either:\n"
                    f"  - {self.reg_file} (registered)\n"
                    f"  - {self.raw_file} (raw)\n"
                    f"Please check that Suite2p processing completed successfully."
                )

        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16

        # Validate file size matches expected shape
        expected_bytes = int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize
        actual_bytes = self.active_file.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(
                f"Binary file {self.active_file.name} is too small!\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"File may be corrupted or ops.npy metadata may be incorrect."
            )
        elif actual_bytes > expected_bytes:
            warnings.warn(
                f"Binary file {self.active_file.name} is larger than expected.\n"
                f"Expected: {expected_bytes:,} bytes for shape {self.shape}\n"
                f"Actual: {actual_bytes:,} bytes\n"
                f"Extra data will be ignored.",
                UserWarning,
            )

        self._file = np.memmap(
            self.active_file, mode="r", dtype=self.dtype, shape=self.shape
        )
        self.filenames = [self.active_file]

    def switch_channel(self, use_raw=False):
        """Switch between raw and registered data channels."""
        new_file = self.raw_file if use_raw else self.reg_file
        if not new_file.exists():
            raise FileNotFoundError(new_file)
        self._file = np.memmap(new_file, mode="r", dtype=self.dtype, shape=self.shape)
        self.active_file = new_file

    def __getitem__(self, key):
        return self._file[key]

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        n = min(10, self.nframes) if self.nframes >= 10 else self.nframes
        return np.stack([self._file[i] for i in range(n)], axis=0)

    @property
    def ndim(self):
        return len(self.shape)

    def close(self):
        """Close the memory-mapped file."""
        self._file._mmap.close()  # type: ignore

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
        """Write Suite2pArray to disk in various formats."""
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
        arrays = []
        names = []

        # Try to load both files if they exist
        if self.raw_file.exists():
            try:
                raw = Suite2pArray(self.raw_file)
                arrays.append(raw)
                names.append("raw")
            except Exception as e:
                logger.warning(f"Could not open raw file {self.raw_file}: {e}")

        if self.reg_file.exists():
            try:
                reg = Suite2pArray(self.reg_file)
                arrays.append(reg)
                names.append("registered")
            except Exception as e:
                logger.warning(f"Could not open registered file {self.reg_file}: {e}")

        # If neither file could be loaded, show the currently active file
        if not arrays:
            arrays.append(self)
            if self.active_file == self.raw_file:
                names.append("raw")
            elif self.active_file == self.reg_file:
                names.append("registered")
            else:
                names.append(self.active_file.name)

        figure_kwargs = kwargs.get("figure_kwargs", {"size": (800, 1000)})
        histogram_widget = kwargs.get("histogram_widget", True)
        window_funcs = kwargs.get("window_funcs", None)

        import fastplotlib as fpl

        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,
            figure_shape=(1, len(arrays)),
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )
