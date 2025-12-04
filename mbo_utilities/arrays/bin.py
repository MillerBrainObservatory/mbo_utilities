"""
Binary array reader/writer.

This module provides BinArray for reading and writing Suite2p-format binary files
(.bin) without requiring ops.npy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin
from mbo_utilities.util import load_npy

logger = log.get("arrays.bin")


@dataclass
class BinArray(ReductionMixin):
    """
    Read/write raw binary files (Suite2p format) without requiring ops.npy.

    This class provides a lightweight interface for working with raw binary
    files (.bin) directly, without needing the full Suite2p context that
    Suite2pArray provides. Useful for workflows that manipulate individual
    binary files (e.g., data_raw.bin vs data.bin).

    Parameters
    ----------
    filename : str or Path
        Path to the binary file
    shape : tuple, optional
        Shape of the data as (nframes, Ly, Lx). If None and file exists,
        will try to infer from adjacent ops.npy file.
    dtype : np.dtype, default=np.int16
        Data type of the binary file
    metadata : dict, optional
        Additional metadata to store with the array

    Examples
    --------
    >>> # Read existing binary with known shape
    >>> arr = BinArray("data_raw.bin", shape=(1000, 512, 512))
    >>> frame = arr[0]

    >>> # Create new binary file
    >>> arr = BinArray("output.bin", shape=(100, 256, 256))
    >>> arr[0] = my_data
    """

    filename: str | Path
    shape: tuple = None
    dtype: np.dtype = field(default=np.int16)
    metadata: dict = field(default_factory=dict)
    _file: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.filename = Path(self.filename)
        self.dtype = np.dtype(self.dtype)

        # If file exists and shape not provided, try to infer from ops.npy
        if self.filename.exists() and self.shape is None:
            ops_file = self.filename.parent / "ops.npy"
            if ops_file.exists():
                try:
                    ops = load_npy(ops_file).item()
                    Ly = ops.get("Ly")
                    Lx = ops.get("Lx")
                    nframes = ops.get("nframes", ops.get("n_frames"))
                    if all(x is not None for x in [Ly, Lx, nframes]):
                        self.shape = (nframes, Ly, Lx)
                        # Optionally copy metadata from ops
                        self.metadata.update(ops)
                        logger.debug(f"Inferred shape from ops.npy: {self.shape}")
                except Exception as e:
                    logger.warning(f"Could not read ops.npy: {e}")

            if self.shape is None:
                raise ValueError(
                    f"Cannot infer shape for {self.filename}. "
                    "Provide shape=(nframes, Ly, Lx) or ensure ops.npy exists."
                )

        # Creating new file
        if not self.filename.exists():
            if self.shape is None:
                raise ValueError(
                    "Must provide shape=(nframes, Ly, Lx) when creating new file"
                )
            mode = "w+"
        else:
            mode = "r+"

        self._file = np.memmap(
            self.filename, mode=mode, dtype=self.dtype, shape=self.shape
        )
        self.filenames = [self.filename]

    def __getitem__(self, key):
        return self._file[key]

    def __setitem__(self, key, value):
        """Allow assignment to the memmap."""
        if np.asarray(value).dtype != self.dtype:
            max_val = (
                np.iinfo(self.dtype).max - 1
                if np.issubdtype(self.dtype, np.integer)
                else None
            )
            if max_val:
                self._file[key] = np.clip(value, None, max_val).astype(self.dtype)
            else:
                self._file[key] = value.astype(self.dtype)
        else:
            self._file[key] = value

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        """Return first 10 frames for quick inspection."""
        n = min(10, self.shape[0]) if self.shape[0] >= 10 else self.shape[0]
        return np.array([self._file[i] for i in range(n)])

    @property
    def ndim(self):
        return len(self.shape)


    @property
    def nframes(self):
        return self.shape[0]

    @property
    def Ly(self):
        return self.shape[1]

    @property
    def Lx(self):
        return self.shape[2]

    @property
    def file(self):
        """Alias for _file, for backwards compatibility with BinaryFile API."""
        return self._file

    def flush(self):
        """Flush the memmap to disk."""
        self._file.flush()

    def close(self):
        """Close the memmap file."""
        if hasattr(self._file, "_mmap"):
            self._file._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _imwrite(
        self,
        outpath: Path | str,
        planes=None,
        target_chunk_mb: int = 50,
        ext: str = ".bin",
        progress_callback=None,
        debug: bool = False,
        overwrite: bool = False,
        output_name: str | None = None,
        **kwargs,
    ):
        """Write BinArray to disk in various formats."""
        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)

        ext_clean = ext.lower().lstrip(".")

        # For binary output, use direct memmap copy (faster)
        if ext_clean == "bin":
            md = dict(self.metadata) if self.metadata else {}
            md["Ly"] = self.Ly
            md["Lx"] = self.Lx
            md["nframes"] = self.nframes
            md["num_frames"] = self.nframes

            if output_name is None:
                output_name = "data_raw.bin"
            outfile = outpath / output_name

            if not outfile.exists() or overwrite:
                logger.info(f"Writing binary to {outfile}")
                new_file = np.memmap(
                    outfile, mode="w+", dtype=self.dtype, shape=self.shape
                )
                new_file[:] = self._file[:]
                new_file.flush()
                del new_file
            else:
                logger.info(f"Binary file already exists: {outfile}")

            # Write ops.npy
            ops_file = outpath / "ops.npy"
            np.save(ops_file, md)
            logger.info(f"Wrote ops.npy to {ops_file}")
            return outpath

        # For other formats, use common implementation
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            output_name=output_name,
            **kwargs,
        )
