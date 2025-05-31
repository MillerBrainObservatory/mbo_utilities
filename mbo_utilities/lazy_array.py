from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Tuple, Any, Protocol

import numpy as np
import tifffile
import dask.array as da

from mbo_utilities.metadata import is_raw_scanimage
from .file_io import Scan_MBO, read_scan, get_files
from mbo_utilities.metadata import has_mbo_metadata, get_metadata


CHUNKS_4D = {
    0: 1,
    1: "auto",
    2: -1,
    3: -1
}

CHUNKS_3D = {
    0: 1,
    1: -1,
    2: -1
}

class Loader(Protocol):
    def load(self) -> Tuple[Any, List[str]]:
        ...

@dataclass
class MBOScanLoader:
    paths: list[Path]
    roi: int | Sequence[int] | None = None

    def load(self) -> Scan_MBO:
        scan = read_scan(self.paths, roi=self.roi)
        scan.fpath = Path(self.paths[0].parent)
        scan.read_data(self.paths)
        return scan

@dataclass
class MBOTiffLoader:
    paths: list[Path]
    shape: tuple[int, ...] = ()
    _chunks: tuple[int, ...] | dict | None = None

    def __post_init__(self):
        self.chunks = CHUNKS_4D

    @property
    def chunks(self) -> tuple[int, ...] | dict:
        if self._chunks is None:
            self._chunks = CHUNKS_3D
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        self._chunks = value

    def load(self) -> tuple[da.Array, list[str]]:

        # open each plane as a memmap
        mms: list[np.ndarray] = []
        for p in self.paths:
            mm = tifffile.memmap(p, mode="r").view(np.int16)
            mms.append(mm)

        # wrap every mem-map in a dask.Array
        planes = []
        for mm in mms:
            if mm.ndim == 3:
                da_mm = da.from_array(mm, chunks=self._chunks)
                da_mm = da_mm[None, ...]
            else:
                da_mm = da.from_array(mm, chunks=self._chunks)
            planes.append(da_mm)

        stack = da.concatenate(planes, axis=0)       # (Z,T,Y,X)
        # should make this a parameter
        out   = stack.transpose(1, 0, 2, 3)          # (T,Z,Y,X)
        return out


@dataclass
class NpyLoader:
    path: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.path), mmap_mode="r")
        return arr

@dataclass
class TifLoader:
    path: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = tifffile.memmap(str(self.path))
        return arr


@dataclass
class LazyArrayLoader:
    inputs: str | Path | Sequence[str | Path]
    roi: int | None = None
    loader: Any = field(init=False)

    def __post_init__(self):
        # normalize into a list of Path
        if isinstance(self.inputs, (str, Path)):
            p = Path(self.inputs)
            if not p.exists():
                raise ValueError(f"Input path does not exist: {p}")
            if p.is_dir():
                self.inputs = [Path(f) for f in get_files(p)]
            else:
                self.inputs = [p]
        else:
            self.inputs = [Path(p) for p in self.inputs]

        if not self.inputs:
            raise ValueError("No input files found.")

        # filter only supported extensions
        supported = {".npy", ".tif", ".tiff", ".bin"}
        filtered = [p for p in self.inputs if p.suffix.lower() in supported]
        if not filtered:
            raise ValueError(f"No supported files in {self.inputs}")
        self.inputs = filtered

        # check for mixed‐type in a single directory
        exts = {p.suffix.lower() for p in self.inputs}
        if len(exts) > 1:
            raise ValueError(f"Multiple file types found in directory: {exts!r}")

        # dispatch on the first file
        first = Path(self.inputs[0])
        if first.suffix in [".tif", ".tiff"]:
            if is_raw_scanimage(first):
                self.loader = MBOScanLoader(self.inputs, roi=self.roi)
            elif has_mbo_metadata(first):
                self.loader = MBOTiffLoader(self.inputs)
            else:
                raise ValueError("Unsupported TIFF file type or missing metadata.")
        elif first.suffix.lower() == ".bin":
            # meta = first.parent.joinpath("ops.npy")
            # if meta.is_file():
            #     print(f"Metadata found: {meta}")
            #     metadata = np.load(meta, allow_pickle=True).item()
            raise NotImplementedError("BIN files with metadata are not yet supported.")
        else:
            raise TypeError(f"Unsupported file type: {first.suffix}")

    def load(self) -> Any:
        return self.loader.load()
