from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Tuple, Any, Protocol
import numpy as np
import tifffile

from mbo_utilities import is_raw_scanimage
from mbo_utilities.file_io import Scan_MBO, read_scan, get_files
from mbo_utilities.metadata import has_mbo_metadata


class Loader(Protocol):
    def load(self) -> Tuple[Any, List[str]]:
        ...

@dataclass
class RawScanLoader:
    paths: list[Path]
    roi: int | None = None

    def load(self) -> Tuple[Scan_MBO, List[str | Path] | str | Path]:
        scan = read_scan(self.paths, roi=self.roi)
        scan.read_data([str(p) for p in self.paths], dtype=np.int16)
        return scan, [str(p) for p in self.paths]

@dataclass
class MboDataLoader:
    paths: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        # placeholder: read MBO metadata to define stacking axis
        arrays = [np.load(str(p), mmap_mode="r") for p in self.paths]
        stacked = np.stack(arrays, axis=0)
        return stacked, [str(p) for p in self.paths]

@dataclass
class NpyLoader:
    path: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.path), mmap_mode="r")
        return arr, [str(self.path)]

@dataclass
class TifLoader:
    path: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = tifffile.memmap(str(self.path))
        return arr, [str(self.path)]


@dataclass
class LazyArrayLoader:
    inputs: str | Path | Sequence[str | Path]
    roi: int | None = None
    loader: Any = field(init=False)

    def __post_init__(self):
        # 1) Normalize into a list of Path
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

        # 2) Filter only supported extensions
        supported = {".npy", ".tif", ".tiff", ".bin"}
        filtered = [p for p in self.inputs if p.suffix.lower() in supported]
        if not filtered:
            raise ValueError(f"No supported files in {self.inputs}")
        self.inputs = filtered

        # 3) Check for mixed‐type in a single directory
        exts = {p.suffix.lower() for p in self.inputs}
        if len(exts) > 1:
            raise ValueError(f"Multiple file types found in directory: {exts!r}")

        # 4) Dispatch on the first file
        first = Path(self.inputs[0])
        if first.suffix in [".tif", ".tiff"]:
            if is_raw_scanimage(first):
                self.loader = RawScanLoader(self.inputs, roi=self.roi)
            elif has_mbo_metadata(first):
                self.loader = MboDataLoader(self.inputs)
            else:
                raise ValueError("Unsupported TIFF file type or missing metadata.")
        elif first.suffix.lower() == ".bin":
            meta = first.parent.joinpath("ops.npy")
            if meta.is_file():
                print(f"Metadata found: {meta}")
                metadata = np.load(meta, allow_pickle=True).item()
                raise NotImplementedError("BIN files with metadata are not yet supported.")
        elif first.suffix.lower() in {".tif", ".tiff"}:
            self.loader = TifLoader(first)
        else:
            raise TypeError(f"Unsupported file type: {first.suffix}")

    def load(self) -> Tuple[Any, List[str]]:
        return self.loader.load()
