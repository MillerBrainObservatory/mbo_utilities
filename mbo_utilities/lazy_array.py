from __future__ import annotations
import time

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Tuple, Any,  Callable

import h5py
import numpy as np
import tifffile
import dask.array as da
from numpy import memmap, ndarray

from . import log
from mbo_utilities.file_io import Scan_MBO, read_scan, get_files
from mbo_utilities.metadata import is_raw_scanimage
from mbo_utilities.metadata import has_mbo_metadata
from ._writers import _save_data

try:
    from suite2p.io import BinaryFile
    HAS_SUITE2P = True
except ImportError:
    HAS_SUITE2P = False
    BinaryFile = None

try:
    from mbo_utilities.pipelines.masknmf import load_from_dir
    HAS_MASKNMF = True
except ImportError:
    HAS_MASKNMF = False
    load_from_dir = None

logger = log.get("lazy_array")

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}

SUPPORTED_FTYPES = (
    ".npy",
    ".tif",
    ".tiff",
    ".bin",
    ".h5",
    ".zarr",
)

def supports_roi(obj):
    return hasattr(obj, "selected_roi") and hasattr(obj, "num_rois")

def iter_rois(obj):
    if not supports_roi(obj):
        yield None
        return

    selected_roi = getattr(obj, "selected_roi", None)
    num_rois = getattr(obj, "num_rois", 1)

    if selected_roi is None:
        yield from range(1, num_rois + 1)
    elif isinstance(selected_roi, int):
        yield selected_roi
    elif isinstance(selected_roi, (list, tuple)):
        yield from selected_roi
    else:
        yield selected_roi

@dataclass
class DemixingResultsArray:
    plane_dir: Path

    def load(self):
        data = load_from_dir(self.plane_dir)
        return data["pmd_demixer"].results

class _Suite2pLazyArray:
    def __init__(self, ops: dict):
        Ly = ops["Ly"]
        Lx = ops["Lx"]
        n_frames = ops.get("nframes", ops.get("n_frames", None))
        if n_frames is None:
            raise ValueError(
                f"Could not locate 'nframes' or `n_frames` in ops: {ops.keys()}"
            )
        reg_file = ops.get("reg_file", ops.get("raw_file", None))
        if reg_file is None:
            raise ValueError(
                f"Could not locate 'reg_file' or 'raw_file' in ops: {ops.keys()}"
            )
        self._bf = BinaryFile(  # type: ignore  # noqa
            Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_frames
        )
        self.shape = (n_frames, Ly, Lx)
        self.ndim = 3
        self.dtype = np.int16

    def __len__(self) -> None | int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._bf[key]

        if isinstance(key, slice):
            idxs = range(*key.indices(self.shape[0]))   # type: ignore
            return np.stack([self._bf[i] for i in idxs], axis=0)

        t, y, x = key
        if isinstance(t, int):
            frame = self._bf[t]
            return frame[y, x]
        elif isinstance(t, range):
            idxs = t
        else:
            idxs = range(*t.indices(self.shape[0]))
        return np.stack([self._bf[i][y, x] for i in idxs], axis=0)

    def min(self) -> float:
        return float(self._bf[0].min())

    def max(self) -> float:
        return float(self._bf[0].max())

    def __array__(self):
        n = min(10, self.shape[0])
        return np.stack([self._bf[i] for i in range(n)], axis=0)

    def close(self):
        self._bf.file.close()

@dataclass
class Suite2pArray:
    fpath: Path | str
    metadata: dict
    shape: tuple[int, ...] = ()
    nframes: int = 0
    frame_rate: float = 0.0

    def __post_init__(self):
        if not HAS_SUITE2P:
            logger.info("No suite2p detected, cannot initialize Suite2pLoader.")
            return None
        if isinstance(self.fpath, list):
            self.metadata = np.load([p for p in self.fpath if p.suffix == ".npy"][0], allow_pickle=True).item()
            self.fpath = [p for p in self.fpath if p.suffix == ".bin"][0]
        if isinstance(self.metadata, (str, Path)):
            self.metadata = np.load(self.metadata, allow_pickle=True).item()
        self.nframes = self.metadata["nframes"]
        self.Lx = self.metadata["Lx"]
        self.Ly = self.metadata["Ly"]

    def load(self) -> _Suite2pLazyArray:
        """
        Instead of returning a raw np.memmap, wrap the binary in a LazySuite2pMovie.
        That way we never load all frames at once—ImageWidget (or anything else) can
        index it on demand.
        """
        return _Suite2pLazyArray(self.metadata)

@dataclass
class MBOScanArray:
    fpath: list[Path]
    _roi: int | Sequence[int] | None = None
    fix_phase: bool = False
    upsample: int = 4
    max_offset: int = 3
    data: Scan_MBO | None = field(init=False, default=None)
    shape: tuple[int, ...] = field(init=False, default=())
    _metadata: dict = field(init=False, default_factory=dict)

    def __getitem__(self, item):
        """Allow indexing into the Scan_MBO data."""
        return self.data[item]

    def __post_init__(self):
        self.data = Scan_MBO(self.roi,)
        self.data.read_data(self.fpath)  # metadata is set
        print('setting roi')
        self.data.selected_roi = self.roi
        self._metadata.update(self.data.metadata)
        self.shape = self.data.shape

    @property
    def roi(self):
        if self.data is not None:
            return self.data.selected_roi
        return self._roi

    @roi.setter
    def roi(self, value: int | Sequence[int] | None):
        """Set the ROI for the Scan_MBO data."""
        if value is None or isinstance(value, int):
            self._roi = value
        elif isinstance(value, (list, tuple)):
            self._roi = list(value)
        else:
            raise ValueError(f"Invalid ROI type: {type(value)}. Must be int, list, or None.")
        self.data.selected_roi = self._roi

    def _imwrite(
            self,
            outpath: Path | str,
            overwrite = False,
            target_chunk_mb = 50,
            ext = '.tiff',
            progress_callback = None,
            debug = None,
            planes = None,
    ):
        start_time = time.time()
        target = outpath if self.roi is None else outpath / f"roi{self.roi}"
        target.mkdir(exist_ok=True)

        md = self.metadata.copy()
        self.selected_roi = self.roi
        md["roi"] = self.roi
        _save_data(
            self,
            target,
            planes=planes,
            overwrite=overwrite,
            ext=ext,
            target_chunk_mb=target_chunk_mb,
            metadata=md,
            progress_callback=progress_callback,
            debug=debug,
        )

        elapsed_time = time.time() - start_time
        print(f"Done saving ROI {roi}: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")

    @property
    def metadata(self):  # anything you like
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError(f"Metadata must be a dictionary, got {type(value)} instead.")
        self._metadata.update(value)

class _LazyH5Dataset:
    def __init__(self, fpath: Path | str, ds: str = "mov"):
        self._f = h5py.File(fpath, "r")
        self._d = self._f[ds]
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.ndim = self._d.ndim

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        return self._d[key]

    def min(self) -> float:
        return float(self._d[0].min())

    def max(self) -> float:
        return float(self._d[0].max())

    def __array__(self):
        n = min(10, self.shape[0])
        return self._d[:n]

    def close(self):
        self._f.close()

@dataclass
class H5Array:
    fpath: Path | str
    dataset: str = "mov"

    def load(self) -> _LazyH5Dataset:
        return _LazyH5Dataset(self.fpath, self.dataset)

@dataclass
class MBOTiffArray:
    fpath: list[Path]
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

    def load(self) -> memmap[Any, Any] | Any:
        # open each plane as a memmap
        mms: list[np.ndarray] = []
        if isinstance(self.fpath, list):
            if len(self.fpath) > 1:
                for p in self.fpath:
                    mm = tifffile.memmap(p, mode="r")
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

                stack = da.concatenate(planes, axis=0)  # (Z,T,Y,X)
                # should make this a parameter
                out = stack.transpose(1, 0, 2, 3)  # (T,Z,Y,X)
                return out
            elif len(self.fpath) == 1:
                # if there is only one file, just return a memmap
                try:
                    return tifffile.memmap(self.fpath[0], mode="r")
                except (ValueError, MemoryError) as e:
                    logger.debug(
                        f"cannot memmap TIFF file {self.fpath[0]}: {e}\n"
                        f" falling back to imread"
                    )
                    return tifffile.imread(self.fpath[0], mode="r")
        try:
            return tifffile.memmap(self.fpath[0], mode="r")
        except (ValueError, MemoryError) as e:
            logger.debug(
                f"cannot memmap TIFF file {self.fpath[0]}: {e}\n"
                f" falling back to imread"
            )
            return tifffile.imread(self.fpath[0], mode="r")


@dataclass
class NpyArray:
    fpath: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.fpath), mmap_mode="r")
        return arr


@dataclass
class TiffArray:
    fpath: list[Path]

    def load(self) -> np.memmap | ndarray:
        try:
            return tifffile.memmap(str(self.fpath))
        except (ValueError, MemoryError) as e:
            print(
                f"cannot memmap TIFF file {self.fpath}: {e}\n"
                f" falling back to imread"
            )
            return tifffile.imread(str(self.fpath), mode="r")


def imwrite(
        lazy_array,
        outpath: str | Path,
        planes: list | tuple = None,
        metadata: dict = None,
        overwrite: bool = True,
        ext: str = ".tiff",
        order: list | tuple = None,
        target_chunk_mb: int = 20,
        progress_callback: Callable = None,
        debug: bool = False,
):
    # Logging
    if debug:
        logger.setLevel(logging.INFO)
        logger.info("Debug mode enabled; setting log level to INFO.")
        logger.propagate = True  # send to terminal
    else:
        logger.setLevel(logging.WARNING)
        logger.info("Debug mode disabled; setting log level to WARNING.")
        logger.propagate = False  # don't send to terminal

    # save path
    outpath = Path(outpath)
    if not outpath.parent.is_dir():
        raise ValueError(f"{outpath} is not inside a valid directory.")
    outpath.mkdir(exist_ok=True)

    # Determine number of planes from lazy_array attributes
    # fallback to shape
    num_planes = 1
    if hasattr(lazy_array, "num_planes"):
        num_planes = lazy_array.num_planes
    elif hasattr(lazy_array, "num_channels"):
        num_planes = lazy_array.num_channels
    if hasattr(lazy_array, "metadata"):
        if "num_planes" in lazy_array.metadata:
            num_planes = lazy_array.metadata["num_planes"]
        elif "num_channels" in lazy_array.metadata:
            num_planes = lazy_array.metadata["num_channels"]
    elif hasattr(lazy_array, 'ndim') and lazy_array.ndim >= 3:
        num_planes = lazy_array.shape[1] if lazy_array.ndim == 4 else 1
    else:
        raise ValueError("Cannot determine the number of planes.")

    # convert to 0 based indexing
    if isinstance(planes, int):
        planes = [planes - 1]
    elif planes is None:
        planes = list(range(num_planes))
    else:
        planes = [p - 1 for p in planes]

    # make sure indexes are valid
    over_idx = [p for p in planes if p < 0 or p >= num_planes]
    if over_idx:
        raise ValueError(
            f"Invalid plane indices {', '.join(map(str, [p + 1 for p in over_idx]))}; must be in range 1…{lazy_array.num_channels}"
        )

    if order is not None:
        if len(order) != len(planes):
            raise ValueError(
                f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
            )
        planes = [planes[i] for i in order]

    # Handle metadata
    file_metadata = lazy_array.metadata or {}
    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Provided metadata must be a dictionary, got {type(metadata)} instead."
            )
        file_metadata.update(metadata)

    file_metadata["save_path"] = str(outpath.resolve())
    if hasattr(lazy_array, "metadata"):
        lazy_array.metadata.update(file_metadata)

    if hasattr(lazy_array, "_imwrite"):
        return lazy_array._imwrite(  # noqa
            outpath,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            ext=ext,
            progress_callback=progress_callback,
            planes=planes,
            debug=debug
        )
    else:
        raise TypeError(f"{type(lazy_array)} does not implement an `imwrite()` method.")

def imread(
        inputs: str | Path | Sequence[str | Path],
        *,
        roi: int | None = None,
        **kwargs
):
    if isinstance(inputs, np.ndarray):
        return inputs
    if isinstance(inputs, Scan_MBO):
        return inputs

    if isinstance(inputs, (str, Path)):
        p = Path(inputs)
        if not p.exists():
            raise ValueError(f"Input path does not exist: {p}")
        paths = [Path(f) for f in get_files(p)] if p.is_dir() else [p]
    elif isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], np.ndarray):
            return inputs
        paths = [Path(p) for p in inputs if isinstance(p, (str, Path))]
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    if not paths:
        raise ValueError("No input files found.")

    filtered = [p for p in paths if p.suffix.lower() in SUPPORTED_FTYPES]
    if not filtered:
        raise ValueError(f"No supported files in {inputs}")
    paths = filtered

    exts = {p.suffix.lower() for p in paths}
    first = paths[0]

    if len(exts) > 1:
        if exts == {".bin", ".npy"}:
            npy_file = first.parent / "ops.npy"
            bin_file = first.parent / "data_raw.bin"
            md = np.load(str(npy_file), allow_pickle=True).item()
            return Suite2pArray(bin_file, md)
        raise ValueError(f"Multiple file types found in input: {exts!r}")

    if first.suffix in [".tif", ".tiff"]:
        if is_raw_scanimage(first):
            return MBOScanArray(paths, **kwargs)
        if has_mbo_metadata(first):
            return MBOTiffArray(paths, **kwargs)
        return TiffArray(paths)

    if first.suffix == ".bin":
        npy_file = first.parent / "ops.npy"
        bin_file = first.parent / "data_raw.bin"
        if npy_file.exists():
            md = np.load(str(npy_file), allow_pickle=True).item()
            return Suite2pArray(bin_file, md)
        raise NotImplementedError("BIN files with metadata are not yet supported.")

    if first.suffix == ".h5":
        return H5Array(first)

    if first.suffix == ".npy" and (first.parent / "pmd_demixer.npy").is_file():
        return DemixingResultsArray(first.parent)

    raise TypeError(f"Unsupported file type: {first.suffix}")
