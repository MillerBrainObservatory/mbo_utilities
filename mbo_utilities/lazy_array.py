from __future__ import annotations
import time

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, List, Tuple, Any, Protocol, Callable

import h5py
import numpy as np
import tifffile
import dask.array as da
from numpy import memmap, ndarray

from . import log
from mbo_utilities.file_io import Scan_MBO, read_scan, get_files, _make_json_serializable
from mbo_utilities.metadata import is_raw_scanimage, get_metadata
from mbo_utilities.metadata import has_mbo_metadata
from mbo_utilities.assembly import _save_data

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

class Loader(Protocol):
    fpath: Path | str | Sequence[Path | str]
    def load(self) -> Tuple[Any, List[str]]: ...

@dataclass
class DemixingResultsLoader:
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
        self._bf = BinaryFile(
            Ly=Ly, Lx=Lx, filename=str(reg_file), n_frames=n_frames  # type: ignore # noqa
        )
        self.shape = (n_frames, Ly, Lx)
        self.ndim = 3
        self.dtype = np.int16

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._bf[key]

        if isinstance(key, slice):
            idxs = range(*key.indices(self.shape[0]))
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
class Suite2pLoader:
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
class MBOScanLoader:
    fpath: list[Path]
    roi: int | Sequence[int] | None = None
    _scan: Scan_MBO | None = field(init=False, default=None)

    def __post_init__(self):
        self._scan = read_scan(self.fpath, roi=self.roi)

    def load(self) -> Scan_MBO | None | list[Scan_MBO | None]:
        if self._scan.selected_roi is None:
            return self._scan
        out = []
        for r in range(1, self._scan.num_rois + 1):
            s = copy.copy(self._scan)
            s.selected_roi = r
            s.phasecorr_method = "frame"
            out.append(s)
        return out

    def write(self, scan: Scan_MBO, **kwargs):

        planes = kwargs.get("planes")
        overwrite = kwargs.get("overwrite", True)
        ext = kwargs.get("ext", ".tiff")
        trim_edge = kwargs.get("trim_edge", (0, 0, 0, 0))
        fix_phase = kwargs.get("fix_phase", False)
        save_phase_png = kwargs.get("save_phase_png", False)
        target_chunk_mb = kwargs.get("target_chunk_mb", 20)
        metadata = kwargs.get("metadata", {})
        progress_callback = kwargs.get("progress_callback")
        upsample = kwargs.get("upsample", 20)
        debug = kwargs.get("debug", False)
        savedir = Path(metadata["save_path"])

        start_time = time.time()
        for roi in [None] if scan.selected_roi is None else [scan.selected_roi]:
            logger.info(f"Writing ROI {roi} to {savedir}")
            target = savedir if roi is None else savedir / f"roi{roi}"
            target.mkdir(exist_ok=True)

            scan.selected_roi = roi
            md = metadata.copy()
            if roi is not None:
                md["roi"] = roi

            _save_data(
                scan,
                target,
                planes=planes,
                overwrite=overwrite,
                ext=ext,
                trim_edge=trim_edge,
                fix_phase=fix_phase,
                save_phase_png=save_phase_png,
                target_chunk_mb=target_chunk_mb,
                metadata=md,
                progress_callback=progress_callback,
                upsample=upsample,
                debug=debug,
            )

        elapsed_time = time.time() - start_time
        print(f"Done saving ROI {roi}: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")

    @property
    def metadata(self):  # anything you like
        return getattr(self._scan, "metadata", None)



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
class H5Loader:
    fpath: Path | str
    dataset: str = "mov"

    def load(self) -> _LazyH5Dataset:
        return _LazyH5Dataset(self.fpath, self.dataset)

@dataclass
class MBOTiffLoader:
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
class NpyLoader:
    fpath: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.fpath), mmap_mode="r")
        return arr


@dataclass
class TifLoader:
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


@dataclass
class LazyArrayLoader2:
    inputs: str | Path | Sequence[str | Path]
    rois: int | None = None
    metadata: dict | None = None
    loader: Any = field(init=False)

    def __post_init__(self):

        if isinstance(self.inputs, np.ndarray):
            # return a list of the input array
            self.loader = lambda: [self.inputs]
            self.fpath = None
            return

        paths: list[Path]
        if isinstance(self.inputs, (str, Path)):
            p = Path(self.inputs)
            if not p.exists():
                raise ValueError(f"Input path does not exist: {p}")
            if p.is_dir():
                paths = [Path(f) for f in get_files(p)]
            else:
                paths = [Path(p)]
        elif isinstance(self.inputs, (list, tuple)):
            # list of numpy arrays, return as is
            if isinstance(self.inputs[0], np.ndarray):
                self.loader = lambda: self.inputs
                self.fpath = None
                return
            # convert all inputs to Path objects
            if all(isinstance(p, (str, Path)) for p in self.inputs):
                paths = [Path(p) for p in self.inputs]
            else:
                raise TypeError(
                    f"Unsupported input type in sequence: {type(self.inputs[0])}. "
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(self.inputs)}. "
                "Expected str, Path, or a sequence of str/Path."
            )

        if not paths:
            raise ValueError("No input files found.")

        # filter by supported file types
        filtered = [p for p in paths if p.suffix.lower() in SUPPORTED_FTYPES]
        if not filtered:
            raise ValueError(f"No supported files in {self.inputs}")
        self.fpath = filtered

        exts = {p.suffix.lower() for p in filtered}
        first = filtered[0]

        # now parse for loader
        if len(exts) > 1:
            # if there is a single .bin and .npy, its a suite2p bin
            # leaving the code below because
            if exts == {".bin", ".npy"}:
                parent = first.parent
                npy_file = parent / "ops.npy"
                bin_file = parent / "data_raw.bin"
                md = np.load(str(npy_file), allow_pickle=True).item()
                self.loader = Suite2pLoader(bin_file, md)
                return
            raise ValueError(f"Multiple file types found in directory: {exts!r}")
        if first.suffix in [".tif", ".tiff"]:
            if is_raw_scanimage(first):
                self.loader = MBOScanLoader(filtered, roi=self.rois)
            elif has_mbo_metadata(first):
                self.loader = MBOTiffLoader(filtered)
            else:
                self.loader = TifLoader(filtered)
        elif first.suffix.lower() == ".bin":
            npy_file = first.parent.joinpath("ops.npy")
            bin_file = first.parent.joinpath("data_raw.bin")
            if npy_file.is_file():
                metadata = np.load(npy_file, allow_pickle=True).item()
                self.loader = Suite2pLoader(bin_file, metadata)
            else:
                raise NotImplementedError("BIN files with metadata are not yet supported.")
        elif first.suffix.lower() == ".h5":
            self.loader = H5Loader(first)
        elif first.suffix.lower() == ".zarr":
            raise NotImplementedError("Zarr files are not yet supported.")
        elif first.suffix.lower() == ".npy":
            logger.info(f"Checking for demixer in {first.parent}")
            if (first.parent / "pmd_demixer.npy").is_file():
                logger.info(f"Found demixer in {first.parent}, loading demixer results.")
                self.loader = DemixingResultsLoader(first.parent)
                return
        else:
            logger.error(f"Unsupported file type: {first.suffix}")
            raise TypeError(f"Unsupported file type: {first.suffix}")

    def load(self) -> Any:
        if hasattr(self.loader, "load"):
            return self.loader.load()
        return self.loader()


@dataclass
class LazyArrayLoader:
    inputs: str | Path | Sequence[str | Path]
    rois: int | None = None
    loader: Any = field(init=False)
    _metadata: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.inputs, np.ndarray):
            # If input is already an ndarray, set a basic loader
            self.loader = lambda: [self.inputs]
            self.fpath = None
            return
        if isinstance(self.inputs, Scan_MBO):
            # If input is a Scan_MBO object, set the loader to return it
            self.loader = lambda: [self.inputs]
            self.fpath = None
            return

        paths: list[Path]
        if isinstance(self.inputs, (str, Path)):
            p = Path(self.inputs)
            if not p.exists():
                raise ValueError(f"Input path does not exist: {p}")
            paths = [Path(f) for f in get_files(p)] if p.is_dir() else [p]
        elif isinstance(self.inputs, (list, tuple)):
            if isinstance(self.inputs[0], np.ndarray):
                self.loader = lambda: self.inputs
                self.fpath = None
                return
            paths = [Path(p) for p in self.inputs if isinstance(p, (str, Path))]
        else:
            raise TypeError(
                f"Unsupported input type: {type(self.inputs)}. "
                "Expected str, Path, or a sequence of str/Path."
            )

        if not paths:
            raise ValueError("No input files found.")

        # Filter for supported file types
        filtered = [p for p in paths if p.suffix.lower() in SUPPORTED_FTYPES]
        if not filtered:
            raise ValueError(f"No supported files in {self.inputs}")
        self.fpath = filtered

        exts = {p.suffix.lower() for p in filtered}
        first = filtered[0]

        # Select appropriate loader
        if len(exts) > 1:
            if exts == {".bin", ".npy"}:
                parent = first.parent
                npy_file = parent / "ops.npy"
                bin_file = parent / "data_raw.bin"
                md = np.load(str(npy_file), allow_pickle=True).item()
                self.loader = Suite2pLoader(bin_file, md)
                self._extract_metadata()
                return
            raise ValueError(f"Multiple file types found in directory: {exts!r}")

        # Handling file loaders
        self._select_loader(first, filtered, exts)
        self._extract_metadata()

    def _select_loader(self, first, filtered, exts):
        if first.suffix in [".tif", ".tiff"]:
            if is_raw_scanimage(first):
                self.loader = MBOScanLoader(filtered, roi=self.rois)
            elif has_mbo_metadata(first):
                self.loader = MBOTiffLoader(filtered)
            else:
                self.loader = TifLoader(filtered)
        elif first.suffix.lower() == ".bin":
            npy_file = first.parent.joinpath("ops.npy")
            bin_file = first.parent.joinpath("data_raw.bin")
            if npy_file.is_file():
                metadata = np.load(npy_file, allow_pickle=True).item()
                self.loader = Suite2pLoader(bin_file, metadata)
            else:
                raise NotImplementedError("BIN files with metadata are not yet supported.")
        elif first.suffix.lower() == ".h5":
            self.loader = H5Loader(first)
        elif first.suffix.lower() == ".npy":
            if (first.parent / "pmd_demixer.npy").is_file():
                self.loader = DemixingResultsLoader(first.parent)
        else:
            logger.error(f"Unsupported file type: {first.suffix}")
            raise TypeError(f"Unsupported file type: {first.suffix}")

    def _extract_metadata(self):
        """Extracts and initializes metadata."""
        if isinstance(self.loader, MBOScanLoader):
            example_tiff = self.fpath[0]
            logger.info("Extracting metadata from first TIFF file.")
            self._metadata = get_metadata(example_tiff)
            logger.info(f"Metadata keys: {list(self._metadata.keys())}")
            si_metadata = _make_json_serializable(read_scan([example_tiff]).metadata)
            self._metadata.update({"si": si_metadata})
        else:
            self._metadata = {}

    def save_as(
            self,
            savedir: str | Path,
            planes: list | tuple = None,
            metadata: dict = None,
            overwrite: bool = True,
            ext: str = ".tiff",
            order: list | tuple = None,
            trim_edge: list | tuple = (0, 0, 0, 0),
            fix_phase: bool = False,
            save_phase_png: bool = False,
            target_chunk_mb: int = 20,
            progress_callback: Callable = None,
            upsample: int = 20,
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

        self = LazyArrayLoader(self)

        # save path
        savedir = Path(savedir)
        if not savedir.parent.is_dir():
            raise ValueError(f"{savedir} is not inside a valid directory.")
        savedir.mkdir(exist_ok=True)

        # Determine number of planes from lazy_array attributes
        # fallback to shape
        if hasattr(self, "num_planes"):
            num_planes = self.num_planes
        elif hasattr(self, "num_channels"):
            num_planes = self.num_channels
        if hasattr(self, "metadata"):
            if "num_planes" in self.metadata:
                num_planes = self.metadata["num_planes"]
            elif "num_channels" in self.metadata:
                num_planes = self.metadata["num_channels"]
        elif hasattr(self, 'ndim') and self.ndim >= 3:
            num_planes = self.shape[1] if self.ndim == 4 else 1
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
                f"Invalid plane indices {', '.join(map(str, [p + 1 for p in over_idx]))}; must be in range 1…{self.num_channels}"
            )

        if debug:
            logger.info(f"Total number of planes: {num_planes}")
            logger.info(f"Planes to be saved: {planes}")

        if order is not None:
            if len(order) != len(planes):
                raise ValueError(
                    f"The length of the `order` ({len(order)}) does not match the number of planes ({len(planes)})."
                )
            planes = [planes[i] for i in order]

        # Handle metadata
        file_metadata = self.metadata or {}
        if metadata:
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Provided metadata must be a dictionary, got {type(metadata)} instead."
                )
            file_metadata.update(metadata)

        file_metadata["save_path"] = str(savedir.resolve())
        logger.info(f"Final metadata: {file_metadata}")

        self = self.load()
        roi_list = list(iter_rois(self))

        # Determine ROI list based on lazy_array's properties
        for roi in roi_list:
            logger.info(f"Saving ROI {roi}" + (f" of {self.num_rois}" if supports_roi(self) else ""))
        self.loader.write(self)

    @property
    def metadata(self):
        return self._metadata

    def load(self) -> Any:
        if hasattr(self.loader, "load"):
            return self.loader.load()
        return self.loader()
