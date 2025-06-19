from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, List, Sequence

import h5py
import numpy as np
import tifffile
from dask import array as da
from fsspec.implementations.reference import ReferenceFileSystem
from numpy import memmap, ndarray
from zarr import open as zarr_open
from zarr.storage import FsspecStore

from mbo_utilities.metadata import get_metadata
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities._writers import _save_data
from mbo_utilities.file_io import _multi_tiff_to_fsspec, HAS_ZARR, CHUNKS, _convert_range_to_slice, expand_paths
from mbo_utilities.util import subsample_array
from mbo_utilities.pipelines import load_from_dir, HAS_MASKNMF, HAS_SUITE2P

from mbo_utilities import log
from mbo_utilities.roi import iter_rois
from mbo_utilities.phasecorr import ALL_PHASECORR_METHODS, nd_windowed
from mbo_utilities.scanreader import scans, utils
from mbo_utilities.scanreader.multiroi import ROI

CHUNKS_4D = {0: 1, 1: "auto", 2: -1, 3: -1}
CHUNKS_3D = {0: 1, 1: -1, 2: -1}

logger = log.get("array_types")

@dataclass
class DemixingResultsArray:
    plane_dir: Path

    def load(self):
        data = load_from_dir(self.plane_dir)
        return data["pmd_demixer"].results

    def imshow(self, **kwargs):
        """
        Display the demixing results as an image widget.
        """
        pass
        # return imshow_lazy_array(self.load(), **kwargs)


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
    filenames: Path | str
    metadata: dict
    shape: tuple[int, ...] = ()
    nframes: int = 0
    frame_rate: float = 0.0

    def __post_init__(self):
        if not HAS_SUITE2P:
            logger.info("No suite2p detected, cannot initialize Suite2pLoader.")
            return None
        if isinstance(self.filenames, list):
            self.metadata = np.load([p for p in self.filenames if p.suffix == ".npy"][0], allow_pickle=True).item()
            self.filenames = [p for p in self.filenames if p.suffix == ".bin"][0]
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

    def imshow(self, **kwargs):
        """
        Display the Suite2p movie as an image widget.
        """
        pass
        # from mbo_utilities.graphics.display import imshow_lazy_array
        # return imshow_lazy_array(self.load(), **kwargs)


class _LazyH5Dataset:
    def __init__(self, filenames: Path | str, ds: str = "mov"):
        self._f = h5py.File(filenames, "r")
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
    filenames: Path | str
    dataset: str = "mov"

    def load(self) -> _LazyH5Dataset:
        return _LazyH5Dataset(self.filenames, self.dataset)


@dataclass
class MBOTiffArray:
    filenames: list[Path]
    _chunks: tuple[int, ...] | dict | None = None
    roi: Any = None
    _dask_array: da.Array = field(default=None, init=False, repr=False)

    @property
    def chunks(self) -> tuple[int, ...] | dict:
        return self._chunks or CHUNKS_4D

    @chunks.setter
    def chunks(self, value):
        self._chunks = value

    def _build_dask_array(self) -> da.Array:
        if len(self.filenames) == 1:
            arr = tifffile.memmap(self.filenames[0], mode="r")
            return da.from_array(arr, chunks=self.chunks)

        planes = []
        for p in self.filenames:
            mm = tifffile.memmap(p, mode="r")
            if mm.ndim == 3:
                mm = mm[None, ...]
            planes.append(da.from_array(mm, chunks=self.chunks))

        dstack = da.concatenate(planes, axis=0)  # (Z, T, Y, X)
        return dstack.transpose(1, 0, 2, 3)       # (T, Z, Y, X)

    @property
    def dask(self) -> da.Array:
        if self._dask_array is None:
            self._dask_array = self._build_dask_array()
        return self._dask_array

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> np.ndarray:
        return self.dask[key]

    def min(self) -> float:
        return float(self.dask[0].min().compute())

    def max(self) -> float:
        return float(self.dask[0].max().compute())

    @property
    def ndim(self) -> int:
        return self.dask.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dask.shape)

    def imshow(self):
        pass

@dataclass
class NpyArray:
    filenames: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.filenames), mmap_mode="r")
        return arr


@dataclass
class TiffArray:
    filenames: list[Path]

    def load(self) -> np.memmap | ndarray:
        try:
            return tifffile.memmap(str(self.filenames))
        except (ValueError, MemoryError) as e:
            print(
                f"cannot memmap TIFF file {self.filenames}: {e}\n"
                f" falling back to imread"
            )
            return tifffile.imread(str(self.filenames), mode="r")


class MboRawArray(scans.ScanMultiROI):
    """
    A subclass of ScanMultiROI that ignores the num_fields dimension
    and reorders the output to [time, z, x, y].
    """

    def __init__(
            self,
            files: str | Path | list = None,
            roi: int | Sequence[int] | None = None,
            fix_phase: bool = True,
            phasecorr_method: str = "frame",
            border: int | tuple[int, int, int, int] = 3,
            upsample: int =5,
            max_offset: int = 4,
    ):
        super().__init__(join_contiguous=True)
        if files:
            self.read_data(files)
        self._metadata = {} # set when pages are read
        self.roi = roi  # alias
        self._roi = roi
        self._fix_phase = fix_phase
        self._phasecorr_method = phasecorr_method
        self.border: int | tuple[int, int, int, int] = border
        self.max_offset: int = max_offset
        self.upsample: int = upsample
        self.pbar = None
        self.show_pbar = False
        self._offset = 0.0
        self.use_zarr = True

        # Debugging toggles
        self.debug_flags = {
            "frame_idx": True,
            "roi_array_shape": False,
            "phase_offset": False,
        }
        self.logger = logger
        self.logger.info(
            f"Initializing MBO Scan with parameters:\n"
            f"roi: {roi}, "
            f"fix_phase: {fix_phase}, "
            f"phasecorr_method: {phasecorr_method}, "
            f"border: {border}, "
            f"upsample: {upsample}, "
            f"max_offset: {max_offset}"
        )

    def save_fsspec(self, filenames):
        base_dir = Path(filenames[0]).parent

        combined_json_path = base_dir / "combined_refs.json"

        if combined_json_path.is_file():
            # delete it, its cheap to create
            logger.debug(f"Removing existing combined reference file: {combined_json_path}")
            combined_json_path.unlink()

        print(f"Generating combined kerchunk reference for {len(filenames)} files…")
        combined_refs = _multi_tiff_to_fsspec(tif_files=filenames, base_dir=base_dir)

        with open(combined_json_path, "w") as _f:
            json.dump(combined_refs, _f)

        print(f"Combined kerchunk reference written to {combined_json_path}")
        self.reference = combined_json_path
        return combined_json_path

    def as_dask(self):
        """
        Convert the current scan data to a Dask array.
        This will create a Dask array in the same directory as the reference file.
        """
        if not HAS_ZARR:
            raise ImportError("Zarr is not installed. Please install it to use this method.")
        if not Path(self.reference).is_file():
            raise FileNotFoundError(
                f"Reference file {self.reference} does not exist. "
                "Please call save_fsspec() first."
            )
        return da.from_zarr(
            FsspecStore(ReferenceFileSystem(str(self.reference))),
            chunks=CHUNKS,
        )

    def as_zarr(self):
        """
        Convert the current scan data to a Zarr array.
        This will create a Zarr store in the same directory as the reference file.
        """
        if not HAS_ZARR:
            raise ImportError("Zarr is not installed. Please install it to use this method.")
        if not Path(self.reference).is_file():
            raise FileNotFoundError(
                f"Reference file {self.reference} does not exist. "
                "Please call save_fsspec() first."
            )
        return zarr_open(
            FsspecStore(ReferenceFileSystem(str(self.reference))),
            mode="r",
        )

    def read_data(self, filenames, dtype=np.int16):
        filenames = expand_paths(filenames)
        self.save_fsspec(filenames)
        super().read_data(filenames, dtype)
        self._metadata = get_metadata(self.tiff_files[0].filehandle.path)  # from the file
        self._metadata.update({"si": _make_json_serializable(self.tiff_files[0].scanimage_metadata)})
        self._rois = self._create_rois()
        self.fields = self._create_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

    @property
    def metadata(self):
        md = self._metadata.copy()
        md.update({
            "fix_phase": self.fix_phase,
            "phasecorr_method": self.phasecorr_method,
            "offset": self.offset,
            "border": self.border,
            "upsample": self.upsample,
            "max_offset": self.max_offset,
        })
        return md

    @metadata.setter
    def metadata(self, value):
        self._metadata.update(value)

    @property
    def rois(self):
        """ROI's hold information about the size, position and shape of the ROIs."""
        return self._rois

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: float | np.ndarray):
        """
        Set the phase offset for phase correction.
        If value is a scalar, it applies the same offset to all frames.
        If value is an array, it must match the number of frames.
        """
        if isinstance(value, int):
            self._offset = float(value)
        self._offset = value

    @property
    def phasecorr_method(self):
        """
        Get the current phase correction method.
        Options are 'subpix' or 'mean'.
        """
        return self._phasecorr_method

    @phasecorr_method.setter
    def phasecorr_method(self, value: str):
        """
        Set the phase correction method.
        Options are 'two_step', 'subpix', or 'crosscorr'.
        """
        if value not in ALL_PHASECORR_METHODS:
            raise ValueError(
                f"Unsupported phase correction method: {value}. "
                f"Supported methods are: {ALL_PHASECORR_METHODS}"
            )
        self._phasecorr_method = value

    @property
    def fix_phase(self):
        """
        Get whether phase correction is applied.
        If True, phase correction is applied to the data.
        """
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value: bool):
        """
        Set whether to apply phase correction.
        If True, phase correction is applied to the data.
        """
        if not isinstance(value, bool):
            raise ValueError("do_phasecorr must be a boolean value.")
        self._fix_phase = value

    @property
    def roi(self):
        """
        Get the current ROI index.
        If roi is None, returns -1 to indicate no specific ROI.
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        """
        Set the current ROI index.
        If value is None, sets roi to -1 to indicate no specific ROI.
        """
        self._roi = value

    @property
    def num_rois(self) -> int:
        return len(self.rois)

    @property
    def xslices(self):
        return self.fields[0].xslices

    @property
    def yslices(self):
        return self.fields[0].yslices

    @property
    def output_xslices(self):
        return self.fields[0].output_xslices

    @property
    def output_yslices(self):
        return self.fields[0].output_yslices

    def _read_pages(self, frames, chans, yslice=slice(None), xslice=slice(None), **kwargs):
        C = self.num_channels
        pages = [f * C + c for f in frames for c in chans]

        H = len(utils.listify_index(yslice, self._page_height))
        W = len(utils.listify_index(xslice, self._page_width))

        if getattr(self, "use_zarr", False):
            zarray = self.as_zarr()
            buf = np.empty((len(pages), H, W), dtype=self.dtype)
            for i, page in enumerate(pages):
                f, c = divmod(page, C)
                buf[i] = zarray[f, c, yslice, xslice]

            if self.fix_phase:
                self.logger.debug(f"Applying phase correction with strategy: {self.phasecorr_method}")
                buf, self.offset = nd_windowed(
                    buf,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                )
            return buf.reshape(len(frames), len(chans), H, W)

        # TIFF path
        buf = np.empty((len(pages), H, W), dtype=self.dtype)
        start = 0
        for tf in self.tiff_files:
            end = start + len(tf.pages)
            idxs = [i for i, p in enumerate(pages) if start <= p < end]
            if not idxs:
                start = end
                continue

            frame_idx = [pages[i] - start for i in idxs]
            chunk = tf.asarray(key=frame_idx)[..., yslice, xslice]

            if self.fix_phase:
                self.logger.debug(f"Applying phase correction with strategy: {self.phasecorr_method}")
                corrected, self.offset = nd_windowed(
                    chunk,
                    method=self.phasecorr_method,
                    upsample=self.upsample,
                    max_offset=self.max_offset,
                    border=self.border,
                )
                buf[idxs] = corrected
            else:
                buf[idxs] = chunk
            start = end

        return buf.reshape(len(frames), len(chans), H, W)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        t_key, z_key, _, _ = tuple(_convert_range_to_slice(k) for k in key) + (slice(None),) * (4 - len(key))
        frames = utils.listify_index(t_key, self.num_frames)
        chans = utils.listify_index(z_key, self.num_channels)
        if not frames or not chans:
            return np.empty(0)

        logger.debug(
            f"Phase-corrected: {self.fix_phase}/{self.phasecorr_method},"
            f" channels: {chans},"
            f" roi: {self.roi}",
        )
        out = self.process_rois(frames, chans)

        squeeze = []
        if isinstance(t_key, int):
            squeeze.append(0)
        if isinstance(z_key, int):
            squeeze.append(1)
        if squeeze:
            if isinstance(out, tuple):
                out = tuple(np.squeeze(x, axis=tuple(squeeze)) for x in out)
            else:
                out = np.squeeze(out, axis=tuple(squeeze))
        return out

    def process_rois(self, frames, chans):
        if self.roi is not None:
            if isinstance(self.roi, list):
                return tuple(self.process_single_roi(roi_idx - 1, frames, chans) for roi_idx in self.roi)
            elif self.roi == 0:
                return tuple(self.process_single_roi(roi_idx, frames, chans) for roi_idx in range(self.num_rois))
            elif isinstance(self.roi, int):
                return self.process_single_roi(self.roi - 1, frames, chans)
        else:
            H_out, W_out = self.field_heights[0], self.field_widths[0]
            out = np.zeros((len(frames), len(chans), H_out, W_out), dtype=self.dtype)
            for roi_idx in range(self.num_rois):
                roi_data = self.process_single_roi(roi_idx, frames, chans)
                oys, oxs = self.fields[0].output_yslices[roi_idx], self.fields[0].output_xslices[roi_idx]
                out[:, :, oys, oxs] = roi_data
            return out

    def process_single_roi(self, roi_idx, frames, chans):
        return self._read_pages(
            frames,
            chans,
            yslice=self.fields[0].yslices[roi_idx],
            xslice=self.fields[0].xslices[roi_idx],
        )

    @property
    def total_frames(self):
        return sum(len(tf.pages) // self.num_channels for tf in self.tiff_files)

    @property
    def num_planes(self):
        """LBM alias for num_channels."""
        return self.num_channels

    def min(self):
        """
        Returns the minimum value of the first tiff page.
        """
        page = self.tiff_files[0].pages[0]
        return np.min(page.asarray())

    def max(self):
        """
        Returns the maximum value of the first tiff page.
        """
        page = self.tiff_files[0].pages[0]
        return np.max(page.asarray())

    @property
    def shape(self):
        """Shape is relative to the current ROI."""
        if self.roi is not None:
            if not isinstance(self.roi, (list, tuple)):
                if self.roi > 0:
                    s = self.fields[0].output_xslices[self.roi - 1]
                    width = s.stop - s.start
                    return (
                        self.total_frames,
                        self.num_channels,
                        self.field_heights[0],
                        width,
                    )
        # roi = None, or a list/tuple indicates the shape should be relative to the full FOV
        return (
            self.total_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def shape_full(self):
        return (
            self.total_frames,
            self.num_channels,
            self.field_heights[0],
            self.field_widths[0],
        )

    @property
    def ndim(self):
        return 4

    @property
    def size(self):
        return (
            self.num_frames
            * self.num_channels
            * self.field_heights[0]
            * self.field_widths[0]
        )

    @property
    def scanning_depths(self):
        """
        We override this because LBM should always be at a single scanning depth.
        """
        return [0]

    def _create_rois(self):
        """
        Create scan rois from the configuration file. Override the base method to force
        ROI's that have multiple 'zs' to a single depth.
        """
        try:
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        except KeyError:
            raise RuntimeError("This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata.")
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]

        # discard empty/malformed ROIs
        roi_infos = list(filter(lambda r: isinstance(r["zs"], (int, float, list)), roi_infos))

        # LBM uses a single depth that is not stored in metadata,
        # so force this to be 0.
        for roi_info in roi_infos:
            roi_info["zs"] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def __array__(self):
        """
        Convert the scan data to a NumPy array.
        Calculate the size of the scan and subsample to keep under memory limits.
        """
        return subsample_array(self, ignore_dims=[-1, -2, -3])

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
        for roi in iter_rois(self):
            print(roi)
            target = outpath if roi is None else outpath / f"roi{roi}"
            target.mkdir(exist_ok=True)

            md = self.metadata.copy()
            self.roi = roi
            md["roi"] = roi
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

    def imshow(self, **kwargs):
        pass
        # from mbo_utilities.graphics.display import imshow_lazy_array
        # return imshow_lazy_array(self, **kwargs)
