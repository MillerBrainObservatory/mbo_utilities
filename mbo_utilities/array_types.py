from __future__ import annotations

import copy
import json
from dataclasses import field
from typing import Any, Tuple, List, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import h5py
import tifffile
from dask import array as da
from fsspec.implementations.reference import ReferenceFileSystem
from numpy import ndarray
from zarr import open as zarr_open
from zarr.storage import FsspecStore
import fastplotlib as fpl

from mbo_utilities.metadata import get_metadata
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities._writers import _write_plane
from mbo_utilities.file_io import (
    _multi_tiff_to_fsspec,
    HAS_ZARR,
    CHUNKS,
    _convert_range_to_slice,
    expand_paths,
)
from mbo_utilities.util import subsample_array
from mbo_utilities.pipelines import HAS_SUITE2P, HAS_MASKNMF, load_from_dir

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
        if not HAS_MASKNMF:
            raise ImportError(
                "MaskNMF is not installed. Cannot display demixing results."
            )
        import fastplotlib as fpl

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (800, 1000),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=self.load(),
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )


@dataclass
class Suite2pArray:
    metadata: dict

    def __post_init__(self):
        self.Ly = self.metadata["Ly"]
        self.Lx = self.metadata["Lx"]
        self.nframes = self.metadata.get("nframes", self.metadata.get("n_frames"))
        if self.nframes is None:
            raise ValueError("Missing 'nframes' or 'n_frames' in metadata")

        self.filename = self.metadata.get("reg_file", self.metadata.get("raw_file"))
        if self.filename is None:
            raise ValueError("Missing 'reg_file' or 'raw_file' in metadata")

        self.filename = str(self.filename)
        self.shape = (self.nframes, self.Ly, self.Lx)
        self.dtype = np.int16
        self._file = np.memmap(
            self.filename, mode="r", dtype=self.dtype, shape=self.shape
        )

    def __getitem__(self, key):
        return self._file[key]

    def __len__(self):
        return self.shape[0]

    def __array__(self):
        n = min(10, self.nframes)
        return np.stack([self._file[i] for i in range(n)], axis=0)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def min(self):
        return float(self._file[0].min())

    @property
    def max(self):
        return float(self._file[0].max())

    def close(self):
        self._file._mmap.close()


class H5Array:
    def __init__(self, filenames: Path | str, dataset: str = "mov"):
        self.filenames = Path(filenames)
        self._f = h5py.File(self.filenames, "r")
        self._d = self._f[dataset]
        self.shape = self._d.shape
        self.dtype = self._d.dtype
        self.ndim = self._d.ndim

    @property
    def num_planes(self) -> int:
        # TODO: not sure what to do here
        return 14

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        # Expand ellipsis to match ndim
        if Ellipsis in key:
            idx = key.index(Ellipsis)
            n_missing = self.ndim - (len(key) - 1)
            key = key[:idx] + (slice(None),) * n_missing + key[idx + 1 :]

        # Remove None axes (np.newaxis) and track their positions
        slices = []
        result_shape = []
        dim = 0
        for k in key:
            if k is None:
                result_shape.append(1)
            else:
                slices.append(k)
                dim += 1

        data = self._d[tuple(slices)]

        for i, k in enumerate(key):
            if k is None:
                data = np.expand_dims(data, axis=i)

        return data

    def min(self) -> float:
        return float(self._d[0].min())

    def max(self) -> float:
        return float(self._d[0].max())

    def __array__(self):
        n = min(10, self.shape[0])
        return self._d[:n]

    def close(self):
        self._f.close()

    @property
    def metadata(self) -> dict:
        return dict(self._f.attrs)

    def _imwrite(self, outpath, **kwargs):
        _write_plane(
            self._d, Path(outpath),
            ext=kwargs.get("ext", ".tiff"),
            overwrite=kwargs.get("overwrite", False),
            metadata=self.metadata,
            target_chunk_mb=kwargs.get("target_chunk_mb", 20),
            progress_callback=kwargs.get("progress_callback", None),
            debug=kwargs.get("debug", False),
        )

    def _imwrite2(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
    ):
        target = outpath
        target.mkdir(exist_ok=True)
        md = self.metadata.copy()
        _write_plane(
            self,
            target,
            overwrite=overwrite,
            ext=ext,
            target_chunk_mb=target_chunk_mb,
            metadata=md,
            progress_callback=progress_callback,
            debug=debug,
        )


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
        return dstack.transpose(1, 0, 2, 3)  # (T, Z, Y, X)

    @property
    def dask(self) -> da.Array:
        if self._dask_array is None:
            self._dask_array = self._build_dask_array()
        return self._dask_array

    def __getitem__(self, key: int | slice | tuple[int, ...]) -> np.ndarray:
        return self.dask[key]

    def __getattr__(self, attr):
        return getattr(self.dask, attr)

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

    @property
    def metadata(self) -> dict:
        """
        Return metadata from the first TIFF file.
        Assumes all files have the same metadata structure.
        """
        if not self.filenames:
            return {}
        return get_metadata(self.filenames[0])

    def imshow(self, **kwargs) -> fpl.ImageWidget:
        if len(self.filenames) == 1:
            data = tifffile.memmap(self.filenames[0], mode="r")
        else:
            data = self.dask
        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (800, 1000),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=data,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )

    def _imwrite(
            self,
            outpath: Path | str,
            overwrite=False,
            target_chunk_mb=50,
            ext=".tiff",
            progress_callback=None,
            debug=None,
            planes=None,
    ):
        if "plane" in self.metadata.keys():
            plane = self.metadata["plane"]
        else:
            raise ValueError("Cannot determine plane from metadata.")

        outpath = Path(outpath)
        ext = ext.lower().lstrip(".")

        if ext in {"bin"}:
            fname = "data_raw.bin"
        else:
            fname = f"plane{plane:03d}.{ext}"

        if outpath.is_dir():
            target = outpath.joinpath(fname)
        else:
            target = outpath.parent.joinpath(fname)

        _write_plane(
            self,
            target,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            metadata=self.metadata,
            progress_callback=progress_callback,
            debug=debug,
            dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
            plane_index=None,  # convert to 0-based index
        )


# NOT YET IMPLEMENTED FULLY
@dataclass
class NpyArray:
    filenames: list[Path]

    def load(self) -> Tuple[np.ndarray, List[str]]:
        arr = np.load(str(self.filenames), mmap_mode="r")
        return arr


# NOT YET IMPLEMENTED FULLY
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
        upsample: int = 5,
        max_offset: int = 4,
    ):
        super().__init__(join_contiguous=True)
        self._metadata = {}  # set when pages are read
        self._fix_phase = fix_phase
        self._phasecorr_method = phasecorr_method
        self.border: int | tuple[int, int, int, int] = border
        self.max_offset: int = max_offset
        self.upsample: int = upsample
        self.use_zarr = False
        self.reference = ""
        self.roi = roi  # alias
        self._roi = roi
        self.pbar = None
        self.show_pbar = False
        self._offset = 0.0

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
        if files:
            self.read_data(files)

    def save_fsspec(self, filenames):
        base_dir = Path(filenames[0]).parent

        combined_json_path = base_dir / "combined_refs.json"

        if combined_json_path.is_file():
            # delete it, its cheap to create
            logger.debug(
                f"Removing existing combined reference file: {combined_json_path}"
            )
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
            raise ImportError(
                "Zarr is not installed. Please install it to use this method."
            )
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
            raise ImportError(
                "Zarr is not installed. Please install it to use this method."
            )
        if not Path(self.reference).is_file():
            return None
        return zarr_open(
            FsspecStore(ReferenceFileSystem(str(self.reference))),
            mode="r",
        )

    def read_data(self, filenames, dtype=np.int16):
        filenames = expand_paths(filenames)
        # try:
        #     self.save_fsspec(filenames)
        #     self.use_zarr = True
        # except Exception as e:
        #     logger.error(f"Failed to save fsspec: {e}")
        self.use_zarr = False
        self.reference = None
        super().read_data(filenames, dtype)
        self._metadata = get_metadata(
            self.tiff_files[0].filehandle.path
        )  # from the file
        self.metadata = (
            {"si": _make_json_serializable(self.tiff_files[0].scanimage_metadata)}
        )
        self._rois = self._create_rois()
        self.fields = self._create_fields()
        if self.join_contiguous:
            self._join_contiguous_fields()

    @property
    def metadata(self):
        md = self._metadata.copy()
        md.update(
            {
                "fix_phase": self.fix_phase,
                "phasecorr_method": self.phasecorr_method,
                "offset": self.offset,
                "border": self.border,
                "upsample": self.upsample,
                "max_offset": self.max_offset,
                "num_frames": self.num_frames,
            }
        )
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

    def _read_pages(
        self, frames, chans, yslice=slice(None), xslice=slice(None), **kwargs
    ):
        C = self.num_channels
        pages = [f * C + c for f in frames for c in chans]

        H = len(utils.listify_index(yslice, self._page_height))
        W = len(utils.listify_index(xslice, self._page_width))

        if getattr(self, "use_zarr", False):
            zarray = self.as_zarr()
            if zarray is not None:
                buf = np.empty((len(pages), H, W), dtype=self.dtype)
                for i, page in enumerate(pages):
                    f, c = divmod(page, C)
                    buf[i] = zarray[f, c, yslice, xslice]

                if self.fix_phase:
                    self.logger.debug(
                        f"Applying phase correction with strategy: {self.phasecorr_method}"
                    )
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
                self.logger.debug(
                    f"Applying phase correction with strategy: {self.phasecorr_method}"
                )
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
        t_key, z_key, _, _ = tuple(_convert_range_to_slice(k) for k in key) + (
            slice(None),
        ) * (4 - len(key))
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
                return tuple(
                    self.process_single_roi(roi_idx - 1, frames, chans)
                    for roi_idx in self.roi
                )
            elif self.roi == 0:
                return tuple(
                    self.process_single_roi(roi_idx, frames, chans)
                    for roi_idx in range(self.num_rois)
                )
            elif isinstance(self.roi, int):
                return self.process_single_roi(self.roi - 1, frames, chans)
        else:
            H_out, W_out = self.field_heights[0], self.field_widths[0]
            out = np.zeros((len(frames), len(chans), H_out, W_out), dtype=self.dtype)
            for roi_idx in range(self.num_rois):
                roi_data = self.process_single_roi(roi_idx, frames, chans)
                oys, oxs = (
                    self.fields[0].output_yslices[roi_idx],
                    self.fields[0].output_xslices[roi_idx],
                )
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
            roi_infos = self.tiff_files[0].scanimage_metadata["RoiGroups"][
                "imagingRoiGroup"
            ]["rois"]
        except KeyError:
            raise RuntimeError(
                "This file is not a raw-scanimage tiff or is missing tiff.scanimage_metadata."
            )
        roi_infos = roi_infos if isinstance(roi_infos, list) else [roi_infos]

        # discard empty/malformed ROIs
        roi_infos = list(
            filter(lambda r: isinstance(r["zs"], (int, float, list)), roi_infos)
        )

        # LBM uses a single depth that is not stored in metadata,
        # so force this to be 0.
        for roi_info in roi_infos:
            roi_info["zs"] = [0]

        rois = [ROI(roi_info) for roi_info in roi_infos]
        return rois

    def _create_fields(self):
        """Go over each slice depth and each roi generating the scanned fields."""
        fields = []
        previous_lines = 0
        for slice_id, scanning_depth in enumerate(self.scanning_depths):
            next_line_in_page = 0  # each slice is one tiff page
            for roi_id, roi in enumerate(self.rois):
                new_field = roi.get_field_at(scanning_depth)
                if new_field is not None:
                    print(f"Scanning depth {scanning_depth}, ROI {roi_id} ")
                    # if next_line_in_page + new_field.height > self._page_height:
                    #     error_msg = (
                    #         "Overestimated number of fly to lines ({}) at "
                    #         "scanning depth {}".format(
                    #             self._num_fly_to_lines, scanning_depth
                    #         )
                    #     )
                    #     raise RuntimeError(error_msg)

                    # Set xslice and yslice (from where in the page to cut it)
                    new_field.yslices = [
                        slice(next_line_in_page, next_line_in_page + new_field.height)
                    ]
                    new_field.xslices = [slice(0, new_field.width)]

                    # Set output xslice and yslice (where to paste it in output)
                    new_field.output_yslices = [slice(0, new_field.height)]
                    new_field.output_xslices = [slice(0, new_field.width)]

                    # Set slice and roi id
                    new_field.slice_id = slice_id
                    new_field.roi_ids = [roi_id]

                    offsets = self._compute_offsets(
                        new_field.height, previous_lines + next_line_in_page
                    )
                    new_field.offsets = [offsets]
                    next_line_in_page += new_field.height + self._num_fly_to_lines
                    fields.append(new_field)
            previous_lines += self._num_lines_between_fields
        return fields


    def __array__(self):
        """
        Convert the scan data to a NumPy array.
        Calculate the size of the scan and subsample to keep under memory limits.
        """
        return subsample_array(self, ignore_dims=[-1, -2, -3])

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
    ):
        # convert to 0 based indexing
        if isinstance(planes, int):
            planes = [planes - 1]
        elif planes is None:
            planes = list(range(self.num_planes))
        else:
            planes = [p - 1 for p in planes]
        for roi in iter_rois(self):
            for plane in planes:
                self.roi = roi
                if roi is None:
                    fname = f"plane{plane+1:02d}_stitched{ext}"
                else:
                    fname = f"plane{plane+1:02d}_roi{roi}{ext}"
                target = outpath.joinpath(fname)
                target.parent.mkdir(exist_ok=True)

                md = self.metadata.copy()
                md["plane"] = plane + 1  # back to 1-based indexing
                md["roi"] = roi
                _write_plane(
                    self,
                    target,
                    overwrite=overwrite,
                    target_chunk_mb=target_chunk_mb,
                    metadata=md,
                    progress_callback=progress_callback,
                    debug=debug,
                    dshape=(self.shape[0], self.shape[-1], self.shape[-2]),
                    plane_index=plane,
                )

    def imshow(self, **kwargs):
        arrays = []
        names = []
        # if roi is None, use a single array.roi = None
        # if roi is 0, get a list of all ROIs by deeepcopying the array and setting each roi
        for roi in iter_rois(self):
            arr = copy.copy(self)
            arr.roi = roi
            arrays.append(arr)
            names.append(f"ROI {roi}" if roi else "Full Image")

        figure_shape = (1, len(arrays))

        histogram_widget = kwargs.get("histogram_widget", True)
        figure_kwargs = kwargs.get(
            "figure_kwargs",
            {
                "size": (800, 1000),
            },
        )
        window_funcs = kwargs.get("window_funcs", None)
        return fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=histogram_widget,
            figure_kwargs=figure_kwargs,  # "canvas": canvas},
            figure_shape=figure_shape,
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs=window_funcs,
        )
