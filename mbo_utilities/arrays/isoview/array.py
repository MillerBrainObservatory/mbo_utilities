"""Lazy 5D array reader for isoview lightsheet microscopy data.

A single :class:`IsoviewArray` class handles all four output trees of
the Keller-lab IsoView pipeline. The ``kind`` argument (auto-detected by
default) selects which scanner and channel-labeling scheme to use:

- ``"corrected"`` — per-camera corrected output at
  ``<root>.corrected/SPM##/TM######/SPM##_TM######_CM##.<ext>``.
- ``"fused"`` — multi-fused output at
  ``<root>.fused/<method>/{SPM##|TM######}/
  SPM##_TM######_CM##_CM##_VW##(.fusedStack)?.<ext>``.
  Tiled projects keep an ``SPM##`` nesting; timelapse projects have
  ``TM######`` directly under the method dir.
- ``"raw"`` — raw acquisition stacks at
  ``<root>/SPC##_TM#####_ANG###_CM#_CHN##_PH#.stack``.
- ``"clusterpt"`` — clusterPT KLB outputs at
  ``<root>/TM######/SPM##_TM######_CM##_CHN##.klb``.

All four kinds expose the same ``(T, C, Z, Y, X)`` shape, indexing, and
metadata interface — the per-kind variation lives in a small registry,
not in subclasses.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from mbo_utilities.arrays._base import ReductionMixin, Shape5DMixin
from mbo_utilities.lazy_array import register_array_class
from mbo_utilities.log import get as _get_logger
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline


logger = _get_logger("arrays.isoview")


ISOVIEW_ZARR_CHUNK_ZYX: tuple[int, int, int] = (1, -1, -1)
"""Per-plane zarr chunk policy for isoview volumes, ``(z, y, x)``: one Z-plane
per chunk, full Y, full X. ``-1`` = full extent of that axis. Single source of
truth for isoview chunking; resolve against a real shape with
:func:`isoview_zarr_chunks`. ``~/repos/isoview`` imports this."""


def isoview_zarr_chunks(shape_zyx: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve :data:`ISOVIEW_ZARR_CHUNK_ZYX` against a ``(z, y, x)`` shape."""
    return tuple(
        dim if chunk < 0 else min(chunk, dim)
        for chunk, dim in zip(ISOVIEW_ZARR_CHUNK_ZYX, shape_zyx)
    )


ISOVIEW_ZARR_SHARD_TARGET_MB: int = 512
"""Target on-disk shard size for isoview volumes, in MiB.

Shards span a Z-slab (full Y, full X) sized to roughly this budget rather
than the whole volume. A whole-volume shard forces zarr to allocate the
entire array to decode *any* chunk from it (the 3.9 GiB transient that
OOM'd parallel fusion workers reading per-camera masks); a Z-slab shard
caps that transient to one slab while keeping the file count bounded (a
handful of shards per array). ``~/repos/isoview`` imports this."""


def isoview_zarr_shards(
    shape_zyx: tuple[int, ...],
    *,
    itemsize: int = 2,
    target_mb: int | None = None,
) -> tuple[int, ...]:
    """Resolve a memory-bounded ``(z, y, x)`` shard for an isoview volume.

    The shard is ``(slab_z, Y, X)`` with ``slab_z`` chosen so one shard is
    ~``target_mb`` MiB (default :data:`ISOVIEW_ZARR_SHARD_TARGET_MB`),
    clamped to ``[1, Z]``. Because the per-plane chunk has ``z == 1`` (see
    :data:`ISOVIEW_ZARR_CHUNK_ZYX`) any ``slab_z`` tiles the chunk grid
    cleanly, so callers can use the result directly as the shard shape.
    """
    z, y, x = (int(s) for s in shape_zyx[:3])
    budget = (target_mb or ISOVIEW_ZARR_SHARD_TARGET_MB) * 1024 * 1024
    plane_bytes = max(1, y * x * int(itemsize))
    slab_z = max(1, min(z, budget // plane_bytes))
    return (slab_z, y, x)


_TM_PATTERN = re.compile(r"TM(\d{5,6})")
# A tiled output dir: SPM## (legacy) or a specimen_name grid token (e.g.
# TL000 — letters then a trailing 3-digit XYZ index). Excludes TM######
# so tiled (SPM/grid) stays distinguishable from timelapse (TM) folders.
_SPM_PATTERN = re.compile(r"^(?:SPM\d+|(?!TM\d+$)[A-Za-z][A-Za-z0-9]*\d{3})$")
_FUSED_SUFFIX = ".fused"
# Match the `.fused` token at the end of a directory name, optionally
# followed by user-added qualifiers like ``.fused-dx-fixed`` or
# ``.fused_v2`` — anything starting with ``.fused`` then end-of-string
# or one of ``-/./_``. ``.fusedfoo`` deliberately does not match.
_FUSED_TAIL_RE = re.compile(r"\.fused(?:[-._].*)?$")
_CORRECTED_TAIL_RE = re.compile(r"\.corrected(?:[-._].*)?$")
_AUX_PATTERNS = (
    "Mask", "mask", "minIntensity", "coords", "Projection", "configuration",
    "transformation", "intensityCorrection", "referenceMinIntensity",
    "scores", "jobCompleted", "Background_", "transformedMask", "mask2D",
)


def _is_aux(path: Path) -> bool:
    return any(tag in path.name for tag in _AUX_PATTERNS)


def _has_tm_pattern(name: str) -> bool:
    return _TM_PATTERN.search(name) is not None


def _extract_timepoint(folder_name: str) -> int:
    m = _TM_PATTERN.search(folder_name)
    if m is None:
        raise ValueError(f"No TM pattern in folder name: {folder_name}")
    return int(m.group(1))


def _find_tm_folders(base_path: Path) -> list[Path]:
    """Return TM###### subfolders sorted by timepoint."""
    folders = []
    for d in base_path.iterdir():
        if d.is_dir() and _has_tm_pattern(d.name):
            folders.append((_extract_timepoint(d.name), d))
    folders.sort(key=lambda x: x[0])
    return [p for _, p in folders]


def _chunks_touched(shape, chunks, key) -> int:
    n = 1
    for d, c, k in zip(shape, chunks, key):
        if isinstance(k, int):
            continue
        if isinstance(k, slice):
            start, stop, _ = k.indices(d)
            if stop <= start:
                return 0
            n *= (stop - 1) // c - start // c + 1
        else:
            n *= max(1, (d + c - 1) // c)
    return n


class LazyVolume:
    """Lazy 3D volume reader for one zarr/tif/klb/stack file.

    Adapted from ``isoview/io.py``. Reports a ``(Z, Y, X)`` shape and
    forwards ``__getitem__`` to the backing format's native indexing so
    callers can read narrow Y×X slabs without materializing the full
    volume.

    .stack inputs need a ``dimensions=(width, height, depth)`` tuple
    because the raw binary carries no shape header.
    """

    def __init__(self, path: str | Path, dimensions: tuple[int, int, int] | None = None):
        self._path = Path(path)
        self._dimensions = dimensions
        self._arr = None
        self._shape: tuple[int, int, int] | None = None
        self._dtype: np.dtype | None = None
        self._attrs: dict = {}
        self._init()

    @staticmethod
    def _is_tiff(name: str) -> bool:
        n = name.lower()
        return n.endswith((".tif", ".tiff"))

    def _init(self) -> None:
        p = self._path
        if p.suffix.lower() == ".zarr":
            import zarr
            store = zarr.storage.LocalStore(str(p))
            root = zarr.open_group(store=store, mode="r")
            arr = root["0"] if "0" in root else zarr.open_array(store=store, mode="r")
            shape = tuple(int(s) for s in arr.shape)
            while len(shape) > 3 and shape[0] == 1:
                shape = shape[1:]
            self._shape = shape
            self._dtype = arr.dtype
            self._arr = arr
            try:
                self._attrs = dict(root.attrs)
            except Exception:
                self._attrs = {}
        elif self._is_tiff(p.name):
            import tifffile
            tf = tifffile.TiffFile(str(p))
            series = tf.series[0]
            shape = tuple(int(s) for s in series.shape)
            while len(shape) > 3 and shape[0] == 1:
                shape = shape[1:]
            if len(shape) == 2:
                shape = (1,) + shape
            self._shape = shape
            self._dtype = series.dtype
            self._arr = tf
        elif p.suffix.lower() == ".klb":
            import pyklb
            header = pyklb.readheader(str(p))
            dims = [int(d) for d in header["imagesize_tczyx"] if int(d) > 1]
            if len(dims) >= 3:
                self._shape = tuple(dims[-3:])
            else:
                self._shape = (1,) * (3 - len(dims)) + tuple(dims)
            self._dtype = np.dtype(header["datatype"])
            self._attrs["_klb_header"] = header
        elif p.suffix.lower() == ".stack":
            if self._dimensions is None:
                raise ValueError(f"dimensions=(W,H,D) required for .stack: {p}")
            width, height, _ = self._dimensions
            total = p.stat().st_size // 2
            depth = total // (height * width)
            self._shape = (depth, int(height), int(width))
            self._dtype = np.dtype("<u2")
            self._arr = np.memmap(p, dtype="<u2", mode="r", shape=self._shape)
        else:
            raise ValueError(f"unsupported volume format: {p.suffix}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def attrs(self) -> dict:
        return self._attrs

    @property
    def chunks(self):
        return getattr(self._arr, "chunks", None)

    def __getitem__(self, key):
        if isinstance(self._arr, (np.ndarray, np.memmap)):
            return np.asarray(self._arr[key])

        if self._is_tiff(self._path.name):
            return self._read_tiff(key)

        if self._path.suffix.lower() == ".zarr":
            ndim_extra = len(self._arr.shape) - len(self._shape)
            if isinstance(key, tuple):
                user_key = key
            else:
                user_key = (key,)
            if ndim_extra > 0:
                full_key = (0,) * ndim_extra + user_key
            else:
                full_key = user_key
            return np.asarray(self._arr[full_key])

        if self._path.suffix.lower() == ".klb":
            if self._arr is None:
                roi = self._klb_readroi(key)
                if roi is not None:
                    return roi
                import pyklb
                self._arr = pyklb.readfull(str(self._path))
            return np.asarray(self._arr[key])

        return np.asarray(self._arr[key])

    def _klb_readroi(self, key):
        """Read a contiguous (z, y, x) sub-volume via pyklb.readroi.

        Returns ``None`` when the key can't be expressed as a bounding
        box (fancy index, step != 1, empty slice) or asks for the whole
        volume — the caller falls back to readfull in those cases.
        """
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        if len(key) > 3:
            return None
        nz, ny, nx = self._shape

        bounds: list[tuple[int, int]] = []
        squeeze: list[bool] = []
        for k, n in zip(key, (nz, ny, nx)):
            if isinstance(k, (int, np.integer)):
                i = int(k) + (n if int(k) < 0 else 0)
                if i < 0 or i >= n:
                    return None
                bounds.append((i, i))
                squeeze.append(True)
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                if step != 1 or stop <= start:
                    return None
                bounds.append((start, stop - 1))
                squeeze.append(False)
            else:
                return None

        if all(b == (0, n - 1) for b, n in zip(bounds, (nz, ny, nx))):
            return None

        import pyklb
        arr = pyklb.readroi(
            str(self._path),
            [b[0] for b in bounds],
            [b[1] for b in bounds],
        )
        arr = np.asarray(arr)
        target = tuple(b[1] - b[0] + 1 for b in bounds)
        if arr.shape != target:
            arr = arr.reshape(target)
        for axis in (2, 1, 0):
            if squeeze[axis]:
                slicer = [slice(None)] * arr.ndim
                slicer[axis] = 0
                arr = arr[tuple(slicer)]
        return arr

    def _read_tiff(self, key):
        nz = self._shape[0]
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        z_key, y_key, x_key = key

        if isinstance(z_key, (int, np.integer)):
            z_idx = int(z_key) if z_key >= 0 else nz + int(z_key)
            page = self._arr.pages[z_idx].asarray()
            return page[y_key, x_key]
        if isinstance(z_key, slice):
            zs = range(*z_key.indices(nz))
        elif isinstance(z_key, (list, np.ndarray)):
            zs = list(z_key)
        else:
            zs = range(nz)
        stacked = np.stack(
            [self._arr.pages[int(zi)].asarray() for zi in zs], axis=0
        )
        return stacked[:, y_key, x_key]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = np.asarray(self[:])
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def close(self) -> None:
        close = getattr(self._arr, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self._arr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"LazyVolume({self._path.name}, shape={self._shape}, dtype={self._dtype})"


def _parse_tile_grid_position(specimen_name: str):
    """Decode a tiled ``specimen_name``'s trailing XYZ grid index.

    Tiled acquisitions name each tile ``<prefix><X><Y><Z>`` where the last
    three digits are the tile's X, Y, Z index in the grid: ``TL000`` is the
    reference tile, ``TL100`` is one tile over in X, ``TL001`` one in Z.
    Returns ``(prefix, x, y, z)`` or ``None`` when the name has no trailing
    three-digit run.
    """
    m = re.search(r"(\d{3})$", specimen_name)
    if not m:
        return None
    d = m.group(1)
    return specimen_name[: m.start()], int(d[0]), int(d[1]), int(d[2])


def _parse_isoview_xml(xml_path: Path) -> dict:
    """Parse an isoview ``push_config`` XML metadata sidecar.

    Extracts the full attribute set written by the microscope plus
    derived fields used by the metadata viewer. Mirrors the field set in
    isoview/io.py:read_xml_metadata.

    Keys produced (when present in source XML):
      data_header, specimen_name (+ parsed tile_name/tile_x/tile_y/tile_z
      when it ends in three grid digits), timestamp, time_point,
      stage_x/stage_y/stage_z (parsed from specimen_XYZT), angle,
      camera_index, camera_type,
      camera_roi, wavelength, illumination_arms, illumination_filter,
      exposure_time, detection_filter, detection_objective (+ objective_mag),
      dimensions (np.ndarray, n_cameras × 3), z_step, y_step,
      stack_direction, planes, laser_power, experiment_notes.

    Derived: zplanes, fps, vps, camera_pixel_size_um, pixel_resolution_um.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    meta: dict = {}
    CAMERA_PX_UM = 6.5

    for info in root.iter("info"):
        a = info.attrib
        # dimensions: "WxHxD" possibly comma-separated for multi-camera
        if "dimensions" in a:
            cam_dims = []
            for cam in a["dimensions"].split(","):
                parts = [int(x) for x in cam.strip().split("x")]
                if len(parts) == 3:
                    cam_dims.append(parts)
            if cam_dims:
                meta["dimensions"] = np.array(cam_dims)

        # plain attribute lifts (key, type cast)
        for key, cast in (
            ("z_step", float), ("y_step", float),
            ("exposure_time", float), ("angle", float),
            ("time_point", int), ("time_step", float),
            ("camera_pixel_pitch_um", float),
            ("specimen_name", str), ("data_header", str), ("timestamp", str),
            ("camera_index", str), ("camera", str),
            ("camera_type", str), ("camera_roi", str), ("wavelength", str),
            ("camera_orientation", str), ("magnification", str),
            ("illumination_arms", str), ("illumination_filter", str),
            ("detection_filter", str), ("detection_objective", str),
            ("stack_direction", str), ("planes", str),
            ("laser_power", str), ("experiment_notes", str),
            ("software_version", str), ("z_offset_planes", str),
        ):
            if key in a:
                try:
                    meta[key] = cast(a[key])
                except (ValueError, TypeError):
                    meta[key] = a[key]

        # stage coords from specimen_XYZT: "X=3000.000_Y=770.000_Z=770.000_T=0.0"
        if "specimen_XYZT" in a:
            try:
                parts = dict(p.split("=") for p in a["specimen_XYZT"].split("_"))
                meta["stage_x"] = float(parts.get("X", 0))
                meta["stage_y"] = float(parts.get("Y", 0))
                meta["stage_z"] = float(parts.get("Z", 0))
            except (ValueError, KeyError):
                pass

        # per-tile stride in um/tile step, e.g. "400_300_200"
        # microscope writes "tile_strides_xyz_um" (plural); older hand-injected
        # files used the singular. Accept both.
        _stride = a.get("tile_strides_xyz_um") or a.get("tile_stride_xyz_um")
        if _stride:
            meta["tile_stride_xyz_um"] = _stride
            try:
                sx, sy, sz = (float(v) for v in re.split(r"[_,]", _stride))
                meta["tile_stride_x"], meta["tile_stride_y"], meta["tile_stride_z"] = (
                    sx, sy, sz,
                )
            except (ValueError, TypeError):
                pass

    # tiled specimen_name "<prefix><X><Y><Z>" -> tile grid index per axis
    if "specimen_name" in meta:
        grid = _parse_tile_grid_position(str(meta["specimen_name"]))
        if grid is not None:
            meta["tile_name"], meta["tile_x"], meta["tile_y"], meta["tile_z"] = grid
            # physical offset from the <prefix>000 tile = grid index * stride (um)
            if {"tile_stride_x", "tile_stride_y", "tile_stride_z"} <= meta.keys():
                meta["tile_offset_x"] = grid[1] * meta["tile_stride_x"]
                meta["tile_offset_y"] = grid[2] * meta["tile_stride_y"]
                meta["tile_offset_z"] = grid[3] * meta["tile_stride_z"]

    # detection_objective magnification: "SpecialOptics 16x/0.70" -> 16.0
    if "detection_objective" in meta:
        obj = str(meta["detection_objective"]).split(",")[0]
        m = re.search(r"(\d+(?:\.\d+)?)x", obj, re.IGNORECASE)
        if m:
            meta["objective_mag"] = float(m.group(1))

    if "dimensions" in meta:
        dims = meta["dimensions"]
        if dims.ndim > 1:
            dims = dims[0]
        meta["zplanes"] = int(dims[-1])

    if "exposure_time" in meta and meta["exposure_time"] > 0:
        meta["fps"] = round(1000.0 / meta["exposure_time"], 2)
        if "zplanes" in meta:
            meta["vps"] = round(meta["fps"] / meta["zplanes"], 2)

    # pixel resolution = camera pixel pitch / magnification. Prefer the XML's
    # camera_pixel_pitch_um (per view) and magnification (per camera, first
    # value); fall back to the legacy 6.5um and detection_objective mag.
    pitch = float(meta.get("camera_pixel_pitch_um", CAMERA_PX_UM))
    meta["camera_pixel_size_um"] = pitch
    mag = None
    if "magnification" in meta:
        try:
            mag = float(str(meta["magnification"]).split(",")[0])
        except (ValueError, TypeError):
            mag = None
    if mag is None:
        mag = meta.get("objective_mag")
    if mag:
        meta["pixel_resolution_um"] = round(pitch / mag, 5)

    return meta


def _find_isoview_xml(base_path: Path) -> Path | None:
    for pattern in (
        "ch00_spec00.xml", "ch0.xml", "ch*.xml",
        "TL*_ch*.xml", "SPM*_TL*_VW*.xml", "SPM*_TM*_VW*.xml", "*.xml",
    ):
        matches = list(base_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_xml_channel(stem: str) -> int | None:
    """Infer channel number from an XML filename.

    Tries ``ch##`` (raw root), ``CHN##`` (corrected/fused copies),
    then legacy ``VW##``. Returns ``None`` when no channel marker found.
    """
    for tag in ("ch", "CHN", "VW"):
        m = re.search(rf"{tag}(\d+)", stem)
        if m:
            return int(m.group(1))
    return None


def _collect_isoview_xmls(input_dir: Path, specimen: int) -> list[Path]:
    """Find per-channel XML files under one directory.

    Glob priority (first non-empty wins):
      1. ``ch*_spec{specimen:02d}.xml`` — raw acquisition root.
      2. ``ch*.xml`` without ``_spec`` suffix — single-specimen raw root.
      3. ``*_CHN*.xml`` — corrected/fused copies in the current dir.
      4. ``*_VW*.xml`` — legacy CHN naming.
      5. Same patterns under ``SPM{specimen:02d}/``.
      6. Same patterns under each ``TM######/`` child (covers
         timelapse-corrected layouts where XML lives per timepoint).
    """
    if not input_dir.is_dir():
        return []

    out: list[Path] = []

    spec_xmls = sorted(input_dir.glob(f"ch*_spec{specimen:02d}.xml"))
    if spec_xmls:
        return spec_xmls

    for f in sorted(input_dir.glob("ch*.xml")):
        if "_spec" not in f.stem:
            out.append(f)
    if out:
        return out

    out.extend(sorted(input_dir.glob("*_CHN*.xml")))
    if out:
        return out
    out.extend(sorted(input_dir.glob("*_VW*.xml")))
    if out:
        return out

    spm_dir = input_dir / f"SPM{specimen:02d}"
    if spm_dir.is_dir():
        out.extend(sorted(spm_dir.glob("*_CHN*.xml")))
        if out:
            return out
        out.extend(sorted(spm_dir.glob("*_VW*.xml")))
        if out:
            return out

    # timelapse-corrected: XML lives inside each TM######/, identical
    # across timepoints — sample the first one.
    tm_dirs = _find_tm_folders(input_dir) if input_dir.is_dir() else []
    if not tm_dirs and spm_dir.is_dir():
        tm_dirs = _find_tm_folders(spm_dir)
    if tm_dirs:
        first_tm = tm_dirs[0]
        out.extend(sorted(first_tm.glob("*_CHN*.xml")))
        if out:
            return out
        out.extend(sorted(first_tm.glob("*_VW*.xml")))

    return out


def _meta_eq(v1, v2) -> bool:
    """Equality for metadata values, treating numpy arrays element-wise."""
    if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
        return v1.shape == v2.shape and np.allclose(v1, v2)
    return v1 == v2


def _split_common_vs_tile(
    per_tile_common: dict[int, dict],
) -> tuple[dict, dict[int, dict]]:
    """Factor per-tile metadata dicts into shared vs per-tile fields.

    A field present and equal across every tile lands in ``shared``;
    any field that is missing from a tile or differs between tiles lands
    in ``per_tile[ti]`` for each tile that carries it.
    """
    tiles = sorted(per_tile_common)
    all_keys: set[str] = set()
    for d in per_tile_common.values():
        all_keys.update(d.keys())

    shared: dict = {}
    per_tile: dict[int, dict] = {}
    for key in all_keys:
        present = [(ti, per_tile_common[ti][key]) for ti in tiles if key in per_tile_common[ti]]
        values = [v for _, v in present]
        if len(present) == len(tiles) and all(_meta_eq(values[0], v) for v in values[1:]):
            shared[key] = values[0]
        else:
            for ti, v in present:
                per_tile.setdefault(ti, {})[key] = v
    return shared, per_tile


# fields NOT split on their comma: either the comma is internal to one value,
# or the comma pair describes the camera pair as a whole (stack_direction
# "-Z,+Z" is kept intact and copied to both cameras of the view).
_NOT_PER_CAMERA_FIELDS = frozenset({
    "laser_power", "specimen_XYZT", "z_offset_planes", "y_offset_planes",
    "experiment_notes", "timestamp", "stack_direction",
})

# per-tile stride fields surfaced in each tile's metadata section
_TILE_STRIDE_KEYS = (
    "tile_stride_xyz_um", "tile_stride_x", "tile_stride_y", "tile_stride_z",
)


def _cameras_for_xml(meta: dict) -> list[int]:
    """Cameras covered by one per-view XML: from the ``camera`` attribute
    (``"01"`` -> [0, 1], ``"23"`` -> [2, 3]), else by hardware convention
    (Z-scan -> 0,1; Y-scan -> 2,3)."""
    cam_attr = str(meta.get("camera", "")).strip()
    if cam_attr.isdigit():
        return [int(c) for c in cam_attr]
    direction = str(meta.get("stack_direction", "")).upper()
    if "Z" in direction:
        return [0, 1]
    if "Y" in direction:
        return [2, 3]
    return []


def _build_cameras_views(parsed: list[dict], common_keys: set) -> dict:
    """Fold per-view XMLs into a single per-camera metadata section.

    Each XML covers a camera pair (cameras 0,1 = Z-scan; 2,3 = Y-scan).
    A comma-separated field carries one value per camera and is split into
    ``cameras[cam]``; ``stack_direction`` is kept whole (``-Z,+Z`` describes
    the pair, copied to both cameras). A non-comma field that differs between
    views (e.g. ``illumination_arms``, ``z_step``/``y_step``) is copied whole
    to each camera in that view; fields already shared across all XMLs stay in
    the common metadata (``common_keys``) and are not duplicated here.
    Per-camera ``pixel_resolution_um`` is derived from
    ``camera_pixel_pitch_um`` and the camera's ``magnification``.
    """
    cameras: dict[int, dict] = {}
    for meta in parsed:
        cams = _cameras_for_xml(meta)
        if not cams:
            continue
        for key, val in meta.items():
            if key == "camera":
                continue  # "01"/"23" pair label: resolves cams, not surfaced
            per_cam = (
                key not in _NOT_PER_CAMERA_FIELDS
                and isinstance(val, str)
                and "," in val
                and len(val.split(",")) == len(cams)
            )
            if per_cam:
                for cam, part in zip(cams, val.split(",")):
                    cameras.setdefault(cam, {})[key] = part.strip()
            elif key == "dimensions" and getattr(val, "ndim", 0) == 2 \
                    and val.shape[0] == len(cams):
                for i, cam in enumerate(cams):
                    cameras.setdefault(cam, {})["dimensions"] = val[i].tolist()
            elif key not in common_keys:
                for cam in cams:
                    cameras.setdefault(cam, {})[key] = val
        # per-camera resolution = view pixel pitch / camera magnification
        pitch = meta.get("camera_pixel_pitch_um")
        if pitch:
            for cam in cams:
                mag_str = cameras.get(cam, {}).get("magnification")
                if not mag_str:
                    continue
                try:
                    cameras[cam]["pixel_resolution_um"] = round(
                        float(pitch) / float(mag_str), 5
                    )
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
    return cameras


def _read_all_isoview_xml(
    candidate_dirs: list[Path],
    specimen: int = 0,
) -> tuple[dict, dict]:
    """Read XMLs across one or more candidate directories, split by camera.

    ``candidate_dirs`` is tried in order. The first directory yielding any
    XML files wins (downstream dirs are not merged) — this matches isoview's
    expectation that the raw acquisition root is the canonical source and
    corrected/fused copies are byte-identical fallbacks.

    Returns ``(common_metadata, per_camera_metadata)`` where:
      - ``common_metadata`` carries every field present in *all* parsed XMLs
        with a matching value (numpy arrays compared with ``np.allclose``).
      - ``per_camera_metadata[cam_idx]`` carries the per-camera fields (comma
        split) plus the per-view fields that differ between views.

    Two convenience fields are synthesized when possible:
      - ``camera_view_map``: ``{0:0, 1:0}`` for Z-scan and ``{2:90, 3:90}``
        for Y-scan, derived from ``stack_direction`` across all XMLs.
      - ``axial_step``: the first non-empty ``z_step`` or ``y_step`` value
        (same physical spacing, axis renamed by acquisition mode).
    """
    xml_files: list[Path] = []
    for d in candidate_dirs:
        if d is None:
            continue
        xml_files = _collect_isoview_xmls(d, specimen)
        if xml_files:
            break
    if not xml_files:
        return {}, {}

    parsed: list[dict] = []
    channels: list[int | None] = []
    for xml_file in xml_files:
        try:
            meta = _parse_isoview_xml(xml_file)
        except Exception as exc:
            logger.warning("failed to parse %s: %s", xml_file, exc)
            continue
        parsed.append(meta)
        channels.append(_find_xml_channel(xml_file.stem))

    if not parsed:
        return {}, {}

    # synthesize camera_view_map from stack_direction when present.
    # Fall back to the canonical 4-camera default for datasets whose XML
    # omits stack_direction but uses two scan-aligned channels (the
    # IsoView Z+Y dual-view setup, ~2015 vintage).
    camera_view_map: dict[int, int] = {}
    for meta in parsed:
        direction = str(meta.get("stack_direction", "")).upper()
        if "Z" in direction:
            camera_view_map[0] = 0
            camera_view_map[1] = 0
        elif "Y" in direction:
            camera_view_map[2] = 90
            camera_view_map[3] = 90
    if not camera_view_map and len(parsed) >= 2:
        arms_seen = {str(m.get("illumination_arms")) for m in parsed if m.get("illumination_arms") is not None}
        if len(arms_seen) >= 2:
            camera_view_map = {0: 0, 1: 0, 2: 90, 3: 90}

    all_keys: set[str] = set()
    for meta in parsed:
        all_keys.update(meta.keys())

    common: dict = {}
    for key in all_keys:
        values = [meta.get(key) for meta in parsed if key in meta]
        # common = present in EVERY parsed XML with a matching value; a field
        # in only one view (e.g. z_step on the Z-scan view) is view-specific
        # and folds into that view's cameras instead.
        if len(values) == len(parsed) and all(
            _meta_eq(values[0], v) for v in values[1:]
        ):
            common[key] = values[0]

    # per-camera section: comma fields split, view-specific fields folded in
    cameras = _build_cameras_views(parsed, set(common))

    if camera_view_map:
        common["camera_view_map"] = camera_view_map

    # z_step and y_step are the same physical spacing under different
    # acquisition orientations — collapse to axial_step for callers that
    # don't want to special-case scan direction.
    if "axial_step" not in common:
        for meta in parsed:
            if "z_step" in meta:
                common["axial_step"] = meta["z_step"]
                break
            if "y_step" in meta:
                common["axial_step"] = meta["y_step"]
                break

    return common, cameras


def _ome_block(zarr_attrs: dict) -> dict:
    """Return the OME block whether NGFF 0.5 nests under ``ome`` or not."""
    if not zarr_attrs:
        return {}
    if "ome" in zarr_attrs and isinstance(zarr_attrs["ome"], dict):
        return zarr_attrs["ome"]
    return zarr_attrs


def _extract_zarr_scale(zarr_attrs: dict) -> dict:
    """Pull dz/dy/dx + stage_x/y/z from OME-Zarr coordinateTransformations.

    Reads scale (pixel spacing) and an optional translation block (the
    isoview writer emits one when stage position is known) at the first
    multiscales level. Both NGFF v0.4 (multiscales at top level) and
    v0.5 (multiscales under ``ome``) layouts are accepted.
    """
    out: dict = {}
    block = _ome_block(zarr_attrs)
    multiscales = block.get("multiscales") if block else None
    if not multiscales:
        return out
    ms = multiscales[0] if isinstance(multiscales, list) else multiscales
    datasets = ms.get("datasets", []) if isinstance(ms, dict) else []
    if not datasets:
        return out
    transforms = datasets[0].get("coordinateTransformations", [])
    for t in transforms:
        if t.get("type") == "scale":
            scale = t.get("scale", [])
            if len(scale) >= 3:
                out["dz"] = float(scale[-3])
                out["dy"] = float(scale[-2])
                out["dx"] = float(scale[-1])
        elif t.get("type") == "translation":
            tr = t.get("translation", [])
            if len(tr) >= 3:
                out["stage_z"] = float(tr[-3])
                out["stage_y"] = float(tr[-2])
                out["stage_x"] = float(tr[-1])
    return out


def _extract_isoview_attrs(zarr_attrs: dict) -> dict:
    """Return the ``ome.isoview`` attrs block, or empty when absent.

    Carries specimen, specimen_name, timepoint, camera_pair, view,
    channel, stage_um, xml_metadata. Written by isoview's writer so
    downstream readers don't need to re-parse the XML sidecar.
    """
    block = _ome_block(zarr_attrs)
    iv = block.get("isoview") if block else None
    return dict(iv) if isinstance(iv, dict) else {}


def _extract_tiff_scale(tif) -> dict:
    """Pull dz/dy/dx and frame rate from ImageJ TIFF metadata."""
    out: dict = {}
    ij = getattr(tif, "imagej_metadata", None) or {}
    if "spacing" in ij:
        try:
            out["dz"] = float(ij["spacing"])
        except (ValueError, TypeError):
            pass
    if "finterval" in ij and ij.get("finterval", 0) > 0:
        out["fs"] = 1.0 / float(ij["finterval"])
    elif "fps" in ij:
        out["fs"] = float(ij["fps"])
    try:
        page = tif.pages[0]
        for tag, key in (("XResolution", "dx"), ("YResolution", "dy")):
            t = page.tags.get(tag)
            if t and t.value and t.value[0] > 0:
                ppum = float(t.value[0]) / float(t.value[1])
                out[key] = 1.0 / ppum
    except Exception:
        pass
    return out


def _resolve_corrected_spm_dir(p: Path) -> Path | None:
    """Locate the SPM## directory under a .corrected/ tree.

    Accepts the SPM## itself, a TM###### child, a ``.corrected/`` root,
    or any descendant inside one. Returns ``None`` when no SPM## under a
    .corrected ancestor is found.
    """
    if p.is_file():
        p = p.parent
    if _SPM_PATTERN.match(p.name):
        return p
    if _has_tm_pattern(p.name) and p.parent and _SPM_PATTERN.match(p.parent.name):
        return p.parent
    if _CORRECTED_TAIL_RE.search(p.name):
        spms = sorted(
            d for d in p.iterdir()
            if d.is_dir() and _SPM_PATTERN.match(d.name)
        )
        return spms[0] if spms else None
    for ancestor in p.parents:
        if _CORRECTED_TAIL_RE.search(ancestor.name):
            spms = sorted(
                d for d in ancestor.iterdir()
                if d.is_dir() and _SPM_PATTERN.match(d.name)
            )
            if spms:
                return spms[0]
            return None
    return None


def _is_fused_root(p: Path) -> bool:
    return _FUSED_TAIL_RE.search(p.name) is not None


def _is_method_dir(p: Path) -> bool:
    """A method dir is a directory whose parent ends in ``.fused`` and whose
    own name is NOT an SPM##/TM###### tile. (New layout drops the method
    level — SPM## sits directly under ``.fused`` — so those aren't methods.)
    """
    return (
        p.parent is not None
        and _is_fused_root(p.parent)
        and not (_SPM_PATTERN.match(p.name) or _has_tm_pattern(p.name))
    )


def _resolve_fused_method_dir(p: Path) -> Path | None:
    """Locate the ``<raw>.fused/<method>/`` directory holding fused outputs.

    Accepts the method dir itself, an ``SPM##`` child (tiled), a
    ``TM######`` child (timelapse), the ``<raw>.fused/`` root (picks
    the preferred method via :func:`_first_method_dir`), or a
    ``.corrected/`` root whose ``.fused/`` sibling exists. Returns
    ``None`` when no method dir is reachable.
    """
    if p.is_file():
        p = p.parent
    if _is_method_dir(p):
        return p
    parents = list(p.parents)
    if parents and _is_method_dir(parents[0]) and (
        _SPM_PATTERN.match(p.name) or _has_tm_pattern(p.name)
    ):
        return parents[0]
    if _is_fused_root(p) and p.is_dir():
        return _first_method_dir(p)
    m = _CORRECTED_TAIL_RE.search(p.name)
    if m:
        stem = p.name[: m.start()]
        # Prefer the literal ``.fused`` sibling; fall back to any
        # ``<stem>.fused*`` variant created with a custom suffix.
        literal = p.parent / f"{stem}{_FUSED_SUFFIX}"
        if literal.is_dir():
            return _first_method_dir(literal)
        for sibling in sorted(p.parent.iterdir()) if p.parent.is_dir() else []:
            if sibling.is_dir() and sibling.name.startswith(f"{stem}.fused"):
                if _is_fused_root(sibling):
                    return _first_method_dir(sibling)
    for ancestor in parents:
        if _is_fused_root(ancestor) and ancestor.is_dir():
            return _first_method_dir(ancestor)
    return None


def _first_method_dir(fused_root: Path) -> Path | None:
    """Pick the fused scan root under a ``<raw>.fused/`` tree.

    New layout (SPM##/TM###### directly under ``.fused``, no method dir):
    the ``.fused`` root itself is the scan root. Old layout (method-named
    subdirs): prefer ``geometric``, else the alphabetically first method.
    Returns ``None`` when nothing scannable is present.
    """
    if not fused_root.is_dir():
        return None
    candidates = sorted(
        d for d in fused_root.iterdir()
        if d.is_dir() and d.name != "projections"
    )
    if not candidates:
        return None
    # New layout: tiles sit directly under .fused — scan the root itself.
    if any(
        _SPM_PATTERN.match(d.name) or _has_tm_pattern(d.name)
        for d in candidates
    ):
        return fused_root
    for c in candidates:
        if c.name == "geometric":
            return c
    return candidates[0]


# group(1): leading tile token — SPM## (legacy/timelapse) or a tiled
# specimen_name grid token (e.g. TL000); the token has no underscore.
_CORRECTED_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_(CM|VW)(\d+)(?:_(?:CHN|CH)(\d+))?\.(?:ome\.tif|tif|tiff|ome\.zarr|zarr|klb)$"
)
# VW##_VW## is a fused camera pair (the two view angles); a lone VW## is a
# single-camera passthrough. CH## is the wavelength channel.
_FUSED_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_VW(\d+)(?:_VW(\d+))?_CH(\d+)"
    r"(?:\.fusedStack)?\.(?:ome\.tif|tif|tiff|ome\.zarr|zarr|klb)$"
)
# legacy fused naming: CM##_CM##_VW##(_CHN##); a lone CM##_VW## is single-camera.
_FUSED_LEGACY_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_CM(\d+)(?:_CM(\d+))?_(?:VW|CHN)(\d+)(?:_CHN(\d+))?"
    r"(?:\.fusedStack)?\.(?:ome\.tif|tif|tiff|ome\.zarr|zarr|klb)$"
)


def _match_fused(name: str):
    """``(tile, a0, a1, chn)`` for a fused filename (a1=-1 when single), with
    a0/a1 as view angles. Matches new VW##_VW##_CH## and legacy CM##_CM##_VW##.
    Returns ``None`` if neither matches."""
    m = _FUSED_RE.match(name)
    if m:
        a1 = int(m.group(4)) if m.group(4) is not None else -1
        return m.group(1), int(m.group(3)), a1, int(m.group(5))
    m = _FUSED_LEGACY_RE.match(name)
    if m:
        c0 = int(m.group(3))
        c1 = int(m.group(4)) if m.group(4) is not None else -1
        chn = int(m.group(6)) if m.group(6) is not None else 0
        a0 = CAMERA_VIEW_ANGLE.get(c0, c0)
        a1 = CAMERA_VIEW_ANGLE.get(c1, c1) if c1 >= 0 else -1
        return m.group(1), a0, a1, chn
    return None
_RAW_STACK_RE = re.compile(
    r"SPC(\d+)_TM(\d+)_ANG\d+_CM(\d+)_CHN(\d+)_PH\d+\.stack$"
)
_KLB_TM_RE = re.compile(r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)\.klb$")

# Camera index -> view angle (deg): the four cameras sit at 0/90/180/270 around
# the sample (CM0=VW00 ref, CM2=VW90, CM1=VW180, CM3=VW270). User-facing labels
# are VW{angle}; raw .stack/.klb filenames keep their CM# (microscope naming).
CAMERA_VIEW_ANGLE = {0: 0, 1: 180, 2: 90, 3: 270}
VIEW_ANGLE_CAMERA = {v: k for k, v in CAMERA_VIEW_ANGLE.items()}


def camera_view_label(cam: int) -> str:
    """``VW{angle}`` label for a camera index (falls back to the index)."""
    return f"VW{CAMERA_VIEW_ANGLE.get(int(cam), int(cam)):02d}"


def camera_from_view_label(label: str) -> "int | None":
    """Camera index for a ``VW{angle}`` label, or ``None`` if it isn't one."""
    m = re.search(r"VW(\d+)", str(label))
    return VIEW_ANGLE_CAMERA.get(int(m.group(1))) if m else None


def _token_to_camera(token: str, num: int) -> int:
    """Camera index from a filename view token: ``CM##`` is the literal camera
    (raw/legacy); ``VW{angle}`` maps the view angle back to the camera."""
    return int(num) if str(token).upper() == "CM" else VIEW_ANGLE_CAMERA.get(int(num), int(num))

# Per-kind filename regex whose group(1) is the specimen (tile) number.
# Used to recover the SPC/SPM id of each tile slot for per-tile XML reads.
_KIND_SPECIMEN_RE = {
    "raw": _RAW_STACK_RE,
    "corrected": _CORRECTED_RE,
    "fused": _FUSED_RE,
    "clusterpt": _KLB_TM_RE,
}

# projection group(1): leading tile token (SPM## or grid token, no underscore).
# group(3) is the camera/angle number (CM index or VW angle) — camera_view_label
# yields the right VW label for either, so the CM|VW prefix is non-capturing.
_PROJ_FLAT_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_(?:CM|VW)(\d+)(?:_(?:CHN|CH)\d+)?\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
# Capture the first token's prefix (CM|VW) so the fused view is read from the
# leading token, matching _match_fused: new VW##_VW##_CH## names put the view
# angle in the leading token (the later number is then CH), while legacy
# CM##_CM##_VW## names put a camera index there (mapped via CAMERA_VIEW_ANGLE).
_PROJ_FUSED_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_(CM|VW)(\d+)(?:_(?:CM|VW)(\d+))?_(?:VW|CHN|CH)(\d+)(?:_(?:CHN|CH)(\d+))?\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_PROJ_VW_ONLY_RE = re.compile(
    r"^([^_]+)(?:_TM(\d+))?_(?:VW|CHN)(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_AXIS_ORDER = ("xy", "xz", "yz")

# Acquisition-time MIPs written by the microscope live under
# <base>/projections/<raw_name>/ as CV (=xy, max-Z) and OV (=xz, max-Y); the
# scope writes no yz. Two redundant variants exist: per-timepoint
# MI_Projection_* and across-time Global_MI_Projection_* (no TM).
_MIC_VIEW_AXIS = {"cv": "xy", "ov": "xz"}
_MIC_PERTM_RE = re.compile(
    r"^MI_Projection_(CV|OV)_SPC(\d+)_TM(\d+)_ANG\d+_CM(\d+)_CH\d+_PH\d+\.tif$",
    re.IGNORECASE,
)
_MIC_GLOBAL_RE = re.compile(
    r"^Global_MI_Projection_(CV|OV)_CM(\d+)_SPC(\d+)_CH\d+\.tif$",
    re.IGNORECASE,
)


def _finalize_projections(
    axes: set, views: set, files: dict
) -> dict | None:
    if not files:
        return None
    return {
        "axes": [a for a in _AXIS_ORDER if a in axes],
        "views": sorted(views),
        "files": files,
    }


def _scan_flat_projections(proj_dir: Path) -> dict | None:
    """Scan a flat ``*.{raw,corrected}.projections/`` directory.

    Files look like ``SPM##_TM##_CM##.{xy,xz,yz}Projection.tif`` with no
    nested TM folders. Tiled acquisitions (single TM, multiple SPM) key
    each slot by its SPM so every spatial tile gets its own slot,
    mirroring :func:`_scan_raw`; timelapse keys by TM. Returns the
    standard ``{"axes", "views", "files"}`` dict or ``None`` when empty.
    """
    if not proj_dir.is_dir():
        return None
    parsed: list[tuple[str, int, int, str, Path]] = []
    spm_set: set[str] = set()
    tm_set: set[int] = set()
    for f in proj_dir.iterdir():
        if not f.is_file():
            continue
        m = _PROJ_FLAT_RE.match(f.name)
        if not m:
            continue
        spm, tm, cm, axis = m.groups()
        cm = int(cm)
        tm = int(tm) if tm is not None else 0
        spm_set.add(spm)
        tm_set.add(tm)
        parsed.append((spm, tm, cm, axis.lower(), f))
    if not parsed:
        return None
    use_spm = len(tm_set) == 1 and len(spm_set) > 1
    axes: set[str] = set()
    views: set[str] = set()
    files: dict[tuple[str, str, int], Path] = {}
    for spm, tm, cm, axis, f in parsed:
        view = camera_view_label(cm)
        axes.add(axis)
        views.add(view)
        files[(axis, view, spm if use_spm else tm)] = f
    return _finalize_projections(axes, views, files)


def _iter_fused_leaf_dirs(root: Path):
    """Yield ``(timepoint_or_None, leaf_dir)`` for every dir directly holding
    fused volume files, under either layout:

      new: ``<.fused>/SPM##/[TM######/]``  (mirrors corrected; no method dir)
      old: ``<.fused>/<method>/<SPM##|TM######>/``

    ``timepoint`` comes from a ``TM######`` dir name, else ``None`` (the
    timepoint is then read from the filename at scan time). ``projections/``
    and ``.zarr`` volume dirs are not descended into.
    """
    if not root.is_dir():
        return

    def _has_fused(d: Path) -> bool:
        try:
            return any(_match_fused(c.name) is not None for c in d.iterdir())
        except OSError:
            return False

    def _walk(d: Path):
        if _has_fused(d):
            tm = _extract_timepoint(d.name) if _has_tm_pattern(d.name) else None
            yield tm, d
            return
        for sub in sorted(d.iterdir()):
            if (
                sub.is_dir()
                and sub.name != "projections"
                and sub.suffix.lower() != ".zarr"
            ):
                yield from _walk(sub)

    yield from _walk(root)


def _collect_fused_proj_files(
    d: Path, tm_from_dir: int | None, axes: set, views: set, files: dict
) -> None:
    """Match fused projection TIFFs in one flat dir, accumulating results.

    ``tm_from_dir`` overrides the timepoint when the enclosing dir names it
    (legacy per-leaf layout); ``None`` reads it from the filename.
    """
    if not d.is_dir():
        return
    for f in d.iterdir():
        if not f.is_file() or "Projection" not in f.name:
            continue
        m = _PROJ_FUSED_RE.match(f.name)
        vw_only = False
        if not m:
            m = _PROJ_VW_ONLY_RE.match(f.name)
            vw_only = m is not None
        if not m:
            continue
        if vw_only:
            spm_str, tm_str, vw, axis = m.groups()
            view = f"VW{int(vw):02d}"
        else:
            spm_str, tm_str, pfx, n0, _n1, _mid, _chn, axis = m.groups()
            a0 = int(n0) if pfx.upper() == "VW" else CAMERA_VIEW_ANGLE.get(int(n0), int(n0))
            view = f"VW{a0:02d}"
        axis = axis.lower()
        # tiled projections (flat dir, no TM in name) key by the leading tile
        # token (SPM## or grid name), matching _scan_fused's per-tile slots.
        if tm_from_dir is not None:
            t = tm_from_dir
        elif tm_str is not None:
            t = int(tm_str)
        else:
            t = spm_str
        axes.add(axis)
        views.add(view)
        files[(axis, view, t)] = f


def _scan_fused_projections(method_dir: Path) -> dict | None:
    """Scan a fused method tree for projection TIFFs.

    Prefers the shared ``method_dir/projections/`` dir (alongside the
    SPM##/TM###### leaves); falls back to the legacy per-leaf layout
    ``method_dir/{SPM##|TM######}/*.Projection.tif``. Both ``CM##_CM##_VW##``
    and ``VW##``-only naming variants are matched; view is labeled by the
    fused VW number.
    """
    if not method_dir.is_dir():
        return None
    axes: set[str] = set()
    views: set[str] = set()
    files: dict[tuple[str, str, int], Path] = {}
    flat = method_dir / "projections"
    if flat.is_dir():
        _collect_fused_proj_files(flat, None, axes, views, files)
    else:
        for tm_from_dir, leaf in _iter_fused_leaf_dirs(method_dir):
            _collect_fused_proj_files(leaf, tm_from_dir, axes, views, files)
    return _finalize_projections(axes, views, files)


def _scan_corrected(spm_dir: Path):
    """Discover per-camera corrected volumes under one SPM## tree.

    Layout::

        timelapse: <root>.corrected/SPM##/TM######/SPM##_TM######_CM##(_VW##)?.<ext>
        tiled:     <root>.corrected/SPM##/SPM##_CM##(_VW##|_CHN##)?.<ext>

    Returns ``(tp_paths, view_keys, channel_names, is_tiled)`` where:
      - tp_paths: dict[int, dict[int, Path]] keyed by timepoint index
        then camera index.
      - view_keys: sorted list of camera indices.
      - channel_names: ``["CM00", "CM01", ...]``.
      - is_tiled: True when the .corrected/ root has multiple SPM##
        siblings (tiled acquisition); False otherwise.

    Tiled acquisitions fold every sibling ``SPM##`` tile into the T axis
    (one tile per timepoint index); timelapse acquisitions keep the
    single SPM## and fold its ``TM######`` folders into T.
    """
    parent = spm_dir.parent
    sibling_spms = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and _SPM_PATTERN.match(d.name)
    ) if parent.is_dir() else []
    is_tiled = (
        _CORRECTED_TAIL_RE.search(parent.name) is not None
        and len(sibling_spms) > 1
    )

    def _read_cams(folder: Path, ti: int) -> None:
        for f in sorted(folder.iterdir()):
            if not (f.is_file() or f.suffix.lower() == ".zarr"):
                continue
            if _is_aux(f):
                continue
            m = _CORRECTED_RE.match(f.name)
            if not m:
                continue
            cam = _token_to_camera(m.group(3), m.group(4))
            tp_paths.setdefault(ti, {})[cam] = f
            cams.add(cam)

    tp_paths: dict[int, dict[int, Path]] = {}
    cams: set[int] = set()

    if is_tiled:
        # one timepoint per spatial tile; cameras stay on the C axis.
        for ti, spm in enumerate(sibling_spms):
            _read_cams(spm, ti)
    else:
        if _has_tm_pattern(spm_dir.name):
            tm_dirs = [spm_dir]
        else:
            # timelapse nests TM######/ under SPM##; a non-tiled SPM with
            # the camera files directly under it is a single timepoint.
            tm_dirs = _find_tm_folders(spm_dir) or [spm_dir]
        for ti, tm in enumerate(tm_dirs):
            _read_cams(tm, ti)

    view_keys = sorted(cams)
    channel_names = [camera_view_label(c) for c in view_keys]
    return tp_paths, view_keys, channel_names, is_tiled


def _scan_fused(method_dir: Path):
    """Discover fused-view volumes under one ``<raw>.fused/<method>/`` tree.

    Layouts::

        timelapse: <method>/TM######/SPM##_TM######_CM##_CM##_VW##(.fusedStack)?.<ext>
        tiled:     <method>/SPM##/SPM##_CM##_CM##_VW##(_CHN##)?.<ext>   (no TM)

    Returns ``(tp_paths, view_keys, channel_names, is_tiled)`` where
    ``view_keys`` are ``(cam0, cam1, vw, chn)`` tuples (``chn=-1`` when
    the filename omits the trailing ``_CHN##``) and channel names are
    ``["VW00_fused", "VW00_CHN01_fused", ...]``. ``is_tiled`` is True
    when leaves are SPM## (tiled) instead of TM###### (timelapse). For
    tiled mode the SPM number drives the timepoint index (one slot per
    tile); for timelapse mode the enclosing TM dir does.
    """
    leaves = list(_iter_fused_leaf_dirs(method_dir))
    is_tiled = any(tm_from_dir is None for tm_from_dir, _ in leaves)
    if not leaves:
        return {}, [], [], is_tiled

    by_tm: dict[int, dict[tuple, Path]] = {}
    views: set[tuple[int, int, int]] = set()

    for tm_from_dir, leaf in leaves:
        for f in sorted(leaf.iterdir()):
            if not (f.is_file() or f.suffix.lower() == ".zarr"):
                continue
            if _is_aux(f):
                continue
            parsed = _match_fused(f.name)
            if parsed is None:
                continue
            tile_tok, a0, a1, chn = parsed
            # tiled leaves (no TM dir) key the slot by the leading tile token
            # (SPM## or grid name) so each tile gets its own timepoint;
            # timelapse keys by the TM dir.
            slot = tm_from_dir if tm_from_dir is not None else tile_tok
            key = (a0, a1, chn)
            by_tm.setdefault(slot, {})[key] = f
            views.add(key)

    if not by_tm:
        return {}, [], [], is_tiled

    tp_paths = {ti: by_tm[tm] for ti, tm in enumerate(sorted(by_tm))}
    view_keys = sorted(views)
    channel_names = [
        (f"VW{a0:02d}_VW{a1:02d}_CH{chn:02d}_fused" if a1 >= 0
         else f"VW{a0:02d}_CH{chn:02d}_fused")
        for a0, a1, chn in view_keys
    ]
    return tp_paths, view_keys, channel_names, is_tiled


def _scan_raw(base_path: Path):
    """Discover flat raw acquisition stacks.

    Layout::

        <root>/SPC##_TM#####_ANG###_CM#_CHN##_PH#.stack

    Requires a sibling XML metadata file (``ch00_spec00.xml`` etc.) so
    we can compute the per-volume (Z, Y, X) shape from the binary size.

    Timelapse data (multiple TM, single SPC) uses TM as the T axis.
    Tiled data (single TM, multiple SPC) uses SPC as the T axis so each
    spatial tile shows up as its own slot.
    """
    by_key: dict[int, dict[tuple[int, int], Path]] = {}
    views: set[tuple[int, int]] = set()
    spc_set: set[int] = set()
    tm_set: set[int] = set()
    parsed: list[tuple[int, int, int, int, Path]] = []
    for f in sorted(base_path.iterdir()):
        if not f.is_file():
            continue
        m = _RAW_STACK_RE.match(f.name)
        if not m:
            continue
        spc = int(m.group(1))
        tm = int(m.group(2))
        cam = int(m.group(3))
        chn = int(m.group(4))
        spc_set.add(spc)
        tm_set.add(tm)
        views.add((cam, chn))
        parsed.append((spc, tm, cam, chn, f))

    if not parsed:
        return {}, [], [], False

    use_spc = len(tm_set) == 1 and len(spc_set) > 1
    for spc, tm, cam, chn, f in parsed:
        key = spc if use_spc else tm
        by_key.setdefault(key, {})[(cam, chn)] = f

    sorted_keys = sorted(by_key)
    tp_paths = {ti: by_key[k] for ti, k in enumerate(sorted_keys)}
    view_keys = sorted(views)
    channel_names = [camera_view_label(cm) for cm, ch in view_keys]
    return tp_paths, view_keys, channel_names, use_spc


def _scan_klb_tm(base_path: Path):
    """Discover clusterPT KLB outputs.

    Layout::

        <root>/TM######/SPM##_TM######_CM##_CHN##.klb
    """
    if _has_tm_pattern(base_path.name):
        tm_dirs = [base_path]
    else:
        tm_dirs = _find_tm_folders(base_path)
    if not tm_dirs:
        return {}, [], [], False

    tp_paths: dict[int, dict[tuple, Path]] = {}
    views: set[tuple[int, int]] = set()
    for ti, tm in enumerate(tm_dirs):
        for f in sorted(tm.glob("*.klb")):
            m = _KLB_TM_RE.match(f.name)
            if not m:
                continue
            key = (int(m.group(3)), int(m.group(4)))
            tp_paths.setdefault(ti, {})[key] = f
            views.add(key)

    view_keys = sorted(views)
    channel_names = [camera_view_label(cm) for cm, ch in view_keys]
    return tp_paths, view_keys, channel_names, False


_PIPELINE_INFOS = (
    PipelineInfo(
        name="isoview-corrected",
        description="IsoView corrected per-camera output",
        input_patterns=[
            "**/*.corrected*/SPM??/TM??????/SPM??_TM??????_CM??.ome.zarr",
            "**/*.corrected*/SPM??/TM??????/SPM??_TM??????_CM??.zarr",
            "**/*.corrected*/SPM??/TM??????/SPM??_TM??????_CM??.tif",
        ],
        output_patterns=[], input_extensions=["zarr", "tif"],
        output_extensions=[], marker_files=[], category="reader",
    ),
    PipelineInfo(
        name="isoview-fused",
        description="IsoView fused output",
        input_patterns=[
            # new layout: <.fused>/SPM##/[TM######/] (no method dir)
            "**/*.fused*/SPM??/TM??????/SPM??_TM??????_CM??_CM??_VW??.*",
            "**/*.fused*/SPM??/SPM??_TM??????_CM??_CM??_VW??.*",
            # old layout: <.fused>/<method>/<SPM##|TM######>/
            "**/*.fused*/*/TM??????/*.fusedStack.*",
            "**/*.fused*/*/TM??????/SPM??_TM??????_CM??_CM??_VW??.*",
            "**/*.fused*/*/SPM??/*.fusedStack.*",
            "**/*.fused*/*/SPM??/SPM??_TM??????_CM??_CM??_VW??.*",
        ],
        output_patterns=[], input_extensions=["klb", "tif", "zarr"],
        output_extensions=[], marker_files=[], category="reader",
    ),
    PipelineInfo(
        name="isoview-raw",
        description="IsoView raw acquisition stacks",
        input_patterns=["**/SPC??_TM?????_ANG???_CM?_CHN??_PH?.stack"],
        output_patterns=[], input_extensions=["stack"],
        output_extensions=[], marker_files=["ch00_spec00.xml", "ch0.xml"],
        category="reader",
    ),
    PipelineInfo(
        name="isoview-clusterpt",
        description="IsoView clusterPT KLB output",
        input_patterns=["**/TM??????/SPM??_TM??????_CM??_CHN??.klb"],
        output_patterns=[], input_extensions=["klb"],
        output_extensions=[], marker_files=[], category="reader",
    ),
)
for _info in _PIPELINE_INFOS:
    register_pipeline(_info)


def _sibling_raw_root(arr: "IsoviewArray") -> Path | None:
    """Locate the raw acquisition root next to a ``.corrected/`` or ``.fused/`` tree.

    For arr.scan_root = ``<dataset>.corrected/SPM##`` (corrected) or
    ``<dataset>.fused/<method>/`` (fused), the sibling raw root is
    ``<dataset>/`` — the directory holding the original ``ch*_spec##.xml``
    and ``SPC##_TM*.stack`` files.

    Returns ``None`` when the path is not under a ``.corrected/`` or
    ``.fused/`` tree.
    """
    def _strip(name: str) -> str | None:
        m = _CORRECTED_TAIL_RE.search(name)
        if m:
            return name[: m.start()]
        m = _FUSED_TAIL_RE.search(name)
        if m:
            return name[: m.start()]
        return None

    candidates = [arr.scan_root, *arr.scan_root.parents]
    for d in candidates:
        stem = _strip(d.name)
        if stem is None:
            continue
        sibling = d.parent / stem
        return sibling if sibling.is_dir() else None
    return None


def _corrected_xml_dirs(arr: "IsoviewArray") -> list[Path]:
    """Candidate XML dirs for the corrected kind."""
    raw = _sibling_raw_root(arr)
    dirs: list[Path] = []
    if raw is not None:
        dirs.append(raw)
    dirs.append(arr.scan_root)
    return dirs


def _fused_xml_dirs(arr: "IsoviewArray") -> list[Path]:
    """Candidate XML dirs for the fused kind."""
    raw = _sibling_raw_root(arr)
    dirs: list[Path] = []
    if raw is not None:
        dirs.append(raw)
    # fused tree: method_dir/{SPM##|TM######}/ holds per-channel XMLs
    # alongside volumes. Take the first leaf dir under the method dir.
    for _t, leaf in _iter_fused_leaf_dirs(arr.scan_root):
        dirs.append(leaf)
        break
    return dirs


def _raw_xml_dirs(arr: "IsoviewArray") -> list[Path]:
    return [arr.scan_root]


def _klb_xml_dirs(arr: "IsoviewArray") -> list[Path]:
    """clusterPT XMLs may live at TM/SPM level; raw root if reachable."""
    dirs: list[Path] = [arr.scan_root]
    raw = _sibling_raw_root(arr)
    if raw is not None:
        dirs.append(raw)
    return dirs


def _corrected_projections(arr: "IsoviewArray") -> dict | None:
    corrected_root = arr.scan_root.parent
    if not _CORRECTED_TAIL_RE.search(corrected_root.name):
        return None
    # Projections live alongside the SPM## tiles, in <.corrected>/projections/.
    # Fall back to the legacy sibling <root>.corrected.projections/ and the
    # old per-SPM <.corrected>/SPM##/projections/ for existing datasets.
    return (
        _scan_flat_projections(corrected_root / "projections")
        or _scan_flat_projections(arr.scan_root / "projections")
        or _scan_flat_projections(
            corrected_root.parent / f"{corrected_root.name}.projections"
        )
    )


def _fused_projections(arr: "IsoviewArray") -> dict | None:
    return _scan_fused_projections(arr.scan_root)


def _raw_projections(arr: "IsoviewArray") -> dict | None:
    return _scan_flat_projections(
        arr.scan_root.parent / f"{arr.scan_root.name}.raw.projections"
    )


def _scan_microscope_projections(mic_dir: Path) -> dict | None:
    """Index acquisition-time MIPs written by the microscope, or ``None``.

    Keys are ``(axis, spc, cam, tm)`` where ``axis`` is ``xy``/``xz`` (from
    CV/OV) and ``tm`` is ``None`` for the across-time ``Global_MI`` variant,
    used as a fallback when no per-timepoint file exists.
    """
    if not mic_dir.is_dir():
        return None
    index: dict[tuple[str, int, int, int | None], Path] = {}
    for f in mic_dir.iterdir():
        if not f.is_file():
            continue
        m = _MIC_PERTM_RE.match(f.name)
        if m:
            view, spc, tm, cam = m.groups()
            index[(_MIC_VIEW_AXIS[view.lower()], int(spc), int(cam), int(tm))] = f
            continue
        m = _MIC_GLOBAL_RE.match(f.name)
        if m:
            view, cam, spc = m.groups()
            index.setdefault(
                (_MIC_VIEW_AXIS[view.lower()], int(spc), int(cam), None), f
            )
    return index or None


def _microscope_sources(index: dict, spc: int, tm: int, cam: int) -> dict | None:
    """Return ``{"xy": path, "xz": path}`` for one camera (per-timepoint file
    preferred over the across-time fallback), or ``None`` unless both axes are
    present so the caller falls back to reading the raw volume.
    """
    out = {}
    for axis in ("xy", "xz"):
        p = index.get((axis, spc, cam, tm)) or index.get((axis, spc, cam, None))
        if p is None:
            return None
        out[axis] = p
    return out


def make_raw_projections(
    raw_dir: str | Path,
    *,
    overwrite: bool = False,
    progress_callback=None,
) -> dict:
    """Write per-camera XY and XZ max-projections for a raw isoview acquisition.

    When the microscope wrote acquisition-time MIPs under
    ``<base>/projections/<raw_name>/`` (CV=xy, OV=xz), they are copied directly
    and no raw volume is read. Otherwise each ``(specimen, timepoint, camera)``
    raw ``.stack`` volume (Z, Y, X) is read once and reduced over Z/Y to the
    XY/XZ MIPs. YZ is not produced (the scope writes none; only the
    orthogonal-views preview used it). Output goes to the flat sibling
    ``<raw_dir>.raw.projections/`` — the same location and names
    :func:`_raw_projections` (and the GUI previews) read. Names are
    ``<tile>_CM##.{xy,xz}Projection.tif`` for tiled acquisitions (``<tile>`` =
    specimen_name grid token, else SPM##) and ``SPM##_TM######_CM##.…`` for
    timelapse. Existing files are skipped unless ``overwrite``.

    CHN in a raw filename marks the camera's view, not a color channel,
    so views collapse to one projection set per camera (lowest CHN wins).

    Returns ``{"dir": Path, "written": int, "skipped": int, "total": int}``
    where ``written``/``skipped`` count cameras (each writes two planes).
    """
    import tifffile

    raw_dir = Path(raw_dir)
    if raw_dir.is_file():
        raw_dir = raw_dir.parent

    xml_path = _find_isoview_xml(raw_dir)
    if xml_path is None:
        raise ValueError(f"no isoview XML sidecar with dimensions under {raw_dir}")
    xml_meta = _parse_isoview_xml(xml_path)
    if "dimensions" not in xml_meta:
        raise ValueError(f"isoview XML at {xml_path} has no dimensions field")
    w, h, _d = xml_meta["dimensions"][0]
    w, h = int(w), int(h)

    best: dict[tuple[int, int, int], tuple[int, Path]] = {}
    for f in sorted(raw_dir.iterdir()):
        if not f.is_file():
            continue
        m = _RAW_STACK_RE.match(f.name)
        if not m:
            continue
        spc, tm, cam, chn = (int(g) for g in m.groups())
        key = (spc, tm, cam)
        prev = best.get(key)
        if prev is None or chn < prev[0]:
            best[key] = (chn, f)

    proj_dir = raw_dir.parent / f"{raw_dir.name}.raw.projections"
    mic_index = _scan_microscope_projections(
        raw_dir.parent / "projections" / raw_dir.name
    )
    total = len(best)
    written = skipped = 0
    if total:
        proj_dir.mkdir(parents=True, exist_ok=True)

    # tiled (single TM, multiple SPC): name each tile by its specimen_name grid
    # token (else SPM##) and drop the TM, matching isoview's renamed outputs.
    spc_vals = {spc for (spc, _tm, _cam) in best}
    tm_vals = {tm for (_spc, tm, _cam) in best}
    tiled = len(tm_vals) == 1 and len(spc_vals) > 1
    spc_to_token: dict[int, str] = {}
    if tiled:
        for spc in spc_vals:
            name = None
            xmls = _collect_isoview_xmls(raw_dir, spc)
            if xmls:
                try:
                    name = _parse_isoview_xml(xmls[0]).get("specimen_name")
                except Exception:
                    name = None
            spc_to_token[spc] = (
                str(name)
                if name and _parse_tile_grid_position(str(name)) is not None
                else f"SPM{spc:02d}"
            )

    for i, ((spc, tm, cam), (_chn, path)) in enumerate(sorted(best.items())):
        if spc in spc_to_token:
            base = f"{spc_to_token[spc]}_CM{cam:02d}"
        else:
            base = f"SPM{spc:02d}_TM{tm:06d}_CM{cam:02d}"
        # stack is (Z, Y, X): axis 0 -> XY (max-Z), axis 1 -> XZ (max-Y)
        targets = {
            "xy": (0, proj_dir / f"{base}.xyProjection.tif"),
            "xz": (1, proj_dir / f"{base}.xzProjection.tif"),
        }
        if not overwrite and all(p.exists() for _, p in targets.values()):
            skipped += 1
            if progress_callback is not None:
                progress_callback(i + 1, total, base)
            continue
        src = _microscope_sources(mic_index, spc, tm, cam) if mic_index else None
        if src is not None:
            for axis_name, (_axis, out_path) in targets.items():
                if out_path.exists() and not overwrite:
                    continue
                tifffile.imwrite(
                    out_path, tifffile.imread(src[axis_name]), compression="zstd"
                )
        else:
            vol = np.asarray(LazyVolume(path, dimensions=(w, h, 0))[:])
            for axis, out_path in targets.values():
                if out_path.exists() and not overwrite:
                    continue
                tifffile.imwrite(
                    out_path, np.max(vol, axis=axis), compression="zstd"
                )
        written += 1
        if progress_callback is not None:
            progress_callback(i + 1, total, base)

    return {"dir": proj_dir, "written": written, "skipped": skipped, "total": total}


_KINDS: dict[str, dict] = {
    "corrected": {
        "stack_type": "isoview-corrected",
        "resolve": _resolve_corrected_spm_dir,
        "scan": _scan_corrected,
        "projections": _corrected_projections,
        "xml_dirs": _corrected_xml_dirs,
        "needs_raw_dims": False,
    },
    "fused": {
        "stack_type": "isoview-fused",
        "resolve": _resolve_fused_method_dir,
        "scan": _scan_fused,
        "projections": _fused_projections,
        "xml_dirs": _fused_xml_dirs,
        "needs_raw_dims": False,
    },
    "raw": {
        "stack_type": "isoview-raw",
        "resolve": lambda p: p if p.is_dir() else p.parent,
        "scan": _scan_raw,
        "projections": _raw_projections,
        "xml_dirs": _raw_xml_dirs,
        "needs_raw_dims": True,
    },
    "clusterpt": {
        "stack_type": "isoview-clusterpt",
        "resolve": lambda p: p if p.is_dir() else p.parent,
        "scan": _scan_klb_tm,
        "projections": None,
        "xml_dirs": _klb_xml_dirs,
        "needs_raw_dims": False,
    },
}


def detect_isoview_kind(path: str | Path) -> str | None:
    """Pick the kind string for a directory, or ``None`` when nothing matches.

    Resolution order (most specific first):
      1. inside (or at) a ``<raw>.fused/`` tree → ``"fused"``
      2. inside a ``.corrected/SPM##`` tree → ``"corrected"``
      3. ``.stack`` files at the top level → ``"raw"``
      4. TM######/*.klb tree → ``"clusterpt"``
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        p = p.parent

    if _is_fused_root(p) or any(_is_fused_root(a) for a in p.parents):
        if _resolve_fused_method_dir(p) is not None:
            return "fused"

    if _resolve_corrected_spm_dir(p) is not None:
        return "corrected"

    if any(_RAW_STACK_RE.match(f.name) for f in p.iterdir() if f.is_file()):
        return "raw"

    tm_dirs = _find_tm_folders(p) if p.is_dir() else []
    if not tm_dirs and _has_tm_pattern(p.name):
        tm_dirs = [p]
    if tm_dirs and any(d.glob("*.klb") for d in tm_dirs):
        return "clusterpt"

    return None


class IsoviewArray(ReductionMixin, Shape5DMixin):
    """Lazy ``(T, C, Z, Y, X)`` reader for any isoview output tree.

    One class, four kinds (``"corrected"``, ``"fused"``, ``"raw"``,
    ``"clusterpt"``). ``IsoviewArray(path)`` auto-detects which kind the
    path belongs to via :func:`detect_isoview_kind`; pass ``kind=`` to
    force a specific reader. Per-kind logic (scanner, path resolver,
    projections dir, channel naming) lives in the :data:`_KINDS` table —
    everything else is shared.
    """

    # imread() dispatch: wins over generic ZarrArray (corrected/fused are
    # .ome.zarr). suite2p outputs are claimed by Suite2pArray at a higher
    # priority, so a suite2p folder inside a .corrected tree stays suite2p.
    PRIORITY = 90

    @classmethod
    def can_open(cls, file: Path | str) -> bool:
        try:
            return detect_isoview_kind(file) is not None
        except Exception:
            return False

    # Pre-cache the display range so the histogram opens at a sensible
    # default for IsoView 16-bit fluorescence data instead of probing
    # the first frame (which can be slow for large fused volumes and
    # often yields a range biased by registration zero-fill).
    # _compute_frame_vminmax short-circuits when these are non-None.
    _cached_vmin = 0.0
    _cached_vmax = 1000.0

    def __init__(self, path: str | Path, kind: str | None = None):
        self.base_path = Path(path)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        if kind is None:
            kind = detect_isoview_kind(self.base_path)
            if kind is None:
                raise ValueError(
                    f"Not an isoview tree (or kind not recognized): {self.base_path}"
                )
        if kind not in _KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_KINDS)}; got {kind!r}"
            )
        self.kind: str = kind
        self._kind_cfg = _KINDS[kind]
        self.stack_type: str = self._kind_cfg["stack_type"]

        scan_root = self._kind_cfg["resolve"](self.base_path)
        if scan_root is None:
            raise ValueError(f"Not a {self.stack_type} tree: {self.base_path}")
        self.scan_root: Path = scan_root

        # raw .stack needs the XML sidecar BEFORE we can probe shape.
        self._raw_dims: tuple[int, int, int] | None = None
        self._metadata: dict = {}
        self._camera_metadata: dict[int, dict] = {}
        self._tile_metadata: dict[int, dict] = {}
        if self._kind_cfg["needs_raw_dims"]:
            self._probe_raw_xml()

        tp_paths, view_keys, channel_names, is_tiled = self._kind_cfg["scan"](scan_root)
        if not tp_paths or not view_keys:
            raise ValueError(
                f"No {self.stack_type} volumes discovered under {scan_root}"
            )

        self._tp_paths = tp_paths
        self._view_keys = list(view_keys)
        self._channel_names = list(channel_names)
        self._is_tiled: bool = bool(is_tiled)
        self._timepoints = sorted(tp_paths.keys())
        self._cache: dict[tuple[int, int], np.ndarray] = {}

        # Snapshot the (timepoint, view) → path mapping into a flat list
        # at construction time. The scanner already verified every entry
        # exists; re-stat'ing on every `arr.filenames` access turned the
        # property into 1,204+ stat calls per frame for time-lapse
        # datasets. Build once, return a defensive copy on access.
        self._filenames_snapshot: list[Path] = [
            Path(p)
            for ti in self._timepoints
            for vk in self._view_keys
            for p in (self._tp_paths.get(ti, {}).get(vk),)
            if p is not None
        ]

        first_tp = self._timepoints[0]
        first_view = self._view_keys[0]
        first_path = self._tp_paths[first_tp][first_view]
        self._probe_shape(first_path)

        # When per-view crops are applied at fusion time, sibling views
        # in the same fused tree can have different (Z, Y, X) shapes.
        # Probe each view once and store per-view shape; the global
        # ``.shape`` then reports the per-axis maximum, and short slabs
        # get zero-padded into the result slot in __getitem__.
        self._view_shapes: dict = {first_view: (self._nz, self._ny, self._nx)}
        for vk in self._view_keys[1:]:
            p = self._tp_paths[first_tp].get(vk)
            if p is None:
                continue
            try:
                with LazyVolume(p, dimensions=self._stack_dimensions()) as v:
                    self._view_shapes[vk] = tuple(int(d) for d in v.shape)
            except Exception:
                continue
        self._nz = max(s[0] for s in self._view_shapes.values())
        self._ny = max(s[1] for s in self._view_shapes.values())
        self._nx = max(s[2] for s in self._view_shapes.values())

        # XML metadata sweep — common fields merge into self._metadata,
        # per-camera fields land in self._camera_metadata. Raw kind has
        # already loaded its XML during _probe_raw_xml; running this
        # second pass is harmless because it just re-confirms the same
        # values and populates _camera_metadata with any per-channel
        # differences.
        self._load_xml_metadata()

    def _stack_dimensions(self) -> tuple[int, int, int] | None:
        """(W, H, D) for .stack inputs that need it; None for everything else."""
        return self._raw_dims

    def _probe_raw_xml(self) -> None:
        """Pull (W, H) plus metadata from the XML sidecar of a raw .stack tree."""
        xml_path = _find_isoview_xml(self.scan_root)
        if xml_path is None and self.scan_root.parent.exists():
            xml_path = _find_isoview_xml(self.scan_root.parent)
        if xml_path is None:
            raise ValueError(
                f"raw isoview at {self.scan_root}: no XML sidecar with dimensions"
            )
        xml_meta = _parse_isoview_xml(xml_path)
        self._metadata.update(xml_meta)
        if "dimensions" not in xml_meta:
            raise ValueError(
                f"raw isoview at {self.scan_root}: XML has no dimensions field"
            )
        w, h, _d = xml_meta["dimensions"][0]
        self._raw_dims = (int(w), int(h), None)

    def _load_xml_metadata(self) -> None:
        """Run the per-kind XML sweep and merge results into metadata.

        Always called from ``__init__``; silently no-ops when the kind
        has no ``xml_dirs`` entry or no XML files are discovered.
        Populates:
          - ``self._metadata`` with fields that match across all parsed XMLs
          - ``self._camera_metadata`` with per-channel fields that differ
        """
        xml_dirs_fn = self._kind_cfg.get("xml_dirs")
        if xml_dirs_fn is None:
            return
        if self._is_tiled:
            self._load_tiled_xml_metadata(xml_dirs_fn)
            return
        try:
            candidate_dirs = xml_dirs_fn(self)
        except Exception as exc:
            logger.debug("xml_dirs lookup failed for %s: %s", self.kind, exc)
            return
        if not candidate_dirs:
            return
        common, cameras = _read_all_isoview_xml(candidate_dirs)
        # don't clobber raw kind's authoritative dimensions array
        if "dimensions" in self._metadata:
            common.pop("dimensions", None)
        self._metadata.update(common)
        if cameras:
            self._camera_metadata.update(cameras)

    def _load_tiled_xml_metadata(self, xml_dirs_fn) -> None:
        """Read each tile's XML and split fields into shared/camera/tile.

        Shared fields (identical across every tile) merge into
        ``self._metadata``; per-camera fields (scan direction, illumination,
        z/y step) into ``self._camera_metadata``. Fields that differ per tile
        (stage_x/y/z, specimen_name, tile grid index) land in
        ``self._tile_metadata`` keyed by tile (T-axis) index, along with the
        per-tile stride (``tile_stride_*``).

        Reuses :func:`_read_all_isoview_xml` per tile (which already does
        the per-camera split + camera_view_map synthesis) and then folds
        the per-tile ``common`` dicts into shared vs per-tile.
        """
        try:
            base_dirs = xml_dirs_fn(self)
        except Exception as exc:
            logger.debug("xml_dirs lookup failed for %s: %s", self.kind, exc)
            base_dirs = []

        specimens = self._tile_specimen_ids()
        per_tile_common: dict[int, dict] = {}
        cameras: dict[int, dict] = {}
        for ti in self._timepoints:
            spc = specimens.get(ti, ti)
            dirs = self._tile_xml_dirs(ti, base_dirs)
            common_ti, cameras_ti = _read_all_isoview_xml(dirs, specimen=spc)
            if not common_ti and not cameras_ti:
                continue
            per_tile_common[ti] = common_ti
            # camera intrinsics are identical across tiles — keep the first
            # tile's split as the shared cameras section.
            if cameras_ti and not cameras:
                cameras = cameras_ti

        if not per_tile_common:
            return

        shared, per_tile = _split_common_vs_tile(per_tile_common)
        if "dimensions" in self._metadata:
            shared.pop("dimensions", None)
        # Surface the tile stride per tile (it is identical across tiles, so
        # _split_common_vs_tile parks it in `shared`); move it into each
        # tile's section so the strides live alongside the per-tile offsets.
        for ti, d in per_tile_common.items():
            for k in _TILE_STRIDE_KEYS:
                if k in d:
                    per_tile.setdefault(ti, {})[k] = d[k]
                shared.pop(k, None)
        # Drop tile-varying fields seeded into _metadata by the single-XML
        # probe (e.g. stage_x/y/z, specimen_name from spec00) so they don't
        # linger as stale "shared" values once they live in the tiles section.
        for key in {k for d in per_tile.values() for k in d}:
            if key != "dimensions":
                self._metadata.pop(key, None)
        self._metadata.update(shared)
        if cameras:
            self._camera_metadata.update(cameras)
        self._tile_metadata.update(per_tile)

    def _tile_specimen_ids(self) -> dict[int, int]:
        """Map each tile (T-axis) index to its specimen (SPC/SPM) number.

        Recovered from the volume filename via the per-kind specimen regex.
        Renamed tiled outputs lead with a specimen_name grid token (e.g.
        ``TL100``) that carries no integer index, so the leading token only
        yields a number for raw stacks (``SPC##``) and ``SPM##`` outputs;
        otherwise the tile index is used as the fallback specimen id.
        """
        rx = _KIND_SPECIMEN_RE.get(self.kind)
        out: dict[int, int] = {}
        for ti in self._timepoints:
            spc = None
            if rx is not None:
                for p in self._tp_paths.get(ti, {}).values():
                    if p is None:
                        continue
                    m = rx.match(Path(p).name)
                    if not m:
                        continue
                    tok = m.group(1)
                    if tok.isdigit():
                        spc = int(tok)
                    else:
                        mm = re.match(r"SP[MC](\d+)$", tok)
                        spc = int(mm.group(1)) if mm else None
                    if spc is not None:
                        break
            out[ti] = spc if spc is not None else ti
        return out

    def _tile_xml_dirs(self, ti: int, base_dirs: list[Path]) -> list[Path]:
        """Candidate XML dirs for one tile: the tile's own volume dirs
        first (where per-tile sidecars live for corrected/fused), then the
        kind's base dirs (raw root etc.) as fallback.
        """
        dirs: list[Path] = []
        seen: set[Path] = set()
        for p in self._tp_paths.get(ti, {}).values():
            if p is None:
                continue
            parent = Path(p).parent
            if parent not in seen:
                seen.add(parent)
                dirs.append(parent)
        for d in base_dirs or []:
            if d is not None and d not in seen:
                seen.add(d)
                dirs.append(d)
        return dirs

    def _probe_shape(self, sample_path: Path) -> None:
        # Finalize raw_dims depth from on-disk file size before LazyVolume needs it.
        if self._kind_cfg["needs_raw_dims"] and self._raw_dims is not None:
            w, h, _ = self._raw_dims
            total = sample_path.stat().st_size // 2
            depth = total // (h * w)
            self._raw_dims = (w, h, depth)

        with LazyVolume(sample_path, dimensions=self._stack_dimensions()) as v:
            self._nz, self._ny, self._nx = v.shape
            self._dtype = v.dtype
            if v.attrs:
                self._metadata.update(_extract_zarr_scale(v.attrs))
                iv = _extract_isoview_attrs(v.attrs)
                if iv:
                    # surface structured identifiers as top-level fields
                    # so callers (GUI metadata panel, BigStitcher submit)
                    # see specimen/timepoint/etc. without digging.
                    for key in (
                        "specimen", "specimen_name", "timepoint",
                        "camera_pair", "view", "channel",
                    ):
                        if key in iv:
                            self._metadata.setdefault(key, iv[key])
                    # merge embedded xml_metadata last so it doesn't
                    # overwrite identifiers above
                    xml_meta = iv.get("xml_metadata") or {}
                    for k, val in xml_meta.items():
                        self._metadata.setdefault(k, val)
            if v._path.suffix.lower() in (".tif", ".tiff") and v._arr is not None:
                self._metadata.update(_extract_tiff_scale(v._arr))

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (len(self._timepoints), len(self._view_keys), self._nz, self._ny, self._nx)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return self.shape

    @property
    def is_tiled(self) -> bool:
        return self._is_tiled

    def summary_stats_dim_role(self, name: str):
        """Tiled datasets fold spatial tiles into T, so T is a group (one stats
        series per tile), never collapsed. Otherwise use the default mapping."""
        from mbo_utilities.arrays.features._summary_stats import (
            StatsDimRole,
            default_dim_role,
        )

        if name.upper() == "T" and self._is_tiled:
            return (False, StatsDimRole.GROUP)
        return default_dim_role(name)

    def _summary_stats_store_path(self) -> str | None:
        """Cache dataset-level summary stats in the first zarr leaf store.

        Raw kinds back onto .klb/.tif and have no zarr attrs, so persistence
        is skipped there (None); corrected/fused are .ome.zarr.
        """
        for tp in self._timepoints:
            paths = self._tp_paths.get(tp, {})
            for vk in self._view_keys:
                p = paths.get(vk)
                if p is not None and str(p).lower().endswith(".zarr"):
                    return str(p)
        return None

    @property
    def slider_dim_labels(self) -> tuple[str, ...]:
        """User-facing slider labels for non-singleton T, C, Z axes.

        Tiled datasets fold spatial tiles into the T axis, so T is
        labeled ``"Tile"``; timepoint datasets label it ``"Timepoint"``.
        The C axis is ``"View"`` (VW##) for every kind.
        """
        t_label = "Tile" if self._is_tiled else "Timepoint"
        c_label = "View"
        labels: list[str] = []
        if len(self._timepoints) > 1:
            labels.append(t_label)
        if len(self._view_keys) > 1:
            labels.append(c_label)
        if self._nz > 1:
            labels.append("Zplane")
        return tuple(labels)

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def num_timepoints(self) -> int:
        return len(self._timepoints)

    @property
    def num_planes(self) -> int:
        return self._nz

    @property
    def num_views(self) -> int:
        return len(self._view_keys)

    @property
    def num_color_channels(self) -> int:
        """Number of distinct optical color channels (excitation wavelengths).

        A color channel is one wavelength, NOT a camera: isoview images each
        wavelength with several cameras/views, so this is usually 1 even
        though the C axis (:attr:`num_views`) holds 2-4 cameras. C-axis
        consumers (writers, consolidate) iterate :attr:`num_views`. Counted
        from distinct ``wavelength`` values across the XML sidecars; 1 when no
        wavelength metadata is present.
        """
        wavelengths = set()
        w = self._metadata.get("wavelength")
        if w not in (None, ""):
            wavelengths.add(str(w))
        for cam in self._camera_metadata.values():
            cw = cam.get("wavelength")
            if cw not in (None, ""):
                wavelengths.add(str(cw))
        return len(wavelengths) or 1

    @property
    def views(self) -> list:
        return list(self._view_keys)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def camera_metadata(self) -> dict[int, dict]:
        """Per-camera XML metadata, keyed by camera index (0-3).

        Carries the comma-separated fields split per camera (``magnification``,
        ``camera_type``, ``camera_roi``, ``detection_objective``,
        ``dimensions``, ``planes``) plus per-view fields that differ between
        views folded onto each camera (``stack_direction`` as the whole
        ``-Z,+Z`` pair, ``illumination_arms``, ``z_step``/``y_step``) and the
        derived per-camera ``pixel_resolution_um``. Fields shared across all
        cameras live on :attr:`metadata`. Empty when no XML sidecar was found.
        """
        return {k: dict(v) for k, v in self._camera_metadata.items()}

    @property
    def tile_metadata(self) -> dict[int, dict]:
        """Per-tile XML metadata, keyed by tile (T-axis) index.

        Populated only for tiled acquisitions. Carries fields that differ
        between tiles (stage_x/y/z, specimen_name,
        tile_name/tile_x/tile_y/tile_z grid index, tile_offset_*) plus the
        per-tile stride (``tile_stride_*``); fields shared across tiles live
        on :attr:`metadata` and per-camera fields on :attr:`camera_metadata`.
        Empty for timelapse / non-tiled datasets.
        """
        return {k: dict(v) for k, v in self._tile_metadata.items()}

    def tile_label(self, slot) -> str:
        """Grid-token display label (e.g. ``TL010``) for a tiled slot.

        ``slot`` is a scrubber/projection token: an ``SPM##``/``SPC##``
        string, a bare index (int or digit string), or a ``specimen_name``
        grid token. Returns the tile's ``specimen_name`` from
        :attr:`tile_metadata` when known, else the slot's own ``SPM##`` form.
        """
        m = re.match(r"^(?:SP[MC])?(\d+)$", str(slot), re.IGNORECASE)
        if m is None:
            # already a grid token (or unrecognized) — show as-is
            return str(slot)
        spec = int(m.group(1))
        ti = {v: k for k, v in self._tile_specimen_ids().items()}.get(spec, spec)
        tile = self._tile_metadata.get(ti)
        if tile and tile.get("specimen_name"):
            return str(tile["specimen_name"])
        return f"SPM{spec:02d}"

    @property
    def filenames(self) -> list[Path]:
        return list(self._filenames_snapshot)

    @property
    def metadata(self) -> dict:
        meta = dict(self._metadata)
        meta["Ly"] = self._ny
        meta["Lx"] = self._nx
        meta["num_zplanes"] = self._nz
        meta["nplanes"] = self._nz
        meta["num_planes"] = self._nz
        nt = len(self._timepoints)
        if self._is_tiled:
            # The T axis holds spatial tiles, not time samples. Report one
            # timepoint and the tile count separately so readers don't treat
            # tiles as a time series.
            meta["num_tiles"] = nt
            meta["num_timepoints"] = 1
            meta["nframes"] = 1
            meta["num_frames"] = 1
        else:
            meta["num_timepoints"] = nt
            meta["nframes"] = nt
            meta["num_frames"] = nt
        meta["num_color_channels"] = self.num_color_channels
        meta["channel_names"] = list(self._channel_names)
        meta["num_views"] = self.num_views
        meta["dtype"] = str(self._dtype)
        meta["stack_type"] = self.stack_type
        meta["pipeline_stage"] = self.stack_type
        meta["shape"] = self.shape
        meta["view_keys"] = list(self._view_keys)
        if "pixel_resolution_um" in meta:
            px = float(meta["pixel_resolution_um"])
            meta.setdefault("dx", px)
            meta.setdefault("dy", px)
        if "z_step" in meta:
            meta.setdefault("dz", float(meta["z_step"]))
        elif "axial_step" in meta:
            meta.setdefault("dz", float(meta["axial_step"]))
        elif "y_step" in meta:
            meta.setdefault("dz", float(meta["y_step"]))
        # fs (the T-axis rate) is not reliably encoded in the acquisition
        # XML, so it is left unset for the user to enter via the metadata
        # editor. fps stays available for reference but no longer seeds fs
        # automatically. vps (volumes/sec) is a temporal volume rate and is
        # undefined for a single-timepoint tiled stack, so drop it there.
        if self._is_tiled:
            meta.pop("vps", None)
            meta.pop("volumes_per_second", None)
        if self._camera_metadata:
            # The GUI metadata viewer renders a dedicated "Cameras" panel
            # from metadata["cameras"]. Surface per-camera fields there;
            # the IsoviewArray.camera_metadata property is the programmatic
            # accessor for the same data.
            meta["cameras"] = {k: dict(v) for k, v in self._camera_metadata.items()}
        if self._tile_metadata:
            # Tiled-only: the GUI metadata viewer renders a "Tiles" panel
            # from metadata["tiles"]; IsoviewArray.tile_metadata is the
            # programmatic accessor for the same per-tile data.
            meta["tiles"] = {k: dict(v) for k, v in self._tile_metadata.items()}
        return meta

    @metadata.setter
    def metadata(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError(f"metadata must be a dict, got {type(value)}")
        self._metadata.update(value)

    def __len__(self) -> int:
        return len(self._timepoints)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (5 - len(key))
        t_key, c_key, z_key, y_key, x_key = key

        nt = len(self._timepoints)
        nc = len(self._view_keys)
        t_indices = _to_indices(t_key, nt)
        c_indices = _to_indices(c_key, nc)

        z_size = _axis_size(z_key, self._nz)
        y_size = _axis_size(y_key, self._ny)
        x_size = _axis_size(x_key, self._nx)

        out_shape = (len(t_indices), len(c_indices), z_size, y_size, x_size)
        result = np.empty(out_shape, dtype=self._dtype)

        # _read_slab reads only the requested slab and never fills the
        # whole-volume cache. route single (t, view) reads through it, and
        # any single-plane read (z is an int) regardless of how many
        # timepoints are asked for. zstats strides T at a fixed Z; without
        # the single-plane carve-out each sampled timepoint would cache a
        # full Z*Y*X volume that is never released. genuine multi-plane
        # bulk reads still cache the full volume so the same (t, view) is
        # not re-decompressed at every Z.
        single_tv = len(t_indices) == 1 and len(c_indices) == 1
        single_plane = isinstance(z_key, (int, np.integer))

        for ti, t_idx in enumerate(t_indices):
            for ci, c_idx in enumerate(c_indices):
                view_key = self._view_keys[c_idx]
                path = self._tp_paths.get(self._timepoints[t_idx], {}).get(view_key)
                if path is None:
                    result[ti, ci] = 0
                    continue

                if single_tv or single_plane:
                    slab = self._read_slab(path, t_idx, c_idx, z_key, y_key, x_key)
                else:
                    vol = self._read_volume(path, t_idx, c_idx)
                    if _int_out_of_bounds((z_key, y_key, x_key), vol.shape):
                        slab = self._empty_slab(z_key, y_key, x_key, vol.shape)
                    else:
                        slab = vol[z_key, y_key, x_key]

                if isinstance(z_key, int):
                    slab = slab[np.newaxis, ...]
                if isinstance(y_key, int):
                    slab = slab[:, np.newaxis, :]
                if isinstance(x_key, int):
                    slab = slab[:, :, np.newaxis]
                # Per-view crops can make sibling views smaller than the
                # global max shape advertised in .shape — drop the slab
                # into the top-left corner of the result slot and leave
                # padding pixels at zero.
                target = result[ti, ci]
                if slab.shape == target.shape:
                    target[...] = slab
                else:
                    sz, sy, sx = (min(slab.shape[i], target.shape[i]) for i in range(3))
                    target[...] = 0
                    target[:sz, :sy, :sx] = slab[:sz, :sy, :sx]

        int_indexed = [
            isinstance(t_key, int), isinstance(c_key, int),
            isinstance(z_key, int), isinstance(y_key, int),
            isinstance(x_key, int),
        ]
        for ax in range(4, -1, -1):
            if int_indexed[ax] and result.shape[ax] == 1:
                result = np.squeeze(result, axis=ax)
        return result

    def _empty_slab(self, z_key, y_key, x_key, vol_shape) -> np.ndarray:
        def dim(k, n):
            if isinstance(k, (int, np.integer)):
                return None
            if isinstance(k, slice):
                return len(range(*k.indices(n)))
            if isinstance(k, (list, np.ndarray)):
                return len(k)
            return n

        nz, ny, nx = vol_shape
        out_dims = [d for d in (dim(z_key, nz), dim(y_key, ny), dim(x_key, nx)) if d is not None]
        if not out_dims:
            out_dims = [1]
        return np.zeros(tuple(out_dims), dtype=self._dtype)

    def _read_volume(self, path: Path, t_idx: int, c_idx: int) -> np.ndarray:
        cache_key = (t_idx, c_idx)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        with LazyVolume(path, dimensions=self._stack_dimensions()) as v:
            data = np.asarray(v[:])
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        self._cache[cache_key] = data
        return data

    def _read_slab(
        self, path: Path, t_idx: int, c_idx: int, z_key, y_key, x_key
    ) -> np.ndarray:
        cached = self._cache.get((t_idx, c_idx))
        if cached is not None:
            shape = cached.shape
            if _int_out_of_bounds((z_key, y_key, x_key), shape):
                return self._empty_slab(z_key, y_key, x_key, shape)
            return cached[z_key, y_key, x_key]
        with LazyVolume(path, dimensions=self._stack_dimensions()) as v:
            if _int_out_of_bounds((z_key, y_key, x_key), v.shape):
                return self._empty_slab(z_key, y_key, x_key, v.shape)
            slab = np.asarray(v[z_key, y_key, x_key])
            if logger.isEnabledFor(logging.DEBUG) and v.chunks is not None:
                touched = _chunks_touched(
                    v._arr.shape, v.chunks,
                    (0,) * (len(v._arr.shape) - 3) + (z_key, y_key, x_key),
                )
                chunk_bytes = int(np.prod(v.chunks)) * v.dtype.itemsize
                decompressed = touched * chunk_bytes
                logger.debug(
                    "%s narrow zarr read: t=%d c=%d returned=%d decompressed=%d ratio=%.1fx",
                    type(self).__name__, t_idx, c_idx, slab.nbytes, decompressed,
                    decompressed / max(1, slab.nbytes),
                )
        return slab

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        out = np.asarray(self[0, 0])
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def astype(self, dtype, *args, **kwargs) -> np.ndarray:
        return np.asarray(self).astype(dtype, *args, **kwargs)

    def close(self) -> None:
        self._cache.clear()

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite: bool = False,
        target_chunk_mb: int = 50,
        ext: str = ".tiff",
        progress_callback=None,
        debug: bool = False,
        planes: list[int] | int | None = None,
        **kwargs,
    ):
        from mbo_utilities.arrays._base import _imwrite_base
        return _imwrite_base(
            self, outpath, planes=planes, ext=ext, overwrite=overwrite,
            target_chunk_mb=target_chunk_mb, progress_callback=progress_callback,
            debug=debug, **kwargs,
        )

    def projections(self) -> dict | None:
        """Return projection TIFFs for this stack, or ``None`` when none exist.

        Each kind points at its own projection directory (sibling
        ``.projections/`` for raw/corrected, the Results subtree for
        fused). Result format::

            {"axes": ["xy", "xz", "yz"],
             "views": ["CM00", ...],
             "files": {(axis, view, t): Path, ...}}
        """
        fn = self._kind_cfg["projections"]
        if fn is None:
            return None
        return fn(self)

    def __repr__(self):
        return (
            f"IsoviewArray(kind={self.kind!r}, shape={self.shape}, "
            f"dtype={self.dtype}, channels={self._channel_names})"
        )


def _int_out_of_bounds(keys, shape) -> bool:
    for k, n in zip(keys, shape):
        if isinstance(k, (int, np.integer)):
            i = int(k)
            if i < 0:
                i += n
            if i < 0 or i >= n:
                return True
    return False


def _to_indices(k, max_val: int) -> list[int]:
    if isinstance(k, (int, np.integer)):
        return [int(k) if k >= 0 else max_val + int(k)]
    if isinstance(k, slice):
        return list(range(*k.indices(max_val)))
    if isinstance(k, (list, np.ndarray)):
        return list(k)
    return list(range(max_val))


def _axis_size(k, max_val: int) -> int:
    if isinstance(k, int):
        return 1
    if isinstance(k, slice):
        return len(range(*k.indices(max_val)))
    if isinstance(k, (list, np.ndarray)):
        return len(k)
    return max_val


def isoview_to_ome_zarr(
    src: str | Path,
    out: str | Path,
    *,
    kind: str | None = None,
    timepoints: list[int] | int | None = None,
    channels: list[int] | int | None = None,
    planes: list[int] | int | None = None,
    overwrite: bool = False,
    target_chunk_mb: int = 64,
    sharded: bool = True,
    compressor: str = "zstd",
    compression_level: int = 3,
    shuffle: str | None = None,
    pyramid: bool = False,
    pyramid_max_layers: int = 4,
    pyramid_method: str = "mean",
    output_suffix: str | None = None,
    progress_callback=None,
    show_progress: bool = True,
    debug: bool = False,
):
    """Convert one isoview output tree to an OME-Zarr v0.5 group.

    ``kind`` forces a specific reader (``"corrected"``, ``"fused"``,
    ``"raw"``, ``"clusterpt"``). When omitted, the path is auto-detected
    via :func:`detect_isoview_kind`.
    """
    arr = IsoviewArray(src, kind=kind)
    return arr._imwrite(
        out, ext=".zarr", overwrite=overwrite,
        target_chunk_mb=target_chunk_mb,
        progress_callback=progress_callback,
        show_progress=show_progress, debug=debug,
        planes=planes, frames=timepoints, channels=channels,
        sharded=sharded, compressor=compressor,
        compression_level=compression_level, shuffle=shuffle,
        pyramid=pyramid, pyramid_max_layers=pyramid_max_layers,
        pyramid_method=pyramid_method,
        output_suffix=output_suffix or arr.stack_type,
    )


register_array_class(IsoviewArray)
