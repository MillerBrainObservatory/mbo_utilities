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

from mbo_utilities.arrays._base import Shape5DMixin
from mbo_utilities.log import get as _get_logger
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline


logger = _get_logger("arrays.isoview")


_TM_PATTERN = re.compile(r"TM(\d{5,6})")
_SPM_PATTERN = re.compile(r"^SPM\d+$")
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

        # klb fallback — materialize then slice
        if self._arr is None and self._path.suffix.lower() == ".klb":
            import pyklb
            self._arr = pyklb.readfull(str(self._path))
        return np.asarray(self._arr[key])

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


def _parse_isoview_xml(xml_path: Path) -> dict:
    """Parse an isoview ``push_config`` XML metadata sidecar.

    Extracts the full attribute set written by the microscope plus
    derived fields used by the metadata viewer. Mirrors the field set in
    isoview/io.py:read_xml_metadata.

    Keys produced (when present in source XML):
      data_header, specimen_name, timestamp, time_point, specimen_XYZT
      (+ parsed stage_x/stage_y/stage_z), angle, camera_index, camera_type,
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
            ("specimen_name", str), ("data_header", str), ("timestamp", str),
            ("specimen_XYZT", str), ("camera_index", str),
            ("camera_type", str), ("camera_roi", str), ("wavelength", str),
            ("illumination_arms", str), ("illumination_filter", str),
            ("detection_filter", str), ("detection_objective", str),
            ("stack_direction", str), ("planes", str),
            ("laser_power", str), ("experiment_notes", str),
            ("software_version", str), ("z_offset_planes", str),
            ("specimen_drift", str),
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

    meta["camera_pixel_size_um"] = CAMERA_PX_UM
    if "objective_mag" in meta:
        meta["pixel_resolution_um"] = CAMERA_PX_UM / meta["objective_mag"]

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
      - ``common_metadata`` carries every field whose value matches across
        all parsed XMLs (numpy arrays compared with ``np.allclose``).
      - ``per_camera_metadata[cam_idx]`` carries the channel-specific
        fields when at least one XML disagrees on that key.

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

    def _eq(v1, v2) -> bool:
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return v1.shape == v2.shape and np.allclose(v1, v2)
        return v1 == v2

    all_keys: set[str] = set()
    for meta in parsed:
        all_keys.update(meta.keys())

    common: dict = {}
    per_cam: dict[int, dict] = {}
    for key in all_keys:
        values = [meta.get(key) for meta in parsed if key in meta]
        if not values:
            continue
        if all(_eq(values[0], v) for v in values[1:]):
            common[key] = values[0]
        else:
            for ch, meta in zip(channels, parsed):
                if ch is None or key not in meta:
                    continue
                per_cam.setdefault(ch, {})[key] = meta[key]

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

    return common, per_cam


def _extract_zarr_scale(zarr_attrs: dict) -> dict:
    """Pull dz/dy/dx from OME-Zarr multiscales coordinateTransformations."""
    out: dict = {}
    multiscales = zarr_attrs.get("multiscales") if zarr_attrs else None
    if not multiscales:
        return out
    ms = multiscales[0] if isinstance(multiscales, list) else multiscales
    datasets = ms.get("datasets", []) if isinstance(ms, dict) else []
    if not datasets:
        return out
    for t in datasets[0].get("coordinateTransformations", []):
        if t.get("type") == "scale":
            scale = t.get("scale", [])
            if len(scale) >= 3:
                out["dz"] = float(scale[-3])
                out["dy"] = float(scale[-2])
                out["dx"] = float(scale[-1])
            break
    return out


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
    """A method dir is any directory whose parent ends in ``.fused``."""
    return p.parent is not None and _is_fused_root(p.parent)


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
    """Pick a method dir under a ``<raw>.fused/`` root.

    Prefers ``geometric`` when present; otherwise the alphabetically
    first subdirectory. Returns ``None`` when the root has no method
    dirs yet.
    """
    if not fused_root.is_dir():
        return None
    candidates = sorted(d for d in fused_root.iterdir() if d.is_dir())
    if not candidates:
        return None
    for c in candidates:
        if c.name == "geometric":
            return c
    return candidates[0]


_CORRECTED_RE = re.compile(
    r"SPM(\d+)_TM(\d+)_CM(\d+)(?:_(?:VW|CHN)(\d+))?\.(?:ome\.tif|tif|tiff|zarr|klb)$"
)
_FUSED_RE = re.compile(
    r"SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_(?:VW|CHN)(\d+)"
    r"(?:\.fusedStack)?\.(?:ome\.tif|tif|tiff|zarr|klb)$"
)
_RAW_STACK_RE = re.compile(
    r"SPC(\d+)_TM(\d+)_ANG\d+_CM(\d+)_CHN(\d+)_PH\d+\.stack$"
)
_KLB_TM_RE = re.compile(r"SPM(\d+)_TM(\d+)_CM(\d+)_CHN(\d+)\.klb$")

_PROJ_FLAT_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.(xy|xz|yz)Projection\.tif$", re.IGNORECASE
)
_PROJ_FUSED_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_(?:VW|CHN)(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_PROJ_VW_ONLY_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_(?:VW|CHN)(\d+)\.(xy|xz|yz)Projection\.tif$",
    re.IGNORECASE,
)
_AXIS_ORDER = ("xy", "xz", "yz")


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
    nested TM folders. Returns the standard
    ``{"axes", "views", "files"}`` dict or ``None`` when empty.
    """
    axes: set[str] = set()
    views: set[str] = set()
    files: dict[tuple[str, str, int], Path] = {}
    if not proj_dir.is_dir():
        return None
    for f in proj_dir.iterdir():
        if not f.is_file():
            continue
        m = _PROJ_FLAT_RE.match(f.name)
        if not m:
            continue
        _, tm, cm, axis = m.groups()
        axis = axis.lower()
        view = f"CM{int(cm):02d}"
        axes.add(axis)
        views.add(view)
        files[(axis, view, int(tm))] = f
    return _finalize_projections(axes, views, files)


def _iter_fused_leaf_dirs(method_dir: Path):
    """Yield (timepoint, leaf_dir) for both timelapse and tiled fused layouts.

    Timelapse: ``method_dir/TM######/`` — one entry per TM, t comes
    from the TM number.
    Tiled: ``method_dir/SPM##/`` — files share a single TM (extracted
    from filenames at scan time). Yields ``(None, spm_dir)`` so callers
    derive t from each file's name.
    """
    if not method_dir.is_dir():
        return
    for sub in sorted(method_dir.iterdir()):
        if not sub.is_dir():
            continue
        if _has_tm_pattern(sub.name):
            yield _extract_timepoint(sub.name), sub
        elif _SPM_PATTERN.match(sub.name):
            yield None, sub


def _scan_fused_projections(method_dir: Path) -> dict | None:
    """Scan a fused method tree for projection TIFFs.

    Walks ``method_dir/{SPM##|TM######}/*.{xy,xz,yz}Projection.tif`` —
    both the ``CM##_CM##_VW##`` and the ``VW##``-only naming variants.
    View is labeled by the fused VW number (``VW00``, ``VW90``).
    """
    if not method_dir.is_dir():
        return None
    axes: set[str] = set()
    views: set[str] = set()
    files: dict[tuple[str, str, int], Path] = {}
    for tm_from_dir, leaf in _iter_fused_leaf_dirs(method_dir):
        for f in leaf.iterdir():
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
                _, tm_str, vw, axis = m.groups()
            else:
                _, tm_str, _cm0, _cm1, vw, axis = m.groups()
            axis = axis.lower()
            view = f"VW{int(vw):02d}"
            t = tm_from_dir if tm_from_dir is not None else int(tm_str)
            axes.add(axis)
            views.add(view)
            files[(axis, view, t)] = f
    return _finalize_projections(axes, views, files)


def _scan_corrected(spm_dir: Path):
    """Discover per-camera corrected volumes under one SPM## tree.

    Layout::

        <root>.corrected/SPM##/TM######/SPM##_TM######_CM##(_VW##)?.<ext>

    Returns ``(tp_paths, view_keys, channel_names)`` where:
      - tp_paths: dict[int, dict[int, Path]] keyed by timepoint index
        then camera index.
      - view_keys: sorted list of camera indices.
      - channel_names: ``["CM00", "CM01", ...]``.
    """
    if _has_tm_pattern(spm_dir.name):
        tm_dirs = [spm_dir]
    else:
        tm_dirs = _find_tm_folders(spm_dir)
    if not tm_dirs:
        return {}, [], []

    tp_paths: dict[int, dict[int, Path]] = {}
    cams: set[int] = set()

    for ti, tm in enumerate(tm_dirs):
        for f in sorted(tm.iterdir()):
            if not (f.is_file() or f.suffix.lower() == ".zarr"):
                continue
            if _is_aux(f):
                continue
            m = _CORRECTED_RE.match(f.name)
            if not m:
                continue
            cam = int(m.group(3))
            tp_paths.setdefault(ti, {})[cam] = f
            cams.add(cam)

    view_keys = sorted(cams)
    channel_names = [f"CM{c:02d}" for c in view_keys]
    return tp_paths, view_keys, channel_names


def _scan_fused(method_dir: Path):
    """Discover fused-view volumes under one ``<raw>.fused/<method>/`` tree.

    Layouts::

        timelapse: <method>/TM######/SPM##_TM######_CM##_CM##_VW##(.fusedStack)?.<ext>
        tiled:     <method>/SPM##/SPM##_TM######_CM##_CM##_VW##(.fusedStack)?.<ext>

    Returns ``(tp_paths, view_keys, channel_names)`` where ``view_keys``
    are ``(cam0, cam1, vw)`` tuples and channel names are
    ``["VW00_fused", ...]``. For tiled mode each file's TM number drives
    the timepoint index; for timelapse mode the enclosing TM dir does.
    """
    leaves = list(_iter_fused_leaf_dirs(method_dir))
    if not leaves:
        return {}, [], []

    by_tm: dict[int, dict[tuple, Path]] = {}
    views: set[tuple[int, int, int]] = set()

    for tm_from_dir, leaf in leaves:
        for f in sorted(leaf.iterdir()):
            if not (f.is_file() or f.suffix.lower() == ".zarr"):
                continue
            if _is_aux(f):
                continue
            m = _FUSED_RE.match(f.name)
            if not m:
                continue
            tm = tm_from_dir if tm_from_dir is not None else int(m.group(2))
            key = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
            by_tm.setdefault(tm, {})[key] = f
            views.add(key)

    if not by_tm:
        return {}, [], []

    tp_paths = {ti: by_tm[tm] for ti, tm in enumerate(sorted(by_tm))}
    view_keys = sorted(views)
    channel_names = [f"VW{vw:02d}_fused" for _, _, vw in view_keys]
    return tp_paths, view_keys, channel_names


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
        return {}, [], []

    use_spc = len(tm_set) == 1 and len(spc_set) > 1
    for spc, tm, cam, chn, f in parsed:
        key = spc if use_spc else tm
        by_key.setdefault(key, {})[(cam, chn)] = f

    sorted_keys = sorted(by_key)
    tp_paths = {ti: by_key[k] for ti, k in enumerate(sorted_keys)}
    view_keys = sorted(views)
    channel_names = [f"CM{cm}_CHN{ch:02d}" for cm, ch in view_keys]
    return tp_paths, view_keys, channel_names


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
        return {}, [], []

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
    channel_names = [f"CM{cm:02d}_CHN{ch:02d}" for cm, ch in view_keys]
    return tp_paths, view_keys, channel_names


_PIPELINE_INFOS = (
    PipelineInfo(
        name="isoview-corrected",
        description="IsoView corrected per-camera output",
        input_patterns=[
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
    return _scan_flat_projections(
        corrected_root.parent / f"{corrected_root.name}.projections"
    )


def _fused_projections(arr: "IsoviewArray") -> dict | None:
    return _scan_fused_projections(arr.scan_root)


def _raw_projections(arr: "IsoviewArray") -> dict | None:
    return _scan_flat_projections(
        arr.scan_root.parent / f"{arr.scan_root.name}.raw.projections"
    )


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


class IsoviewArray(Shape5DMixin):
    """Lazy ``(T, C, Z, Y, X)`` reader for any isoview output tree.

    One class, four kinds (``"corrected"``, ``"fused"``, ``"raw"``,
    ``"clusterpt"``). ``IsoviewArray(path)`` auto-detects which kind the
    path belongs to via :func:`detect_isoview_kind`; pass ``kind=`` to
    force a specific reader. Per-kind logic (scanner, path resolver,
    projections dir, channel naming) lives in the :data:`_KINDS` table —
    everything else is shared.
    """

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
        if self._kind_cfg["needs_raw_dims"]:
            self._probe_raw_xml()

        tp_paths, view_keys, channel_names = self._kind_cfg["scan"](scan_root)
        if not tp_paths or not view_keys:
            raise ValueError(
                f"No {self.stack_type} volumes discovered under {scan_root}"
            )

        self._tp_paths = tp_paths
        self._view_keys = list(view_keys)
        self._channel_names = list(channel_names)
        self._timepoints = sorted(tp_paths.keys())
        self._cache: dict[tuple[int, int], np.ndarray] = {}

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
        try:
            candidate_dirs = xml_dirs_fn(self)
        except Exception as exc:
            logger.debug("xml_dirs lookup failed for %s: %s", self.kind, exc)
            return
        if not candidate_dirs:
            return
        common, per_cam = _read_all_isoview_xml(candidate_dirs)
        # don't clobber raw kind's authoritative dimensions array
        if "dimensions" in self._metadata:
            common.pop("dimensions", None)
        self._metadata.update(common)
        if per_cam:
            self._camera_metadata.update(per_cam)

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
            if v._path.suffix.lower() in (".tif", ".tiff") and v._arr is not None:
                self._metadata.update(_extract_tiff_scale(v._arr))

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (len(self._timepoints), len(self._view_keys), self._nz, self._ny, self._nx)

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return self.shape

    @property
    def ndim(self) -> int:
        return 5

    @property
    def dims(self) -> tuple[str, str, str, str, str]:
        return ("T", "C", "Z", "Y", "X")

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
        return len(self._view_keys)

    @property
    def views(self) -> list:
        return list(self._view_keys)

    @property
    def channel_names(self) -> list[str]:
        return list(self._channel_names)

    @property
    def camera_metadata(self) -> dict[int, dict]:
        """Per-channel XML metadata, keyed by channel index.

        Only populated for fields that differ between channels (e.g.
        ``wavelength``, ``illumination_arms``, ``z_step`` vs ``y_step``).
        Fields that match across every channel live on
        :attr:`metadata` instead. Returns an empty dict when no XML
        sidecar was found or every field agreed.
        """
        return {k: dict(v) for k, v in self._camera_metadata.items()}

    @property
    def filenames(self) -> list[Path]:
        out: list[Path] = []
        for ti in self._timepoints:
            for vk in self._view_keys:
                p = self._tp_paths.get(ti, {}).get(vk)
                if p is not None and Path(p).exists():
                    out.append(Path(p))
        return out

    @property
    def metadata(self) -> dict:
        meta = dict(self._metadata)
        meta["Ly"] = self._ny
        meta["Lx"] = self._nx
        meta["num_zplanes"] = self._nz
        meta["nplanes"] = self._nz
        meta["num_planes"] = self._nz
        nt = len(self._timepoints)
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
        meta["views"] = list(self._view_keys)
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
        # fs is the T-axis rate. For volumetric lightsheet, that's the
        # volume rate (vps = fps / zplanes), not the per-plane camera
        # rate (fps = 1000/exposure_time_ms). Fall back to fps only when
        # zplanes is unknown / vps wasn't derived.
        if "vps" in meta:
            meta.setdefault("fs", float(meta["vps"]))
        elif "fps" in meta:
            meta.setdefault("fs", float(meta["fps"]))
        if self._camera_metadata:
            # The GUI metadata viewer renders a dedicated "Cameras" panel
            # from metadata["cameras"]. Surface per-camera fields there;
            # the IsoviewArray.camera_metadata property is the programmatic
            # accessor for the same data.
            meta["cameras"] = {k: dict(v) for k, v in self._camera_metadata.items()}
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

        # narrow zarr reads only pay off when one (t, view) is hit per call;
        # bulk reads (zstats striding T at fixed Z) benefit from the cached
        # full volume so the same (t, view) isn't re-decompressed at every Z.
        single_tv = len(t_indices) == 1 and len(c_indices) == 1

        for ti, t_idx in enumerate(t_indices):
            for ci, c_idx in enumerate(c_indices):
                view_key = self._view_keys[c_idx]
                path = self._tp_paths.get(self._timepoints[t_idx], {}).get(view_key)
                if path is None:
                    raise FileNotFoundError(
                        f"missing volume at t={t_idx} view={view_key} "
                        f"in {self.scan_root}"
                    )

                if single_tv:
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
        out = self[:]
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
