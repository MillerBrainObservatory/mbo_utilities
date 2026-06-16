"""Consolidate an isoview output tree (corrected or fused) into one OME-Zarr group.

Reads via :class:`mbo_utilities.arrays.isoview.IsoviewArray`, walks the
companion files (3D masks, 2D coordinate / depth masks, projection
TIFFs, backgrounds, raw XMLs, min-intensity NPZs), and writes one
Zarr v3 group following the OME-NGFF 0.5 spec for the spec-covered
pieces (multiscales, omero, labels). Items the spec has no first-class
home for live as top-level sibling groups (2D masks, backgrounds,
``/metadata/``) — each with its own ``multiscales`` block so Fiji can
drag-drop them directly.

Output structure (both kinds share most of this)::

    <out>.zarr/
    ├── zarr.json                            ome.{version, multiscales, omero}
    ├── 0/ .. n/                             image pyramid (T,C,Z,Y,X) uint16
    ├── projections/
    │   ├── max_xy/                          max over Z, (T,C,1,Y,X), own pyramid
    │   ├── max_xz/                          max over Y, (T,C,1,Z,X), own pyramid
    │   └── max_yz/                          max over X, (T,C,1,Z,Y), own pyramid
    ├── raw/
    │   ├── projections/
    │   │   └── max_xy/                      corrected only — disk-read raw
    │   │                                    XY proj (T,C,1,Y,X), own pyramid
    │   ├── background/                      (1,C,1,Y,X) uncompressed
    │   └── metadata/
    │       └── <stem>/                      1D uint8, verbatim XML bytes
    ├── labels/
    │   ├── zarr.json                        ome.labels = ["background_mask"]
    │   └── background_mask/                 parent-mirroring pyramid, uint8
    │                                        corrected: per-camera mask
    │                                        fused:     combined mask (cam0 ∪ cam1)
    ├── xy_mask/, xz_mask/                   corrected only — (T,C,1,d0,d1)
    ├── fusion_mask/                         fused only — (T,C,1,Y,X)
    ├── mask2D_cam0/, mask2D_cam1/           fused only — (T,C,1,d0,d1)
    ├── transformedMask2D_cam1/              fused only — (T,C,1,d0,d1)
    ├── min_intensity/                       corrected only — (T,C,2), no multiscales
    └── metadata/
        └── attrs                            common + per_camera + isoview_config
                                             (parsed/derived; raw XML bytes
                                              live at /raw/metadata/)

All projections are stored with the singleton in the **Z** slot (not in Y
or X) so the last two axes are always real spatial dimensions — required
for Fiji's 5D OME-NGFF reader to open them as images.

Every image array is written as full 5D TCZYX (singleton dims padded
for projections, backgrounds, and 2D mask groups) so Fiji's OME-NGFF
reader — which is hardcoded for 5D and crashes on lower-rank arrays —
can drag-drop-open any of them directly. The one exception is
``min_intensity``, which is bookkeeping data and carries no
multiscales block.

Public entry: :func:`consolidate_isoview` (kind dispatch).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
import zarr
from zarr.codecs import (
    BloscCodec,
    GzipCodec,
    ZstdCodec,
)

from mbo_utilities.arrays.features._pyramid import downsample_block
from mbo_utilities.arrays.isoview.array import (
    IsoviewArray,
    _extract_timepoint,
    _find_tm_folders,
    _resolve_corrected_spm_dir,
    _scan_flat_projections,
    _sibling_raw_root,
    detect_isoview_kind,
)
from mbo_utilities.log import get as _get_logger

logger = _get_logger("arrays.isoview.consolidate")


ProgressCallback = Callable[[str, int, int], None]


# === Pyramid factors ===

def _compute_anisotropic_mags(
    voxel_size: tuple[float, float, float],
    shape: tuple[int, int, int],
    max_layers: int,
    min_size: int = 64,
) -> list[tuple[int, int, int]]:
    """Per-level (Z, Y, X) downsample factors using the webknossos algorithm.

    At each step doubles the dim(s) with the smallest effective physical
    size (``mag * voxel_size``), so anisotropic data converges toward
    isotropy before downsampling further. Stops when any Y/X dim would
    fall below ``min_size``.

    Returns cumulative mag tuples starting at ``(1, 1, 1)``. For your
    typical isoview corrected layout (dz=6.22, dxy=0.325), the first
    ~4 levels are all ``(1, 2**k, 2**k)`` — Z is already coarse enough
    that the algorithm never picks it.

    Adapted from isoview/io.py:_compute_anisotropic_mags.
    """
    mags: list[tuple[int, int, int]] = [(1, 1, 1)]
    current_mag = [1, 1, 1]

    for _ in range(max_layers):
        effective = [m * v for m, v in zip(current_mag, voxel_size)]
        min_eff = min(effective)

        # candidate A: double all dims
        cand_a = [m * 2 for m in current_mag]
        eff_a = [m * v for m, v in zip(cand_a, voxel_size)]

        # candidate B: double only the smallest-effective dim(s)
        cand_b = list(current_mag)
        for i, e in enumerate(effective):
            if math.isclose(e, min_eff, rel_tol=1e-6):
                cand_b[i] *= 2
        eff_b = [m * v for m, v in zip(cand_b, voxel_size)]

        ratio_a = max(eff_a) / min(eff_a) if min(eff_a) > 0 else float("inf")
        ratio_b = max(eff_b) / min(eff_b) if min(eff_b) > 0 else float("inf")
        chosen = cand_a if ratio_a <= ratio_b else cand_b

        out_shape = [s // m for s, m in zip(shape, chosen)]
        # min_size applies to Y/X only (Z can be small at deep levels)
        if any(d < min_size for d in out_shape[1:]):
            break

        current_mag = chosen
        mags.append(tuple(current_mag))

    return mags


# === Companion-file scanning ===

_MASK_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.segmentationMask\."
    r"(?:ome\.tif|tif|tiff|zarr|klb)$"
)
_XY_MASK_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.xyMask\.(?:ome\.tif|tif|tiff)$"
)
_XZ_MASK_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.xzMask\.(?:ome\.tif|tif|tiff)$"
)
_MIN_INT_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)\.minIntensity\.npz$"
)

# Fused tree (under <raw>.fused/<method>/{SPM##|TM######}/) — pair-level
# files use the CM##_CM##_VW## naming; single-camera-in-pair files
# (mask2D, transformedMask2D) drop one CM and stay keyed by the
# remaining (cam, vw).
_FUSED_MASK_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_VW(\d+)(?:_CHN(\d+))?\.mask\."
    r"(?:ome\.tif|tif|tiff|zarr|klb)$"
)
_FUSION_MASK_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_CM(\d+)_VW(\d+)(?:_CHN(\d+))?\.fusionMask\.(?:tif|tiff)$"
)
_MASK2D_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_VW(\d+)(?:_CHN(\d+))?\.mask2D\.(?:tif|tiff)$"
)
_TRANSFORMED_MASK2D_RE = re.compile(
    r"^SPM(\d+)_TM(\d+)_CM(\d+)_VW(\d+)(?:_CHN(\d+))?\.transformedMask2D\.(?:tif|tiff)$"
)


def _scan_corrected_companions(spm_dir: Path) -> dict[str, dict[int, dict[int, Path]]]:
    """Locate companion files under ``SPM##/TM######/``.

    Returns ``{kind: {t_idx: {cam: Path}}}`` for each of
    ``segmentation``, ``xy_mask``, ``xz_mask``, ``min_intensity``.
    Missing entries are simply omitted; downstream writers treat absent
    items as "skip this subgroup".
    """
    tm_dirs = _find_tm_folders(spm_dir)
    if not tm_dirs:
        return {}

    patterns: dict[str, re.Pattern[str]] = {
        "segmentation": _MASK_RE,
        "xy_mask": _XY_MASK_RE,
        "xz_mask": _XZ_MASK_RE,
        "min_intensity": _MIN_INT_RE,
    }
    out: dict[str, dict[int, dict[int, Path]]] = {k: {} for k in patterns}

    for ti, tm in enumerate(tm_dirs):
        for f in tm.iterdir():
            for kind, pattern in patterns.items():
                m = pattern.match(f.name)
                if m:
                    cam = int(m.group(3))
                    out[kind].setdefault(ti, {})[cam] = f
                    break

    return out


def _scan_fused_companions(method_dir: Path) -> dict[str, dict[int, dict[tuple, Path]]]:
    """Locate companion files under one ``<raw>.fused/<method>/`` tree.

    Returns the four fused-specific companion kinds, keyed differently
    based on file scope:

    Pair-level keys = ``(cam0, cam1, vw, chn)`` — match the tuples
    returned by :func:`IsoviewArray(kind="fused").views`. ``chn=-1``
    when the filename omits the trailing ``_CHN##``:
      - ``"mask"``         — 3D combined ``mask.zarr``
      - ``"fusion_mask"``  — 2D blending boundary ``fusionMask.tif``

    Single-cam-in-pair keys = ``(cam, vw, chn)`` — one file per source
    camera within each pair:
      - ``"mask2D"``               — 2D depth-encoded slice mask
      - ``"transformedMask2D"``    — 2D slice mask AFTER camera transform
                                     (only the ``cam1`` of each pair)

    Layouts (both supported, mirroring :func:`_scan_fused`):
      timelapse: ``<method>/TM######/...`` — one TM per leaf dir
      tiled:     ``<method>/SPM##/...``    — TM number lives in filenames
    """
    from .array import _iter_fused_leaf_dirs, _FUSED_RE

    if not method_dir.is_dir():
        return {}

    out: dict[str, dict[int, dict[tuple, Path]]] = {
        "mask": {},
        "fusion_mask": {},
        "mask2D": {},
        "transformedMask2D": {},
    }

    leaves = list(_iter_fused_leaf_dirs(method_dir))
    if not leaves:
        return out

    # Build the canonical TM->ti mapping from the *volume* files, so
    # companion ti's stay aligned with the image array even when a
    # companion happens to be absent for some timepoint.
    volume_tms: set[int] = set()
    for tm_from_dir, leaf in leaves:
        for f in leaf.iterdir():
            m = _FUSED_RE.match(f.name)
            if m is None:
                continue
            volume_tms.add(
                tm_from_dir if tm_from_dir is not None else int(m.group(2))
            )
    if not volume_tms:
        return out
    tm_to_ti = {tm: ti for ti, tm in enumerate(sorted(volume_tms))}

    for tm_from_dir, leaf in leaves:
        for f in leaf.iterdir():
            name = f.name
            m_mask = _FUSED_MASK_RE.match(name)
            m_fus = _FUSION_MASK_RE.match(name)
            m_m2 = _MASK2D_RE.match(name)
            m_tm = _TRANSFORMED_MASK2D_RE.match(name)
            m = m_mask or m_fus or m_m2 or m_tm
            if m is None:
                continue
            tm = tm_from_dir if tm_from_dir is not None else int(m.group(2))
            ti = tm_to_ti.get(tm)
            if ti is None:
                continue
            if m_mask:
                chn = int(m_mask.group(6)) if m_mask.group(6) is not None else -1
                key = (int(m_mask.group(3)), int(m_mask.group(4)), int(m_mask.group(5)), chn)
                out["mask"].setdefault(ti, {})[key] = f
            elif m_fus:
                chn = int(m_fus.group(6)) if m_fus.group(6) is not None else -1
                key = (int(m_fus.group(3)), int(m_fus.group(4)), int(m_fus.group(5)), chn)
                out["fusion_mask"].setdefault(ti, {})[key] = f
            elif m_m2:
                chn = int(m_m2.group(5)) if m_m2.group(5) is not None else -1
                key = (int(m_m2.group(3)), int(m_m2.group(4)), chn)
                out["mask2D"].setdefault(ti, {})[key] = f
            elif m_tm:
                chn = int(m_tm.group(5)) if m_tm.group(5) is not None else -1
                key = (int(m_tm.group(3)), int(m_tm.group(4)), chn)
                out["transformedMask2D"].setdefault(ti, {})[key] = f

    return out


def _remap_single_cam_to_pair(
    per_cam_vw: dict[int, dict[tuple[int, int, int], Path]],
    pair_view_keys: list[tuple[int, int, int, int]],
    cam_position: int,  # 0 → cam0 of each pair; 1 → cam1
) -> dict[int, dict[tuple, Path]]:
    """Pivot a (cam, vw, chn)-keyed scanner to match pair-tuple view_keys.

    The mask2D / transformedMask2D scanners key by (cam, vw, chn) — one
    entry per source camera. For consolidation we want them indexed
    by the pair (cam0, cam1, vw, chn) so they line up with the channel
    axis of the consolidated array. This walks each pair, looks up
    the right single-cam entry, and rebuilds the scanner shape.
    ``chn=-1`` denotes filenames that omit the trailing ``_CHN##``.
    """
    out: dict[int, dict[tuple, Path]] = {}
    for ti, by_cam in per_cam_vw.items():
        per_pair: dict[tuple, Path] = {}
        for pair in pair_view_keys:
            cam0, cam1, vw, chn = pair
            cam = cam0 if cam_position == 0 else cam1
            p = by_cam.get((cam, vw, chn))
            if p is not None:
                per_pair[pair] = p
        if per_pair:
            out[ti] = per_pair
    return out


# === Codec construction ===

def _make_compressors(name: str, level: int, itemsize: int) -> list | None:
    """Build the compressor list passed to ``zarr.create_array``.

    ``None`` means "no compressor"; zarr will still apply its default
    bytes serialization. Used for both sharded and unsharded arrays.
    """
    if name in (None, "none"):
        return None
    if name == "gzip":
        return [GzipCodec(level=level)]
    if name == "zstd":
        return [ZstdCodec(level=level)]
    if name == "blosc-lz4":
        return [BloscCodec(cname="lz4", clevel=level, typesize=itemsize)]
    if name == "blosc-zstd":
        return [
            BloscCodec(
                cname="zstd", clevel=level, typesize=itemsize, shuffle="bitshuffle"
            )
        ]
    raise ValueError(
        f"unknown compressor {name!r}; expected one of "
        "'none', 'gzip', 'zstd', 'blosc-lz4', 'blosc-zstd'"
    )


def _v2_compressor(name: str, level: int):
    """Numcodecs codec for Zarr v2 writes (BigStitcher mirror).

    v2 has no sharding and takes a single compressor (not a list of
    codecs). Mirrors :func:`_make_compressors` for the BigStitcher
    on-the-fly path.
    """
    if name in (None, "none"):
        return None
    import numcodecs
    if name == "gzip":
        return numcodecs.GZip(level=level)
    if name == "zstd":
        return numcodecs.Zstd(level=level)
    if name == "blosc-lz4":
        return numcodecs.Blosc(cname="lz4", clevel=level)
    if name == "blosc-zstd":
        return numcodecs.Blosc(
            cname="zstd", clevel=level, shuffle=numcodecs.Blosc.BITSHUFFLE
        )
    raise ValueError(
        f"unknown compressor {name!r}; expected one of "
        "'none', 'gzip', 'zstd', 'blosc-lz4', 'blosc-zstd'"
    )


def _v2_attrs(attrs: dict) -> dict:
    """Translate any ``"version": "0.5"`` entries to ``"0.4"``.

    OME-NGFF 0.5 corresponds to Zarr v3; BigStitcher only reads 0.4 +
    Zarr v2. The omero/multiscales/labels block contents are otherwise
    identical between the two spec versions so the rest passes through.
    """
    if not isinstance(attrs, dict):
        return attrs
    out: dict = {}
    for k, v in attrs.items():
        if k == "version" and v == "0.5":
            out[k] = "0.4"
        elif isinstance(v, dict):
            out[k] = _v2_attrs(v)
        elif isinstance(v, list):
            out[k] = [_v2_attrs(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


# === Output group + array creation ===

def _open_output_group(out_path: Path, overwrite: bool):
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists (pass overwrite=True): {out_path}"
            )
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(out_path), mode="w", zarr_format=3)


def _create_sharded_array(
    group,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    compressor: str,
    compression_level: int,
):
    """Create one zarr v3 array with inner chunks wrapped in a shard.

    ``chunk_shape`` is the inner chunk size within a shard (what gets
    decompressed for one narrow read). ``shard_shape`` is the outer
    on-disk granule. zarr-python builds the ``ShardingCodec`` from the
    ``chunks=`` / ``shards=`` kwargs. When the two are equal, sharding
    is effectively a no-op and we pass ``shards=None`` to skip the
    extra metadata layer.
    """
    itemsize = np.dtype(dtype).itemsize
    compressors = _make_compressors(compressor, compression_level, itemsize)
    shards = shard_shape if tuple(shard_shape) != tuple(chunk_shape) else None
    full_name = f"{group.path.rstrip('/')}/{name}" if group.path else name
    return zarr.create_array(
        store=group.store,
        name=full_name,
        shape=shape,
        dtype=dtype,
        chunks=chunk_shape,
        shards=shards,
        compressors=compressors,
        overwrite=True,
        zarr_format=3,
        dimension_names=("t", "c", "z", "y", "x"),
    )


# === OME metadata builders ===

_AXES_5D = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _scale_5d(dt_s, dz, dy, dx, mag_zyx):
    """Build a (T, C, Z, Y, X) scale list for one pyramid level."""
    mz, my, mx = mag_zyx
    return [float(dt_s), 1.0, float(dz * mz), float(dy * my), float(dx * mx)]


def _multiscales_block(
    name: str,
    axes: list[dict],
    dataset_paths: list[str],
    scales: list[list[float]],
    method: str = "median",
) -> dict:
    """Assemble one multiscales entry."""
    datasets = [
        {"path": p, "coordinateTransformations": [{"type": "scale", "scale": s}]}
        for p, s in zip(dataset_paths, scales)
    ]
    return {
        "name": name,
        "axes": axes,
        "datasets": datasets,
        "type": method,
        "metadata": {
            "method": "mbo_utilities.arrays.features._pyramid.downsample_block",
            "downsample": method,
        },
    }


def _omero_block(channel_names: list[str], cam_metadata: dict, default_z: int) -> dict:
    """Build a per-channel omero block.

    ``label`` is the CM## name, ``window.min/max`` come from the dtype
    range, ``window.start/end`` are placeholders the GUI overrides. We
    fold per-camera fields from cam_metadata into a non-standard
    ``isoview`` sub-dict so the GUI's "Cameras" panel still has them
    without colliding with the OMERO schema.
    """
    channels = []
    for c_idx, label in enumerate(channel_names):
        cm = cam_metadata.get(c_idx, {})
        ch = {
            "label": label,
            "color": "FFFFFF",
            "active": True,
            "coefficient": 1,
            "family": "linear",
            "inverted": False,
            # start/end must DIFFER from min/max for BDV/Fiji to apply
            # contrast on open. 0..1000 is the empirical accurate range
            # for IsoView fluorescence counts; users can stretch from
            # there in the viewer if their data goes higher.
            "window": {"min": 0, "max": 65535, "start": 0, "end": 1000},
        }
        if cm:
            ch["isoview"] = {k: _json_safe(v) for k, v in cm.items()}
        channels.append(ch)
    return {
        "channels": channels,
        "rdefs": {"defaultT": 0, "defaultZ": default_z, "model": "color"},
    }


def _json_safe(v):
    """Coerce numpy scalars / arrays to JSON-encodable types for attrs."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


# === Image pyramid writer ===

def _write_image_pyramid(
    group,
    iso_arr: IsoviewArray,
    mags: list[tuple[int, int, int]],
    dx: float,
    dy: float,
    dz: float,
    dt_s: float,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> tuple[list[str], list[list[float]]]:
    """Stream the corrected image into ``/0/``..``/N/``.

    Level 0 reads each ``(t, c)`` volume from the lazy IsoviewArray
    (already chunked appropriately on disk) and writes into the new
    consolidated zarr. Higher levels read back the previous level's
    on-disk array and downsample via windowed median (webknossos
    default for intensity data).
    """
    nt, nc, nz, ny, nx = iso_arr.shape
    dtype = iso_arr.dtype
    paths: list[str] = []
    scales: list[list[float]] = []

    for level_idx, mag in enumerate(mags):
        mz, my, mx = mag
        out_nz = max(1, nz // mz)
        out_ny = max(1, ny // my)
        out_nx = max(1, nx // mx)
        shape = (nt, nc, out_nz, out_ny, out_nx)
        chunk = (1, 1, 1, out_ny, out_nx)
        shard = (1, 1, out_nz, out_ny, out_nx)

        arr = _create_sharded_array(
            group, str(level_idx), shape, dtype, chunk, shard,
            compressor, compression_level,
        )
        paths.append(str(level_idx))
        scales.append(_scale_5d(dt_s, dz, dy, dx, mag))

        if level_idx == 0:
            for ti in range(nt):
                for ci in range(nc):
                    vol = np.asarray(iso_arr[ti, ci])  # (Z, Y, X)
                    arr[ti, ci, :, :, :] = vol
                    if progress_callback:
                        progress_callback("image_l0", ti * nc + ci + 1, nt * nc)
        else:
            prev_mag = mags[level_idx - 1]
            factor = (mag[0] // prev_mag[0], mag[1] // prev_mag[1], mag[2] // prev_mag[2])
            prev_path = f"{group.path.rstrip('/')}/{level_idx - 1}" if group.path else str(level_idx - 1)
            prev_arr = zarr.open_array(store=group.store, path=prev_path, mode="r")
            for ti in range(nt):
                for ci in range(nc):
                    vol = prev_arr[ti, ci, :, :, :]
                    out = downsample_block(vol, factor, method="median")
                    # downsample_block may return shape `(out_nz, out_ny, out_nx)`
                    # one larger than what shape said when nz % mz != 0; trim.
                    out = out[:out_nz, :out_ny, :out_nx]
                    arr[ti, ci, :, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"image_l{level_idx}", ti * nc + ci + 1, nt * nc
                        )

    return paths, scales


# === Label writers ===

def _read_companion(path: Path) -> np.ndarray:
    """Read one companion file (zarr / tif / klb) as a numpy array."""
    suffix = path.suffix.lower()
    if suffix == ".zarr":
        g = zarr.open(str(path), mode="r")
        arr = g["0"] if "0" in g else g
        data = np.asarray(arr[:])
        while data.ndim > 3 and data.shape[0] == 1:
            data = data[0]
        return data
    if suffix in (".tif", ".tiff") or path.name.lower().endswith(".ome.tif"):
        return np.asarray(tifffile.imread(str(path)))
    if suffix == ".klb":
        import pyklb
        return pyklb.readfull(str(path))
    raise ValueError(f"unsupported companion format: {path}")


def _write_segmentation_pyramid(
    labels_group,
    scanner: dict[int, dict[int, Path]],
    iso_arr: IsoviewArray,
    mags: list[tuple[int, int, int]],
    dx: float,
    dy: float,
    dz: float,
    dt_s: float,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> list[str] | None:
    """Write /labels/segmentation/{0..N}/ mirroring the image pyramid.

    Per the OME spec the labels' datasets list MUST have the same
    number of entries as the parent image's. Downsampling uses the
    windowed mode (webknossos default for segmentation): a majority
    vote that never invents a label value, preserving the integer-label
    contract the spec requires for label pixels.
    """
    if not scanner:
        return None

    nt, nc, nz, ny, nx = iso_arr.shape
    seg_group = labels_group.create_group("background_mask", overwrite=True)
    paths: list[str] = []
    scales: list[list[float]] = []

    for level_idx, mag in enumerate(mags):
        mz, my, mx = mag
        out_nz = max(1, nz // mz)
        out_ny = max(1, ny // my)
        out_nx = max(1, nx // mx)
        shape = (nt, nc, out_nz, out_ny, out_nx)
        chunk = (1, 1, 1, out_ny, out_nx)
        shard = (1, 1, out_nz, out_ny, out_nx)

        arr = _create_sharded_array(
            seg_group, str(level_idx), shape, np.dtype("uint8"),
            chunk, shard, compressor, compression_level,
        )
        paths.append(str(level_idx))
        scales.append(_scale_5d(dt_s, dz, dy, dx, mag))

        if level_idx == 0:
            for ti in range(nt):
                for ci in range(nc):
                    p = scanner.get(ti, {}).get(iso_arr.views[ci])
                    if p is None:
                        continue
                    mask = _read_companion(p).astype(np.uint8)
                    arr[ti, ci, :, :, :] = mask
                    if progress_callback:
                        progress_callback("seg_l0", ti * nc + ci + 1, nt * nc)
        else:
            prev_mag = mags[level_idx - 1]
            factor = (mag[0] // prev_mag[0], mag[1] // prev_mag[1], mag[2] // prev_mag[2])
            prev_path = f"{seg_group.path.rstrip('/')}/{level_idx - 1}"
            prev_arr = zarr.open_array(store=seg_group.store, path=prev_path, mode="r")
            for ti in range(nt):
                for ci in range(nc):
                    vol = prev_arr[ti, ci, :, :, :]
                    out = downsample_block(vol, factor, method="mode")
                    out = out[:out_nz, :out_ny, :out_nx]
                    arr[ti, ci, :, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"seg_l{level_idx}", ti * nc + ci + 1, nt * nc
                        )

    # ome metadata on the background_mask subgroup
    seg_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(
                "background_mask", _AXES_5D, paths, scales, method="mode"
            )
        ],
        "image-label": {
            "version": "0.5",
            "source": {"image": "../../"},
        },
    }
    return paths


def _write_aux_2d_per_tc(
    root,
    name: str,
    scanner: dict[int, dict[int, Path]],
    iso_arr: IsoviewArray,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Write ``/<name>/0/`` — a 2D-per-(t,c) auxiliary mask at top level.

    Used for the ``xyMask`` / ``xzMask`` coordinate maps (corrected) and
    the ``fusion_mask`` / ``mask2D_*`` slice masks (fused). Each becomes
    its own multiscale group at the zarr root so Fiji's OME-NGFF reader
    can drag-drop the folder directly.

    Output shape: ``(T, C, 1, dim0, dim1)`` — singleton Z padded in so the
    Fiji OME-NGFF reader (which only handles 5D) accepts the array. The
    source 2D plane is preserved bit-exact at ``arr[t, c, 0, :, :]``.
    No pyramid — these arrays are small (~70 KB per (t, c) for a
    typical isoview dataset).
    """
    if not scanner:
        return

    # probe the first available source for shape + dtype
    sample_path = next(
        (p for per_ti in scanner.values() for p in per_ti.values()), None
    )
    if sample_path is None:
        return
    sample = _read_companion(sample_path)
    if sample.ndim == 3 and sample.shape[0] == 1:
        sample = sample[0]
    if sample.ndim != 2:
        logger.warning(
            "skipping aux %s: unexpected source ndim %d (path=%s)",
            name, sample.ndim, sample_path,
        )
        return
    dim0, dim1 = sample.shape
    src_dtype = sample.dtype

    nt = iso_arr.num_timepoints
    nc = iso_arr.num_color_channels
    shape = (nt, nc, 1, dim0, dim1)
    chunk = (1, 1, 1, dim0, dim1)

    sub = root.create_group(name, overwrite=True)
    arr = _create_sharded_array(
        sub, "0", shape, src_dtype, chunk, chunk,
        compressor, compression_level,
    )

    for ti in range(nt):
        for ci in range(nc):
            p = scanner.get(ti, {}).get(iso_arr.views[ci])
            if p is None:
                continue
            data = _read_companion(p)
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            if data.shape != (dim0, dim1):
                logger.warning(
                    "aux %s: shape mismatch at t=%d c=%d "
                    "(expected %s, got %s) — skipping",
                    name, ti, ci, (dim0, dim1), data.shape,
                )
                continue
            arr[ti, ci, 0, :, :] = data
            if progress_callback:
                progress_callback(f"aux_{name}", ti * nc + ci + 1, nt * nc)

    sub.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(
                name, _AXES_5D, ["0"], [[1.0, 1.0, 1.0, 1.0, 1.0]],
            )
        ],
    }
    sub.attrs["description"] = (
        f"Isoview-internal 2D mask per (t, c). Source plane shape: "
        f"(dim0={dim0}, dim1={dim1}); padded with singleton Z for the "
        f"Fiji 5D reader. Unit scales (axes are isoview-specific)."
    )


# === max projections ===

def _write_disk_xy_projections(
    root,
    projections: dict | None,
    iso_arr: IsoviewArray,
    tm_int_by_index: dict[int, int],
    mags_yx: list[tuple[int, int]],
    dx: float,
    dy: float,
    dt_s: float,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Read pre-computed XY projection TIFFs and write ``/raw/projections/max_xy/``.

    Used only for the corrected pipeline's *raw* projections (the only
    case where projections aren't derivable from the consolidated
    volume — they're max-projections of the raw acquisition stacks,
    not the corrected output). For projections of the corrected /
    fused volume, use :func:`_write_computed_projections` instead.

    Output: ``/raw/projections/max_xy/{0..N}/`` shape ``(T, C, 1, Y, X)``
    — 5D with a singleton Z so Fiji's 5D-only OME-NGFF reader can
    drag-drop the folder directly.
    """
    if not projections or not projections.get("files"):
        return

    nt = iso_arr.num_timepoints
    channel_names = list(iso_arr.channel_names)
    nc = len(channel_names)

    # discover ny, nx, dtype from the first available file
    sample_path = next(iter(projections["files"].values()))
    sample = tifffile.imread(str(sample_path))
    if sample.ndim != 2:
        sample = np.squeeze(sample)
    ny, nx = sample.shape
    dtype = sample.dtype

    # ``raw/projections/`` may already exist if a future raw-XZ/YZ writer
    # gets added later; require_group is a get-or-create.
    raw_group = root.require_group("raw")
    raw_projs = raw_group.require_group("projections")
    xy_group = raw_projs.create_group("max_xy", overwrite=True)

    paths: list[str] = []
    scales: list[list[float]] = []

    for level_idx, mag in enumerate(mags_yx):
        my, mx = mag
        out_ny = max(1, ny // my)
        out_nx = max(1, nx // mx)
        shape = (nt, nc, 1, out_ny, out_nx)
        chunk = (1, 1, 1, out_ny, out_nx)
        shard = chunk
        arr = _create_sharded_array(
            xy_group, str(level_idx), shape, dtype, chunk, shard,
            compressor, compression_level,
        )
        paths.append(str(level_idx))
        # 5D scale with Z mag fixed at 1 (singleton).
        scales.append(_scale_5d(dt_s, 1.0, dy, dx, (1, my, mx)))

        if level_idx == 0:
            for ti in range(nt):
                tm_int = tm_int_by_index.get(ti)
                if tm_int is None:
                    continue
                # projections["files"] is keyed by (axis, view_label, tm_int);
                # view labels for corrected/raw flat scans are "CM##".
                for ci in range(nc):
                    label = channel_names[ci]
                    p = projections["files"].get(("xy", label, tm_int))
                    if p is None:
                        continue
                    img = np.asarray(tifffile.imread(str(p)))
                    if img.ndim != 2:
                        img = np.squeeze(img)
                    arr[ti, ci, 0, :, :] = img.astype(dtype, copy=False)
                    if progress_callback:
                        progress_callback(
                            "raw_max_xy_l0",
                            ti * nc + ci + 1, nt * nc,
                        )
        else:
            prev_path = f"{xy_group.path.rstrip('/')}/{level_idx - 1}"
            prev_arr = zarr.open_array(store=xy_group.store, path=prev_path, mode="r")
            prev_mag = mags_yx[level_idx - 1]
            factor2 = (mag[0] // prev_mag[0], mag[1] // prev_mag[1])
            for ti in range(nt):
                for ci in range(nc):
                    img = prev_arr[ti, ci, 0, :, :]
                    out = downsample_block(img, factor2, method="median")
                    out = out[:out_ny, :out_nx]
                    arr[ti, ci, 0, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"raw_max_xy_l{level_idx}",
                            ti * nc + ci + 1, nt * nc,
                        )

    xy_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block("max_xy", _AXES_5D, paths, scales)
        ],
    }


def _write_computed_projections(
    root,
    iso_arr: IsoviewArray,
    image_mags: list[tuple[int, int, int]],
    dx: float,
    dy: float,
    dz: float,
    dt_s: float,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Compute XY/XZ/YZ max-projections and write under ``/projections/``.

    Each (t, c) volume is read **once** from the lazy IsoviewArray and
    reduced along all three spatial axes; the three writes share that
    single read.

      - ``/projections/max_xy/`` — max over Z, plane (Y, X) → (T,C,1,Y,X)
      - ``/projections/max_xz/`` — max over Y, plane (Z, X) → (T,C,1,Z,X)
      - ``/projections/max_yz/`` — max over X, plane (Z, Y) → (T,C,1,Z,Y)

    All three are stored as 5D TCZYX with the singleton in the **Z** slot
    so the last two axes are always real spatial dimensions. Storing the
    YZ/XZ projections with X=1 or Y=1 instead (the prior layout) made the
    Fiji 5D OME-NGFF reader treat the projection as a degenerate
    1-pixel-wide image and refused to open it.

    Per-level scales reflect the *actual* axis content: ``max_xz``'s "y"
    slot carries Z values, so its y-scale is ``dz * mz``.
    """
    nt, nc, nz, ny, nx = iso_arr.shape
    dtype = iso_arr.dtype

    projections_group = root.create_group("projections", overwrite=True)

    # Each spec maps source-volume axis indices (Z=0, Y=1, X=2) onto the
    # output's "y" and "x" slots. The reduced axis becomes the singleton Z.
    #   reduce       — which source axis to max-project
    #   y_src, x_src — source-axis index for the output's y / x slot
    #   d_y,   d_x   — physical voxel size for the output's y / x slot
    axis_specs: dict[str, dict] = {
        "max_xy": {"reduce": 0, "y_src": 1, "x_src": 2, "d_y": dy, "d_x": dx},
        "max_xz": {"reduce": 1, "y_src": 0, "x_src": 2, "d_y": dz, "d_x": dx},
        "max_yz": {"reduce": 2, "y_src": 0, "x_src": 1, "d_y": dz, "d_x": dy},
    }
    src_dims = (nz, ny, nx)

    def _shape_for(spec: dict, mag: tuple[int, int, int]) -> tuple[int, ...]:
        dim_y = max(1, src_dims[spec["y_src"]] // mag[spec["y_src"]])
        dim_x = max(1, src_dims[spec["x_src"]] // mag[spec["x_src"]])
        return (nt, nc, 1, dim_y, dim_x)

    def _scale_for(spec: dict, mag: tuple[int, int, int]) -> list[float]:
        my = mag[spec["y_src"]]
        mx = mag[spec["x_src"]]
        return [float(dt_s), 1.0, 1.0, float(spec["d_y"] * my), float(spec["d_x"] * mx)]

    # Create all level arrays up front so the level-0 loop can write
    # to each axis in one pass without re-iterating the volume.
    axis_state: dict[str, dict] = {}
    for axis, spec in axis_specs.items():
        sub = projections_group.create_group(axis, overwrite=True)
        arrays: list = []
        paths: list[str] = []
        scales: list[list[float]] = []
        for level_idx, mag in enumerate(image_mags):
            shape5 = _shape_for(spec, mag)
            chunk5 = (1, 1, 1, shape5[3], shape5[4])
            arr = _create_sharded_array(
                sub, str(level_idx), shape5, dtype,
                chunk5, chunk5, compressor, compression_level,
            )
            arrays.append(arr)
            paths.append(str(level_idx))
            scales.append(_scale_for(spec, mag))
        sub.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                _multiscales_block(axis, _AXES_5D, paths, scales)
            ],
        }
        axis_state[axis] = {"arrays": arrays, "reduce": spec["reduce"]}

    # Level 0: one volume read per (t, c), three projection writes.
    for ti in range(nt):
        for ci in range(nc):
            vol = np.asarray(iso_arr[ti, ci])         # (Z, Y, X)
            for state in axis_state.values():
                proj = vol.max(axis=state["reduce"])  # 2D
                state["arrays"][0][ti, ci, 0, :, :] = proj.astype(dtype, copy=False)
            if progress_callback:
                progress_callback("max_proj_l0", ti * nc + ci + 1, nt * nc)

    # Higher levels per axis: read prev level's 2D plane, median-
    # downsample 2D, write back. Both arrays have layout (T,C,1,Y',X')
    # so the 2D plane comes from arr[t, c, 0, :, :] directly.
    for axis, state in axis_state.items():
        arrays = state["arrays"]
        for level_idx in range(1, len(arrays)):
            arr = arrays[level_idx]
            prev_arr = arrays[level_idx - 1]
            factor = (
                max(1, prev_arr.shape[3] // arr.shape[3]),
                max(1, prev_arr.shape[4] // arr.shape[4]),
            )
            target = (arr.shape[3], arr.shape[4])
            for ti in range(nt):
                for ci in range(nc):
                    prev_2d = prev_arr[ti, ci, 0, :, :]
                    if all(f == 1 for f in factor):
                        out = prev_2d
                    else:
                        out = downsample_block(prev_2d, factor, method="median")
                        out = out[: target[0], : target[1]]
                    arr[ti, ci, 0, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"{axis}_l{level_idx}",
                            ti * nc + ci + 1, nt * nc,
                        )


# === /min_intensity ===

def _write_min_intensity(
    root,
    scanner: dict[int, dict[int, Path]],
    iso_arr: IsoviewArray,
    compressor: str,
    compression_level: int,
) -> None:
    """Write top-level ``/min_intensity/0/`` as (T, C, 2) float32.

    The ``minIntensity.npz`` archives carry either a scalar or a
    ``[mask_pct, stack_pct]`` 2-vector per (t, c). We normalize to the
    2-vector form by duplicating the scalar; downstream consumers (the
    multifuse blending stage) treat the two slots identically when no
    mask is present, so the duplication is information-preserving.
    """
    if not scanner:
        return

    nt = iso_arr.num_timepoints
    nc = iso_arr.num_color_channels
    data = np.zeros((nt, nc, 2), dtype=np.float32)

    for ti, per_cam in scanner.items():
        for ci, view in enumerate(iso_arr.views):
            p = per_cam.get(view)
            if p is None:
                continue
            with np.load(p) as npz:
                val = np.asarray(npz["min_intensity"])
            if val.ndim == 0:
                data[ti, ci, 0] = float(val)
                data[ti, ci, 1] = float(val)
            else:
                flat = val.ravel().astype(np.float32)
                data[ti, ci, 0] = float(flat[0])
                data[ti, ci, 1] = float(flat[1]) if flat.size > 1 else float(flat[0])

    g = root.create_group("min_intensity", overwrite=True)
    arr = _create_sharded_array(
        g, "0", data.shape, data.dtype, data.shape, data.shape,
        compressor, compression_level,
    )
    arr[:] = data
    g.attrs["description"] = (
        "Background percentile baseline per (t, c). Two slots: "
        "[mask_pct, stack_pct] when segmentation enabled; both equal "
        "to stack_pct otherwise. Not an image — no multiscales block."
    )


# === /metadata ===

def _write_metadata(root, iso_arr: IsoviewArray) -> None:
    """Inline parsed XML + isoview_config into ``/metadata`` attrs."""
    g = root.create_group("metadata", overwrite=True)
    g.attrs["common"] = _json_safe(iso_arr._metadata)
    if iso_arr._camera_metadata:
        g.attrs["per_camera"] = _json_safe(iso_arr._camera_metadata)

    # try to inline isoview_config.json if present at the dataset root
    raw_root = _sibling_raw_root(iso_arr)
    if raw_root is not None:
        cfg = raw_root.parent / "isoview_config.json"
        if cfg.is_file():
            try:
                with open(cfg) as f:
                    g.attrs["isoview_config"] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("could not read %s: %s", cfg, exc)


def _write_backgrounds(root, raw_root: Path | None) -> None:
    """Store ``Background_*.tif`` images as top-level ``/backgrounds/``.

    Shape ``(1, C, 1, Y, X)`` — 5D TCZYX with singleton T and Z so
    Fiji's 5D-only OME-NGFF reader can drag-drop the folder. The raw
    acquisition root holds one background image per camera (e.g.
    ``Background_0.tif`` ... ``Background_3.tif``). They're stored
    *uncompressed* and pixel-exact so the array round-trips bit-for-bit
    against the source TIFFs — the user's explicit requirement for
    these is "not compressed or altered in any way".

    No-op when ``raw_root`` is unreachable or has no ``Background_*.tif``.
    """
    if raw_root is None or not raw_root.is_dir():
        return
    bg_files = sorted(raw_root.glob("Background_*.tif"))
    if not bg_files:
        return

    # probe shape + dtype from the first file
    sample = tifffile.imread(str(bg_files[0]))
    if sample.ndim == 3 and sample.shape[0] == 1:
        sample = sample[0]
    if sample.ndim != 2:
        logger.warning(
            "skipping backgrounds: first source has ndim %d (path=%s)",
            sample.ndim, bg_files[0],
        )
        return
    ny, nx = sample.shape
    dtype = sample.dtype
    nc = len(bg_files)

    # Lives under /raw/ alongside other raw-acquisition artifacts
    # (raw/projections/max_xy, raw/metadata/<xml stems>).
    raw_group = root.require_group("raw")
    bg_group = raw_group.create_group("background", overwrite=True)
    arr = _create_sharded_array(
        bg_group, "0",
        shape=(1, nc, 1, ny, nx),
        dtype=dtype,
        chunk_shape=(1, 1, 1, ny, nx),
        shard_shape=(1, 1, 1, ny, nx),
        compressor="none",
        compression_level=0,
    )

    source_names: list[str] = []
    for i, bg in enumerate(bg_files):
        img = tifffile.imread(str(bg))
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if img.shape != (ny, nx):
            logger.warning(
                "background %s: shape %s != %s, skipping",
                bg.name, img.shape, (ny, nx),
            )
            continue
        arr[0, i, 0, :, :] = img
        source_names.append(bg.name)

    bg_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(
                "background", _AXES_5D, ["0"],
                [[1.0, 1.0, 1.0, 1.0, 1.0]],
            )
        ],
    }
    bg_group.attrs["description"] = (
        "Per-camera background images from the raw acquisition root. "
        "Stored uncompressed and pixel-exact (no scaling, no codec). "
        "Padded to 5D (T=1, Z=1) for Fiji's OME-NGFF reader."
    )
    bg_group.attrs["source_files"] = source_names


def _write_raw_xml(root, raw_root: Path | None) -> None:
    """Store each ``*.xml`` from the raw root as a 1D ``uint8`` byte array.

    Path: ``/raw/metadata/<stem>/``. The array is the literal bytes of
    the source file — no encoding, no parsing, no compression. Use
    ``arr[:].tobytes().decode("utf-8")`` to round-trip back to text.
    Original filename stays in ``arr.attrs["filename"]``.

    Provides a redundant copy of the acquisition XMLs alongside the
    parsed/merged metadata in ``/metadata.attrs["common"]``.
    """
    if raw_root is None or not raw_root.is_dir():
        return
    xml_files = sorted(raw_root.glob("*.xml"))
    if not xml_files:
        return

    raw_group = root.require_group("raw")
    xml_group = raw_group.create_group("metadata", overwrite=True)

    for xml_path in xml_files:
        raw = xml_path.read_bytes()
        data = np.frombuffer(raw, dtype=np.uint8)
        # zarr path names with dots can confuse some readers — use stem,
        # round-trip the original filename via the array's own attrs.
        safe = xml_path.stem.replace(".", "_").replace("/", "_")
        full_name = f"{xml_group.path.rstrip('/')}/{safe}"
        arr = zarr.create_array(
            store=xml_group.store,
            name=full_name,
            shape=data.shape,
            dtype=np.uint8,
            chunks=data.shape,
            compressors=None,
            overwrite=True,
            zarr_format=3,
        )
        arr[:] = data
        arr.attrs["filename"] = xml_path.name
        arr.attrs["size_bytes"] = int(len(raw))

    xml_group.attrs["description"] = (
        "Verbatim bytes of each XML file from the raw acquisition root. "
        "Each array is uint8; decode via arr[:].tobytes().decode('utf-8')."
    )


# === Public entry point ===

def _setup_consolidation(
    iso: IsoviewArray,
    out: str | Path,
    *,
    overwrite: bool,
    pyramid: bool,
    pyramid_max_layers: int,
):
    """Shared boot for any kind: open the output group, compute mags,
    pull physical scales from metadata. Returns ``(root, mags, dx, dy,
    dz, dt_s, out_path)`` used by both per-kind consolidators."""
    _nt, _nc, nz, ny, nx = iso.shape
    md = iso.metadata
    dx = float(md.get("dx", 1.0))
    dy = float(md.get("dy", 1.0))
    dz = float(md.get("dz", 1.0))
    fs = float(md.get("fs", 1.0))
    dt_s = 1.0 / fs if fs > 0 else 1.0

    if pyramid:
        mags = _compute_anisotropic_mags(
            (dz, dy, dx), (nz, ny, nx), pyramid_max_layers,
        )
    else:
        mags = [(1, 1, 1)]

    out_path = Path(out)
    if out_path.suffix.lower() != ".zarr":
        out_path = out_path.with_suffix(".zarr")
    root = _open_output_group(out_path, overwrite)
    return root, mags, dx, dy, dz, dt_s, out_path


def _consolidate_corrected(
    src_path: Path,
    out: str | Path,
    *,
    overwrite: bool,
    pyramid: bool,
    pyramid_max_layers: int,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> Path:
    """Internal: corrected-pipeline consolidator.

    Uses :func:`_write_computed_projections` for the corrected
    projections (``projections/max_xy``, ``max_xz``, ``max_yz`` derived
    from the volume in one read per (t,c)). The on-disk
    ``<root>.corrected.projections/`` flat dir is ignored — the
    spec-correct equivalent comes out of the volume, and the disk files
    would just duplicate it. The raw projections
    (``<root>.raw.projections/``) ARE pulled from disk because they
    derive from raw data the consolidated zarr doesn't carry; they
    land at ``/raw/projections/max_xy/``.
    """
    spm_dir = _resolve_corrected_spm_dir(src_path)
    if spm_dir is None:
        raise ValueError(f"not an isoview corrected tree: {src_path}")

    iso = IsoviewArray(spm_dir, kind="corrected")
    nt, nc, nz, ny, nx = iso.shape
    root, mags, dx, dy, dz, dt_s, out_path = _setup_consolidation(
        iso, out, overwrite=overwrite,
        pyramid=pyramid, pyramid_max_layers=pyramid_max_layers,
    )

    logger.info(
        "consolidate corrected: src=%s out=%s shape=%s pyramid=%d levels=%d",
        spm_dir, out_path, iso.shape, int(pyramid), len(mags),
    )

    # main image pyramid
    img_paths, img_scales = _write_image_pyramid(
        root, iso, mags, dx, dy, dz, dt_s,
        compressor, compression_level, progress_callback,
    )

    # labels — only the OME-spec-compliant 3D segmentation lives here.
    # The IsoView xyMask / xzMask coordinate maps are written as
    # top-level sibling groups since their axes don't map to the OME
    # labels projection schema.
    companions = _scan_corrected_companions(spm_dir)
    labels = root.create_group("labels", overwrite=True)
    label_names: list[str] = []
    if _write_segmentation_pyramid(
        labels, companions.get("segmentation", {}), iso, mags,
        dx, dy, dz, dt_s, compressor, compression_level, progress_callback,
    ) is not None:
        label_names.append("background_mask")
    labels.attrs["ome"] = {"version": "0.5", "labels": label_names}

    _write_aux_2d_per_tc(
        root, "xy_mask", companions.get("xy_mask", {}), iso,
        compressor, compression_level, progress_callback,
    )
    _write_aux_2d_per_tc(
        root, "xz_mask", companions.get("xz_mask", {}), iso,
        compressor, compression_level, progress_callback,
    )

    # corrected projections/max_xy / max_xz / max_yz: one volume read per (t,c).
    _write_computed_projections(
        root, iso, mags,
        dx, dy, dz, dt_s, compressor, compression_level, progress_callback,
    )

    # raw projections: come from a flat sibling dir (not from the
    # consolidated volume — they're max-Z of the *raw* stacks).
    raw_root = _sibling_raw_root(iso)
    raw_proj_dir = (
        raw_root.parent / f"{raw_root.name}.raw.projections"
        if raw_root is not None else None
    )
    raw_proj = (
        _scan_flat_projections(raw_proj_dir)
        if raw_proj_dir is not None and raw_proj_dir.is_dir()
        else None
    )
    if raw_proj is not None:
        tm_dirs = _find_tm_folders(spm_dir)
        tm_int_by_index = {
            ti: _extract_timepoint(d.name) for ti, d in enumerate(tm_dirs)
        }
        # Y/X-only mag list for the 4D projection group
        mags_yx: list[tuple[int, int]] = []
        seen_yx: set[tuple[int, int]] = set()
        for _mz, my, mx in mags:
            if (my, mx) not in seen_yx:
                seen_yx.add((my, mx))
                mags_yx.append((my, mx))
        _write_disk_xy_projections(
            root, raw_proj, iso, tm_int_by_index,
            mags_yx, dx, dy, dt_s, compressor, compression_level,
            progress_callback,
        )

    _write_min_intensity(
        root, companions.get("min_intensity", {}), iso,
        compressor, compression_level,
    )
    _write_metadata(root, iso)
    _write_backgrounds(root, raw_root)
    _write_raw_xml(root, raw_root)

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(spm_dir.name, _AXES_5D, img_paths, img_scales)
        ],
        "omero": _omero_block(
            iso.channel_names, iso._camera_metadata, default_z=nz // 2
        ),
    }
    root.attrs["isoview"] = {
        "schema_version": "0.1",
        "kind": "corrected",
        "source": str(spm_dir),
        "shape": list(iso.shape),
    }
    logger.info("consolidate: wrote %s", out_path)
    return out_path


def _consolidate_fused(
    src_path: Path,
    out: str | Path,
    *,
    overwrite: bool,
    pyramid: bool,
    pyramid_max_layers: int,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> Path:
    """Internal: fused-pipeline consolidator.

    Layout differences vs corrected (see ``docs/isoview.md`` for the
    full inventory):
      - Channel axis indexes fused view PAIRS (``VW00_fused``,
        ``VW90_fused``), not single cameras.
      - Combined 3D mask (``.mask.zarr`` = ``cam0_mask ∪ cam1_transformed_mask``)
        goes into ``/labels/segmentation/``.
      - Four top-level 2D mask groups instead of corrected's two:
        ``fusion_mask``, ``mask2D_cam0``, ``mask2D_cam1``,
        ``transformedMask2D_cam1``.
      - No ``min_intensity`` (correction-stage artifact, not produced
        by fusion).
      - No ``raw/projections/max_xy`` (fusion's input is corrected, not raw).
    """
    iso = IsoviewArray(src_path, kind="fused")
    method_dir = iso.scan_root  # <raw>.fused/<method>/
    nt, nc, nz, ny, nx = iso.shape
    root, mags, dx, dy, dz, dt_s, out_path = _setup_consolidation(
        iso, out, overwrite=overwrite,
        pyramid=pyramid, pyramid_max_layers=pyramid_max_layers,
    )

    logger.info(
        "consolidate fused: src=%s out=%s shape=%s pyramid=%d levels=%d",
        method_dir, out_path, iso.shape, int(pyramid), len(mags),
    )

    img_paths, img_scales = _write_image_pyramid(
        root, iso, mags, dx, dy, dz, dt_s,
        compressor, compression_level, progress_callback,
    )

    # Combined 3D fusion mask → /labels/segmentation/.
    # OME-spec-correct: ``mask`` is a binary integer-label array
    # (uint8 after _read_companion + ``.astype``), same (T,C,Z,Y,X)
    # shape as the parent image, pyramid mirrors the image.
    companions = _scan_fused_companions(method_dir)
    labels = root.create_group("labels", overwrite=True)
    label_names: list[str] = []
    if _write_segmentation_pyramid(
        labels, companions.get("mask", {}), iso, mags,
        dx, dy, dz, dt_s, compressor, compression_level, progress_callback,
    ) is not None:
        label_names.append("background_mask")
    labels.attrs["ome"] = {"version": "0.5", "labels": label_names}

    # 2D mask groups — keyed by pair (fusion_mask) or by single camera
    # within the pair (mask2D, transformedMask2D). The single-cam ones
    # get pivoted to per-pair indexing so the shared aux writer can
    # use them.
    pair_view_keys = list(iso.views)
    _write_aux_2d_per_tc(
        root, "fusion_mask", companions.get("fusion_mask", {}), iso,
        compressor, compression_level, progress_callback,
    )
    _write_aux_2d_per_tc(
        root, "mask2D_cam0",
        _remap_single_cam_to_pair(
            companions.get("mask2D", {}), pair_view_keys, cam_position=0,
        ),
        iso, compressor, compression_level, progress_callback,
    )
    _write_aux_2d_per_tc(
        root, "mask2D_cam1",
        _remap_single_cam_to_pair(
            companions.get("mask2D", {}), pair_view_keys, cam_position=1,
        ),
        iso, compressor, compression_level, progress_callback,
    )
    _write_aux_2d_per_tc(
        root, "transformedMask2D_cam1",
        _remap_single_cam_to_pair(
            companions.get("transformedMask2D", {}),
            pair_view_keys, cam_position=1,
        ),
        iso, compressor, compression_level, progress_callback,
    )

    # projections/max_xy / max_xz / max_yz from the consolidated volume
    _write_computed_projections(
        root, iso, mags,
        dx, dy, dz, dt_s, compressor, compression_level, progress_callback,
    )

    # metadata + backgrounds + raw XML (same as corrected)
    _write_metadata(root, iso)
    raw_root = _sibling_raw_root(iso)
    _write_backgrounds(root, raw_root)
    _write_raw_xml(root, raw_root)

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(method_dir.name, _AXES_5D, img_paths, img_scales)
        ],
        "omero": _omero_block(
            iso.channel_names, iso._camera_metadata, default_z=nz // 2
        ),
    }
    root.attrs["isoview"] = {
        "schema_version": "0.1",
        "kind": "fused",
        "source": str(method_dir),
        "shape": list(iso.shape),
    }
    logger.info("consolidate: wrote %s", out_path)
    return out_path


def consolidate_isoview(
    src: str | Path,
    out: str | Path,
    *,
    kind: str | None = None,
    overwrite: bool = False,
    pyramid: bool = True,
    pyramid_max_layers: int = 4,
    compressor: str = "zstd",
    compression_level: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Consolidate one isoview output tree into one OME-Zarr group.

    Parameters
    ----------
    src
        Any path inside the source tree. For ``kind="corrected"`` this
        is resolved to the ``SPM##`` directory under ``.corrected/``;
        for ``kind="fused"`` it's resolved to the
        ``<raw>.fused/<method>/`` directory. Auto-detected when
        ``kind=None`` via :func:`detect_isoview_kind`.
    out
        Output ``.zarr`` path. Created (parents OK) or overwritten when
        ``overwrite=True``.
    kind
        ``"corrected"`` | ``"fused"`` | ``None``. ``None`` auto-detects.
    pyramid, pyramid_max_layers
        OME-NGFF resolution pyramid config. Levels are computed
        anisotropy-aware via the webknossos algorithm (in practice:
        downsample Y/X by 2 per level until <64 voxels).
    compressor, compression_level
        Codec for inner chunks. Matches the existing rechunked corrected
        tree at ``zstd / level 3``.
    progress_callback
        Optional ``(stage, current, total) -> None`` hook.

    Returns
    -------
    Path
        Path to the written ``.zarr`` group.
    """
    src_path = Path(src)
    if kind is None:
        kind = detect_isoview_kind(src_path)
        if kind is None:
            raise ValueError(
                f"cannot auto-detect isoview kind at {src_path}; pass kind=..."
            )

    if kind == "corrected":
        return _consolidate_corrected(
            src_path, out, overwrite=overwrite,
            pyramid=pyramid, pyramid_max_layers=pyramid_max_layers,
            compressor=compressor, compression_level=compression_level,
            progress_callback=progress_callback,
        )
    if kind == "fused":
        return _consolidate_fused(
            src_path, out, overwrite=overwrite,
            pyramid=pyramid, pyramid_max_layers=pyramid_max_layers,
            compressor=compressor, compression_level=compression_level,
            progress_callback=progress_callback,
        )
    raise ValueError(
        f"kind must be 'corrected' or 'fused'; got {kind!r}"
    )


def to_bigstitcher(
    src: str | Path,
    dest: str | Path | None = None,
    *,
    compressor: str = "zstd",
    compression_level: int = 3,
    overwrite: bool = False,
) -> Path:
    """Mirror a consolidated v3 zarr to a transient v2 zarr for BigStitcher.

    BigStitcher (and BigDataViewer's older ImageLoader path) only read
    Zarr v2 / OME-NGFF 0.4. This writes a fresh v2 group at ``dest``
    that contains the same image pyramid + omero/multiscales metadata
    as the source v3 group, with sharding dropped (v2 doesn't support
    it) and the OME version string flipped from ``"0.5"`` to ``"0.4"``.

    The mirror is meant to be **transient** — produce it when you need
    BigStitcher access, delete it when you're done. Auxiliary groups
    (``projections/``, ``labels/``, ``raw/``, ``xy_mask`` etc.) are
    *not* copied — BigStitcher only consumes the main image pyramid.

    BigStitcher import flow:
      1. Run this function to produce ``<src>.v2.zarr``.
      2. In Fiji: ``Plugins → BigStitcher → Define a new dataset``,
         pick "Zeiss Lightsheet ... → Generic OME-Zarr" (or the closest
         loader your version exposes).
      3. Point it at the ``.v2.zarr`` directory. BigStitcher reads
         ``omero.channels[].label`` for channel names, ``window`` for
         contrast, and the multiscales axes for voxel size, then
         generates its own ``dataset.xml``.
      4. Use BigStitcher's UI to assign per-camera angles (the OME-NGFF
         spec has no first-class home for them, so this is a one-time
         manual step per dataset).

    Parameters
    ----------
    src
        Path to a consolidated v3 ``.zarr`` (output of
        :func:`consolidate_isoview`).
    dest
        Where to write the v2 mirror. Defaults to ``<src>.v2.zarr``
        next to ``src``.
    compressor, compression_level
        Codec for the v2 chunks. Defaults match the v3 source so
        on-disk size stays comparable. ``"none"`` skips compression.
    overwrite
        Replace ``dest`` if it exists.

    Returns
    -------
    Path
        Path to the v2 ``.zarr`` directory.
    """
    src_path = Path(src)
    if not src_path.is_dir():
        raise FileNotFoundError(f"source v3 zarr not found: {src_path}")
    if dest is None:
        dest_path = src_path.with_suffix(".v2.zarr")
    else:
        dest_path = Path(dest)
        if dest_path.suffix.lower() != ".zarr":
            dest_path = dest_path.with_suffix(".zarr")

    if dest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"v2 mirror already exists (pass overwrite=True): {dest_path}"
            )
        shutil.rmtree(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    src_root = zarr.open_group(str(src_path), mode="r")
    src_attrs = dict(src_root.attrs)
    multiscales = src_attrs.get("ome", {}).get("multiscales") or []
    if not multiscales:
        raise ValueError(
            f"no multiscales metadata in {src_path}; not a consolidated isoview zarr"
        )

    dst_root = zarr.open_group(str(dest_path), mode="w", zarr_format=2)
    dst_root.attrs.update(_v2_attrs(src_attrs))

    v2_codec = _v2_compressor(compressor, compression_level)

    # Image pyramid only — BigStitcher consumes /0/../N/ and ignores
    # everything else. Each level keeps the source's chunk shape and
    # gets copied (T,C)-slab-by-slab to keep peak memory bounded.
    level_paths = [d["path"] for d in multiscales[0]["datasets"]]
    for lvl in level_paths:
        src_arr = zarr.open_array(store=src_root.store, path=lvl, mode="r")
        dst_arr = zarr.create_array(
            store=dst_root.store,
            name=lvl,
            shape=src_arr.shape,
            dtype=src_arr.dtype,
            chunks=src_arr.chunks,
            compressors=[v2_codec] if v2_codec is not None else None,
            zarr_format=2,
            overwrite=True,
        )
        nt = src_arr.shape[0]
        nc = src_arr.shape[1]
        for ti in range(nt):
            for ci in range(nc):
                dst_arr[ti, ci] = src_arr[ti, ci]

    logger.info("to_bigstitcher: wrote %s (v2 mirror of %s)", dest_path, src_path)
    return dest_path
