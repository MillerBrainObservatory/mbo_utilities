"""Consolidate an isoview corrected output tree into a single OME-Zarr group.

Reads via :class:`mbo_utilities.arrays.isoview.IsoviewArray`, walks the
companion files (segmentation masks, axis-mask projections, min-intensity
NPZs, sibling projection trees), and writes one Zarr v3 group following
the OME-NGFF 0.5 spec for the spec-covered pieces (multiscales, omero,
labels) plus a custom ``/isoview/`` subgroup for the items the spec has
no first-class home for (intensity projections, min_intensity, parsed
XML provenance).

Output structure::

    <out>.zarr/
    ├── zarr.json                            ome.{version, multiscales, omero}
    ├── 0/ .. n/                             image pyramid (T,C,Z,Y,X) uint16
    ├── labels/
    │   ├── zarr.json                        ome.labels = ["segmentation"]
    │   └── segmentation/                    parent-mirroring pyramid, uint8
    └── isoview/                             custom namespace
        ├── projections/{corrected,raw}/xy/  (T,C,Y,X) uint16, own pyramid
        ├── aux/{xy_mask,xz_mask}/0/         (T,C,d0,d1) uint16, source-faithful
        ├── backgrounds/0/                   (C,Y,X) uncompressed, pixel-exact
        ├── min_intensity/0/                 (T,C,2) float32
        └── provenance/
            ├── attrs                        common + per_camera + isoview_config
            └── xml_raw/<stem>/              1D uint8, verbatim XML bytes
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
from mbo_utilities.arrays.isoview import (
    IsoviewArray,
    _extract_timepoint,
    _find_tm_folders,
    _resolve_corrected_spm_dir,
    _scan_flat_projections,
    _sibling_raw_root,
)
from mbo_utilities.log import get as _get_logger

logger = _get_logger("arrays.isoview_consolidate")


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
    )


# === OME metadata builders ===

_AXES_5D = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]
_AXES_4D = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _scale_5d(dt_s, dz, dy, dx, mag_zyx):
    """Build a (T, C, Z, Y, X) scale list for one pyramid level."""
    mz, my, mx = mag_zyx
    return [float(dt_s), 1.0, float(dz * mz), float(dy * my), float(dx * mx)]


def _scale_4d(dt_s, dy, dx, mag_yx):
    """Build a (T, C, Y, X) scale list for one pyramid level."""
    my, mx = mag_yx
    return [float(dt_s), 1.0, float(dy * my), float(dx * mx)]


def _multiscales_block(
    name: str, axes: list[dict], dataset_paths: list[str], scales: list[list[float]]
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
        "type": "gaussian",
        "metadata": {
            "method": "mbo_utilities.arrays.features._pyramid.downsample_block",
            "downsample": "gaussian",
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
            "window": {"min": 0, "max": 65535, "start": 0, "end": 65535},
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
    on-disk array and downsample via gaussian blur + subsample.
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
                    out = downsample_block(vol, factor, method="gaussian")
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
    number of entries as the parent image's. Downsampling uses
    nearest-neighbor because gaussian on a binary mask + threshold
    produces shrunken regions; nearest preserves the integer-label
    contract that the spec requires for label pixels.
    """
    if not scanner:
        return None

    nt, nc, nz, ny, nx = iso_arr.shape
    seg_group = labels_group.create_group("segmentation", overwrite=True)
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
                    out = downsample_block(vol, factor, method="nearest")
                    out = out[:out_nz, :out_ny, :out_nx]
                    arr[ti, ci, :, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"seg_l{level_idx}", ti * nc + ci + 1, nt * nc
                        )

    # ome metadata on the segmentation subgroup
    seg_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block("segmentation", _AXES_5D, paths, scales)
        ],
        "image-label": {
            "version": "0.5",
            "source": {"image": "../../"},
        },
    }
    return paths


def _write_aux_2d_per_tc(
    iso_group,
    name: str,
    scanner: dict[int, dict[int, Path]],
    iso_arr: IsoviewArray,
    compressor: str,
    compression_level: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Write /isoview/aux/<name>/0/ — a 2D-per-(t,c) auxiliary array.

    Used for the ``xyMask`` / ``xzMask`` coordinate maps the IsoView
    pipeline emits. These don't fit the OME label projection schema
    (their axes don't map cleanly to "drop Z" or "drop X"; e.g. the
    "xzMask" is shape ``(Y, Z)``, not ``(Z, X)`` as the name implies),
    so we store them faithfully under a custom path without OME
    labels metadata.

    Output shape: ``(T, C, dim0, dim1)`` matching the source 2D shape.
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
    shape = (nt, nc, dim0, dim1)
    chunk = (1, 1, dim0, dim1)

    aux_group = (
        iso_group["aux"] if "aux" in iso_group
        else iso_group.create_group("aux", overwrite=False)
    )
    sub = aux_group.create_group(name, overwrite=True)
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
            arr[ti, ci, :, :] = data
            if progress_callback:
                progress_callback(f"aux_{name}", ti * nc + ci + 1, nt * nc)

    sub.attrs["description"] = (
        f"Isoview-internal 2D coordinate mask per (t, c). "
        f"Source TIFF shape preserved: (dim0={dim0}, dim1={dim1}). "
        f"Axes are isoview-specific and don't fit the OME-NGFF labels "
        f"spec, hence the custom /isoview/aux/ path."
    )


# === /isoview/projections (intensity max-projections) ===

def _write_intensity_projections(
    iso_group,
    name: str,  # "corrected" | "raw"
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
    """Write /isoview/projections/{name}/xy/ from a flat-projections scan.

    Shape: ``(T, C, Y, X)``. Own multiscale pyramid (Y/X only). Reads
    each TIFF lazily; first TIFF probes dtype + shape.
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

    parent_group = iso_group.create_group("projections", overwrite=False) \
        if "projections" not in iso_group else iso_group["projections"]
    src_group = parent_group.create_group(name, overwrite=True)
    xy_group = src_group.create_group("xy", overwrite=True)

    paths: list[str] = []
    scales: list[list[float]] = []

    for level_idx, mag in enumerate(mags_yx):
        my, mx = mag
        out_ny = max(1, ny // my)
        out_nx = max(1, nx // mx)
        shape = (nt, nc, out_ny, out_nx)
        chunk = (1, 1, out_ny, out_nx)
        shard = chunk
        arr = _create_sharded_array(
            xy_group, str(level_idx), shape, dtype, chunk, shard,
            compressor, compression_level,
        )
        paths.append(str(level_idx))
        scales.append(_scale_4d(dt_s, dy, dx, mag))

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
                    arr[ti, ci, :, :] = img.astype(dtype, copy=False)
                    if progress_callback:
                        progress_callback(
                            f"proj_{name}_l0",
                            ti * nc + ci + 1, nt * nc,
                        )
        else:
            prev_path = f"{xy_group.path.rstrip('/')}/{level_idx - 1}"
            prev_arr = zarr.open_array(store=xy_group.store, path=prev_path, mode="r")
            prev_mag = mags_yx[level_idx - 1]
            factor2 = (mag[0] // prev_mag[0], mag[1] // prev_mag[1])
            for ti in range(nt):
                for ci in range(nc):
                    img = prev_arr[ti, ci, :, :]
                    out = downsample_block(img, factor2, method="gaussian")
                    out = out[:out_ny, :out_nx]
                    arr[ti, ci, :, :] = out
                    if progress_callback:
                        progress_callback(
                            f"proj_{name}_l{level_idx}",
                            ti * nc + ci + 1, nt * nc,
                        )

    xy_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(f"projections/{name}/xy", _AXES_4D, paths, scales)
        ],
    }


# === /isoview/min_intensity ===

def _write_min_intensity(
    iso_group,
    scanner: dict[int, dict[int, Path]],
    iso_arr: IsoviewArray,
    compressor: str,
    compression_level: int,
) -> None:
    """Write /isoview/min_intensity/0/ as (T, C, 2) float32.

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

    g = iso_group.create_group("min_intensity", overwrite=True)
    arr = _create_sharded_array(
        g, "0", data.shape, data.dtype, data.shape, data.shape,
        compressor, compression_level,
    )
    arr[:] = data
    g.attrs["description"] = (
        "Background percentile baseline per (t, c). Two slots: "
        "[mask_pct, stack_pct] when segmentation enabled; both equal "
        "to stack_pct otherwise."
    )


# === /isoview/provenance ===

def _write_provenance(iso_group, iso_arr: IsoviewArray) -> None:
    """Inline parsed XML + isoview_config into group attrs."""
    g = iso_group.create_group("provenance", overwrite=True)
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


def _write_backgrounds(iso_group, raw_root: Path | None) -> None:
    """Store ``Background_*.tif`` images as a single ``(C, Y, X)`` zarr.

    The raw acquisition root holds one background image per camera
    (e.g. ``Background_0.tif`` ... ``Background_3.tif``). They're stored
    *uncompressed* and pixel-exact so the array round-trips bit-for-bit
    against the source TIFFs — the user's explicit requirement for
    these is "not compressed or altered in any way". One chunk per
    camera plane.

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

    bg_group = iso_group.create_group("backgrounds", overwrite=True)
    arr = _create_sharded_array(
        bg_group, "0",
        shape=(len(bg_files), ny, nx),
        dtype=dtype,
        chunk_shape=(1, ny, nx),
        shard_shape=(1, ny, nx),
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
        arr[i] = img
        source_names.append(bg.name)

    bg_group.attrs["description"] = (
        "Per-camera background images from the raw acquisition root. "
        "Stored uncompressed and pixel-exact (no scaling, no codec)."
    )
    bg_group.attrs["source_files"] = source_names


def _write_raw_xml(iso_group, raw_root: Path | None) -> None:
    """Store each ``*.xml`` from the raw root as a 1D ``uint8`` byte array.

    Path: ``/isoview/provenance/xml_raw/<stem>/``. The array is the
    literal bytes of the source file — no encoding, no parsing, no
    compression. Use ``arr[:].tobytes().decode("utf-8")`` to round-trip
    back to text. Original filename stays in ``arr.attrs["filename"]``.

    Provides a redundant copy of the acquisition XMLs alongside the
    parsed/merged metadata in ``/isoview/provenance.attrs["common"]``.
    """
    if raw_root is None or not raw_root.is_dir():
        return
    xml_files = sorted(raw_root.glob("*.xml"))
    if not xml_files:
        return

    prov = (
        iso_group["provenance"]
        if "provenance" in iso_group
        else iso_group.create_group("provenance", overwrite=False)
    )
    xml_group = prov.create_group("xml_raw", overwrite=True)

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

def consolidate_isoview_corrected(
    src: str | Path,
    out: str | Path,
    *,
    overwrite: bool = False,
    pyramid: bool = True,
    pyramid_max_layers: int = 4,
    compressor: str = "zstd",
    compression_level: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Consolidate a corrected isoview tree into one OME-Zarr group.

    Parameters
    ----------
    src
        Any path inside the corrected tree. Resolved via
        :func:`_resolve_corrected_spm_dir` to the ``SPM##`` directory.
    out
        Output ``.zarr`` path. Created (parents OK) or overwritten when
        ``overwrite=True``.
    pyramid
        Generate the OME-NGFF resolution pyramid. Levels are computed
        anisotropy-aware via the webknossos algorithm — for typical
        isoview corrected data (Z much coarser than Y/X) this reduces
        to "downsample Y/X by 2 per level until <64 voxels".
    pyramid_max_layers
        Cap on additional levels beyond ``/0``. Stops earlier when Y or
        X would fall below 64 voxels.
    compressor, compression_level
        Codec config for inner chunks. ``"zstd"`` with level 3 matches
        the existing rechunked corrected tree.
    progress_callback
        Optional ``(stage, current, total) -> None`` hook fired once
        per ``(t, c)`` slab written.

    Returns
    -------
    Path
        Path to the written ``.zarr`` group.
    """
    src_path = Path(src)
    spm_dir = _resolve_corrected_spm_dir(src_path)
    if spm_dir is None:
        raise ValueError(f"not an isoview corrected tree: {src_path}")

    iso = IsoviewArray(spm_dir, kind="corrected")
    nt, nc, nz, ny, nx = iso.shape
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

    # Y/X-only mag list for the 4D intensity-projection groups
    mags_yx = []
    seen_yx: set[tuple[int, int]] = set()
    for mz, my, mx in mags:
        if (my, mx) not in seen_yx:
            seen_yx.add((my, mx))
            mags_yx.append((my, mx))

    out_path = Path(out)
    if out_path.suffix.lower() != ".zarr":
        out_path = out_path.with_suffix(".zarr")
    root = _open_output_group(out_path, overwrite)

    logger.info(
        "consolidate: src=%s out=%s shape=%s pyramid=%d levels=%d",
        spm_dir, out_path, iso.shape, int(pyramid), len(mags),
    )

    # main image pyramid
    img_paths, img_scales = _write_image_pyramid(
        root, iso, mags, dx, dy, dz, dt_s,
        compressor, compression_level, progress_callback,
    )

    # labels (spec-compliant integer-label arrays only — the 3D
    # segmentation mask). The IsoView "xyMask"/"xzMask" coordinate
    # maps don't fit the OME labels spec (axes don't map cleanly to
    # "drop Z" or "drop X"), so they go under /isoview/aux/ instead.
    companions = _scan_corrected_companions(spm_dir)
    labels = root.create_group("labels", overwrite=True)
    label_names: list[str] = []
    if _write_segmentation_pyramid(
        labels, companions.get("segmentation", {}), iso, mags,
        dx, dy, dz, dt_s, compressor, compression_level, progress_callback,
    ) is not None:
        label_names.append("segmentation")
    labels.attrs["ome"] = {"version": "0.5", "labels": label_names}

    # /isoview custom namespace
    iso_group = root.create_group("isoview", overwrite=True)

    # 2D coordinate masks (xyMask, xzMask) live here, not under /labels/
    _write_aux_2d_per_tc(
        iso_group, "xy_mask", companions.get("xy_mask", {}), iso,
        compressor, compression_level, progress_callback,
    )
    _write_aux_2d_per_tc(
        iso_group, "xz_mask", companions.get("xz_mask", {}), iso,
        compressor, compression_level, progress_callback,
    )

    # intensity max-projections (corrected + raw, when sibling dirs exist)
    raw_root = _sibling_raw_root(iso)
    corr_root = spm_dir.parent  # the .corrected/ dir
    corr_proj_dir = corr_root.parent / f"{corr_root.name}.projections"
    raw_proj_dir = (
        raw_root.parent / f"{raw_root.name}.raw.projections"
        if raw_root is not None else None
    )
    corr_proj = _scan_flat_projections(corr_proj_dir) if corr_proj_dir.is_dir() else None
    raw_proj = (
        _scan_flat_projections(raw_proj_dir)
        if raw_proj_dir is not None and raw_proj_dir.is_dir()
        else None
    )
    # Flat projection scans key files by the raw TM integer parsed from
    # the filename (e.g. TM000005 → 5). IsoviewArray's _timepoints is
    # the index list 0..nt-1, not the raw TM number, so we build an
    # explicit ti → raw-tm-int map from the SPM dir's TM folders.
    tm_dirs = _find_tm_folders(spm_dir)
    tm_int_by_index = {ti: _extract_timepoint(d.name) for ti, d in enumerate(tm_dirs)}
    _write_intensity_projections(
        iso_group, "corrected", corr_proj, iso, tm_int_by_index,
        mags_yx, dx, dy, dt_s, compressor, compression_level, progress_callback,
    )
    _write_intensity_projections(
        iso_group, "raw", raw_proj, iso, tm_int_by_index,
        mags_yx, dx, dy, dt_s, compressor, compression_level, progress_callback,
    )

    # min intensity + provenance + backgrounds + raw XML
    _write_min_intensity(
        iso_group, companions.get("min_intensity", {}), iso,
        compressor, compression_level,
    )
    _write_provenance(iso_group, iso)
    _write_backgrounds(iso_group, raw_root)
    _write_raw_xml(iso_group, raw_root)

    # root-level OME multiscales + omero + custom isoview manifest
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            _multiscales_block(
                spm_dir.name, _AXES_5D, img_paths, img_scales
            )
        ],
        "omero": _omero_block(iso.channel_names, iso._camera_metadata, default_z=nz // 2),
    }
    root.attrs["isoview"] = {
        "schema_version": "0.1",
        "kind": "corrected",
        "source": str(spm_dir),
        "shape": list(iso.shape),
    }

    logger.info("consolidate: wrote %s", out_path)
    return out_path
