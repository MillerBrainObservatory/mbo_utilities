"""Add multi-resolution pyramid levels to an existing single-level BDV
dataset.ome.zarr (the bdv.multimg.zarr format BigStitcher reads).

For each s{setup}-t{tp}.zarr it keeps level 0 as-is and writes levels 1..N
(anisotropy-aware, gzip / little-endian / zarr v2, 3-D block chunks), then
rewrites the group `multiscales` to list every level. Idempotent (skips
sub-zarrs that already have >1 level). dataset.xml is left untouched.

Uses isoview's own pyramid helpers so the result matches new exports exactly.

Usage:
    uv run python scripts/backfill_bdv_pyramid.py <dataset.ome.zarr> [<dataset.xml>]
"""
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import zarr

from isoview.views import _BDV_BLOCK_ZYX, _bdv_pyramid_mags
from isoview.io import _median_downsample


def _voxel_zyx_from_xml(xml_path: Path) -> tuple[float, float, float]:
    """Read voxel size (x y z um) from the first ViewSetup -> (z, y, x)."""
    root = ET.parse(xml_path).getroot()
    vs = root.find(".//ViewSetup/voxelSize/size")
    x, y, z = (float(v) for v in vs.text.split())
    return (z, y, x)


def _write_level(arr_dir: Path, data5d: np.ndarray) -> None:
    bz, by, bx = _BDV_BLOCK_ZYX
    chunks = (1, 1, min(bz, data5d.shape[2]),
              min(by, data5d.shape[3]), min(bx, data5d.shape[4]))
    arr_dir.mkdir(parents=True, exist_ok=True)
    (arr_dir / ".zarray").write_text(json.dumps({
        "shape": list(data5d.shape), "chunks": list(chunks),
        "dtype": "<u2", "fill_value": 0, "order": "C", "filters": None,
        "dimension_separator": "/", "compressor": {"id": "gzip", "level": 5},
        "zarr_format": 2,
    }))
    (arr_dir / ".zattrs").write_text(json.dumps({}))
    zarr.open_group(zarr.storage.LocalStore(str(arr_dir.parent)), mode="a",
                    zarr_format=2)[arr_dir.name][:] = data5d


def _multiscales(mags: list[tuple[int, int, int]]) -> dict:
    return {"multiscales": [{
        "name": "/", "version": "0.4",
        "axes": [
            {"type": "time", "name": "t"}, {"type": "channel", "name": "c"},
            {"type": "space", "name": "z", "unit": "micrometer"},
            {"type": "space", "name": "y", "unit": "micrometer"},
            {"type": "space", "name": "x", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": str(i),
            "coordinateTransformations": [
                # axes order [t,c,z,y,x] -- matches BigStitcher-Spark's own
                # resave output. m is (z, y, x).
                {"type": "scale", "scale": [1.0, 1.0, float(m[0]), float(m[1]), float(m[2])]},
                {"type": "translation", "translation": [0.0] * 5},
            ],
        } for i, m in enumerate(mags)],
        "coordinateTransformations": [{"type": "scale", "scale": [1.0] * 5}],
    }]}


def _already_multilevel(sub: Path) -> bool:
    za = sub / ".zattrs"
    if not za.exists():
        return False
    ms = json.loads(za.read_text()).get("multiscales")
    return bool(ms) and len(ms[0].get("datasets", [])) > 1


def _backfill_one(sub: Path, voxel_zyx) -> int:
    vol = np.asarray(zarr.open(str(sub / "0"), mode="r")[0, 0])  # (z,y,x)
    mags = _bdv_pyramid_mags(tuple(int(s) for s in vol.shape),
                             tuple(float(v) for v in voxel_zyx))
    cur, prev = vol, (1, 1, 1)
    for level, mag in enumerate(mags):
        if level == 0:
            continue
        cur = _median_downsample(cur, tuple(m // p for m, p in zip(mag, prev)))
        prev = mag
        _write_level(sub / str(level), cur.reshape(1, 1, *cur.shape).astype("<u2"))
    (sub / ".zattrs").write_text(json.dumps(_multiscales(mags)))
    return len(mags)


def main() -> None:
    root = Path(sys.argv[1])
    xml = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parent / "dataset.xml"
    voxel = _voxel_zyx_from_xml(xml)
    subs = sorted(p for p in root.glob("s*-t*.zarr") if p.is_dir())
    print(f"root: {root}\nvoxel (z,y,x) um: {voxel}\nsub-zarrs: {len(subs)}")
    done = skipped = 0
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        if _already_multilevel(sub):
            skipped += 1
            continue
        n = _backfill_one(sub, voxel)
        done += 1
        if done % 25 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] backfilled={done} skipped={skipped} "
                  f"levels={n} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"done: backfilled={done} skipped={skipped} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
