"""Re-encode a BDV dataset.ome.zarr from big-endian/zstd to little-endian/gzip
so Fiji / BigStitcher's n5-zarr reader can decode the chunks.

Operates per (setup, timepoint) sub-zarr with an atomic swap; idempotent
(skips sub-zarrs already <u2 + gzip). dataset.xml is left untouched, so the
existing registration / interest points are preserved.

Usage:
    uv run python scripts/reencode_bdv_zarr.py <path-to-dataset.ome.zarr>
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr

TARGET_DTYPE = "<u2"
TARGET_COMPRESSOR = {"id": "gzip", "level": 5}


def _already_converted(sub: Path) -> bool:
    za = sub / "0" / ".zarray"
    if not za.exists():
        return False
    meta = json.loads(za.read_text())
    return meta.get("dtype") == TARGET_DTYPE and meta.get("compressor") == TARGET_COMPRESSOR


def _reencode_one(sub: Path) -> str:
    src_meta = json.loads((sub / "0" / ".zarray").read_text())
    data = np.asarray(zarr.open(str(sub / "0"), mode="r")[:]).astype("<u2")

    tmp = sub.parent / (sub.name + ".reenc")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "0").mkdir(parents=True)
    (tmp / ".zgroup").write_text((sub / ".zgroup").read_text())
    (tmp / ".zattrs").write_text((sub / ".zattrs").read_text())
    (tmp / "0" / ".zattrs").write_text(json.dumps({}))
    (tmp / "0" / ".zarray").write_text(json.dumps({
        "shape": src_meta["shape"],
        "chunks": src_meta["chunks"],
        "dtype": TARGET_DTYPE,
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
        "compressor": TARGET_COMPRESSOR,
        "zarr_format": 2,
    }))
    zarr.open_group(zarr.storage.LocalStore(str(tmp)), mode="a", zarr_format=2)["0"][:] = data

    # verify before swapping
    back = np.asarray(zarr.open(str(tmp / "0"), mode="r")[:])
    if back.shape != data.shape or not np.array_equal(back, data):
        shutil.rmtree(tmp)
        raise RuntimeError(f"verify failed for {sub.name}")

    bak = sub.parent / (sub.name + ".bak")
    if bak.exists():
        shutil.rmtree(bak)
    sub.rename(bak)
    tmp.rename(sub)
    shutil.rmtree(bak)
    return "converted"


def main() -> None:
    root = Path(sys.argv[1])
    subs = sorted(p for p in root.glob("s*-t*.zarr") if p.is_dir())
    print(f"root: {root}")
    print(f"sub-zarrs: {len(subs)}")
    done = skipped = 0
    t0 = time.time()
    for i, sub in enumerate(subs, 1):
        if _already_converted(sub):
            skipped += 1
            continue
        _reencode_one(sub)
        done += 1
        if done % 25 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] converted={done} skipped={skipped} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"done: converted={done} skipped={skipped} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
