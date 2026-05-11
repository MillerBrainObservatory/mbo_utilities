"""One-time rechunk of an IsoView corrected output tree.

Walks `<root>/SPM##/TM######/SPM##_TM######_CM##.zarr` files and
rewrites each in place with chunks ``(1, 1, 1, Y, X)`` (one chunk per
Y×X plane) wrapped in a single shard ``(1, 1, Z, Y, X)``. Drops cold
Y×X plane reads from ~150 ms to ~5 ms — see
``D:/demo/zarr_chunking_benchmark/benchmark.py`` for the evidence.

Each file is rewritten to a sibling ``*.zarr.rechunk-tmp/`` directory,
verified to round-trip, then atomically swapped over the original.
Original is preserved as ``*.zarr.bak/`` until the next file completes
successfully (then deleted) so a power-failure mid-rechunk leaves the
tree in a recoverable state.

Files that already have the target chunk shape are skipped, so re-running
this script is safe and idempotent.

Usage:
    uv run python scripts/rechunk_isoview.py <root> [--include-masks]
                                              [--dry-run] [--no-backup]

Examples:
    # rechunk only the volume zarrs (the GUI hot path)
    uv run python scripts/rechunk_isoview.py \\
        "D:/isoview_pipeline_demo/Dme_E1_..._.corrected/SPM00"

    # also rechunk the segmentation masks
    uv run python scripts/rechunk_isoview.py <root> --include-masks

    # preview what would be rewritten without touching anything
    uv run python scripts/rechunk_isoview.py <root> --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import (
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingCodec,
    ZstdCodec,
)
from zarr.storage import LocalStore


# matches SPM00_TM000005_CM02.zarr but not …segmentationMask.zarr
_VOLUME_GLOB = "**/SPM*_TM*_CM*.zarr"
_MASK_SUFFIX = ".segmentationMask.zarr"


def _is_already_rechunked(arr) -> bool:
    """True if the array's inner chunks are already (1, 1, 1, Y, X)."""
    if arr.ndim != 5:
        # 3D or 4D source — not the usual layout, skip
        return False
    chunks = arr.chunks
    return chunks[0] == 1 and chunks[1] == 1 and chunks[2] == 1


def _open_array(zarr_path: Path):
    z = zarr.open(str(zarr_path), mode="r")
    if hasattr(z, "shape"):
        return z, None
    if "0" in z:
        return z["0"], "0"
    keys = list(z)
    return z[keys[0]], keys[0]


def _human(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


def _disk_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _detect_compressor(arr) -> tuple[str, int]:
    """Best-effort: read the source's existing compressor + level so the
    rechunked file matches as closely as possible. Falls back to
    ``("gzip", 1)`` if we can't introspect.
    """
    # zarr v3: compressor lives in arr.metadata.codecs
    try:
        codecs = arr.metadata.codecs
    except AttributeError:
        return "gzip", 1
    for c in codecs:
        cls_name = type(c).__name__
        if cls_name == "ZstdCodec":
            return "zstd", getattr(c, "level", 1)
        if cls_name == "GzipCodec":
            return "gzip", getattr(c, "level", 1)
        if cls_name == "BloscCodec":
            return "blosc", getattr(c, "clevel", 1)
        # ShardingCodec wraps inner codecs; recurse
        if cls_name == "ShardingCodec":
            inner = getattr(c, "codecs", ())
            for ic in inner:
                if type(ic).__name__ == "ZstdCodec":
                    return "zstd", getattr(ic, "level", 1)
                if type(ic).__name__ == "GzipCodec":
                    return "gzip", getattr(ic, "level", 1)
    return "gzip", 1


def _build_inner_codecs(name: str, level: int) -> list:
    if name == "zstd":
        return [BytesCodec(), ZstdCodec(level=level)]
    return [BytesCodec(), GzipCodec(level=level)]


def rechunk_file(src_path: Path, *, dry_run: bool, keep_backup: bool) -> dict:
    """Rechunk one zarr in place. Returns a status dict for reporting."""
    status = {"path": src_path, "skipped": False, "error": None,
              "before_files": 0, "after_files": 0,
              "before_bytes": 0, "after_bytes": 0,
              "elapsed_s": 0.0}

    src_arr, sub_path = _open_array(src_path)
    status["before_bytes"] = _disk_size(src_path)
    status["before_files"] = sum(1 for f in src_path.rglob("*") if f.is_file())

    if _is_already_rechunked(src_arr):
        status["skipped"] = True
        status["after_files"] = status["before_files"]
        status["after_bytes"] = status["before_bytes"]
        return status

    if src_arr.ndim != 5 or src_arr.shape[0] != 1 or src_arr.shape[1] != 1:
        # not the (1, 1, Z, Y, X) per-(t, c) layout we know how to rechunk
        status["error"] = f"unexpected shape {src_arr.shape}"
        return status

    if dry_run:
        return status

    # read source data once. per-(t, c) volumes are ~108 MB; comfortable to
    # hold one at a time. the original file's chunks are Y×X-tiled, so the
    # full read decompresses every chunk — same cost as the GUI's old hot
    # path, but only paid once per rechunk.
    tmp_path = src_path.with_suffix(src_path.suffix + ".rechunk-tmp")
    bak_path = src_path.with_suffix(src_path.suffix + ".bak")

    # clean any leftovers from a prior interrupted run
    for p in (tmp_path, bak_path):
        if p.exists():
            shutil.rmtree(p)

    t0 = time.perf_counter()
    data = np.asarray(src_arr[:])

    # match the source's compressor settings so disk size stays comparable
    compressor, level = _detect_compressor(src_arr)
    inner_codecs = _build_inner_codecs(compressor, level)

    Y, X = data.shape[-2], data.shape[-1]
    Z = data.shape[-3]

    # inner chunk: one Y×X plane at one z. shard: all Z (entire volume)
    # in one shard. The source layout is per-(t, c) files, so each file
    # holds T=1, C=1, Z=Z; one shard per file is the most you can do.
    inner = (1, 1, 1, Y, X)
    shard = (1, 1, Z, Y, X)

    root = zarr.open_group(str(tmp_path), mode="w", zarr_format=3)
    new_codec = ShardingCodec(
        chunk_shape=inner,
        codecs=inner_codecs,
        index_codecs=[BytesCodec(), Crc32cCodec()],
    )
    out_path = sub_path or "0"
    new_arr = zarr.create(
        store=root.store,
        path=out_path,
        shape=data.shape,
        chunks=shard,
        dtype=data.dtype,
        codecs=[new_codec],
        overwrite=True,
    )
    # carry over user attrs (NOT codec/chunk metadata, that's structural)
    try:
        new_arr.attrs.update(dict(src_arr.attrs))
    except Exception:
        pass
    # OME-Zarr group attrs (multiscales, etc.) live on the parent group
    src_group = zarr.open(str(src_path), mode="r")
    if not hasattr(src_group, "shape"):
        # was a group → carry attrs over
        try:
            root.attrs.update(dict(src_group.attrs))
        except Exception:
            pass

    new_arr[:] = data

    # round-trip verification before we touch the original
    rt_arr, _ = _open_array(tmp_path)
    rt = np.asarray(rt_arr[:])
    if not np.array_equal(rt, data):
        shutil.rmtree(tmp_path, ignore_errors=True)
        status["error"] = "round-trip verification failed"
        return status

    # atomic-ish swap. on Windows shutil.move + rename gets a directory
    # rename which is atomic at the FS level. We move the original to
    # *.bak first, then move tmp into place. If the second move fails,
    # the original is still intact at *.bak.
    src_path.rename(bak_path)
    try:
        tmp_path.rename(src_path)
    except OSError:
        # restore original
        bak_path.rename(src_path)
        shutil.rmtree(tmp_path, ignore_errors=True)
        status["error"] = "atomic swap failed"
        return status

    if not keep_backup:
        shutil.rmtree(bak_path, ignore_errors=True)

    status["after_bytes"] = _disk_size(src_path)
    status["after_files"] = sum(1 for f in src_path.rglob("*") if f.is_file())
    status["elapsed_s"] = time.perf_counter() - t0
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="path to a *.corrected/SPM## directory or its parent")
    ap.add_argument(
        "--include-masks",
        action="store_true",
        help="also rechunk *.segmentationMask.zarr (default: skip)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list files that would be rechunked without writing anything",
    )
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="delete the .bak directory after each successful swap "
             "(default: keep until next file finishes, never accumulates)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"error: root not found: {root}", file=sys.stderr)
        return 2

    candidates = sorted(root.glob(_VOLUME_GLOB))
    if not args.include_masks:
        candidates = [p for p in candidates if _MASK_SUFFIX not in p.name]

    print(f"Found {len(candidates)} candidate zarr files under {root}")
    if not candidates:
        return 0

    if args.dry_run:
        for p in candidates[:10]:
            print(f"  would rechunk: {p}")
        if len(candidates) > 10:
            print(f"  … and {len(candidates) - 10} more")
        return 0

    rechunked = skipped = errored = 0
    total_before = total_after = 0
    t_start = time.perf_counter()

    for i, src in enumerate(candidates):
        try:
            status = rechunk_file(
                src, dry_run=False, keep_backup=not args.no_backup
            )
        except Exception as e:  # noqa: BLE001
            errored += 1
            print(f"  [{i+1}/{len(candidates)}] {src.name}: ERROR {e}")
            continue

        total_before += status["before_bytes"]
        total_after += status["after_bytes"] or status["before_bytes"]

        if status["error"]:
            errored += 1
            print(f"  [{i+1}/{len(candidates)}] {src.name}: {status['error']}")
        elif status["skipped"]:
            skipped += 1
            if i < 5 or i == len(candidates) - 1:
                print(f"  [{i+1}/{len(candidates)}] {src.name}: already rechunked, skip")
        else:
            rechunked += 1
            files_delta = status["after_files"] - status["before_files"]
            bytes_delta = status["after_bytes"] - status["before_bytes"]
            print(
                f"  [{i+1}/{len(candidates)}] {src.name}: "
                f"{status['elapsed_s']:.1f}s "
                f"files {status['before_files']}->{status['after_files']} "
                f"({files_delta:+d}), "
                f"size {_human(status['before_bytes'])}"
                f"->{_human(status['after_bytes'])} "
                f"({bytes_delta / max(1, status['before_bytes']) * 100:+.1f}%)"
            )

    elapsed = time.perf_counter() - t_start
    print()
    print(f"=== summary ===")
    print(f"  rechunked: {rechunked}")
    print(f"  skipped:   {skipped} (already had target chunks)")
    print(f"  errored:   {errored}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  total bytes: {_human(total_before)} -> {_human(total_after)}")
    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main())
