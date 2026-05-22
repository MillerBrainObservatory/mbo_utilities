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
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingCodec,
    ZstdCodec,
)
from zarr.storage import LocalStore


# matches SPM00_TM000005_CM02.zarr (corrected) and
# SPM00_TM000005_CM02_CM03_VW00.zarr (fused). Mask siblings under both
# trees are filtered below unless --include-masks is passed.
_VOLUME_GLOB = "**/SPM*_TM*_CM*.zarr"
# Substrings that mark a file as a mask companion rather than a volume.
# Matches ``foo.mask.zarr``, ``foo.mask.ome.zarr``, ``foo.segmentationMask.zarr``,
# etc. — anything with ``.mask.`` or ``segmentationMask`` in the directory name.
_MASK_MARKERS = (".mask.", "segmentationMask", "fusionMask")


def _is_mask(p: Path) -> bool:
    name = p.name
    return any(m in name for m in _MASK_MARKERS)


def _is_already_rechunked(arr) -> bool:
    """True if the array's inner chunks are already Z-narrow.

    For 5D ``(T, C, Z, Y, X)`` arrays this means chunks[0..2] == (1, 1, 1);
    for 4D ``(T, Z, Y, X)`` it means chunks[0..1] == (1, 1); etc. The
    invariant is: every dim except Y and X must be chunked at 1.
    """
    if arr.ndim < 2:
        return True
    chunks = arr.chunks
    return all(c == 1 for c in chunks[:-2])


def _iter_arrays(group):
    """Yield (path, array) for every sub-array under a zarr group.

    OME-Zarr pyramids land at top-level keys ``"0"``, ``"1"``, etc.; a
    plain array store has no children and reports itself with path ``"0"``.
    """
    if hasattr(group, "shape"):
        # group IS an array — wrap as a single-item iteration
        yield "0", group
        return
    keys = sorted(group.array_keys()) if hasattr(group, "array_keys") else sorted(group)
    for k in keys:
        try:
            arr = group[k]
        except KeyError:
            continue
        if hasattr(arr, "shape"):
            yield k, arr
        else:
            for sub_k, sub_arr in _iter_arrays(arr):
                yield f"{k}/{sub_k}", sub_arr


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


def _detect_compressor(arr) -> dict:
    """Best-effort: read the source's existing compressor settings so the
    rechunked file matches as closely as possible. Returns a dict the
    builder consumes; falls back to ``{"name": "gzip", "level": 1}`` when
    introspection fails.
    """
    try:
        codecs = arr.metadata.codecs
    except AttributeError:
        return {"name": "gzip", "level": 1}
    candidates = list(codecs)
    # ShardingCodec wraps the real compressor inside ``.codecs``; flatten.
    flattened: list = []
    for c in candidates:
        flattened.append(c)
        if type(c).__name__ == "ShardingCodec":
            flattened.extend(getattr(c, "codecs", ()) or ())
    for c in flattened:
        cls_name = type(c).__name__
        if cls_name == "BloscCodec":
            return {
                "name": "blosc",
                "cname": getattr(c, "cname", "zstd"),
                "level": getattr(c, "clevel", 1),
                "shuffle": getattr(c, "shuffle", "bitshuffle"),
            }
        if cls_name == "ZstdCodec":
            return {"name": "zstd", "level": getattr(c, "level", 1)}
        if cls_name == "GzipCodec":
            return {"name": "gzip", "level": getattr(c, "level", 1)}
    return {"name": "gzip", "level": 1}


def _build_inner_codecs(spec: dict) -> list:
    name = spec.get("name", "gzip")
    if name == "blosc":
        cname = spec.get("cname", "zstd")
        # zarr stores enum values internally; accept either str or enum.
        if hasattr(cname, "value"):
            cname = cname.value
        shuffle = spec.get("shuffle", "bitshuffle")
        if hasattr(shuffle, "value"):
            shuffle = shuffle.value
        return [
            BytesCodec(),
            BloscCodec(
                cname=cname,
                clevel=int(spec.get("level", 1)),
                shuffle=shuffle,
            ),
        ]
    if name == "zstd":
        return [BytesCodec(), ZstdCodec(level=int(spec.get("level", 1)))]
    return [BytesCodec(), GzipCodec(level=int(spec.get("level", 1)))]


def _rechunk_one_array(
    src_arr, dest_group, dest_path: str
) -> np.ndarray:
    """Rechunk a single zarr array into ``dest_group`` at ``dest_path``.

    Returns the materialized source data for round-trip verification.
    Chunk shape is ``(1,)*(ndim-2) + (Y, X)`` and the wrapping shard
    spans every non-spatial dim — so the on-disk file count stays
    constant regardless of T/C/Z extent.
    """
    data = np.asarray(src_arr[:])
    compressor_spec = _detect_compressor(src_arr)
    inner_codecs = _build_inner_codecs(compressor_spec)

    if data.ndim < 2:
        # 1D or scalar: just copy as-is, single chunk
        new_arr = zarr.create(
            store=dest_group.store,
            path=dest_path,
            shape=data.shape,
            chunks=data.shape or (1,),
            dtype=data.dtype,
            codecs=inner_codecs,
            overwrite=True,
        )
    else:
        Y, X = data.shape[-2], data.shape[-1]
        # one Y×X plane per inner chunk; outer shard spans all non-spatial
        # dims so file count is one shard per array.
        inner = (1,) * (data.ndim - 2) + (Y, X)
        shard = data.shape
        new_codec = ShardingCodec(
            chunk_shape=inner,
            codecs=inner_codecs,
            index_codecs=[BytesCodec(), Crc32cCodec()],
        )
        new_arr = zarr.create(
            store=dest_group.store,
            path=dest_path,
            shape=data.shape,
            chunks=shard,
            dtype=data.dtype,
            codecs=[new_codec],
            overwrite=True,
        )

    try:
        new_arr.attrs.update(dict(src_arr.attrs))
    except Exception:
        pass
    new_arr[:] = data
    return data


def rechunk_file(src_path: Path, *, dry_run: bool, keep_backup: bool) -> dict:
    """Rechunk one zarr (or OME-Zarr group) in place.

    For pyramidal stores every sub-array under the group (level 0, 1, …)
    is rewritten with the same Y×X-plane chunk layout, and the parent
    group's attrs (multiscales metadata, omero block, etc.) are carried
    over verbatim.
    """
    status = {"path": src_path, "skipped": False, "error": None,
              "before_files": 0, "after_files": 0,
              "before_bytes": 0, "after_bytes": 0,
              "levels": 0, "elapsed_s": 0.0}

    src_group = zarr.open(str(src_path), mode="r")
    status["before_bytes"] = _disk_size(src_path)
    status["before_files"] = sum(1 for f in src_path.rglob("*") if f.is_file())

    arrays = list(_iter_arrays(src_group))
    if not arrays:
        status["error"] = "no arrays found in store"
        return status
    status["levels"] = len(arrays)

    if all(_is_already_rechunked(a) for _, a in arrays):
        status["skipped"] = True
        status["after_files"] = status["before_files"]
        status["after_bytes"] = status["before_bytes"]
        return status

    if dry_run:
        return status

    tmp_path = src_path.with_suffix(src_path.suffix + ".rechunk-tmp")
    bak_path = src_path.with_suffix(src_path.suffix + ".bak")

    for p in (tmp_path, bak_path):
        if p.exists():
            shutil.rmtree(p)

    t0 = time.perf_counter()

    dest_group = zarr.open_group(str(tmp_path), mode="w", zarr_format=3)
    if not hasattr(src_group, "shape"):
        try:
            dest_group.attrs.update(dict(src_group.attrs))
        except Exception:
            pass

    # rewrite each level and remember the read-back data for verification
    written: list[tuple[str, np.ndarray]] = []
    for sub_path, src_arr in arrays:
        out_path = sub_path if sub_path else "0"
        try:
            data = _rechunk_one_array(src_arr, dest_group, out_path)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(tmp_path, ignore_errors=True)
            status["error"] = f"write failed at {out_path}: {exc}"
            return status
        written.append((out_path, data))

    # round-trip verification (each sub-array)
    for out_path, data in written:
        rt = np.asarray(zarr.open_array(str(tmp_path), path=out_path, mode="r")[:])
        if not np.array_equal(rt, data):
            shutil.rmtree(tmp_path, ignore_errors=True)
            status["error"] = f"round-trip mismatch at {out_path}"
            return status

    # atomic-ish swap via .bak indirection.
    src_path.rename(bak_path)
    try:
        tmp_path.rename(src_path)
    except OSError:
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
        candidates = [p for p in candidates if not _is_mask(p)]

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
            levels = status.get("levels", 1)
            print(
                f"  [{i+1}/{len(candidates)}] {src.name}: "
                f"{status['elapsed_s']:.1f}s lvls={levels} "
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
