import argparse
import shutil
import time
from pathlib import Path

def dir_size_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def copy_dir(src: Path, dst: Path) -> dict:
    if dst.exists():
        shutil.rmtree(dst)
    start = time.time()
    shutil.copytree(src, dst)
    elapsed = time.time() - start
    size_bytes = dir_size_bytes(src)
    mb = size_bytes / (1024 ** 2)
    mbps = mb / elapsed if elapsed > 0 else 0
    return {"name": src.name, "size_mb": mb, "elapsed": elapsed, "mbps": mbps}

def main():
    parser = argparse.ArgumentParser(
        description="Copy local result folders (e.g. zarr outputs) to SMB or network path and record metrics."
    )
    parser.add_argument(
        "--src",
        nargs="+",
        required=True,
        help="One or more local source directories to copy (e.g. ./zarr ./zarr-sharded).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination root directory (e.g. \\\\rbo-s1\\S1_DATA\\lbm\\foconnell).",
    )
    args = parser.parse_args()

    dst_root = Path(args.dst)
    if not dst_root.exists():
        print(f"Creating destination root: {dst_root}")
        dst_root.mkdir(parents=True, exist_ok=True)

    results = []
    for src_str in args.src:
        src = Path(src_str).resolve()
        if not src.exists() or not src.is_dir():
            print(f"Skipping missing folder: {src}")
            continue

        print(f"Copying {src} → {dst_root / src.name}")
        metrics = copy_dir(src, dst_root / src.name)
        results.append(metrics)
        print(
            f"{metrics['name']}: {metrics['size_mb']:.1f} MB in "
            f"{metrics['elapsed']:.2f}s ({metrics['mbps']:.1f} MB/s)"
        )

    print("\nSummary:")
    total_mb = sum(m["size_mb"] for m in results)
    total_time = sum(m["elapsed"] for m in results)
    avg_mbps = total_mb / total_time if total_time > 0 else 0
    for m in results:
        print(f"  {m['name']:<15} {m['size_mb']:8.1f} MB  {m['elapsed']:8.2f}s  {m['mbps']:6.1f} MB/s")
    print(f"\nTotal: {total_mb:.1f} MB in {total_time:.2f}s (avg {avg_mbps:.1f} MB/s)")

if __name__ == "__main__":
    main()
