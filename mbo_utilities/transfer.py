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
    mb = size_bytes / (1024**2)
    mbps = mb / elapsed if elapsed > 0 else 0
    return {"name": src.name, "size_mb": mb, "elapsed": elapsed, "mbps": mbps}


def main():
    parser = argparse.ArgumentParser(
        description="Copy local result folders to SMB or network path, with I/O benchmarks."
    )
    parser.add_argument(
        "--src",
        nargs="+",
        required=True,
        help="One or more local source directories to copy.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination root directory.",
    )
    args = parser.parse_args()

    dst_root = Path(args.dst)
    if not dst_root.exists():
        dst_root.mkdir(parents=True, exist_ok=True)

    results = []
    for src_str in args.src:
        src = Path(src_str).resolve()
        if not src.exists() or not src.is_dir():
            continue

        metrics = copy_dir(src, dst_root / src.name)
        results.append(metrics)

    total_mb = sum(m["size_mb"] for m in results)
    total_time = sum(m["elapsed"] for m in results)
    total_mb / total_time if total_time > 0 else 0
    for _m in results:
        pass


if __name__ == "__main__":
    main()
