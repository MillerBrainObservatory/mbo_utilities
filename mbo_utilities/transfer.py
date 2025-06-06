import argparse
import concurrent.futures
import sys
import threading
from pathlib import Path

try:
    from fabric import Connection
except ImportError:
    print(
        "Error: Fabric library is not installed. Please install it with 'pip install fabric'.",
        file=sys.stderr,
    )
    sys.exit(1)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI to sync a local directory to HPC cluster in parallel using Fabric (SFTP over SSH)."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Local path to the data directory (must be a directory).",
    )
    parser.add_argument(
        "--user", type=str, required=True, help="Username for the remote server."
    )
    parser.add_argument(
        "--remote_host",
        type=str,
        default="dtn02-hpc.rockefeller.edu",
        help="Remote server hostname.",
    )
    parser.add_argument(
        "--remote_path",
        type=str,
        help="Path appended to /lustre/fs4/mbo/scratch/{user}.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        help="List of patterns to exclude (e.g., temp/ or .git/ ).",
    )
    return parser.parse_args()


def validate_paths(data_path, remote_path):
    data_path = Path(data_path).resolve()
    if not data_path.exists() or not data_path.is_dir():
        print(
            f"❌ Error: '{data_path}' must exist and be a directory.", file=sys.stderr
        )
        sys.exit(1)
    if " " in remote_path:
        print(
            f"❌ Error: Remote path '{remote_path}' contains spaces.", file=sys.stderr
        )
        sys.exit(1)
    return data_path, remote_path


def transfer_file(host, user, local_file, remote_file_path, pbar, pbar_lock):
    try:
        file_size = local_file.stat().st_size
        transferred = 0
        # Open a dedicated connection for this file transfer
        with Connection(host=host, user=user) as conn:
            with open(local_file, "rb") as lf, conn.sftp() as sftp:
                remote_dir = "/".join(remote_file_path.split("/")[:-1])
                conn.run(f'mkdir -p "{remote_dir}"')
                with sftp.open(remote_file_path, "wb") as rf:
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = lf.read(chunk_size)
                        if not chunk:
                            break
                        rf.write(chunk)
                        transferred += len(chunk)
                        with pbar_lock:
                            pbar.update(len(chunk))
        if transferred < file_size:
            raise ValueError(
                f"Incomplete transfer for {local_file}: {transferred}/{file_size} bytes."
            )
    except Exception as e:
        print(f"❌ Error transferring {local_file}: {e}", file=sys.stderr)


def transfer_files(host, user, local_path, remote_path, exclude_patterns=None):
    print(f"📂 Starting file transfer from {local_path} to {remote_path}.")
    all_files = [f for f in local_path.rglob("*") if f.is_file()]

    # Filter out excluded files
    if exclude_patterns:

        def should_exclude(f):
            rel_path = str(f.relative_to(local_path)).replace("\\", "/")
            return any(rel_path.startswith(p.rstrip("/")) for p in exclude_patterns)

        all_files = [f for f in all_files if not should_exclude(f)]

    total_files = len(all_files)
    total_size = sum(f.stat().st_size for f in all_files)

    if total_files == 0:
        print("⚠️ No files to transfer. Check exclusions or directory.")
        return

    print(f"📦 Total files: {total_files} | Total size: {total_size / 1e9:.2f} GB")
    pbar_lock = threading.Lock()
    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Transferring",
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for f in all_files:
                relative_path = f.relative_to(local_path)
                remote_file_path = f"{remote_path}/{relative_path}"
                print(f"🚀 Queueing {f} -> {remote_file_path}")
                futures.append(
                    executor.submit(
                        transfer_file, host, user, f, remote_file_path, pbar, pbar_lock
                    )
                )
            concurrent.futures.wait(futures)


def main():
    args = parse_args()

    if not args.remote_path:
        remote_path = f"/lustre/fs4/mbo/scratch/{args.user}"
    else:
        remote_path = (
            f"/lustre/fs4/mbo/scratch/{args.user}/{args.remote_path.strip('/')}"
        )

    local_dir, remote_path = validate_paths(args.data, remote_path)
    transfer_files(
        args.remote_host,
        args.user,
        local_dir,
        remote_path,
        exclude_patterns=args.exclude,
    )
    print("✅ Transfer complete!")


if __name__ == "__main__":
    main()
