from collections import defaultdict
from pathlib import Path
import numpy as np


def safe_delete(file_path):
    if file_path.exists():
        try:
            file_path.unlink()
        except PermissionError:
            print(f"Error: Cannot delete {file_path}, it's open elsewhere.")


def group_plane_rois(input_dir):
    input_dir = Path(input_dir)
    grouped = defaultdict(list)

    for d in input_dir.iterdir():
        if d.is_dir() and d.stem.startswith("plane") and "_roi" in d.stem:
            parts = d.stem.split("_")
            if len(parts) == 2 and parts[1].startswith("roi"):
                plane = parts[0]  # "plane01"
                grouped[plane].append(d)

    return grouped


def load_ops(ops_input: str | Path | list[str | Path]) -> dict:
    """Simple utility load a suite2p npy file"""
    if isinstance(ops_input, (str, Path)):
        return np.load(ops_input, allow_pickle=True).item()
    elif isinstance(ops_input, dict):
        return ops_input
    print("Warning: No valid ops file provided, returning empty dict.")
    return {}
