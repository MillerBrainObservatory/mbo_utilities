import json
import time
import uuid as _uuid
from pathlib import Path

import numpy as np
import zarr
from dask import array as da

from mbo_utilities import get_mbo_dirs
from mbo_utilities._parsing import _load_existing, _increment_label, _get_git_commit

def run_benchmark(*arrays, uuid=None):
    """
    Run a benchmark for indexing arrays and save the results to a JSON file.

    This function is a wrapper that sets up the necessary parameters and calls
    the `_benchmark_indexing` function.

    Returns
    -------
    dict
        The results of the benchmarking.
    """
    tests_dir = get_mbo_dirs()["tests"]
    save_path = tests_dir / "benchmark_indexing.json"
    return _benchmark_indexing(arrays, save_path)

def _benchmark_indexing(
    arrays: dict[str, np.ndarray | da.Array | zarr.Array],
    save_path: Path,
    num_repeats: int = 5,
    index_slices: dict[str, tuple[slice | int, ...]] = None,
    label: str = None,
):
    if index_slices is None:
        index_slices = {
            "[:200,0,:,:]": (slice(0, 200), 0, slice(None), slice(None)),
            "[:,0,:40,:40]": (slice(None), 0, slice(0, 40), slice(0, 40)),
        }

    results = {}
    for name, array in arrays.items():
        results[name] = {}
        for label_idx, idx in index_slices.items():
            times = []
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                val = array[idx]
                if isinstance(val, da.Array):
                    val.compute()
                elif hasattr(val, "read"):
                    np.array(val)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            results[name][label_idx] = {
                "min": round(min(times), 3),
                "max": round(max(times), 3),
                "mean": round(sum(times) / len(times), 3),
            }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(save_path)
    final_label = _increment_label(existing, label or "Unnamed Run")

    entry = {
        "uuid": str(_uuid.uuid4()),
        "git_commit": _get_git_commit(),
        "label": final_label,
        "index_slices": list(index_slices.keys()),
        "results": results,
    }

    existing.append(entry)
    save_path.write_text(json.dumps(existing, indent=2))
    return results
