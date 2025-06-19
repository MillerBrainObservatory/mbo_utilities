#!/usr/bin/env python3

import glfw
import time
from typing import Any
from rendercanvas.glfw import GlfwRenderCanvas
from rendercanvas.auto import loop
import io
import json
from pathlib import Path
import tempfile
import shutil
import zarr.core.sync
import asyncio
from zarr.core import sync
import dask.array as da

import numpy as np
import tifffile
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from skimage import data as skdata
import fastplotlib as fpl
from tifffile import TiffFile
from mbo_utilities.lazy_array import imread, imwrite

import uuid
import subprocess

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def dump_kerchunk_reference(tif_path: Path, base_dir: Path) -> dict:
    """
    Create a kerchunk reference for a single TIFF file.

    Parameters
    ----------
    tif_path : Path
        Path to the TIFF file on disk.
    base_dir : Path
        Directory representing the “root” URI for the reference.

    Returns
    -------
    refs : dict
        A kerchunk reference dict (in JSON form) for this single TIFF.
    """
    with tifffile.TiffFile(str(tif_path.expanduser().resolve())) as tif:
        with io.StringIO() as f:
            store = tif.aszarr()
            store.write_fsspec(f, url=base_dir.as_uri())
            refs = json.loads(f.getvalue())
    return refs

def create_combined_kerchunk_reference(tif_files: list[Path], base_dir: Path) -> dict:
    assert len(tif_files) > 1, "Need at least two TIFF files to combine."

    combined_refs: dict[str, str] = {}
    per_file_refs = []
    total_shape = None
    total_chunks = None
    zarr_meta = {}
    for tif_path in tif_files:
        inner_refs = dump_kerchunk_reference(tif_path, base_dir)
        zarr_meta = json.loads(inner_refs.pop(".zarray"))
        inner_refs.pop(".zattrs", None)

        shape = zarr_meta["shape"]
        chunks = zarr_meta["chunks"]

        if total_shape is None:
            total_shape = shape.copy()
            total_chunks = chunks
        else:
            assert shape[1:] == total_shape[1:], f"Shape mismatch in {tif_path}"
            assert chunks == total_chunks, f"Chunk mismatch in {tif_path}"
            total_shape[0] += shape[0]  # accumulate along axis 0

        per_file_refs.append((inner_refs, shape))

    combined_zarr_meta = {
        "shape": total_shape,
        "chunks": total_chunks,
        "dtype": zarr_meta["dtype"],
        "compressor": zarr_meta["compressor"],
        "filters": zarr_meta.get("filters", None),
        "order": zarr_meta["order"],
        "zarr_format": zarr_meta["zarr_format"],
        "fill_value": zarr_meta.get("fill_value", 0),
    }

    combined_refs[".zarray"] = json.dumps(combined_zarr_meta)
    combined_refs[".zattrs"] = json.dumps(
        {"_ARRAY_DIMENSIONS": ["T", "C", "Y", "X"][:len(total_shape)]}
    )

    axis0_offset = 0
    for inner_refs, shape in per_file_refs:
        chunksize0 = total_chunks[0]
        for key, val in inner_refs.items():
            idx = list(map(int, key.strip("/").split(".")))
            idx[0] += axis0_offset // chunksize0
            new_key = ".".join(map(str, idx))
            combined_refs[new_key] = val
        axis0_offset += shape[0]

    return combined_refs

def save_fsspec(tiff_path: str | Path):
    tiff_path = Path(tiff_path).expanduser().resolve()
    if tiff_path.is_file():
        tiff_path = Path(tiff_path).parent

    files = [x for x in Path(tiff_path).glob("*tif*")]

    print(f"Generating combined kerchunk reference for {len(files)} files…")
    combined_refs = create_combined_kerchunk_reference(
        tif_files=files, base_dir=tiff_path
    )
    combined_json_path = tiff_path / "combined_refs.json"
    with open(combined_json_path, "w") as f:
        json.dump(combined_refs, f)
    print(f"Combined kerchunk reference written to {combined_json_path}")
    return combined_json_path

# async def load_zarr_data(spec_path):
def load_zarr_data(spec_path):
    store = zarr.storage.FsspecStore(ReferenceFileSystem(str(spec_path)))
    z_arr = zarr.open(store, mode="r",)
    return z_arr

def _load_existing(save_path: Path) -> list[dict[str, Any]]:
    if not save_path.exists():
        return []
    try:
        return json.loads(save_path.read_text())
    except Exception:
        return []

def _increment_label(existing: list[dict[str, Any]], base_label: str) -> str:
    count = 1
    labels = {e["label"] for e in existing if "label" in e}
    if base_label not in labels:
        return base_label
    while f"{base_label} [{count+1}]" in labels:
        count += 1
    return f"{base_label} [{count+1}]"


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
        "uuid": str(uuid.uuid4()),
        "git_commit": _get_git_commit(),
        "label": final_label,
        "index_slices": list(index_slices.keys()),
        "results": results,
    }

    existing.append(entry)
    save_path.write_text(json.dumps(existing, indent=2))

    return results

if __name__ == "__main__":
    data = imread(
        r"/home/flynn/lbm_data/raw",
    )
    store = data.data.as_zarr()
    mem = np.concat([tifffile.imread(str(p)) for p in data.fpath])
    print(data.shape)
    print(store.shape)
    print(mem.shape)

    _benchmark_indexing(
        arrays={"data": data, "store": store, "mem": mem},
        save_path=Path("/tmp/01/benchmark_results.json"),
        num_repeats=5,
        index_slices={
            "[:20,0,:,:]": (slice(0, 20), 0, slice(None), slice(None)),
            "[:20,0,:40,:40]": (slice(None), 0, slice(0, 40), slice(0, 40)),
        },
        label="Benchmark_v1",
    )
    
    #
    # data.imshow()
    # loop.run()
    # asyncio.run(main())