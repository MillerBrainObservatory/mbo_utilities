#!/usr/bin/env python3

import glfw
from rendercanvas.glfw import GlfwRenderCanvas
from rendercanvas.auto import loop
import io
import json
from pathlib import Path
import tempfile
import shutil
import zarr.core.sync
import asyncio
import zarr
from zarr.core import sync
import dask.array as da

import numpy as np
import tifffile
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from skimage import data as skdata
import fastplotlib as fpl
from tifffile import TiffFile


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
        # Generate an “aszarr” (Kerchunk) store and write it into a StringIO buffer
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

    # axis0_offset = 0
    # for inner_refs, shape in per_file_refs:
    #     chunksize0 = total_chunks[0]
    #     for key, val in inner_refs.items():
    #         idx = list(map(int, key.strip("/").split(".")))
    #         idx[0] += axis0_offset // chunksize0
    #         new_key = ".".join(map(str, idx))
    #         combined_refs[new_key] = val
    #     axis0_offset += shape[0]

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
    arr = da.from_zarr(z_arr, chunks=z_arr.chunks)
    return arr

# async def main():
def get_iw():
    spec_path = save_fsspec("/home/flynn/lbm_data/raw")
    z_arr = load_zarr_data(spec_path)
    iw = fpl.ImageWidget(z_arr)
    return iw

if __name__ == "__main__":
    import sniffio
    libname = sniffio.current_async_library()
    iw = get_iw()
    iw.show()
    loop.run()
    # asyncio.run(main())