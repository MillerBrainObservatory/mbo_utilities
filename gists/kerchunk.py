#!/usr/bin/env python3

import io
import json
import pathlib
import tempfile
import shutil

import numpy as np
import tifffile
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from skimage import data as skdata
import fastplotlib as fpl
from tifffile import TiffFile


def dump_kerchunk_reference(tif_path: pathlib.Path, base_dir: pathlib.Path) -> dict:
    """
    Create a kerchunk reference for a single TIFF file.

    Parameters
    ----------
    tif_path : pathlib.Path
        Path to the TIFF file on disk.
    base_dir : pathlib.Path
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


def create_combined_kerchunk_reference(
    tif_files: list[pathlib.Path], base_dir: pathlib.Path
) -> dict:
    """
    Stack multiple single‐file kerchunk references into one combined reference.

    Each TIFF’s kerchunk reference is prefixed by its index, so that the final
    combined JSON represents a 4D array of shape (2, T, Y, X), where the new
    leading axis corresponds to the index of the TIFF in `tif_files`.

    Parameters
    ----------
    tif_files : list of pathlib.Path
        At least two TIFF paths that have identical shape and chunking.
    base_dir : pathlib.Path
        Base directory used for the “url” of the kerchunk references.

    Returns
    -------
    combined_refs : dict
        A single kerchunk reference dict representing a stacked 4D array.
    """
    assert len(tif_files) > 1, "Need at least two TIFF files to combine."

    combined_refs: dict[str, str] = {}
    ref_shape = None
    ref_chunks = None

    for idx, tif_path in enumerate(tif_files):
        inner_refs = dump_kerchunk_reference(tif_path, base_dir)
        zarr_meta = json.loads(inner_refs.pop(".zarray"))
        inner_refs.pop(".zattrs", None)

        if idx == 0:
            # Base shape and chunks from the first file
            ref_shape = zarr_meta["shape"]
            ref_chunks = zarr_meta["chunks"]

            # Build the combined zarray metadata with a new leading dimension
            combined_zarr_meta = zarr_meta.copy()
            combined_zarr_meta["shape"] = [len(tif_files), *ref_shape]
            combined_zarr_meta["chunks"] = [1, *ref_chunks]
            combined_refs[".zarray"] = json.dumps(combined_zarr_meta)
            combined_refs[".zattrs"] = json.dumps(
                {"_ARRAY_DIMENSIONS": ["Z", "T", "Y", "X"]}
            )
        else:
            pass
            # Ensure every subsequent TIFF has the same shape & chunks
            # assert zarr_meta["shape"] == ref_shape, (
            #     f"Shape mismatch: {tif_path} → {zarr_meta['shape']} vs {ref_shape}"
            # )
            # assert zarr_meta["chunks"] == ref_chunks, (
            #     f"Chunk-size mismatch: {tif_path} → {zarr_meta['chunks']} vs {ref_chunks}"
            # )

        # Prefix each chunk key by its index
        for key, val in inner_refs.items():
            combined_refs[f"{idx}.{key}"] = val

    return combined_refs

def main_mbo():

    tmp_dir = pathlib.Path("/home/flynn/lbm_data/raw/out")
    tmp_dir.mkdir(exist_ok=True)

    files = [x for x in pathlib.Path("/home/flynn/lbm_data/raw").glob("*.tif")]
    tfiles = [TiffFile(str(f)) for f in files]

    print("Generating combined kerchunk reference…")
    combined_refs = create_combined_kerchunk_reference(
        tif_files=files, base_dir=tmp_dir
    )

    combined_json_path = tmp_dir / "combined_refs.json"
    with open(combined_json_path, "w") as f:
        json.dump(combined_refs, f)
    print(f"Combined kerchunk reference written to {combined_json_path}")

    store = zarr.storage.FsspecStore(
        ReferenceFileSystem(combined_refs, asynchronous=True)
    )
    z_arr = zarr.open(store)
    iw = fpl.ImageWidget(z_arr)
    iw.show()

    fpl.loop.run()
    print(f"Combined Zarr has shape {z_arr.shape} and dtype {z_arr.dtype}")
    print(files)

def main_cells3d():
    cells: np.ndarray = skdata.cells3d()

    membranes = cells[:, 0, :, :]
    nuclei = cells[:, 1, :, :]

    tmp_dir = pathlib.Path().cwd().joinpath("temp")
    tmp_dir.mkdir(exist_ok=True)
    try:
        # Paths for output TIFFs
        membranes_tif = tmp_dir / "membranes.tiff"
        nuclei_tif = tmp_dir / "nuclei.tiff"

        print(f"Writing membranes → {membranes_tif}")
        tifffile.imwrite(str(membranes_tif), membranes, photometric="minisblack")
        print(f"Writing nuclei → {nuclei_tif}")
        tifffile.imwrite(str(nuclei_tif), nuclei, photometric="minisblack")

        print("Generating combined kerchunk reference…")
        combined_refs = create_combined_kerchunk_reference(
            tif_files=[membranes_tif, nuclei_tif], base_dir=tmp_dir
        )

        combined_json_path = tmp_dir / "combined_refs.json"
        with open(combined_json_path, "w") as f:
            json.dump(combined_refs, f)
        print(f"Combined kerchunk reference written to {combined_json_path}")

        store = zarr.storage.FsspecStore(
            ReferenceFileSystem(combined_refs, asynchronous=False)
        )
        z_arr = zarr.open(store)
        iw = fpl.ImageWidget(z_arr)
        iw.show()

        fpl.loop.run()
        print(f"Combined Zarr has shape {z_arr.shape} and dtype {z_arr.dtype}")


    finally:
        # 7. Clean up the temporary directory
        print(f"Cleaning up temporary directory {tmp_dir} …")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main_mbo()
