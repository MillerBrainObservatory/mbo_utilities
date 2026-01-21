"""
Demo script for mbo_utilities file format handling.

Reads from D:/demo/raw, demonstrates array types, saves to various formats,
and reloads to verify round-trip behavior.

Output is saved to docs/_static/demo_output.txt for inclusion in documentation.
"""

import os
import sys
import shutil
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

# suppress tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import mbo_utilities as mbo

# output file for docs
DOCS_OUTPUT = Path(__file__).parent.parent / "docs" / "_static" / "demo_output.txt"


def print_arr_info(label: str, arr, fpath=None):
    """Print array information."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    if fpath:
        print(f"  Path: {fpath}")
    print(f"  Type: {type(arr).__name__}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    if hasattr(arr, "dims"):
        print(f"  Dims: {arr.dims}")
    if hasattr(arr, "num_planes"):
        print(f"  Planes: {arr.num_planes}")
    if hasattr(arr, "num_rois"):
        print(f"  ROIs: {arr.num_rois}")
    if hasattr(arr, "stack_type"):
        print(f"  Stack Type: {arr.stack_type}")
    if hasattr(arr, "filenames") and arr.filenames:
        fnames = arr.filenames
        # handle single path vs list
        if isinstance(fnames, (str, Path)):
            print(f"  File: {Path(fnames).name}")
        elif hasattr(fnames, "__len__"):
            if len(fnames) <= 3:
                for f in fnames:
                    print(f"  File: {Path(f).name}")
            else:
                print(f"  Files: {len(fnames)} files")
                print(f"    First: {Path(fnames[0]).name}")
                print(f"    Last: {Path(fnames[-1]).name}")


def clean_output_dir(output_dir: Path):
    """Remove output directory if it exists."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def demo_scanimage_workflow():
    """Demo reading raw ScanImage data and saving to different formats."""
    print("\n" + "#"*60)
    print("# ScanImage Workflow Demo")
    print("#"*60)

    raw_dir = Path("D:/demo/raw")
    output_dir = Path("D:/demo/output_test")
    clean_output_dir(output_dir)

    # find tiff files
    tiff_files = sorted(raw_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No .tif files found in {raw_dir}")
        return

    print(f"\nFound {len(tiff_files)} TIFF files in {raw_dir}")

    # read with imread (auto-detection)
    print("\n--- Reading with mbo.imread() ---")
    arr = mbo.imread(str(raw_dir))
    print_arr_info("Source Array (imread)", arr, raw_dir)

    # show some metadata
    if hasattr(arr, "metadata") and arr.metadata:
        print("\n  Key metadata:")
        for key in ["fs", "dz", "dx", "dy", "num_planes", "num_rois"]:
            if key in arr.metadata:
                print(f"    {key}: {arr.metadata[key]}")

    # save to zarr
    print("\n--- Saving to Zarr ---")
    zarr_path = output_dir / "demo_data.zarr"
    mbo.imwrite(arr, zarr_path, ext=".zarr", frames=range(100))
    print(f"  Saved to: {zarr_path}")

    # reload zarr
    zarr_arr = mbo.imread(str(zarr_path))
    print_arr_info("Reloaded Zarr", zarr_arr, zarr_path)

    # save to tiff
    print("\n--- Saving to TIFF ---")
    tiff_path = output_dir / "demo_data.tiff"
    mbo.imwrite(arr, tiff_path, ext=".tiff", frames=range(100))
    print(f"  Saved to: {tiff_path}")

    # reload tiff
    tiff_arr = mbo.imread(str(tiff_path))
    print_arr_info("Reloaded TIFF", tiff_arr, tiff_path)

    # save to binary (suite2p format)
    print("\n--- Saving to Binary (.bin) ---")
    bin_dir = output_dir / "demo_bin"
    bin_dir.mkdir(exist_ok=True)
    mbo.imwrite(arr, bin_dir, ext=".bin", frames=range(100), planes=1)
    # find the actual bin file and ops.npy created
    bin_files = list(bin_dir.glob("*.bin")) + list(bin_dir.glob("*/*.bin"))
    ops_files = list(bin_dir.glob("**/ops.npy"))
    if bin_files and ops_files:
        bin_file = bin_files[0]
        ops_file = ops_files[0]
        print(f"  Saved to: {bin_file}")
        print(f"  Ops file: {ops_file}")
        # read ops to get shape
        ops = np.load(ops_file, allow_pickle=True).item()
        shape = (ops.get("nframes", ops.get("num_frames", 100)), ops.get("Ly", 550), ops.get("Lx", 448))
        # reload using BinArray directly
        from mbo_utilities.arrays import BinArray
        bin_arr = BinArray(str(bin_file), shape=shape)
        print_arr_info("Reloaded Binary (BinArray)", bin_arr, bin_file)
    elif ops_files:
        # try reading via Suite2pArray from ops parent
        bin_parent = ops_files[0].parent
        print(f"  Saved to: {bin_parent}")
        bin_arr = mbo.imread(str(bin_parent))
        print_arr_info("Reloaded Binary (Suite2p)", bin_arr, bin_parent)
    else:
        print(f"  No binary output found in {bin_dir}")

    # save to hdf5
    print("\n--- Saving to HDF5 ---")
    h5_path = output_dir / "demo_data.h5"
    mbo.imwrite(arr, h5_path, ext=".h5", frames=range(100), planes=1)
    print(f"  Saved to: {h5_path}")

    # reload hdf5
    h5_arr = mbo.imread(str(h5_path))
    print_arr_info("Reloaded HDF5", h5_arr, h5_path)

    # save to npy
    print("\n--- Saving to NPY ---")
    npy_path = output_dir / "demo_data.npy"
    # extract small subset to memory first
    subset = np.array(arr[:50, 0, :, :])  # 50 frames, plane 0
    np_arr = mbo.imread(subset)
    mbo.imwrite(np_arr, npy_path, ext=".npy")
    print(f"  Saved to: {npy_path}")

    # reload npy
    npy_arr = mbo.imread(str(npy_path))
    print_arr_info("Reloaded NPY", npy_arr, npy_path)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


def demo_numpy_workflow():
    """Demo wrapping numpy arrays and saving to different formats."""
    print("\n" + "#"*60)
    print("# NumPy Array Workflow Demo")
    print("#"*60)

    output_dir = Path("D:/demo/output_numpy")
    clean_output_dir(output_dir)

    # create synthetic 4D data
    print("\n--- Creating synthetic 4D volume ---")
    data = np.random.randn(100, 8, 256, 256).astype(np.float32)
    print(f"  Raw numpy shape: {data.shape}")
    print(f"  Raw numpy dtype: {data.dtype}")

    # wrap with imread
    arr = mbo.imread(data)
    print_arr_info("Wrapped NumpyArray", arr)

    # save to each format
    formats = [".zarr", ".tiff", ".bin", ".h5", ".npy"]

    for ext in formats:
        print(f"\n--- Round-trip test: {ext} ---")
        out_path = output_dir / f"synthetic{ext}"

        # for bin, only write single plane (3D) and handle reload differently
        if ext == ".bin":
            bin_dir = output_dir / "synthetic_bin"
            bin_dir.mkdir(exist_ok=True)
            mbo.imwrite(arr, bin_dir, ext=ext, planes=1)
            # find actual files
            bin_files = list(bin_dir.glob("*.bin"))
            ops_files = list(bin_dir.glob("ops.npy"))
            if bin_files and ops_files:
                bin_file = bin_files[0]
                ops = np.load(ops_files[0], allow_pickle=True).item()
                shape = (ops.get("nframes", 100), ops.get("Ly", 256), ops.get("Lx", 256))
                print(f"  Saved to: {bin_file}")
                from mbo_utilities.arrays import BinArray
                reloaded = BinArray(str(bin_file), shape=shape)
                print_arr_info(f"Reloaded {ext}", reloaded, bin_file)
            else:
                print(f"  Binary write failed - no files found")
            continue

        mbo.imwrite(arr, out_path, ext=ext)
        print(f"  Saved to: {out_path}")

        # reload
        reloaded = mbo.imread(str(out_path))
        print_arr_info(f"Reloaded {ext}", reloaded, out_path)

    print("\n" + "="*60)
    print("NumPy workflow demo complete!")
    print("="*60)


def run_demos():
    """Run all demos and return output."""
    # check if raw data exists
    raw_dir = Path("D:/demo/raw")
    if raw_dir.exists() and list(raw_dir.glob("*.tif")):
        demo_scanimage_workflow()
    else:
        print(f"Raw data directory not found or empty: {raw_dir}")
        print("Skipping ScanImage workflow demo.")

    # always run numpy workflow
    demo_numpy_workflow()


if __name__ == "__main__":
    # capture output for docs
    output = StringIO()

    # run with output capture (also print to console)
    import sys
    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, text):
            for w in self.writers:
                w.write(text)
        def flush(self):
            for w in self.writers:
                w.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, output)

    try:
        run_demos()
    finally:
        sys.stdout = original_stdout

    # save to docs
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_OUTPUT, "w") as f:
        f.write(output.getvalue())
    print(f"\nOutput saved to: {DOCS_OUTPUT}")
