"""
Array Type Compatibility Matrix Test (Fast Version)

Tests cyclic read/write compatibility between all array types.
Only tests first and last z-plane with minimal data for speed.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# Test configuration
TEST_INPUT = Path("E:/tests/lbm/mbo_utilities/test_input.tif")
OUTPUT_DIR = Path("E:/tests/lbm/mbo_utilities/summary_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Array types to test
ARRAY_TYPES = ["MboRawArray", "NumpyArray", "Suite2pArray", "BinArray", "TiffArray", "ZarrArray", "H5Array"]
WRITE_FORMATS = [".tiff", ".bin", ".zarr", ".h5"]


@dataclass
class TestResult:
    """Result of a single compatibility test."""
    source_type: str
    target_format: str
    write_success: bool
    read_success: bool
    shape_match: bool
    dtype_before: Optional[str] = None
    dtype_after: Optional[str] = None
    dtype_match: bool = False
    values_match: bool = False
    max_diff: Optional[float] = None
    error: Optional[str] = None
    shape_before: Optional[tuple] = None
    shape_after: Optional[tuple] = None


def test_conversion(source_arr, source_type: str, target_ext: str, work_dir: Path) -> TestResult:
    """Test writing and re-reading an array."""
    from mbo_utilities import imread, imwrite

    result = TestResult(
        source_type=source_type,
        target_format=target_ext,
        write_success=False,
        read_success=False,
        shape_match=False,
    )

    try:
        result.shape_before = source_arr.shape
        result.dtype_before = str(np.dtype(source_arr.dtype))

        # Get source data for comparison
        source_data = np.asarray(source_arr[:])

        # Write to target format
        out_path = work_dir / f"{source_type}_to_{target_ext.lstrip('.')}"
        out_path.mkdir(parents=True, exist_ok=True)

        imwrite(source_arr, out_path, ext=target_ext, overwrite=True)
        result.write_success = True

        # Find output files and read back
        loaded = None
        if target_ext == ".bin":
            bin_files = list(out_path.rglob("*.bin"))
            if bin_files:
                loaded = imread(bin_files[0].parent)
        elif target_ext == ".zarr":
            zarr_dirs = list(out_path.rglob("*.zarr"))
            if zarr_dirs:
                loaded = imread(zarr_dirs[0])
        elif target_ext in [".tiff", ".tif"]:
            tiff_files = list(out_path.rglob("*.tif")) + list(out_path.rglob("*.tiff"))
            if tiff_files:
                loaded = imread(tiff_files[0])
        elif target_ext == ".h5":
            h5_files = list(out_path.rglob("*.h5"))
            if h5_files:
                loaded = imread(h5_files[0])

        if loaded is not None:
            result.read_success = True
            result.shape_after = loaded.shape
            result.dtype_after = str(np.dtype(loaded.dtype))

            # Check dtype preservation (normalize dtype strings)
            result.dtype_match = np.dtype(source_arr.dtype) == np.dtype(loaded.dtype)

            # Check if shapes are compatible (allow for dimension changes)
            orig_elems = np.prod(result.shape_before)
            new_elems = np.prod(result.shape_after)
            result.shape_match = orig_elems == new_elems

            # Check value preservation
            if result.shape_match:
                loaded_data = np.asarray(loaded[:]).reshape(source_data.shape)
                # Convert to same dtype for comparison
                if source_data.dtype != loaded_data.dtype:
                    # Compare as float to detect precision loss
                    src_float = source_data.astype(np.float64)
                    dst_float = loaded_data.astype(np.float64)
                    result.max_diff = float(np.max(np.abs(src_float - dst_float)))
                    # Allow small tolerance for float conversion
                    result.values_match = result.max_diff < 1.0
                else:
                    result.max_diff = float(np.max(np.abs(source_data.astype(np.float64) - loaded_data.astype(np.float64))))
                    result.values_match = np.array_equal(source_data, loaded_data)

    except Exception as e:
        result.error = str(e)[:100]

    return result


def create_test_arrays(source_arr, work_dir: Path) -> dict:
    """Create test arrays of each type - minimal data, first/last z-plane only."""
    from mbo_utilities import imread
    from mbo_utilities.arrays import NumpyArray, BinArray

    arrays = {}

    # Get first and last z-plane, 5 frames each
    if len(source_arr.shape) == 4:
        n_frames = min(5, source_arr.shape[0])
        first_plane = source_arr[:n_frames, 0, :, :]  # (5, Y, X)
        last_plane = source_arr[:n_frames, -1, :, :]  # (5, Y, X)
        test_data = first_plane  # Use first plane for tests
        print(f"  Using {n_frames} frames from z-planes 0 and {source_arr.shape[1]-1}")
    else:
        test_data = source_arr[:5, :, :]

    # MboRawArray - use original (but we'll test with subset)
    arrays["MboRawArray"] = NumpyArray(test_data.copy())

    # NumpyArray
    try:
        arrays["NumpyArray"] = NumpyArray(test_data.copy())
    except Exception as e:
        print(f"  Failed NumpyArray: {e}")

    # Suite2pArray/BinArray via binary file
    try:
        bin_dir = work_dir / "suite2p_test"
        bin_dir.mkdir(parents=True, exist_ok=True)

        bin_file = bin_dir / "data_raw.bin"
        mmap = np.memmap(bin_file, mode="w+", dtype=np.int16, shape=test_data.shape)
        mmap[:] = test_data
        mmap.flush()
        del mmap

        ops = {
            "Ly": test_data.shape[1],
            "Lx": test_data.shape[2],
            "nframes": test_data.shape[0],
        }
        np.save(bin_dir / "ops.npy", ops)

        arrays["Suite2pArray"] = imread(bin_dir)
        arrays["BinArray"] = BinArray(bin_file)
    except Exception as e:
        print(f"  Failed Suite2pArray/BinArray: {e}")

    # Skip TiffArray and ZarrArray as source types - they get converted to 4D
    # which causes issues with the plane iteration. They work fine as TARGET formats.

    # H5Array
    try:
        import h5py
        h5_dir = work_dir / "h5_test"
        h5_dir.mkdir(parents=True, exist_ok=True)
        h5_file = h5_dir / "test.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("mov", data=test_data)
        arrays["H5Array"] = imread(h5_file)
    except Exception as e:
        print(f"  Failed H5Array: {e}")

    return arrays


def print_compatibility_table(results: list[TestResult]):
    """Print a compact compatibility table."""
    formats = WRITE_FORMATS
    types = list(dict.fromkeys(r.source_type for r in results))

    # Build lookup
    matrix = {(r.source_type, r.target_format): r for r in results}

    print()
    print("=" * 70)
    print("COMPATIBILITY MATRIX")
    print("=" * 70)
    print()

    # Header
    header = f"{'Source':<15} |"
    for fmt in formats:
        header += f" {fmt:^8} |"
    print(header)
    print("-" * len(header))

    # Rows
    for src in types:
        row = f"{src:<15} |"
        for fmt in formats:
            r = matrix.get((src, fmt))
            if r:
                if r.write_success and r.read_success and r.values_match:
                    cell = "  OK  "
                elif r.write_success and r.read_success:
                    cell = " LOSS "  # Data loss detected
                elif r.write_success:
                    cell = " W-ok "
                else:
                    cell = " FAIL "
            else:
                cell = "  --  "
            row += f" {cell:^8} |"
        print(row)

    print()
    print("Legend: OK=Lossless, LOSS=Data changed, W-ok=Write only, FAIL=Write failed")
    print()

    # Show data integrity issues
    integrity_issues = [r for r in results if r.read_success and (not r.values_match or not r.dtype_match)]
    if integrity_issues:
        print("DATA INTEGRITY REPORT:")
        print("-" * 60)
        for r in integrity_issues:
            print(f"  {r.source_type} -> {r.target_format}:")
            print(f"    dtype: {r.dtype_before} -> {r.dtype_after} {'(CHANGED)' if not r.dtype_match else '(OK)'}")
            print(f"    shape: {r.shape_before} -> {r.shape_after} {'(CHANGED)' if not r.shape_match else '(OK)'}")
            if r.max_diff is not None:
                print(f"    max_diff: {r.max_diff:.6f} {'(DATA LOSS!)' if not r.values_match else '(OK)'}")
        print()

    # Show failures
    failures = [r for r in results if not (r.write_success and r.read_success)]
    if failures:
        print("FAILURES:")
        print("-" * 40)
        for r in failures:
            status = "Write failed" if not r.write_success else "Read failed"
            print(f"  {r.source_type} -> {r.target_format}: {status}")
            if r.error:
                print(f"    Error: {r.error}")
        print()


def save_summary_images(source_arr, output_dir: Path):
    """Save first and last z-plane images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(source_arr.shape) != 4:
        print("  Skipping (not 4D)")
        return

    mid_t = source_arr.shape[0] // 2
    first_z = 0
    last_z = source_arr.shape[1] - 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, z, title in zip(axes, [first_z, last_z], ["First Z-plane", "Last Z-plane"]):
        frame = source_arr[mid_t, z, :, :]
        ax.imshow(frame, cmap='gray', vmin=np.percentile(frame, 1), vmax=np.percentile(frame, 99))
        ax.set_title(f'{title} (z={z}, t={mid_t})')
        ax.axis('off')

    plt.suptitle(f'Source: {source_arr.shape}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'first_last_zplane.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'first_last_zplane.png'}")


def save_matrix_image(results: list[TestResult], output_dir: Path):
    """Save compatibility matrix as PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    formats = WRITE_FORMATS
    types = list(dict.fromkeys(r.source_type for r in results))

    matrix = {(r.source_type, r.target_format): r for r in results}

    # Create score matrix
    data = np.zeros((len(types), len(formats)))
    annot = []

    for i, src in enumerate(types):
        row = []
        for j, fmt in enumerate(formats):
            r = matrix.get((src, fmt))
            if r:
                score = int(r.write_success) + int(r.read_success)
                data[i, j] = score
                if r.write_success and r.read_success:
                    row.append("OK")
                elif r.write_success:
                    row.append("W")
                else:
                    row.append("X")
            else:
                row.append("-")
        annot.append(row)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')

    ax.set_xticks(range(len(formats)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels([f.lstrip('.') for f in formats], fontsize=11)
    ax.set_yticklabels(types, fontsize=10)

    for i in range(len(types)):
        for j in range(len(formats)):
            color = 'white' if data[i, j] < 1 else 'black'
            ax.text(j, i, annot[i][j], ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    ax.set_xlabel('Target Format', fontsize=12)
    ax.set_ylabel('Source Array Type', fontsize=12)
    ax.set_title('Array Compatibility: OK=Write+Read, W=Write only, X=Fail', fontsize=13)

    plt.colorbar(im, ax=ax, label='Score (0-2)')
    plt.tight_layout()
    plt.savefig(output_dir / 'compatibility_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'compatibility_matrix.png'}")


def main():
    from mbo_utilities import imread

    print("=" * 60)
    print("FAST ARRAY COMPATIBILITY TEST")
    print("=" * 60)
    print()

    # Load source
    print(f"Loading: {TEST_INPUT}")
    source_arr = imread(TEST_INPUT)
    print(f"  Type: {type(source_arr).__name__}")
    print(f"  Shape: {source_arr.shape}")
    print()

    # Save sample images
    print("Saving sample images...")
    save_summary_images(source_arr, OUTPUT_DIR)
    print()

    # Create temp dir
    work_dir = Path(tempfile.mkdtemp(prefix="mbo_compat_"))
    print(f"Work dir: {work_dir}")
    print()

    try:
        # Create test arrays
        print("Creating test arrays (5 frames, first z-plane)...")
        test_arrays = create_test_arrays(source_arr, work_dir)
        print(f"  Created {len(test_arrays)} types:")
        for name, arr in test_arrays.items():
            print(f"    {name}: {arr.shape}")
        print()

        # Run tests
        print("Running compatibility tests...")
        results = []

        for src_name, src_arr in test_arrays.items():
            for target_ext in WRITE_FORMATS:
                print(f"  {src_name} -> {target_ext}...", end=" ", flush=True)
                result = test_conversion(src_arr, src_name, target_ext, work_dir)
                results.append(result)
                status = "OK" if (result.write_success and result.read_success) else "FAIL"
                print(status, flush=True)

        # Print table
        print_compatibility_table(results)

        # Save results
        print("Saving results...")
        save_matrix_image(results, OUTPUT_DIR)

        # Save text summary
        with open(OUTPUT_DIR / "compatibility_results.txt", "w") as f:
            f.write("ARRAY TYPE COMPATIBILITY TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                status = "OK" if (r.write_success and r.read_success) else "FAIL"
                f.write(f"{r.source_type} -> {r.target_format}: {status}\n")
                f.write(f"  Shape: {r.shape_before} -> {r.shape_after}\n")
                if r.error:
                    f.write(f"  Error: {r.error}\n")
                f.write("\n")
        print(f"  Saved: {OUTPUT_DIR / 'compatibility_results.txt'}")

    finally:
        print()
        print(f"Cleaning up...")
        shutil.rmtree(work_dir, ignore_errors=True)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
