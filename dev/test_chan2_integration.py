"""
Test script to validate chan2 integration in run_plane_from_data.
This tests the logic of loading and writing both functional and structural channels.
"""

import numpy as np
from pathlib import Path
import tempfile
import sys

# Add mbo_utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mbo_utilities._writers import _write_bin
from mbo_utilities.file_io import write_ops


def test_chan2_binary_writing():
    """Test that chan2 binaries are written with correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        plane_dir = tmpdir / "plane01_roi01"
        plane_dir.mkdir(parents=True)

        # Create test data
        n_frames_functional = 100
        n_frames_structural = 95
        height, width = 256, 256

        func_data = np.random.randint(
            0, 1000, (n_frames_functional, height, width), dtype=np.int16
        )
        struct_data = np.random.randint(
            0, 1000, (n_frames_structural, height, width), dtype=np.int16
        )

        # Trim both to minimum frames (simulating the GUI behavior)
        min_frames = min(n_frames_functional, n_frames_structural)
        func_data = func_data[:min_frames]
        struct_data = struct_data[:min_frames]

        raw_file = plane_dir / "data_raw.bin"
        chan2_file = plane_dir / "data_chan2.bin"
        ops_path = plane_dir / "ops.npy"

        # Create metadata like the GUI does
        metadata = {
            "num_frames": min_frames,
            "shape": func_data.shape,
            "Ly": height,
            "Lx": width,
            "fs": 15.0,
            "dx": 1.0,
            "dy": 1.0,
            "align_by_chan": 1,  # User setting from GUI
            "reg_tif_chan2": True,  # User setting from GUI
            "chan2_file": str(chan2_file),  # This will be updated to the actual path
        }

        # Write functional channel (channel 1)
        _write_bin(raw_file, func_data, overwrite=True, metadata=metadata)
        print(f"✓ Wrote functional channel: {raw_file}")

        # Write structural channel (channel 2)
        chan2_metadata = metadata.copy()
        chan2_metadata["num_frames"] = struct_data.shape[0]
        chan2_metadata["shape"] = struct_data.shape
        chan2_metadata["chan2_file"] = str(chan2_file.resolve())
        _write_bin(
            chan2_file,
            struct_data,
            overwrite=True,
            metadata=chan2_metadata,
            structural=True,
        )
        print(f"✓ Wrote structural channel: {chan2_file}")

        # Check that ops.npy was created
        assert ops_path.exists(), "ops.npy was not created"
        print(f"✓ ops.npy created at {ops_path}")

        # Load and validate ops
        ops = np.load(ops_path, allow_pickle=True).item()
        print("\nOps dictionary contents:")
        print(f"  nframes_chan1: {ops.get('nframes_chan1')} (expected: {min_frames})")
        print(
            f"  nframes_chan2: {ops.get('nframes_chan2')} (expected: {struct_data.shape[0]})"
        )
        print(f"  raw_file: {ops.get('raw_file')}")
        print(f"  chan2_file: {ops.get('chan2_file')}")
        print(f"  align_by_chan: {ops.get('align_by_chan')} (expected: 1)")
        print(f"  reg_tif_chan2: {ops.get('reg_tif_chan2')} (expected: True)")

        # Validate key fields
        assert ops.get("nframes_chan1") == min_frames, (
            f"Expected nframes_chan1={min_frames}, got {ops.get('nframes_chan1')}"
        )
        assert ops.get("nframes_chan2") == struct_data.shape[0], (
            f"Expected nframes_chan2={struct_data.shape[0]}, got {ops.get('nframes_chan2')}"
        )
        assert ops.get("align_by_chan") == 1, (
            f"Expected align_by_chan=1, got {ops.get('align_by_chan')}"
        )
        assert ops.get("reg_tif_chan2") == True, (
            f"Expected reg_tif_chan2=True, got {ops.get('reg_tif_chan2')}"
        )
        assert str(chan2_file.resolve()) in ops.get("chan2_file", ""), (
            "chan2_file not properly set in ops"
        )

        print("\n✓ All validations passed!")
        return True


if __name__ == "__main__":
    try:
        test_chan2_binary_writing()
        print("\n✓✓✓ Test PASSED ✓✓✓")
    except Exception as e:
        print(f"\n✗✗✗ Test FAILED ✗✗✗")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
