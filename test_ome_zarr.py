"""
Quick test to verify OME-Zarr structure
"""
import numpy as np
from pathlib import Path
import zarr
import json

# Create test data
test_data = np.random.randint(0, 100, (10, 128, 128), dtype=np.int16)

# Test path
test_path = Path("test_ome_output.zarr")

# Import mbo to test
import sys
sys.path.insert(0, str(Path(__file__).parent))
import mbo_utilities as mbo

print("Creating test OME-Zarr...")

# Create a simple test array wrapper
class TestArray:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.metadata = {
            "num_frames": data.shape[0],
            "pixel_resolution": (2.0, 2.0),
            "frame_rate": 10.0,
        }
        self.roi = None

    def __getitem__(self, key):
        return self.data[key]

test_array = TestArray(test_data)

# Write with OME metadata
metadata = {
    "num_frames": 10,
    "pixel_resolution": (2.0, 2.0),
    "frame_rate": 10.0,
}

# Use internal writer directly for testing
from mbo_utilities._writers import _write_zarr

_write_zarr(
    test_path,
    test_data,
    metadata=metadata,
    ome=True,
    overwrite=True
)

print(f"\n✓ Created {test_path}")

# Inspect the structure
print("\n=== Zarr Structure ===")
root = zarr.open(str(test_path), mode="r")
print(f"Root: {root}")
print(f"Root attrs: {dict(root.attrs)}")
print(f"\nRoot tree:")
print(root.tree())

# Check for array "0"
if "0" in root:
    arr = root["0"]
    print(f"\n=== Array '0' ===")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Attrs: {dict(arr.attrs)}")
else:
    print("\n⚠ Array '0' not found!")

# Save metadata to JSON for inspection
metadata_file = test_path.parent / "ome_metadata.json"
with open(metadata_file, "w") as f:
    json.dump(dict(root.attrs), f, indent=2, default=str)
print(f"\n✓ Saved metadata to {metadata_file}")

print("\n=== Check OME-NGFF compliance ===")
attrs = dict(root.attrs)
if "multiscales" in attrs:
    ms = attrs["multiscales"][0]
    print(f"✓ multiscales found")
    print(f"  version: {ms.get('version')}")
    print(f"  axes: {ms.get('axes')}")
    print(f"  datasets: {ms.get('datasets')}")

    # Check dataset path
    if ms.get('datasets'):
        ds_path = ms['datasets'][0].get('path')
        print(f"  → dataset path: '{ds_path}'")
        if ds_path in root or ds_path == "0":
            print(f"  ✓ Dataset '{ds_path}' exists")
        else:
            print(f"  ⚠ Dataset '{ds_path}' NOT FOUND in root")
else:
    print("⚠ No multiscales metadata found!")

print("\n=== Files on disk ===")
for item in sorted(test_path.rglob("*")):
    if item.is_file():
        rel_path = item.relative_to(test_path.parent)
        size = item.stat().st_size
        print(f"  {rel_path} ({size} bytes)")
