"""
Example: Merge z-plane Zarr files into a single OME-Zarr volume with Suite2p labels.

This script demonstrates how to consolidate multiple single z-plane Zarr stores
into a single OME-NGFF v0.5 compliant Zarr volume with optional Suite2p ROI masks.
"""

from pathlib import Path
import mbo_utilities as mbo

# Example 1: Basic merge without Suite2p labels
# ==============================================

# List of z-plane Zarr stores
zarr_files = [
    "data/session1/plane01_stitched.zarr",
    "data/session1/plane02_stitched.zarr",
    "data/session1/plane03_stitched.zarr",
    "data/session1/plane04_stitched.zarr",
    "data/session1/plane05_stitched.zarr",
]

# Merge into a single volume
output = mbo.merge_zarr_zplanes(
    zarr_files,
    "data/session1/volume.zarr",
    metadata={
        "pixel_resolution": (0.65, 0.65),  # x, y in micrometers
        "frame_rate": 30.0,  # Hz
        "dz": 5.0,  # z-step in micrometers
        "name": "example_volume",
    },
)

print(f"Created volume at: {output}")


# Example 2: Merge with Suite2p segmentation masks
# =================================================

# Corresponding Suite2p directories for each z-plane
suite2p_dirs = [
    "data/session1/plane01_stitched/suite2p/plane0",
    "data/session1/plane02_stitched/suite2p/plane0",
    "data/session1/plane03_stitched/suite2p/plane0",
    "data/session1/plane04_stitched/suite2p/plane0",
    "data/session1/plane05_stitched/suite2p/plane0",
]

# Merge with ROI labels
output_with_labels = mbo.merge_zarr_zplanes(
    zarr_files,
    "data/session1/volume_with_labels.zarr",
    suite2p_dirs=suite2p_dirs,
    metadata={
        "pixel_resolution": (0.65, 0.65),
        "frame_rate": 30.0,
        "dz": 5.0,
        "name": "example_volume_with_labels",
    },
    overwrite=True,
    compression_level=1,  # Fast compression
)

print(f"Created volume with labels at: {output_with_labels}")


# Example 3: Using glob patterns to find Zarr files
# ==================================================

from pathlib import Path

# Find all plane Zarr files in a directory
session_dir = Path("data/session1")
zarr_files = sorted(session_dir.glob("plane*_stitched.zarr"))

# Find corresponding Suite2p directories
suite2p_dirs = [
    zf / "suite2p" / "plane0" for zf in zarr_files
]

# Filter to only those that exist
suite2p_dirs = [s2p for s2p in suite2p_dirs if s2p.exists()]

if len(suite2p_dirs) == len(zarr_files):
    output = mbo.merge_zarr_zplanes(
        zarr_files,
        session_dir / "merged_volume.zarr",
        suite2p_dirs=suite2p_dirs,
        metadata={
            "pixel_resolution": (0.65, 0.65),
            "frame_rate": 30.0,
            "dz": 5.0,
        },
    )
    print(f"Merged {len(zarr_files)} z-planes with labels")
else:
    print(
        f"Warning: Found {len(zarr_files)} Zarr files but only "
        f"{len(suite2p_dirs)} Suite2p directories"
    )


# Example 4: View the result in napari
# =====================================

try:
    import napari
    import zarr

    # Open the merged volume
    volume_zarr = zarr.open("data/session1/volume_with_labels.zarr", mode="r")

    # Load image data
    image_data = volume_zarr["0"][:]  # Shape: (T, Z, Y, X)

    # Load labels if they exist
    labels_data = None
    if "labels" in volume_zarr and "0" in volume_zarr["labels"]:
        labels_data = volume_zarr["labels"]["0"][:]  # Shape: (Z, Y, X)

    # Create napari viewer
    viewer = napari.Viewer()

    # Add image (show max projection over time for visualization)
    viewer.add_image(
        image_data.max(axis=0),  # Max project over time -> (Z, Y, X)
        name="imaging_data",
        colormap="gray",
    )

    # Add labels
    if labels_data is not None:
        viewer.add_labels(
            labels_data,
            name="suite2p_rois",
        )

    napari.run()

except ImportError:
    print("napari not installed. Install with: pip install napari[all]")


# Example 5: Inspect the OME-Zarr metadata
# =========================================

import zarr
import json

volume = zarr.open("data/session1/volume.zarr", mode="r")

print("\n" + "=" * 60)
print("OME-Zarr Metadata")
print("=" * 60)

# Print OME metadata
if "ome" in volume.attrs:
    print("\nOME-NGFF metadata:")
    print(json.dumps(dict(volume.attrs["ome"]), indent=2))

# Print array info
if "0" in volume:
    arr = volume["0"]
    print(f"\nImage array:")
    print(f"  Shape: {arr.shape} (T, Z, Y, X)")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Chunks: {arr.chunks}")

# Check for labels
if "labels" in volume:
    print(f"\nLabels found:")
    if "0" in volume["labels"]:
        labels_arr = volume["labels"]["0"]
        print(f"  Shape: {labels_arr.shape} (Z, Y, X)")
        print(f"  Dtype: {labels_arr.dtype}")
        print(f"  Unique labels: {len(np.unique(labels_arr[:]))}")
