"""
Demonstration of SegmentationMixin features in ZarrArray.

1. Creates specific Zarr data (with bright "cells").
2. Runs Cellpose detection (mocked if not installed).
3. Converts between Masks and Suite2p Stats.
4. Saves segmentation to OME-Zarr.
5. Visualizes results.
"""
from pathlib import Path
import numpy as np
import zarr
import shutil
import matplotlib.pyplot as plt
from mbo_utilities.arrays.zarr import ZarrArray
import time

def create_synthetic_data(path, shape=(1, 256, 256)):
    """Create a Zarr with some 'cells' (gaussian blobs)."""
    if Path(path).exists():
        shutil.rmtree(path)
        
    if hasattr(zarr, 'DirectoryStore'):
        store = zarr.DirectoryStore(path)
    elif hasattr(zarr, 'storage') and hasattr(zarr.storage, 'DirectoryStore'):
        store = zarr.storage.DirectoryStore(path)
    else:
        store = path # Fallback to string path for V3/default
        
    z = zarr.open(store, mode='w')
    z.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    
    # Create background noise
    data = np.random.normal(100, 10, shape).astype('int16')
    
    # Add cells
    Y, X = shape[1:]
    # random centers
    centers = np.random.randint(20, 230, (15, 2))
    
    # Simple blobs
    y_grid, x_grid = np.ogrid[:Y, :X]
    true_masks = np.zeros((Y, X), dtype='uint32')
    
    for i, (cy, cx) in enumerate(centers):
        dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        mask = dist < 15
        data[0][mask] += 200 # Bright cell
        true_masks[mask] = i + 1
        
    z.create_dataset("0", data=data, chunks=(1, Y, X), overwrite=True)
    return centers, true_masks

def main():
    work_dir = Path("demo_seg")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    
    zarr_path = work_dir / "data.zarr"
    print(f"Creating synthetic data at {zarr_path}...")
    centers, true_masks = create_synthetic_data(str(zarr_path))
    
    # Open array
    arr = ZarrArray(zarr_path)
    print(f"Opened ZarrArray: {arr.shape} {arr.dims}")
    
    # 1. CELLPOSE DETECTION
    print("\n--- 1. Detect Cellpose ---")
    try:
        from cellpose import models
        print("Cellpose installed, running detection...")
        # Reduce diameter for synthetic blobs
        masks, flows, styles, diams = arr.detect_cellpose(diameter=30, channels=[0,0])
        print(f"Detection complete. Found {masks.max()} ROIs.")
    except ImportError:
        print("Cellpose not installed. Using synthetic truth masks as mock detection.")
        masks = true_masks
        
    # 2. CONVERSION TO SUITE2P STATS
    print("\n--- 2. Convert to Suite2p Stats ---")
    from mbo_utilities.arrays.features import masks_to_stat, stat_to_masks
    
    stat = masks_to_stat(masks)
    print(f"Converted to {len(stat)} Suite2p ROIs.")
    if len(stat) > 0:
        print(f"Example ROI 0: med={stat[0]['med']}, npix={stat[0]['npix']}")
        
    # 3. CONVERT BACK TO MASKS
    print("\n--- 3. Roundtrip Check ---")
    Y, X = arr.shape[-2], arr.shape[-1]
    masks_rec = stat_to_masks(stat, Ly=Y, Lx=X)
    match = np.array_equal(masks, masks_rec)
    print(f"Roundtrip successful? {match}")
    
    # 4. SAVE TO OME-ZARR
    print("\n--- 4. Save Segmentation ---")
    arr.save_segmentation(zarr_path, masks=masks, name="demo_labels")
    print(f"Saved labels to {zarr_path}/labels/demo_labels")
    
    # Verify save
    z = zarr.open(str(zarr_path), mode='r')
    if "labels" in z and "demo_labels" in z["labels"]:
        print("Verification: Labels found in Zarr.")
    else:
        print("Verification: Labels NOT found!")

    # 5. VISUALIZATION
    print("\n--- 5. Visualization ---")
    print("Plotting results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw Image (Frame 0)
    axes[0].imshow(arr[0], cmap='gray', vmin=50, vmax=350)
    axes[0].set_title("Raw Image")
    axes[0].axis('off')
    
    # Masks
    # Create colored masks
    axes[1].imshow(masks, cmap='turbo', interpolation='nearest')
    axes[1].set_title(f"Masks (N={masks.max()})")
    axes[1].axis('off')
    
    # Suite2p Contours (Overlay)
    axes[2].imshow(arr[0], cmap='gray', vmin=50, vmax=350)
    axes[2].set_title("Suite2p Stat Contours")
    axes[2].axis('off')
    
    for roi in stat:
        # Plotting scatter for simplicity of 'xpix'/'ypix'
        # Or simplistic contour
        axes[2].scatter(roi['xpix'], roi['ypix'], s=1, alpha=0.5)
        
    plt.tight_layout()
    plot_path = work_dir / "demo_output.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    print("Done.")

if __name__ == "__main__":
    main()
