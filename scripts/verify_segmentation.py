
import numpy as np
import pytest
from mbo_utilities.arrays.features import masks_to_stat, stat_to_masks, SegmentationMixin, DimLabelsMixin
from mbo_utilities.arrays.zarr import ZarrArray
import zarr
import shutil
import os

def test_conversion_roundtrip():
    print("Testing stat <-> masks conversion...")
    Ly, Lx = 100, 100
    
    # Create synthetic masks
    masks = np.zeros((Ly, Lx), dtype=np.uint32)
    masks[10:20, 10:20] = 1 # ROI 1
    masks[30:40, 30:40] = 2 # ROI 2
    
    # Convert to stat
    stat = masks_to_stat(masks)
    assert len(stat) == 2
    assert stat[0]['id'] == 1
    assert stat[1]['id'] == 2
    assert stat[0]['npix'] == 100
    
    # Convert back to masks
    masks_rec = stat_to_masks(stat, Ly, Lx)
    assert np.array_equal(masks, masks_rec)
    print("Roundtrip success!")

def test_mixin_save():
    print("Testing SegmentationMixin.save_segmentation...")
    work_dir = "verify_seg"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    
    # Create Zarr
    # Create Zarr
    z_path = os.path.join(work_dir, "test.zarr")
    
    # Create valid Zarr
    if hasattr(zarr, 'DirectoryStore'):
        store = zarr.DirectoryStore(z_path)
    elif hasattr(zarr, 'storage') and hasattr(zarr.storage, 'DirectoryStore'):
        store = zarr.storage.DirectoryStore(z_path)
    else:
        store = z_path
        
    z = zarr.open(store, mode='w')
    z.create_dataset("0", shape=(1, 100, 100), dtype='int16')
    z.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]
    
    # Open with ZarrArray
    arr = ZarrArray(z_path)
    
    # Create masks
    masks = np.zeros((100, 100), dtype=np.uint32)
    masks[50:60, 50:60] = 5
    
    # Save
    arr.save_segmentation(z_path, masks=masks, name="test_labels")
    
    # Verify
    z_out = zarr.open(z_path, mode='r')
    assert "labels" in z_out
    assert "test_labels" in z_out["labels"]
    loaded = z_out["labels/test_labels"][:]
    assert np.array_equal(loaded, masks)
    print("Save success!")

if __name__ == "__main__":
    try:
        test_conversion_roundtrip()
        test_mixin_save()
        print("All tests passed.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
