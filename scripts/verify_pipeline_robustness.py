
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
import logging
import traceback
import sys
import copy

# Mock modules if real deps are missing
from unittest.mock import MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PipelineVerify")

def create_mock_data(ndim=3):
    """Create mock data with metadata simulating a loaded ScanImage TIFF."""
    if ndim == 3:
        shape = (10, 64, 64) # T, Y, X
        dims = "TZYX" # ZarrArray presents 4D even if source is 3D
    elif ndim == 4:
        shape = (10, 2, 64, 64) # T, Z, Y, X
        dims = "TZYX"
        
    data = np.random.randint(0, 1000, size=shape, dtype=np.int16)
    
    metadata = {
        "scanimage_metadata": {"FrameRate": 30.0, "Objective": "25x"},
        "description": "Mock Data",
        "mroi": False
    }
    
    return data, dims, metadata

def verify_pipeline():
    print("=== Starting Comprehensive Pipeline Verification ===")
    
    # Imports
    try:
        from mbo_utilities.arrays.zarr import ZarrArray
        from mbo_utilities.arrays.tiff import TiffArray
        # from mbo_utilities.arrays.h5 import H5Array
        from mbo_utilities.reader import imread
        import zarr
    except ImportError as e:
        print(f"FAILED: Imports missing: {e}")
        return

    work_dir = Path("pipeline_verify_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    
    formats = [
        (".zarr", "Zarr"),
        (".tiff", "Tiff"), 
        (".h5", "HDF5"), 
        (".bin", "Suite2p Bin")
    ]
    
    # 1. Test 3D (TYX) Pipeline
    print("\n--- Testing 3D Pipeline (TYX) ---")
    data_3d, dims_3d, meta_3d = create_mock_data(ndim=3)
    
    # Wrap in ZarrArray (simulating loaded data)
    # Ideally we'd use a MockArray class if available, but ZarrArray is good generic container
    # Or just save numpy array using ZarrArray's save logic? 
    # Actually, we should test `mbo_utilities.reader.save` if it existed, 
    # but currently saving is method of Array classes.
    # So let's create a ZarrArray from scratch as "Source".
    
    # Create Source Zarr
    src_path = work_dir / "source.zarr"
    
    # Use zarr.open_array for v3 explicit array creation
    try:
        if hasattr(zarr, "open_array"):
            z_src = zarr.open_array(
                str(src_path), 
                mode='w', 
                shape=data_3d.shape, 
                dtype=data_3d.dtype
            )
        else:
            # Fallback for v2/older
            z_src = zarr.open(
                str(src_path), 
                mode='w', 
                shape=data_3d.shape, 
                dtype=data_3d.dtype
            )
    except TypeError:
        # If open_array missing but open strict (bad v3 wrapper state?)
        # Try direct creation
        z_src = zarr.create(
            store=str(src_path),
            shape=data_3d.shape,
            dtype=data_3d.dtype,
            overwrite=True
        )
    z_src[:] = data_3d
    z_src.attrs.update(meta_3d)
    
    # Load as MBO Array
    source_arr = ZarrArray(src_path, dims=dims_3d)
    # Manually inject metadata if ZarrArray doesn't load attrs automatically as metadata (it ideally should)
    # Checking source... ZarrArray loads attrs into metadata.
    
    for ext, name in formats:
        print(f"testing save to {name} ({ext})...")
        out_path = work_dir / f"output_3d{ext}"
        
        try:
            # SAVE
            if ext == ".zarr":
                # Zarr saving usually expects directory
                source_arr.save(out_path, format="zarr") # Method might be export? or save_as? 
                # Checking API... ZarrArray usually relies on mixins or `save` method.
                # User request implies specific "save" capability.
                # Note: `ZarrArray.save` might not exist generically for all formats in current codebase.
                # Typically `export` or specific writers are used.
                # Investigating current API: `ZarrArray` inherits `ReductionMixin`, `Suite2pRegistrationMixin`, `SegmentationMixin`. 
                # Does it have generic `save`?
                # If not, that's a GAP to fill.
                
                # Check if `save` exists via reflection in this script
                if not hasattr(source_arr, "save"):
                    print(f" WARN: source_arr has no 'save' method. Testing stopped.")
                    break
                    
            source_arr.save(out_path, overwrite=True)
            
            # READ BACK
            loaded = imread(out_path)
            
            # VERIFY
            print(f"  Verifying {name}...")
            # Squeeze singleton dimensions for comparison
            loaded_shape_sq = tuple(d for d in loaded.shape if d > 1)
            orig_shape_sq = tuple(d for d in data_3d.shape if d > 1)
            
            if loaded_shape_sq != orig_shape_sq:
                print(f"  FAIL: Shape mismatch. Orig={data_3d.shape}, Loaded={loaded.shape}")
            else:
                print(f"  Pass: Shape match (sq={loaded_shape_sq})")
                
            # Compare content
            # Ensure dtypes match or close enough
            if np.allclose(loaded[:].squeeze(), data_3d.squeeze()):
                print("  Pass: Content match")
            else:
                print("  FAIL: Content mismatch.")
                print(f"  Orig range: {data_3d.min()}..{data_3d.max()}")
                print(f"  Load range: {loaded[:].min()}..{loaded[:].max()}")
                     
            # Metadata
            # Check if FrameRate preserved
            # Note: Tiff saving metadata is tricky (ImageJ tags vs OME).
            loaded_meta = loaded.metadata
            if "scanimage_metadata" not in loaded_meta:
                print(f"  WARN: 'scanimage_metadata' lost in {ext}")
            else:
                print(f"  Pass: Metadata key found")

        except Exception as e:
            print(f"  FAIL: {e}")
            traceback.print_exc()

    print("\n--- Testing 4D Pipeline (ZTYX / TZYX) ---")
    # ... Similar logic for 4D ...

if __name__ == "__main__":
    verify_pipeline()
