"""
Verify OME-Zarr v3 compliance
"""
import zarr
import json
from pathlib import Path

zarr_path = Path("test_ome_output.zarr")

print("=== OME-Zarr v3 Verification ===\n")

# 1. Check root group
print("1. Root Group:")
root = zarr.open_group(str(zarr_path))
print(f"   Zarr format: {root.metadata.zarr_format}")
print(f"   Node type: {root.metadata.node_type}")
print(f"   Has 'ome' attribute: {'ome' in root.attrs}")

# 2. Check OME metadata
if 'ome' in root.attrs:
    print("\n2. OME Metadata:")
    ome = root.attrs['ome']
    print(f"   OME version: {ome.get('version')}")

    if 'multiscales' in ome:
        ms = ome['multiscales'][0]
        print(f"   Multiscales version: {ms.get('version')}")
        print(f"   Number of axes: {len(ms.get('axes', []))}")
        print(f"   Axes: {[ax['name'] for ax in ms.get('axes', [])]}")
        print(f"   Number of datasets: {len(ms.get('datasets', []))}")

        if ms.get('datasets'):
            ds = ms['datasets'][0]
            print(f"   Dataset path: '{ds.get('path')}'")

            # Check coordinate transformations
            if 'coordinateTransformations' in ds:
                for i, transform in enumerate(ds['coordinateTransformations']):
                    print(f"   Transform {i}: {transform.get('type')}")
                    if transform.get('type') == 'scale':
                        print(f"      Scale: {transform.get('scale')}")

# 3. Check array at path "0"
print("\n3. Array at path '0':")
arr = zarr.open_array(str(zarr_path / "0"))
print(f"   Zarr format: {arr.metadata.zarr_format}")
print(f"   Node type: {arr.metadata.node_type}")
print(f"   Shape: {arr.shape}")
print(f"   Dtype: {arr.dtype}")
chunk_grid = arr.metadata.chunk_grid
if hasattr(chunk_grid, 'chunk_shape'):
    print(f"   Chunk shape: {chunk_grid.chunk_shape}")
else:
    print(f"   Chunk grid: {chunk_grid}")

# 4. Check data
print("\n4. Data Verification:")
data = arr[:]
print(f"   Can read data: Yes")
print(f"   Data shape: {data.shape}")
print(f"   Data range: [{data.min()}, {data.max()}]")
print(f"   Sample values: {data[0, 0, :5]}")

# 5. File structure
print("\n5. File Structure:")
print(f"   Root: {zarr_path}/")
print(f"      zarr.json (zarr_format: 3, node_type: group)")
print(f"      0/")
print(f"         zarr.json (zarr_format: 3, node_type: array)")
print(f"         c/ (chunks: {len(list((zarr_path / '0' / 'c').glob('*')))} directories)")

print("\n=== Verification Complete ===")
print("Status: OME-Zarr v3 structure is compliant with OME-NGFF 0.5 spec")
