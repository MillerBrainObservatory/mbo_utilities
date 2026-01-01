
import zarr
import numpy as np
import os
import shutil

if os.path.exists("debug_work"):
    shutil.rmtree("debug_work")
os.makedirs("debug_work")

print(f"Zarr version: {zarr.__version__}")

# Setup store
store_path = "debug_work/store.zarr"
if hasattr(zarr, 'DirectoryStore'):
    store = zarr.DirectoryStore(store_path)
elif hasattr(zarr, 'storage') and hasattr(zarr.storage, 'DirectoryStore'):
    store = zarr.storage.DirectoryStore(store_path)
else:
    store = store_path

print(f"Store: {store}")

# Test zarr.create
print("\n--- Testing zarr.create ---")
try:
    # Explicit kwargs
    arr = zarr.create(
        store=store,
        path="test_create",
        shape=(100, 100),
        dtype='int16',
        chunks=True,
        overwrite=True
    )
    print("zarr.create success")
except Exception as e:
    print(f"zarr.create failed: {e}")
    # import traceback
    # traceback.print_exc()

# Test zarr.open_array
print("\n--- Testing zarr.open_array ---")
try:
    if hasattr(zarr, 'open_array'):
        arr = zarr.open_array(
            store=store,
            path="test_open_array",
            mode='w',
            shape=(100, 100),
            dtype='int16'
        )
        print("zarr.open_array success")
    else:
        print("zarr.open_array not found")
except Exception as e:
    print(f"zarr.open_array failed: {e}")

# Test Group.create_dataset
print("\n--- Testing Group.create_dataset ---")
try:
    grp = zarr.open(store, mode='a')
    grp.create_dataset(
        "test_dataset",
        shape=(100, 100),
        dtype='int16',
        chunks=True
        # overwrite might not be supported here
    )
    print("Group.create_dataset success")
except Exception as e:
    print(f"Group.create_dataset failed: {e}")

