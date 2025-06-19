import asyncio
import json
import numpy as np
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from zarr.storage import FsspecStore

# Simulated kerchunk reference dict
def make_fake_reference(shape=(10, 1, 64, 64), chunks=(1, 1, 64, 64)):
    import uuid
    ref = {}
    ref[".zarray"] = json.dumps({
        "shape": shape,
        "chunks": chunks,
        "dtype": "int16",
        "compressor": None,
        "filters": None,
        "order": "C",
        "zarr_format": 2,
        "fill_value": 0
    })
    ref[".zattrs"] = json.dumps({"_ARRAY_DIMENSIONS": ["T", "C", "Y", "X"]})
    for i in range(shape[0]):
        ref[f"{i}.0.0.0"] = {
            "offset": 0,
            "length": shape[2] * shape[3] * 2,
            "url": "memory://fake_data",
        }
    return ref

# Emulate render loop trying to access the array
async def simulate_render(z):
    print("Accessing zarr slice...")
    try:
        d = z[0]  # triggers async fetch
        await asyncio.to_thread(np.array, d)  # simulate slow/blocked access
        print("Success")
    except Exception as e:
        print("FAIL:", e)


def main():
    refs = make_fake_reference()
    with open("fake_ref.json", "w") as f:
        json.dump(refs, f)

    fs = ReferenceFileSystem("fake_ref.json", remote_protocol="memory", remote_options={})
    store = FsspecStore(fs)
    z = zarr.open(store, mode="r")

    # this fails under mismatched loops
    asyncio.run(simulate_render(z))

if __name__ == "__main__":
    import fastplotlib as fpl

    arr = np.random.randint(0, 256, size=(64, 64), dtype=np.int16)
    store = zarr.storage.MemoryStore()
    z = zarr.open(store, mode="w", shape=(64, 64), dtype=np.int16)
    z[:] = arr
    print('complete')
    iw = fpl.ImageWidget(data=z)
    iw.show()

    fpl.loop.run()