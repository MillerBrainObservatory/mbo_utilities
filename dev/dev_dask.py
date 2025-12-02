import mbo_utilities as mbo
from pathlib import Path
import fastplotlib as fpl

data_path_a = r"\\rbo-w1\W1_E_USER_DATA\kbarber\11_04_2025_green_full\plane_1"
tiffs = [x for x in Path(data_path_a).glob("*.tif")]

data = mbo.imread(tiffs)

# Visualize the dask graph
print("Dask array info:")
print(f"Shape: {data.shape}")
print(f"Chunks: {data.dask.chunks}")
print(f"Number of tasks: {len(data.dask.dask)}")
print(f"Size per chunk: {data.dask.nbytes / len(data.dask.chunks[0]) / 1e9:.2f} GB")

# Visualize the task graph (requires graphviz)
try:
    data.dask.visualize(filename='dask_graph.png', optimize_graph=False)
    print("Task graph saved to dask_graph.png")
except Exception as e:
    print(f"Could not visualize graph: {e}")
    print("Install graphviz: conda install graphviz python-graphviz")

# Analyze why it's slow - check if chunks are too large
print("\nChunk analysis:")
for i, dim_chunks in enumerate(data.dask.chunks):
    print(f"  Dim {i}: {dim_chunks}")

# Profile a single frame access to see what's slow
import time
print("\nTiming single frame access:")
start = time.time()
frame = data.dask[0, 0, :, :].compute()
end = time.time()
print(f"Single frame took: {end - start:.2f}s")

# Now create ImageWidget
iw = fpl.ImageWidget(
    [data],
    names=["No Plane Alignment"],
)

iw.show()
fpl.loop.run()
# Check if the problem is chunk sizes
print(f"Chunk info: {data.dask.chunks}")
print(f"Array shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"Number of chunks: {data.dask.npartitions}")

# Check memory per chunk
chunk_size_gb = (
    data.dask.chunks[0][0] *
    data.dask.chunks[1][0] *
    data.dask.chunks[2][0] *
    data.dask.chunks[3][0] *
    data.dtype.itemsize / 1e9
)
print(f"Memory per chunk: {chunk_size_gb:.2f} GB")
# Rechunk to smaller time chunks for faster single-frame access
data_rechunked = data.rechunk({0: 1, 1: 1, 2: -1, 3: -1})
# This makes each chunk a single frame (t=1, z=1, full y, full x)

iw = fpl.ImageWidget(
    [data_rechunked],
    names=["No Plane Alignment"],
)