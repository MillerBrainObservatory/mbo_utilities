"""Time the slow path now that vmin/vmax are exposed."""
import time
from pathlib import Path
import numpy as np

P = Path(r"D:\2026_05_light-sheet_workshop\2_zebrafish\zebrafish.corrected")

t0 = time.perf_counter()
from mbo_utilities.reader import imread
print(f"import imread: {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
arr = imread(P)
print(f"imread: {time.perf_counter()-t0:.2f}s  shape={arr.shape}")

# fpl's hasattr + isinstance gates
print(f"hasattr(arr, 'vmin') = {hasattr(arr, 'vmin')}")
print(f"hasattr(arr, 'vmax') = {hasattr(arr, 'vmax')}")
t0 = time.perf_counter()
v = arr.vmin
print(f"arr.vmin = {v!r} ({type(v).__name__}): {time.perf_counter()-t0:.3f}s")
t0 = time.perf_counter()
v = arr.vmax
print(f"arr.vmax = {v!r} ({type(v).__name__}): {time.perf_counter()-t0:.3f}s")
print(f"isinstance check passes: {isinstance(arr.vmin, (float, int, np.number))}")

# what fpl actually does (fast path)
t0 = time.perf_counter()
first = np.asarray(arr[0])
print(f"\nfpl fast path = np.asarray(arr[0]) -> shape={first.shape}: {time.perf_counter()-t0:.2f}s")

# __array__ safety check
t0 = time.perf_counter()
sample = np.asarray(arr)
print(f"np.asarray(arr) (now narrow) -> shape={sample.shape}: {time.perf_counter()-t0:.2f}s")

# narrow indexing - should be small, fast
t0 = time.perf_counter()
narrow = arr[5, 2, 10, 100:110, 200:210]
print(f"arr[5,2,10,100:110,200:210] -> shape={narrow.shape}: {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
plane = arr[3, 1, 40]
print(f"arr[3,1,40] -> shape={plane.shape}: {time.perf_counter()-t0:.2f}s")
