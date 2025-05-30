# /// script
# requires-python = ">=3.13"
# dependencies = ["click", "numpy", "tifffile", "tqdm", "dask[complete]", "fastplotlib[notebook, imgui]"]
# ///
import tifffile, dask.array as da
import mbo_utilities as mbo
from mbo_utilities.lazy_array import LazyArrayLoader

if __name__ == "__main__":
    pathnames = r"D:\tests\data"
    lazy_array = LazyArrayLoader(pathnames).load()
    print(lazy_array)
