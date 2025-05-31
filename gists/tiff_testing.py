# /// script
# requires-python = ">=3.13"
# dependencies = ["click", "numpy", "tifffile", "tqdm", "dask[complete]", "fastplotlib[notebook, imgui]"]
# ///
from mbo_utilities.lazy_array import LazyArrayLoader

if __name__ == "__main__":
    pathnames = r"D:\demo\masknmf\roi1"
    lazy_array = LazyArrayLoader(pathnames).load()
    print(lazy_array)
