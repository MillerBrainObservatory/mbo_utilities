# /// script
# requires-python = ">=3.13"
# dependencies = ["click", "numpy", "tifffile", "tqdm", "dask[complete]", "fastplotlib[notebook, imgui]"]
# ///
import tifffile, dask.array as da
import mbo_utilities as mbo
import fasplotlib as fpl

pathnames = r"D:\demo\raw_data/*"
filenames = mbo.expand_paths(pathnames)
scan = mbo.read_scan(filenames)

iw = fpl.ImageWindow(scan)
iw.show()

if __name__ == "__main__":
    fpl.loop.run()