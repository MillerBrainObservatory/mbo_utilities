# /// script
# requires-python = ">=3.13"
# dependencies = ["click", "numpy", "tifffile", "tqdm", "dask[complete]"]
# ///
import tifffile, dask.array as da

with tifffile.TiffFile(r"D:\W2_DATA\kbarber\2025_03_01\mk301\assembled\plane_07_mk301.tiff") as tif:
    zarr_array = tif.aszarr(series=0, chunkmode="page")
dask_arr = da.from_zarr(zarr_array)
print(dask_arr.shape, dask_arr.chunksize)
