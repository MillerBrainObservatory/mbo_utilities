import numpy as np
from pathlib import Path
import fastplotlib as fpl
import mbo_utilities as mbo
import dask.array as da
from tqdm import tqdm
import zarr
import tifffile


def main():
    base_dir = Path(r"D:\SANDBOX\demo")
    files = mbo.get_files(base_dir, max_depth=1, str_contains='tif')
    scan = mbo.read_scan(files, join_contiguous=True)

    path = Path(r"D:/SANDBOX/")

    mbo.save_as(scan, path, planes=[1, 2])

if __name__ == "__main__":
    main()