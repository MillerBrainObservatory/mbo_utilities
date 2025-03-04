import numpy as np
from pathlib import Path
import fastplotlib as fpl
import mbo_utilities as mbo
import dask.array as da
from tqdm import tqdm
import zarr
import tifffile


def main():
    base_dir = Path(r"D:\W2_DATA\kbarber\2025-02-27\mk301\green")
    files = mbo.get_files(base_dir, max_depth=1, str_contains='tif')
    scan = mbo.read_scan(files, join_contiguous=True)
    out_path = base_dir.parent / 'test'
    out_path.mkdir(parents=True, exist_ok=True)

    mbo.save_as(scan, out_path, planes=[0], trim_edge=[2, 2, 0, 4])

if __name__ == "__main__":
    main()