#!/usr/bin/env python3

import pandas as pd
import glfw
import time
from typing import Any
from rendercanvas.glfw import GlfwRenderCanvas
from rendercanvas.auto import loop
import io
import json
from pathlib import Path
import tempfile
import shutil
import zarr.core.sync
import asyncio
from zarr.core import sync
import dask.array as da

import numpy as np
import tifffile
import zarr
from fsspec.implementations.reference import ReferenceFileSystem
from skimage import data as skdata
import fastplotlib as fpl
from tifffile import TiffFile
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities import get_mbo_dirs

import uuid
import subprocess

tests_dir = get_mbo_dirs()["tests"]

if __name__ == "__main__":
    from mbo_utilities.file_io import Scan_MBO, get_files, ZarrScanView
    from mbo_utilities._benchmark import _benchmark_indexing

    data = imread(r"/home/flynn/lbm_data/raw")
    zarray = data.as_zarr()
    darray = data.as_dask()
    zarray_view = ZarrScanView(
        data.as_zarr(),
        yslices=data.data.yslices,
        xslices=data.data.xslices
    )
    print(f"Zarr shape: {zarray.shape}, Dask shape: {darray.shape}, Zarr-View shape: {zarray_view.shape}")

    # _benchmark_indexing(
    #     label="No Sub",
    #     arrays={
    #         "Zarr": zarray,
    #         "Dask": darray,
    #         "Zarr-View": zarray_view,
    #     },
    #     save_path=tests_dir / "benchmark.json",
    #     num_repeats=5,
    # )
    #
    # fpl.ImageWidget([zarray, darray], names=["Zarr", "Dask"], histogram_widget=True).show()

    # data.imshow()
    # loop.run()