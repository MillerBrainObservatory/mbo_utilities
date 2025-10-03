import mbo_utilities as mbo
import zarr
from pathlib import Path
from pynwb import read_nwb
import fastplotlib as fpl
import numpy as np
import time

from mbo_utilities.metadata import group_si_metadata

full_file = Path(
    r"D:\W2_DATA\kbarber\07_27_2025\mk355\green\mk355_7_27_2025_180mw_right_m2_go_to_2x-mROI-880x1100um_220x550px_2um-px_14p00Hz_00001_00001_00001.tif"
)

start = time.time()
metadata = mbo.get_metadata(
    full_file,
    verbose=True,
)
print(f"Metadata retrieval took {time.time() - start:.2f} seconds")


def nest_keys(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = nest_keys(value)

        if "." in key:
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            result[key] = value
    return result


nested_metadata = nest_keys(metadata)

x = 2
