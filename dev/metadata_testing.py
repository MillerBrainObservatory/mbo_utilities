from pathlib import Path
import numpy as np

import fastplotlib as fpl
import mbo_utilities as mbo
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.file_io import MBO_SUPPORTED_FTYPES

# fpath_og = Path(r"D:\W2_DATA\kbarber\2025_03_01\mk301\green")
fpath_og = Path(r"D:\W2_DATA\foconnell\2025-07-10_Pollen\plane1")
files_og = list(fpath_og.glob("*.tif*"))
fpath = Path(r"D:\W2_DATA\foconnell\2025-07-10_Pollen\plane7")
files = list(fpath.glob("*.tif*"))

import tifffile
tf = tifffile.TiffFile(files[0])
tf_og = tifffile.TiffFile(files_og[0])

md = tf.scanimage_metadata["FrameData"]
md_og = tf_og.scanimage_metadata["FrameData"]

def group_scanimage_metadata(flat: dict) -> dict:
    grouped = {}
    for key, value in flat.items():
        if not key.startswith("SI."):
            continue
        subkeys = key.split(".")[1:]  # remove "SI."
        if not subkeys:
            continue
        root = subkeys[0].split("[")[0]
        path = subkeys[1:]
        group = grouped.setdefault(root, {})

        if not path:
            # Directly assign to the group if no nested path
            grouped[root] = value
            continue

        current = group
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = value
    return grouped

def diff_metadata(g1: dict, g2: dict) -> dict:
    diff = {}
    keys = set(g1) | set(g2)
    for key in keys:
        v1 = g1.get(key)
        v2 = g2.get(key)
        if isinstance(v1, dict) and isinstance(v2, dict):
            subdiff = diff_metadata(v1, v2)
            if subdiff:
                diff[key] = subdiff
        elif v1 != v2:
            diff[key] = {"g1": v1, "g2": v2}
    return diff


groups = group_scanimage_metadata(md)
groups_og = group_scanimage_metadata(md_og)

diff = diff_metadata(groups, groups_og)

x = 2
