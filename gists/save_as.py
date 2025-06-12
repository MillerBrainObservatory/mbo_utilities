# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "mbo_utilities",
#     "fastplotlib",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "dev" }
import tifffile
from pathlib import Path
import tifffile as tiff
import numpy as np

import mbo_utilities as mbo
from mbo_utilities import is_raw_scanimage
from mbo_utilities.metadata import has_mbo_metadata


def find_si_rois(file):
    """
    Find the ROIs in the current ScanImage session.

    Returns
    -------
    list
        List of ROI names.
    """
    with tifffile.TiffFile(file, mode="r") as _tf:
        if is_raw_scanimage(file):
            si_metadata = _tf.scanimage_metadata
        if has_mbo_metadata(file):
            si_metadata = _tf.shaped_metadata[0]["si"]
        rois = si_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
    return rois


def write_u16(infile: str | Path, outfile: str | Path):
    img = tiff.imread(infile).astype(np.int32)
    off  = img.min()
    rng  = img.max() - off
    u16  = (img - off).astype(np.uint16)

    tiff.imwrite(
        outfile,
        u16,
        photometric="minisblack",
        bitspersample=16,
        extratags=[
            (340, "H", 1, (0,),   False),
            (341, "H", 1, (rng if rng < 65536 else 65535,), False),
            (65535, "d", 2, (float(off), float(rng)), False)
        ],
    )



if __name__ == "__main__":
    # metadata = {"plane": 11}
    # mbo.save_nonscan(
    #     input_tiff,
    #     "/home/flynn/lbm_data/assembled",
    #     ext=".tif",
    #     overwrite=True,
    #     metadata=metadata,
    # )
    path = r"D:\W2_DATA\kbarber\2025_03_01\mk301\green"
    savedir = r"D:\tests_bigmem\no_phase"
    test_scan = mbo.read_scan(
        path,
        roi=0,
        phasecorr_method="mean",
    )
    test_scan.roi = 0
    test_scan.fix_phase = False
    mbo.save_as(
        test_scan,
        savedir,
        ext=".tiff",
        overwrite=True,
        planes=[7, 8, 9, 10, 11],
    )
