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
import mbo_utilities as mbo
import tifffile

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


if __name__ == "__main__":
    # input_tiff = "/home/flynn/lbm_data/assembled/roi2/plane11.tif"
    # metadata = {"plane": 11}
    # mbo.save_nonscan(
    #     input_tiff,
    #     "/home/flynn/lbm_data/assembled",
    #     ext=".tif",
    #     overwrite=True,
    #     metadata=metadata,
    # )
    path = r"D:\W2_DATA\kbarber\2025_03_01\mk301\green"
    test_scan = mbo.read_scan(
        path,
        roi=0,
        phasecorr_method="mean",
    )
    savedir = r"D:\W2_DATA\h5"
    test_scan.roi = 0
    mbo.save_as(
        test_scan,
        savedir,
        ext=".h5",
        overwrite=False,
        fix_phase=True,
        debug=True,
        planes=[7, 8, 9, 10, 11],
    )
