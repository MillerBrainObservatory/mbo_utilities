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
    test_scan = mbo.read_scan(
        "/home/flynn/lbm_data/raw",
        roi=0,
        phasecorr_method="mean",
    )
    savedir = r"/home/flynn/lbm_data/bin"
    mbo.save_as(
        test_scan,
        savedir,
        ext=".bin",
        overwrite=True,
        fix_phase=True,
        debug=True,
        planes=[10],
    )
