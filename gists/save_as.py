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
import time
from functools import partial
from typing import Literal

import tifffile
from pathlib import Path
import tifffile as tiff
import numpy as np

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from fastplotlib.widgets import ImageWidget
from scipy.ndimage import fourier_shift

from mbo_utilities.lazy_array import imread, imwrite
import mbo_utilities as mbo
from mbo_utilities import is_raw_scanimage
from mbo_utilities.graphics._imgui import ndim_to_frame
from mbo_utilities.metadata import has_mbo_metadata

from imgui_bundle import imgui


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


class ShiftOffsetWidget(EdgeWindow):
    def __init__(
        self,
        iw: ImageWidget,
        size: int = 300,
        location: Literal["bottom", "right"] = "right",
        title: str = "Shift Pixels",
    ):
        super().__init__(figure=iw.figure, size=size, location=location, title=title)
        self.image_widget = iw
        self._x_shift = 0
        if self.image_widget.frame_apply is None:
            self.image_widget.frame_apply = {"t": self._apply_offset}

    def update(self):
        imgui.text("Shift current frame")
        imgui.spacing()

        changed_x, x = imgui.slider_float("Shift", self._x_shift, -4, 4)
        if changed_x:
            print(f"X shift changed to {x}")
            self._x_shift = x
            self.update_frame_apply()

    def get_raw_frame(self):
        idx = self.image_widget.current_index
        t = idx.get("t", 0)
        return self.image_widget.data[t]

    @staticmethod
    def _apply_offset(frame, shift):
        if shift == 0 or shift == 0.0 or frame.ndim < 2:
            return frame
        rows = frame[1::2]
        f = np.fft.fftn(rows)
        shift_vec = (0, shift)[:rows.ndim]
        rows[:] = np.fft.ifftn(fourier_shift(f, shift_vec)).real
        return frame

    def update_frame_apply(self):
        """Update the frame_apply function of the image widget."""
        self.image_widget.frame_apply = {
            i: partial(self._combined_frame_apply, arr_idx=i)
            for i in range(len(self.image_widget.managed_graphics))
        }

    def _combined_frame_apply(self, frame: np.ndarray, arr_idx: int=0) -> np.ndarray:
        """alter final frame only once, in ImageWidget.frame_apply"""
        frame = self._apply_offset(self.get_raw_frame(), self._x_shift)
        return frame


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Path("/tmp/01").mkdir(exist_ok=True,)
    data = imread(r"/home/flynn/lbm_data/raw")
    data.roi = 0
    assert len(data[0, 0, ...]) == 2, "Data should be 2D after ROI selection"
    assert data[0][0].ndim == 3, "Data should be 2D after ROI selection"
    imwrite(
        data,
        "/tmp/testing",
        ext=".tiff",
        overwrite=True,
        planes=[10, 11],
    )
    data.roi = None
    imwrite(
        data,
        "/tmp/01/testing/full",
        ext=".tiff",
        overwrite=True,
        planes=[10, 11],
    )
    data.roi = [1, 2]
    imwrite(
        data,
        "/tmp/testing/list",
        ext=".tiff",
        overwrite=True,
        planes=[10, 11],
    )
    data.roi = [1, 2]
    imwrite(
        data,
        "/tmp/testing/single",
        ext=".tiff",
        overwrite=True,
        planes=[10, 11],
    )
    files = [x for x in Path("/tmp/01/fsspec").glob("*.tif*")]
    check = tifffile.imread(files[0])
    fpl.ImageWidget(check,
                    histogram_widget=True,
                    figure_kwargs={"size": (800, 1000),},
                    graphic_kwargs={"vmin": check.min(), "vmax": check.max()},
                    window_funcs={"t": (np.mean, 0)},
                   ).show()
    fpl.loop.run()

    # data = scan[:20, 11, :, :]
    # title = f"200 frames, {data.shape[0]} planes, plane {zplane}"
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(data.mean(axis=0)[200:280, 300:380], cmap="gray", vmin=-300, vmax=2500)
    # ax[1].imshow(data.mean(axis=0), cmap="gray", vmin=-300, vmax=2500)
    # plt.title(title)
    # plt.savefig("/tmp/01/both.png")
    # print(data.shape)