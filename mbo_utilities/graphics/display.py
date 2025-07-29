import numpy as np
import copy

from mbo_utilities.roi import iter_rois
from mbo_utilities.array_types import DemixingResultsArray, MBOTiffArray, MboRawArray
import fastplotlib as fpl

from mbo_utilities.pipelines import MBO_DEVICE, HAS_MASKNMF

if HAS_MASKNMF:
    from masknmf.visualization.interactive_guis import make_demixing_video
else:
    make_demixing_video = None


def mbo_tiff_display(array: MBOTiffArray, **kwargs):
    return fpl.ImageWidget(
        data=array,
        histogram_widget=True,
        figure_kwargs={
            "size": (800, 1000),
        },  # "canvas": canvas},
        graphic_kwargs={"vmin": array.min(), "vmax": array.max()},
        window_funcs={"t": (np.mean, 0)},
    )


def mbo_raw_display(array: MboRawArray, **kwargs):
    arrays = []
    names = []
    # if roi is None, use a single array.roi = None
    # if roi is 0, get a list of all ROIs by deeepcopying the array and setting each roi
    for roi in iter_rois(array):
        arr = copy.copy(array)
        arr.roi = roi
        arrays.append(arr)
        names.append(f"ROI {roi}" if roi else "Full Image")

    return fpl.ImageWidget(
        data=arrays,
        names=names,
        histogram_widget=True,
        figure_kwargs={
            "size": (800, 1000),
        },  # "canvas": canvas},
        graphic_kwargs={"vmin": array.min(), "vmax": array.max()},
        window_funcs={"t": (np.mean, 0)},
    )

def imshow_lazy_array(array, **kwargs):
    if hasattr(array, "imshow"):
        return array.imshow(**kwargs)
    if isinstance(array, MBOTiffArray):
        return mbo_tiff_display(array, **kwargs)
    elif isinstance(array, MboRawArray):
        return mbo_raw_display(array, **kwargs)
    raise ValueError("No supported lazy array type found for display.")
