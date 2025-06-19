import numpy as np
import copy

from mbo_utilities.lazy_array import MboRawArray, MBOTiffArray, iter_rois
import fastplotlib as fpl

from mbo_utilities.pipelines import MBO_DEVICE, HAS_MASKNMF

if HAS_MASKNMF:
    from masknmf.visualization.interactive_guis import make_demixing_video
else:
    make_demixing_video = None


def mbo_tiff_display(array: MBOTiffArray, **kwargs):
    iw = fpl.ImageWidget(
        data=array,
        histogram_widget=True,
        figure_kwargs={"size": (800, 1000),},  # "canvas": canvas},
        graphic_kwargs={"vmin": array.min(), "vmax": array.max()},
        window_funcs={"t": (np.mean, 0)},
    )
    widget = kwargs.get("widget", True)
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

        threading = kwargs.get("threading_enabled", True)
        size = kwargs.get("size", 350)
        print("Warning: Double check threading for the input file paths")
        gui = PreviewDataWidget(
            iw=iw,
            fpath=array.fpath,
            threading_enabled=threading,
            size=size
        )
        iw.figure.add_gui(gui)

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

    print((array.min(), array.max()))
    print(arrays)
    iw = fpl.ImageWidget(
        data=arrays,
        names=names,
        histogram_widget=True,
        figure_kwargs={"size": (800, 1000),},  # "canvas": canvas},
        graphic_kwargs={"vmin": array.min(), "vmax": array.max()},
        window_funcs={"t": (np.mean, 0)},
    )
    widget = kwargs.get("widget", True)
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

        threading = kwargs.get("threading_enabled", True)
        size = kwargs.get("size", 350)
        gui = PreviewDataWidget(iw=iw, fpath=array.filenames, threading_enabled=threading, size=size)
        iw.figure.add_gui(gui)

def demixing_display(array, **kwargs):
    if HAS_MASKNMF:
        return make_demixing_video(  # type: ignore  # noqa
            array,
            device=MBO_DEVICE,
            **kwargs
        )

def imshow_lazy_array(array, **kwargs):
    try:
        import fastplotlib as fpl
    except ImportError:
        raise ImportError("fastplotlib is required for image display.")

    if isinstance(array, MBOTiffArray):
        # If the array is a MBOTiffArray, we can directly use it
        iw = fpl.ImageWidget(array, **kwargs)
        return iw
    elif isinstance(array, MboRawArray):
        return mbo_raw_display(array, **kwargs)
    raise ValueError("No supported lazy array type found for display.")