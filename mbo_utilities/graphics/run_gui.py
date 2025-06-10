import copy
from pathlib import Path

import click
import numpy as np

# force import to ensure glfw is initialized before fastplotlib
import glfw
from rendercanvas.glfw import GlfwRenderCanvas, loop
import fastplotlib as fpl
from imgui_bundle import immapp, hello_imgui

from mbo_utilities import get_mbo_dirs
from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.lazy_array import LazyArrayLoader
from mbo_utilities.graphics._file_dialog import FileDialog

try:
    # setup_imgui()
    IMGUI_SETUP_COMPLETE = True
except ImportError:
    IMGUI_SETUP_COMPLETE = False
    print("Failed to set up imgui. GUI functionality may not work as expected.")


def _select_file() -> tuple[str | None, bool, bool]:
    dlg = FileDialog()

    def _render():
        dlg.render()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (1400, 950)
    params.ini_filename = str(Path(get_mbo_dirs()["settings"], "fd_settings.ini").expanduser())
    params.callbacks.show_gui = _render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    hello_imgui.set_assets_folder(str(get_mbo_dirs()["assets"]))
    immapp.run(runner_params=params, add_ons_params=addons)
    return dlg.selected_path, dlg.widget_enabled, dlg.threading_enabled


@click.command()
@click.option("--roi", type=click.IntRange(-1, 10), default=0)
@click.option(
    "--widget/--no-widget",
    default=True,
    help="Enable or disable PreviewDataWidget (default enabled).",
)
@click.option(
    "--threading/--no-threading",
    default=True,
    help="Enable or disable threading (only effects widgets).",
)
@click.argument("data_in", required=False)
def run_gui(data_in=None, widget=None, roi=None, threading=True):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        data_in, widget, threading = _select_file()
        if not data_in:
            click.echo("No file selected, exiting.")
            return

    lazy_obj  = LazyArrayLoader(data_in, rois=roi)

    # arrays  = lazy_obj.loader.rois or [lazy_obj.loader.load()]  # fall back to whole data
    arrays  = lazy_obj.load()
    if isinstance(arrays, list):
        names   = [f"ROI {i+1}" for i in range(len(arrays))]
    else:
        names   = [f"Data with shape: {arrays.shape}"]

    a0 = arrays[0] if isinstance(arrays, list) else arrays
    nx, ny = a0.shape[-2:]

    iw = fpl.ImageWidget(
        data=arrays,
        names=names,
        histogram_widget=True,
        figure_kwargs={"size": (nx * 2, ny * 2),},  # "canvas": canvas},
        graphic_kwargs={"vmin": a0.min(), "vmax": a0.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if widget:
        gui = PreviewDataWidget(iw=iw, fpath=data_in, threading_enabled=threading, size=350)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()

    # if hasattr(data, "rois"):
    #     arrs = []
    #     for roi in range(1, data.num_rois + 1):
    #         scan_copy = copy.copy(data)
    #         scan_copy.selected_roi = roi
    #         arrs.append(scan_copy)
    #
    #     nx, ny = data.shape[-2:]
    #     iw = fpl.ImageWidget(
    #         data=arrs,
    #         names=[f"ROI {i + 1}" for i in range(len(arrs))],
    #         histogram_widget=False,
    #         figure_kwargs={"size": (nx * 2, ny * 2)},
    #         graphic_kwargs={"vmin": data.min(), "vmax": data.max()},
    #         window_funcs={"t": (np.mean, 0)},
    #     )
    #
    #     if widget:
    #         gui = PreviewDataWidget(
    #             iw=iw,
    #             fpath=fpath,
    #             threading_enabled=threading,
    #             size=350,
    #             location="right",
    #             title="Data Preview",
    #             show_title=True,
    #             movable=True,
    #             resizable=True,
    #             scrollable=True,
    #             auto_resize=True,
    #             window_flags=None,
    #         )
    #         iw.figure.add_gui(gui)
    #
    #     iw.show()
    #     fpl.loop.run()
    #
    # else:
    #     if isinstance(data, list):
    #         sample = data[0]
    #     else:
    #         sample = data
    #
    #     if sample.ndim < 2:
    #         raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")
    #
    #     nx, ny = sample.shape[-2:]
    #     iw = fpl.ImageWidget(
    #         data=data,
    #         histogram_widget=True,
    #         figure_kwargs={"size": (nx * 2, ny * 2)},
    #         graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
    #         window_funcs={"t": (np.mean, 0)},
    #     )
    #
    #     if widget:
    #         gui = PreviewDataWidget(
    #             iw=iw,
    #             fpath=fpath,
    #             threading_enabled=threading,
    #             size=350,
    #             location="right",
    #             title="Data Preview",
    #             show_title=True,
    #             movable=True,
    #             resizable=True,
    #             scrollable=True,
    #             auto_resize=True,
    #             window_flags=None,
    #         )
    #         iw.figure.add_gui(gui)
    #
    #     iw.show()
    #     fpl.loop.run()
    #

if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
