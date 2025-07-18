from pathlib import Path
from typing import Any

import click

import fastplotlib as fpl
from imgui_bundle import immapp, hello_imgui

from mbo_utilities.graphics._widgets import draw_metadata_inspector
from mbo_utilities.graphics.display import imshow_lazy_array
from mbo_utilities.lazy_array import imread
from mbo_utilities.graphics._file_dialog import FileDialog
from mbo_utilities.file_io import get_mbo_dirs

try:
    # setup_imgui()
    IMGUI_SETUP_COMPLETE = True
except ImportError:
    IMGUI_SETUP_COMPLETE = False
    print("Failed to set up imgui. GUI functionality may not work as expected.")

try:
    from masknmf.visualization.interactive_guis import make_demixing_video

    HAS_MASKNMF = True
except ImportError:
    HAS_MASKNMF = False
    make_demixing_video = None


def _select_file() -> tuple[Any, Any, Any, bool]:
    dlg = FileDialog()

    def _render():
        dlg.render()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (1400, 950)
    params.ini_filename = str(
        Path(get_mbo_dirs()["settings"], "fd_settings.ini").expanduser()
    )
    params.callbacks.show_gui = _render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    hello_imgui.set_assets_folder(str(get_mbo_dirs()["assets"]))
    immapp.run(runner_params=params, add_ons_params=addons)
    return dlg.selected_path, dlg.widget_enabled, dlg.threading_enabled, dlg.metadata_only


@click.command()
@click.option("--roi", type=click.IntRange(0, 10), default=0)
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
@click.option(
    "--metadata-only/--full-preview",
    default=False,
    help="If enabled, only show extracted metadata.",
)
@click.argument("data_in", required=False)
def run_gui(data_in=None, widget=None, roi=None, threading=True, metadata_only=False):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        data_in, widget, threading, metadata_only = _select_file()
        if not data_in:
            click.echo("No file selected, exiting.")
            return

    data_array = imread(data_in, roi=roi)

    if metadata_only:
        data_array = imread(data_in, roi=roi)  # or whatever loads it
        metadata = data_array.metadata
        if not metadata:
            click.echo("No metadata found.")
            return

        def _render():
            draw_metadata_inspector(metadata)

        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "MBO Metadata Viewer"
        params.app_window_params.window_geometry.size = (1600, 1000)
        params.callbacks.show_gui = _render

        addons = immapp.AddOnsParams()
        addons.with_markdown = True
        addons.with_implot = False
        addons.with_implot3d = False

        immapp.run(runner_params=params, add_ons_params=addons)
        return

    if hasattr(data_array, "imshow"):
        iw = imshow_lazy_array(data_array, widget=widget, threading_enabled=threading)
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            histogram_widget=True,
            figure_kwargs={"size": (800, 1000)},
            graphic_kwargs={"vmin": data_array.min(), "vmax": data_array.max()},
        )
    iw.show()
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

        if hasattr(data_array, "num_rois"):
            rois = data_array.num_rois
        else:
            rois = 1 if roi is None else roi + 1
        gui = PreviewDataWidget(
            iw=iw, fpath=data_array.filenames, threading_enabled=threading, size=350, rois=rois
        )
        iw.figure.add_gui(gui)
    fpl.loop.run()
    return


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
