import copy
from pathlib import Path
from typing import Any
import click
import numpy as np
from mbo_utilities.roi import iter_rois, normalize_roi
from mbo_utilities.graphics._file_dialog import FileDialog, setup_imgui


def _select_file() -> tuple[Any, Any, Any, bool]:
    from mbo_utilities.file_io import get_mbo_dirs
    from imgui_bundle import immapp, hello_imgui

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
    return (
        dlg.selected_path,
        dlg.split_rois,
        dlg.widget_enabled,
        dlg.metadata_only,
    )

@click.command()
@click.option(
    "--roi",
    multiple=True,
    type=int,
    help="ROI index (can pass multiple, e.g. --roi 0 --roi 2). Leave empty for None."
         " If 0 is passed, all ROIs will be shown (only for Raw files).",
    default=None,
)
@click.option(
    "--widget/--no-widget",
    default=True,
    help="Enable or disable PreviewDataWidget for Raw ScanImge tiffs.",
)
@click.option(
    "--metadata-only/--full-preview",
    default=False,
    help="If enabled, only show extracted metadata.",
)
@click.argument("data_in", required=False)
def run_gui(data_in=None, widget=None, roi=None, metadata_only=False):
    """Open a GUI to preview data of any supported type."""
    setup_imgui()  # ensure assets (fonts + icons) are available
    roi_cli = normalize_roi(roi)

    if data_in is None:
        data_in, roi_gui, widget, metadata_only = _select_file()
        if not data_in:
            click.echo("No file selected, exiting.")
            return
    else:
        roi_gui = None

    roi_final = normalize_roi(roi_cli if roi_cli is not None else roi_gui)

    from mbo_utilities.lazy_array import imread
    data_array = imread(data_in, roi=roi_final)

    if metadata_only:
        metadata = data_array.metadata
        if not metadata:
            click.echo("No metadata found.")
            return

        def _render():
            from mbo_utilities.graphics._widgets import draw_metadata_inspector
            draw_metadata_inspector(metadata)

        from imgui_bundle import immapp, hello_imgui
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "MBO Metadata Viewer"
        params.app_window_params.window_geometry.size = (800, 800)
        params.callbacks.show_gui = _render

        addons = immapp.AddOnsParams()
        addons.with_markdown = True
        addons.with_implot = False
        addons.with_implot3d = False

        immapp.run(runner_params=params, add_ons_params=addons)
        return

    import fastplotlib as fpl
    if hasattr(data_array, "rois"):
        arrays = []
        names = []
        for r in iter_rois(data_array):
            arr = copy.copy(data_array)
            arr.roi = r
            arrays.append(arr)
            names.append(f"ROI {r}" if r else "Full Image")

        iw = fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=True,
            figure_kwargs={"size": (800, 1000)},
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs={"t": (np.mean, 0)},
        )
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            histogram_widget=True,
            figure_kwargs={"size": (800, 1000)},
            graphic_kwargs={"vmin": -300, "vmax": 4000},
            window_funcs={"t": (np.mean, 0)},
        )

    iw.show()
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

        gui = PreviewDataWidget(
            iw=iw,
            fpath=data_array.filenames,
            size=300,
        )
        iw.figure.add_gui(gui)
    fpl.loop.run()
    return


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
