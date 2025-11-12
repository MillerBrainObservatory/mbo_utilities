import copy
from pathlib import Path
from typing import Any, Optional, Union
import click
import numpy as np
from mbo_utilities.array_types import iter_rois, normalize_roi
from mbo_utilities.graphics._file_dialog import FileDialog, setup_imgui


def _select_file() -> tuple[Any, Any, Any, bool]:
    """Show file selection dialog and return user choices."""
    from mbo_utilities.file_io import get_mbo_dirs
    from imgui_bundle import immapp, hello_imgui

    dlg = FileDialog()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Utilities – Data Selection"
    params.app_window_params.window_geometry.size = (1400, 950)
    params.ini_filename = str(
        Path(get_mbo_dirs()["settings"], "fd_settings.ini").expanduser()
    )
    params.callbacks.show_gui = dlg.render

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


def _show_metadata_viewer(metadata: dict) -> None:
    """Show metadata in an ImGui window."""
    from imgui_bundle import immapp, hello_imgui
    from mbo_utilities.graphics._widgets import draw_metadata_inspector

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Metadata Viewer"
    params.app_window_params.window_geometry.size = (800, 800)
    params.callbacks.show_gui = lambda: draw_metadata_inspector(metadata)

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)


def _create_image_widget(data_array, widget: bool = True):
    """Create fastplotlib ImageWidget with optional PreviewDataWidget."""
    import fastplotlib as fpl

    # Handle multi-ROI data
    if hasattr(data_array, "rois"):
        arrays = []
        names = []
        for r in iter_rois(data_array):
            arr = copy.copy(data_array)
            arr.fix_phase = False
            arr.roi = r
            arrays.append(arr)
            names.append(f"ROI {r}" if r else "Full Image")

        iw = fpl.ImageWidget(
            data=arrays,
            names=names,
            histogram_widget=True,
            figure_kwargs={"size": (800, 800)},
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )
    else:
        iw = fpl.ImageWidget(
            data=data_array,
            histogram_widget=True,
            figure_kwargs={"size": (800, 800)},
            graphic_kwargs={"vmin": -100, "vmax": 4000},
        )

    iw.show()

    # Add PreviewDataWidget if requested
    if widget:
        from mbo_utilities.graphics.imgui import PreviewDataWidget

        gui = PreviewDataWidget(
            iw=iw,
            fpath=data_array.filenames,
            size=300,
        )
        iw.figure.add_gui(gui)

    return iw


def _is_jupyter() -> bool:
    """Check if running in Jupyter environment."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def _run_gui_impl(
    data_in: Optional[Union[str, Path]] = None,
    roi: Optional[Union[int, tuple[int, ...]]] = None,
    widget: bool = True,
    metadata_only: bool = False,
):
    """Internal implementation of GUI launcher."""
    setup_imgui()  # ensure assets (fonts + icons) are available

    # Handle file selection if no path provided
    if data_in is None:
        data_in, roi_from_dialog, widget, metadata_only = _select_file()
        if not data_in:
            print("No file selected, exiting.")
            return None
        # Use ROI from dialog if not specified in function call
        if roi is None:
            roi = roi_from_dialog

    # Normalize ROI to standard format
    roi = normalize_roi(roi)

    # Load data
    from mbo_utilities.lazy_array import imread
    data_array = imread(data_in, roi=roi)

    # Show metadata viewer if requested
    if metadata_only:
        metadata = data_array.metadata
        if not metadata:
            print("No metadata found.")
            return None
        _show_metadata_viewer(metadata)
        return None

    # Create and show image viewer
    import fastplotlib as fpl
    iw = _create_image_widget(data_array, widget=widget)

    # In Jupyter, just return the widget (user can interact immediately)
    # In standalone, run the event loop
    if _is_jupyter():
        return iw
    else:
        fpl.loop.run()
        return None


def run_gui(
    data_in: Optional[Union[str, Path]] = None,
    roi: Optional[Union[int, tuple[int, ...]]] = None,
    widget: bool = True,
    metadata_only: bool = False,
):
    """
    Open a GUI to preview data of any supported type.

    Works both as a CLI command and as a Python function for Jupyter/scripts.
    In Jupyter, returns the ImageWidget so you can interact with it.
    In standalone mode, runs the event loop (blocking).

    Parameters
    ----------
    data_in : str, Path, optional
        Path to data file or directory. If None, shows file selection dialog.
    roi : int, tuple of int, optional
        ROI index(es) to display. None shows all ROIs for raw files.
    widget : bool, default True
        Enable PreviewDataWidget for raw ScanImage tiffs.
    metadata_only : bool, default False
        If True, only show metadata inspector (no image viewer).

    Returns
    -------
    ImageWidget or None
        In Jupyter: returns the ImageWidget (already shown via iw.show()).
        In standalone: returns None (runs event loop until closed).

    Examples
    --------
    From Python/Jupyter:
    >>> from mbo_utilities.graphics import run_gui
    >>> # Option 1: Just show the GUI
    >>> run_gui("path/to/data.tif")
    >>> # Option 2: Get reference to manipulate it
    >>> iw = run_gui("path/to/data.tif", roi=1, widget=False)
    >>> iw.cmap = "viridis"  # Change colormap

    From command line:
    $ mbo path/to/data.tif
    $ mbo path/to/data.tif --roi 1 --no-widget
    $ mbo path/to/data.tif --metadata-only
    """
    return _run_gui_impl(
        data_in=data_in,
        roi=roi,
        widget=widget,
        metadata_only=metadata_only,
    )


# Create CLI wrapper
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
def _cli_entry(data_in, widget, roi, metadata_only):
    """CLI entry point for mbo-gui command."""
    _run_gui_impl(
        data_in=data_in,
        roi=roi if roi else None,
        widget=widget,
        metadata_only=metadata_only,
    )


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
