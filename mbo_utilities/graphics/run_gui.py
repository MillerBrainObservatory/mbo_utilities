import click
import numpy as np
from icecream import ic

from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.file_io import (
    to_lazy_array,
)
import fastplotlib as fpl
from imgui_bundle import immapp
from mbo_utilities.graphics._file_dialog import FileDialog
from mbo_utilities.graphics._imgui import setup_imgui

try:
    setup_imgui()
    IMGUI_SETUP_COMPLETE = True
except ImportError:
    IMGUI_SETUP_COMPLETE = False
    print("Failed to set up imgui. GUI functionality may not work as expected.")


@click.command()
@click.option("--roi", type=click.IntRange(1, 10), default=None)
@click.option(
    "--widget",
    default=True,
    help="Enable or disable PreviewDataWidget. Default is enabled.",
)
@click.argument("data_in", required=False)
def run_gui(data_in=None, widget=None, roi=None, **kwargs):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        file_dialog = FileDialog()

        def render_file_dialog():
            file_dialog.render()

        immapp.run(render_file_dialog, with_markdown=True, window_size=(1000, 1000))  # type: ignore
        data_in = file_dialog.selected_path
        if not data_in:
            print("No file or folder selected, exiting.")
            return
    ic(data_in)
    data, fpath = to_lazy_array(data_in, roi=roi, **kwargs)

    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    if sample.ndim < 2:
        raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")

    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx * 2, ny * 2)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if widget:
        gui = PreviewDataWidget(iw=iw, fpath=fpath)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()


if __name__ == "__main__":
    run_gui()  # type: ignore
