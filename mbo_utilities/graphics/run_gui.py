import click
import numpy as np

from mbo_utilities.util import is_imgui_installed
from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.file_io import (
    to_lazy_array,
    get_files,
    read_scan,
    _is_arraylike,
)
import fastplotlib as fpl
from imgui_bundle import immapp
from mbo_utilities.graphics._file_dialog import render_file_dialog

global selected


@click.command()
@click.option("--roi", "-r", type=click.IntRange(1, 10), default=None)
@click.option(
    "--gui/--no-gui",
    default=None,
    help="Enable or disable PreviewDataWidget. Default is auto.",
)
@click.argument("data_in", required=False)
def run_gui(data_in=None, gui=None, roi=None, **kwargs):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        immapp.run(render_file_dialog, with_markdown=True, window_size=(1000, 1000))  # type: ignore
        if not selected:
            print("not selected")
        else:
            fpath = selected
        files = get_files(fpath)
        data = read_scan(files)
    elif _is_arraylike(data_in):
        data = data_in
    else:
        data = to_lazy_array(data_in)

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
        figure_kwargs={"size": (nx, ny)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if kwargs.get("gui"):
        gui = PreviewDataWidget(iw=iw)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
