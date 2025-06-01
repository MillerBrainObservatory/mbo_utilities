import copy
from pathlib import Path
from typing import Sequence

import click
import numpy as np
from icecream import ic

import fastplotlib as fpl
from imgui_bundle import immapp

from mbo_utilities.file_io import Scan_MBO
from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.lazy_array import LazyArrayLoader
from mbo_utilities.graphics._file_dialog import FileDialog
from mbo_utilities.graphics._imgui import setup_imgui

try:
    setup_imgui()
    IMGUI_SETUP_COMPLETE = True
except ImportError:
    IMGUI_SETUP_COMPLETE = False
    print("Failed to set up imgui. GUI functionality may not work as expected.")


def to_lazy_array(
    data_in: str | Path | Sequence[str | Path] | Scan_MBO | np.ndarray,
    roi: int | None = None,
) -> tuple[Scan_MBO | np.ndarray | list[np.ndarray], str | None] | None:
    """
    Load any of your supported formats lazily and return (data, filepaths).
    """
    lazy_array = LazyArrayLoader(data_in, roi=roi)
    if hasattr(lazy_array.loader, "paths"):
        return lazy_array.loader.load(), lazy_array.loader.paths
    data = lazy_array.load()
    if isinstance(data, Scan_MBO):
        if hasattr(data, "fpath"):
            path = data.fpath
        else:
            path = data_in
    elif isinstance(data_in, np.ndarray):
        if hasattr(data, "fpath"):
            path = data.fpath
        else:
            path = None
    else:
        return None
    return data, path


@click.command()
@click.option("--roi", type=click.IntRange(-1, 10), default=None)
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
        file_dialog = FileDialog()

        def render_file_dialog():
            file_dialog.render()

        immapp.run(render_file_dialog, with_markdown=True, window_size=(1000, 1000))  # type: ignore  # noqa
        data_in = file_dialog.selected_path
        threading = file_dialog.threading_enabled
        if not data_in:
            print("No file or folder selected, exiting.")
            return
    ic(data_in)
    data, fpath = to_lazy_array(data_in, roi=roi)
    if hasattr(data, "rois"):
        arrs = []
        for roi in range(len(data.rois)):
            scan_copy = copy.copy(data)
            scan_copy.roi = roi + 1
            arrs.append(scan_copy)

        nx, ny = data.shape[-2:]
        iw = fpl.ImageWidget(
            data=arrs,
            names=[f"ROI {i + 1}" for i in range(len(arrs))],
            histogram_widget=False,
            figure_kwargs={"size": (nx * 2, ny * 2)},
            graphic_kwargs={"vmin": data.min(), "vmax": data.max()},
            window_funcs={"t": (np.mean, 0)},
        )

        if widget:
            gui = PreviewDataWidget(
                iw=iw,
                fpath=fpath,
                threading_enabled=threading,
                size=350,
                location="right",
                title="Data Preview",
                show_title=True,
                movable=True,
                resizable=True,
                scrollable=True,
                auto_resize=True,
                window_flags=None,
            )
            iw.figure.add_gui(gui)

        iw.show()
        fpl.loop.run()

    else:
        if isinstance(data, list):
            sample = data[0]
        else:
            sample = data

        if sample.ndim < 2:
            raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")

        nx, ny = sample.shape[-2:]
        iw = fpl.ImageWidget(
            data=data,
            histogram_widget=True,
            figure_kwargs={"size": (nx * 2, ny * 2)},
            graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
            window_funcs={"t": (np.mean, 0)},
        )

        if widget:
            gui = PreviewDataWidget(
                iw=iw,
                fpath=fpath,
                threading_enabled=threading,
                size=350,
                location="right",
                title="Data Preview",
                show_title=True,
                movable=True,
                resizable=True,
                scrollable=True,
                auto_resize=True,
                window_flags=None,
            )
            iw.figure.add_gui(gui)

        iw.show()
        fpl.loop.run()


if __name__ == "__main__":
    run_gui()  # type: ignore # noqa
