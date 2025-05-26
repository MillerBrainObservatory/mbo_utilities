import shutil
from pathlib import Path

import click
import numpy as np
from icecream import ic

from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.file_io import (
    to_lazy_array,
    _is_arraylike,
    _get_mbo_dirs, _get_mbo_project_root,
)
import fastplotlib as fpl
from imgui_bundle import immapp, imgui, hello_imgui
from mbo_utilities.graphics._file_dialog import FileDialog

def setup():
    project_assets: Path = _get_mbo_project_root().joinpath("assets")
    mbo_dirs = _get_mbo_dirs()

    imgui_path = mbo_dirs["base"].joinpath("imgui")
    imgui_path.mkdir(exist_ok=True)

    imgui_ini_path = imgui_path.joinpath("imgui.ini")
    imgui_ini_path.parent.mkdir(exist_ok=True)
    imgui.create_context()
    imgui.get_io().set_ini_filename(str(imgui_ini_path))

    if not project_assets.is_dir():
        ic("Assets folder not found.")
        return

    assets_path = imgui_path.joinpath("assets")
    assets_path.mkdir(exist_ok=True)

    shutil.copytree(project_assets, assets_path, dirs_exist_ok=True)
    hello_imgui.set_assets_folder(str(project_assets))

    font_path = assets_path / "fonts" / "JetBrainsMono-Bold.ttf"
    if font_path.is_file():
        imgui.get_io().fonts.clear()
        imgui.get_io().fonts.add_font_from_file_ttf(str(font_path), 16.0)
    else:
        ic("Font not found:", font_path)


setup()


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
            x = hello_imgui.RunnerParams.ini_filename
            file_dialog.render()
        immapp.run(render_file_dialog, with_markdown=True, window_size=(1000, 1000))  # type: ignore
        data_in = file_dialog.selected_path
    if _is_arraylike(data_in):
        ic(data_in)
        data = data_in
    else:
        ic(data_in)
        data = to_lazy_array(data_in, roi=roi, **kwargs)

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
        figure_kwargs={"size": (nx*2.5, ny*2.5)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if widget:
        fpath = data.fpath if hasattr(data, "fpath") else None
        gui = PreviewDataWidget(iw=iw, fpath=fpath)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()

if __name__ == "__main__":
    run_gui()  # type: ignore