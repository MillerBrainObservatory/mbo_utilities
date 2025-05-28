import shutil

from icecream import ic
from imgui_bundle import imgui, hello_imgui
from pathlib import Path

from mbo_utilities.file_io import (
    _get_mbo_project_root,
    _get_mbo_dirs,
)

def setup_imgui():
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

    font_path = (
        assets_path / "fonts" / "JetBrainsMono" / "JetBrainsMonoNerdFont-Bold.ttf"
    )
    if font_path.is_file():
        imgui.get_io().fonts.clear()
        imgui.get_io().fonts.add_font_from_file_ttf(str(font_path), 16.0)
    else:
        ic("Font not found:", font_path)
