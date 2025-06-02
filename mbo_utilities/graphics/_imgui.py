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
    try:
        if font_path.is_file():
            imgui.get_io().fonts.clear()
            font_path = Path(font_path).expanduser().resolve(strict=True)
            imgui.get_io().fonts.add_font_from_file_ttf(str(font_path), 16.0)
        else:
            ic("Font not found:", font_path)
    except Exception as e:
        ic("Error loading font:", e)


def begin_popup_size():
    width_em = hello_imgui.em_size(1.0)  # 1em in pixels
    win_w = imgui.get_window_width()
    win_h = imgui.get_window_height()

    # 75% of window size in ems
    w = win_w * 0.75 / width_em
    h = win_h * 0.75 / width_em  # same em size applies for height in most UIs

    # Clamp in em units
    w = min(max(w, 20), 60)  # roughly 300–800 px if 1em ≈ 15px
    h = min(max(h, 20), 60)

    return hello_imgui.em_to_vec2(w, h)


def ndim_to_frame(arr, t=0, z=0):
    if arr.ndim == 4:  # TZXY
        return arr[t, z]
    if arr.ndim == 3:  # TXY
        return arr[t]
    if arr.ndim == 2:  # XY
        return arr
    raise ValueError(f"Unsupported data shape: {arr.shape}")
