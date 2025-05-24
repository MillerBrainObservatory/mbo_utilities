from .run_gui import run_gui
from .imgui import PreviewDataWidget
from imgui_bundle import hello_imgui

from pathlib import Path

def setup_imgui():
    from mbo_utilities import get_mbo_project_root, mbo_paths

    assets: Path = get_mbo_project_root.joinpath("assets")
    if Path(assets).is_dir():
        print("yes!")
    hello_imgui.set_assets_folder(str(assets))


__all__ = [
    "SummaryDataWidget",
    "PreviewDataWidget",
    "run_gui",
]
