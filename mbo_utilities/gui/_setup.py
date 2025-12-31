"""imgui and graphics setup for the mbo_utilities graphics module.

this module handles all initialization for imgui_bundle, hello_imgui,
wgpu backend configuration, and qt setup. importing this module
automatically runs setup once.
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

# track initialization state
_initialized = False


def _copy_assets():
    """copy package assets to user config directory."""
    import imgui_bundle
    from mbo_utilities.file_io import get_package_assets_path
    import mbo_utilities as mbo

    package_assets = get_package_assets_path()
    user_assets = Path(mbo.get_mbo_dirs()["base"]) / "imgui" / "assets"

    user_assets.mkdir(parents=True, exist_ok=True)
    if package_assets.is_dir():
        shutil.copytree(package_assets, user_assets, dirs_exist_ok=True)

    # copy imgui_bundle fonts as fallback
    fonts_dst = user_assets / "fonts"
    fonts_dst.mkdir(parents=True, exist_ok=True)
    (user_assets / "static").mkdir(parents=True, exist_ok=True)

    fonts_src = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
    for p in fonts_src.rglob("*"):
        if p.is_file():
            d = fonts_dst / p.relative_to(fonts_src)
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)

    # ensure roboto fonts exist for markdown rendering
    roboto_dir = fonts_dst / "Roboto"
    roboto_dir.mkdir(parents=True, exist_ok=True)
    required = [
        roboto_dir / "Roboto-Regular.ttf",
        roboto_dir / "Roboto-Bold.ttf",
        roboto_dir / "Roboto-RegularItalic.ttf",
        fonts_dst / "fontawesome-webfont.ttf",
    ]
    fallback = next((t for t in roboto_dir.glob("*.ttf")), None)
    for need in required:
        if not need.exists() and fallback and fallback.exists():
            need.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, need)

    # copy FontAwesome 6 font for icons
    fa6_src = fonts_src / "Font_Awesome_6_Free-Solid-900.otf"
    fa6_dst = fonts_dst / "Font_Awesome_6_Free-Solid-900.otf"
    if fa6_src.exists() and not fa6_dst.exists():
        shutil.copy2(fa6_src, fa6_dst)

    return user_assets


def _configure_imgui(user_assets: Path):
    """configure hello_imgui assets folder."""
    from imgui_bundle import hello_imgui

    # set hello_imgui assets folder
    hello_imgui.set_assets_folder(str(user_assets))


def get_default_ini_path(name: str = "imgui_settings") -> str:
    """get path for imgui ini file in the mbo settings directory.

    use this with RunnerParams.ini_filename before calling immapp.run().

    Parameters
    ----------
    name : str
        base name for the ini file (without .ini extension)

    Returns
    -------
    str
        full path to the ini file in ~/mbo/imgui/assets/app_settings/
    """
    import mbo_utilities as mbo

    settings_dir = Path(mbo.get_mbo_dirs()["base"]) / "imgui" / "assets" / "app_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return str(settings_dir / f"{name}.ini")


def _configure_qt_backend():
    """set up qt backend for rendercanvas if pyside6 is available.

    must happen before importing fastplotlib to avoid glfw selection.
    """
    if importlib.util.find_spec("PySide6") is None:
        return

    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PySide6  # noqa: F401

    # fix suite2p pyside6 compatibility
    from PySide6.QtWidgets import QSlider
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks


def _configure_wgpu_backend():
    """configure wgpu instance to skip opengl backend and avoid egl warnings."""
    if sys.platform == "emscripten":
        return

    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        if sys.platform == "win32":
            set_instance_extras(backends=["Vulkan", "DX12"])
        elif sys.platform == "darwin":
            set_instance_extras(backends=["Metal"])
        else:
            set_instance_extras(backends=["Vulkan"])
    except ImportError:
        pass


def setup_imgui():
    """initialize all graphics configuration.

    safe to call multiple times - only runs once.
    configures:
    - qt backend for rendercanvas
    - wgpu backend settings
    - imgui assets and ini file location
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    _configure_qt_backend()
    _configure_wgpu_backend()
    user_assets = _copy_assets()
    _configure_imgui(user_assets)


# run setup on import
setup_imgui()
