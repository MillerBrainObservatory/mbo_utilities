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
_icon_path = None  # cached icon path


def _get_icon_path() -> Path | None:
    """get the application icon path, caching for reuse."""
    global _icon_path
    if _icon_path is not None:
        return _icon_path if _icon_path.exists() else None

    try:
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        # try package assets first, then user assets
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

        if icon_path.exists():
            _icon_path = icon_path
            return _icon_path
    except Exception:
        pass
    return None


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
    also installs hooks to set window icon on all qt windows.
    """
    if importlib.util.find_spec("PySide6") is None:
        return

    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PySide6  # noqa: F401
    from PySide6.QtWidgets import QApplication, QSlider
    from PySide6.QtGui import QIcon

    # fix suite2p pyside6 compatibility
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks

    # get icon path once
    icon_path = _get_icon_path()
    if icon_path is None:
        return

    icon = QIcon(str(icon_path))

    # hook QApplication.__init__ to set icon immediately on creation
    _original_app_init = QApplication.__init__

    def _hooked_app_init(self, *args, **kwargs):
        _original_app_init(self, *args, **kwargs)
        try:
            self.setWindowIcon(icon)
        except Exception:
            pass

    QApplication.__init__ = _hooked_app_init

    # hook rendercanvas QWgpuCanvas to set icon on each window
    try:
        from rendercanvas.qt import QWgpuCanvas

        _original_canvas_init = QWgpuCanvas.__init__
        _original_canvas_show = QWgpuCanvas.show

        def _hooked_canvas_init(self, *args, **kwargs):
            _original_canvas_init(self, *args, **kwargs)
            try:
                self.setWindowIcon(icon)
            except Exception:
                pass

        def _hooked_canvas_show(self):
            # set icon right before show to ensure it's set
            try:
                self.setWindowIcon(icon)
            except Exception:
                pass
            return _original_canvas_show(self)

        QWgpuCanvas.__init__ = _hooked_canvas_init
        QWgpuCanvas.show = _hooked_canvas_show
    except Exception:
        pass

    # if QApplication already exists, set icon now
    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(icon)


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


def set_qt_icon():
    """set the qt application window icon. call after qapplication is created."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon

        icon_path = _get_icon_path()
        if icon_path is None:
            return

        app = QApplication.instance()
        if app is not None:
            icon = QIcon(str(icon_path))
            app.setWindowIcon(icon)
            # also set on all existing windows
            for window in app.topLevelWidgets():
                window.setWindowIcon(icon)
    except Exception:
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
