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


def _install_qt_icon_hook():
    """Install a hook to set Qt icon as soon as QApplication is created."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        # Store original __init__
        _original_init = QApplication.__init__

        def _hooked_init(self, *args, **kwargs):
            # Call original init
            _original_init(self, *args, **kwargs)

            # Set icon immediately after QApplication is created
            try:
                icon_path = get_package_assets_path() / "app_settings" / "icon.png"
                if not icon_path.exists():
                    icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
                if icon_path.exists():
                    self.setWindowIcon(QIcon(str(icon_path)))
            except Exception:
                pass

        # Monkey-patch QApplication.__init__
        QApplication.__init__ = _hooked_init
    except Exception:
        pass


def _configure_qt_backend():
    """set up qt backend for rendercanvas if pyside6 is available.

    must happen before importing fastplotlib to avoid glfw selection.
    """
    if importlib.util.find_spec("PySide6") is not None:
        os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
        import PySide6  # noqa: F401

        # fix suite2p pyside6 compatibility
        from PySide6.QtWidgets import QSlider
        if not hasattr(QSlider, "NoTicks"):
            QSlider.NoTicks = QSlider.TickPosition.NoTicks

        # Install hook to set icon when QApplication is created
        _install_qt_icon_hook()

        # Also monkey-patch rendercanvas Qt window creation
        try:
            from rendercanvas.qt import QWgpuCanvas
            from PySide6.QtGui import QIcon
            from mbo_utilities.file_io import get_package_assets_path
            from mbo_utilities import get_mbo_dirs

            # Store original __init__
            _original_canvas_init = QWgpuCanvas.__init__

            def _hooked_canvas_init(self, *args, **kwargs):
                # Call original init
                _original_canvas_init(self, *args, **kwargs)

                # Set icon on this window
                try:
                    icon_path = get_package_assets_path() / "app_settings" / "icon.png"
                    if not icon_path.exists():
                        icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
                    if icon_path.exists():
                        self.setWindowIcon(QIcon(str(icon_path)))
                except Exception:
                    pass

            # Monkey-patch QWgpuCanvas.__init__
            QWgpuCanvas.__init__ = _hooked_canvas_init
        except Exception:
            pass

        # Also set icon immediately if QApplication already exists
        try:
            from PySide6.QtWidgets import QApplication
            from PySide6.QtGui import QIcon
            from mbo_utilities.file_io import get_package_assets_path
            from mbo_utilities import get_mbo_dirs

            app = QApplication.instance()
            if app is not None:
                icon_path = get_package_assets_path() / "app_settings" / "icon.png"
                if not icon_path.exists():
                    icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
                if icon_path.exists():
                    app.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass


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
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        app = QApplication.instance()
        if app is not None:  # Only set if QApplication already exists
            icon_path = get_package_assets_path() / "app_settings" / "icon.png"
            if not icon_path.exists():
                icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass


def set_window_icon():
    """set window icon for both Qt and GLFW backends."""
    # Try Qt first - set on QApplication
    set_qt_icon()

    # Also set on all Qt windows
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        app = QApplication.instance()
        if app is not None:
            icon_path = get_package_assets_path() / "app_settings" / "icon.png"
            if not icon_path.exists():
                icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

            if icon_path.exists():
                icon = QIcon(str(icon_path))
                # Set on all top-level windows
                for window in app.topLevelWidgets():
                    window.setWindowIcon(icon)
    except Exception:
        pass

    # Try GLFW
    try:
        import glfw
        from PIL import Image
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        # Get the icon path
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

        if icon_path.exists():
            # Load icon image
            img = Image.open(icon_path)
            img = img.convert("RGBA")

            # Get all GLFW windows and set icon
            # Note: This requires the window to be created first
            # We'll set it on the current context window if available
            window = glfw.get_current_context()
            if window:
                glfw.set_window_icon(window, 1, img)
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
