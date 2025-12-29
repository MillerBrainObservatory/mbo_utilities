"""
CLI entry point for mbo_utilities GUI.

This module is designed for fast startup - heavy imports are deferred until needed.
Operations like --download-notebook and --check-install should be near-instant.
"""
import sys
import os
import importlib.util
from pathlib import Path
from typing import Any, Optional, Union

import click


def _set_qt_icon():
    """Set the Qt application window icon. Call after QApplication is created."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        app = QApplication.instance()
        if app is not None:
            # try package assets first, then user assets
            icon_path = get_package_assets_path() / "app_settings" / "icon.png"
            if not icon_path.exists():
                icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"
            if icon_path.exists():
                app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass  # icon is non-critical
