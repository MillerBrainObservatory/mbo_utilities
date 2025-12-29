# Qt Window Icon Setup

## Overview

The MBO Utilities GUI uses a custom icon for the window title bar and Windows taskbar. This document explains how the icon is set and why.

## How It Works

The icon is set in `run_gui.py` via the `_set_qt_icon()` function, called **after** the canvas/window is created and shown:

```python
iw.show()
_set_qt_icon()
```

## Why After Show?

Qt/PySide6 with rendercanvas creates the QApplication and window internally. We cannot set the icon before the window exists. The sequence is:

1. `ImageWidget` is created with `canvas="pyside6"`
2. `iw.show()` creates and shows the Qt window
3. `_set_qt_icon()` sets the icon on the now-existing window

This means there's a brief flash of the default Python icon before the custom icon appears. This is expected behavior.

## The _set_qt_icon() Function

Located in `mbo_utilities/gui/run_gui.py`:

```python
def _set_qt_icon():
    """Set the Qt application window icon."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QIcon
        from mbo_utilities.file_io import get_package_assets_path
        from mbo_utilities import get_mbo_dirs

        app = QApplication.instance()
        if app is None:
            return

        # try package assets first, then user assets
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        if not icon_path.exists():
            icon_path = Path(get_mbo_dirs()["assets"]) / "app_settings" / "icon.png"

        if not icon_path.exists():
            return

        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

        # set on all top-level windows including native handles
        for window in app.topLevelWidgets():
            window.setWindowIcon(icon)
            handle = window.windowHandle()
            if handle:
                handle.setIcon(icon)
            app.processEvents()
    except Exception:
        pass
```

Key points:
- Sets icon on `QApplication` for new windows
- Sets icon on all existing top-level widgets
- Sets icon on native window handles (required for Windows taskbar)
- Calls `processEvents()` to force the update

## Windows Taskbar Icon

For Windows taskbar icon grouping, an AppUserModelID is set at module load in `run_gui.py`:

```python
if sys.platform == 'win32':
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mbo.utilities.gui.1.0')
```

This ensures Windows groups all MBO Utilities windows together with the correct icon.

## Icon Location

The icon file is located at:
- Package: `mbo_utilities/assets/app_settings/icon.png`
- User fallback: `~/mbo/imgui/assets/app_settings/icon.png`

## Troubleshooting

If the icon doesn't appear:
1. Verify the icon.png file exists at the expected path
2. Close the window completely and restart
3. Windows may cache icons - try restarting explorer.exe
