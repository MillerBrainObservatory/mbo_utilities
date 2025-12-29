# Desktop/Window Icon Setup Reference

This document describes all the pathways through which window icons can be set when running `uv run mbo` or any GUI component in mbo_utilities.

## Overview

There are **four distinct icon-setting mechanisms** in play:

1. **hello_imgui** (C++ level) - sets GLFW/SDL window icons from `assets/app_settings/icon.png`
2. **Qt/PySide6** - sets Qt application and window icons via `QApplication.setWindowIcon()`
3. **GLFW direct** - sets GLFW window icons via `glfw.set_window_icon()`
4. **Windows AppUserModelID** - associates the process with a taskbar icon group

## Current Status

| Backend | Icon Working | Mechanism |
|---------|--------------|-----------|
| GLFW (via hello_imgui) | Yes | hello_imgui loads `assets/app_settings/icon.png` automatically |
| Qt/PySide6 | Inconsistent | Multiple hook attempts, timing-dependent |
| Windows Taskbar | Partial | AppUserModelID set, but icon association may fail |

## Architecture Diagram

```
uv run mbo
    │
    ├── [Windows] SetCurrentProcessExplicitAppUserModelID('mbo.utilities.gui.1.0')
    │       └── run_gui.py lines 15-23
    │
    ├── _configure_qt_backend() [if PySide6 available]
    │   │   └── _setup.py lines 122-183
    │   │
    │   ├── _install_qt_icon_hook()
    │   │       └── Monkey-patches QApplication.__init__ to call setWindowIcon()
    │   │
    │   ├── Monkey-patch QWgpuCanvas.__init__
    │   │       └── Calls setWindowIcon() on each rendercanvas Qt window
    │   │
    │   └── Check if QApplication already exists
    │           └── If so, call setWindowIcon() immediately
    │
    ├── fastplotlib.Figure() created
    │       └── Uses rendercanvas internally
    │
    ├── hello_imgui.run() / immapp.run()
    │   │   └── Spawns GLFW or SDL window for ImGui
    │   │
    │   └── [C++ level] abstract_runner.cpp line 650
    │           └── Calls Impl_SetWindowIcon()
    │                   │
    │                   ├── [GLFW] runner_glfw3.cpp lines 138-163
    │                   │       └── Loads assets/app_settings/icon.png via stb_image
    │                   │       └── Calls glfwSetWindowIcon()
    │                   │
    │                   └── [SDL2] runner_sdl2.cpp lines 259-289
    │                           └── Loads assets/app_settings/icon.png via stb_image
    │                           └── Calls SDL_SetWindowIcon()
    │
    └── _set_qt_icon() called after canvas creation
            └── run_gui.py lines 26-53, called at line 472
```

## Detailed Breakdown by Library

### 1. hello_imgui (GLFW working, Qt deprecated)

**Location:** `c:\Users\RBO\repos\hello_imgui\src\hello_imgui\internal\backend_impls\`

**How it works:**
- hello_imgui has a virtual method `Impl_SetWindowIcon()` in `abstract_runner.h`
- Called after GL context creation in `abstract_runner.cpp` line 650
- GLFW implementation (`runner_glfw3.cpp` lines 138-163):
  ```cpp
  void RunnerGlfw3::Impl_SetWindowIcon() {
      std::string iconFile = "app_settings/icon.png";  // hard-coded path
      if (!HelloImGui::AssetExists(iconFile)) return;

      auto imageAsset = HelloImGui::LoadAssetFileData(iconFile.c_str());
      unsigned char* image = stbi_load_from_memory(...);

      GLFWimage icons[1];
      icons[0].width = width;
      icons[0].height = height;
      icons[0].pixels = image;
      glfwSetWindowIcon((GLFWwindow*)mWindow, 1, icons);
  }
  ```

**Critical:** The icon path is hard-coded to `app_settings/icon.png` relative to the assets folder. Your icon at `mbo_utilities/assets/app_settings/icon.png` must be findable by hello_imgui's asset system.

**Qt backend:** The Qt runner in hello_imgui is **deprecated** and does NOT implement `Impl_SetWindowIcon()`. It uses the empty default implementation.

### 2. rendercanvas (NO icon support)

**Location:** `c:\Users\RBO\repos\rendercanvas\`

**Finding:** rendercanvas does **NOT** implement any window icon setup for Qt or GLFW backends.

- No `setWindowIcon()` calls
- No `glfwSetWindowIcon()` calls
- The only "icon" reference is `set_window_iconify_callback` which handles minimize/maximize events

**Implication:** When fastplotlib uses rendercanvas with Qt backend, the icon must be set externally by your application code.

### 3. fastplotlib (NO icon support)

**Location:** `c:\Users\RBO\repos\fastplotlib\`

**Finding:** fastplotlib does **NOT** implement window icon setup.

- Has logo asset at `fastplotlib/assets/fastplotlib_face_logo.png` but only uses it for Jupyter notebook banner
- No `QIcon`, `setWindowIcon`, or GLFW icon code
- Delegates entirely to rendercanvas for window creation

### 4. mbo_utilities icon setup attempts

**Location:** `c:\Users\RBO\repos\mbo_utilities\mbo_utilities\gui\`

**Files involved:**
- `run_gui.py` - Windows AppUserModelID + `_set_qt_icon()` function
- `_setup.py` - Qt hook system + monkey-patching
- `_qt_icon_helper.py` - Simplified Qt icon helper

**Current approach (multiple layers):**

#### Layer 1: Windows AppUserModelID (run_gui.py lines 15-23)
```python
if sys.platform == 'win32':
    myappid = 'mbo.utilities.gui.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
```
This sets the Windows application identity for taskbar grouping.

#### Layer 2: QApplication.__init__ hook (_setup.py lines 91-119)
```python
def _install_qt_icon_hook():
    _original_init = QApplication.__init__

    def _hooked_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        # Set icon immediately after QApplication is created
        icon_path = get_package_assets_path() / "app_settings" / "icon.png"
        self.setWindowIcon(QIcon(str(icon_path)))

    QApplication.__init__ = _hooked_init
```

#### Layer 3: QWgpuCanvas.__init__ hook (_setup.py lines 140-164)
```python
def _hooked_canvas_init(self, *args, **kwargs):
    _original_canvas_init(self, *args, **kwargs)
    self.setWindowIcon(QIcon(str(icon_path)))

QWgpuCanvas.__init__ = _hooked_canvas_init
```

#### Layer 4: Direct icon set after canvas creation (run_gui.py line 472)
```python
iw.show()
_set_qt_icon()  # set qt window icon after canvas is created
```

### 5. PySide6/Qt icon internals

**Location:** `c:\Users\RBO\repos\pyside-pyside-setup\`

**How QIcon works:**
- `QIcon("path/to/icon.png")` loads PNG, ICO, SVG files
- `QApplication.setWindowIcon(icon)` sets default icon for all windows
- `QWidget.setWindowIcon(icon)` sets icon for specific window
- Windows expects `.ico` format for native handling but Qt converts PNG internally

**Known behaviors:**
- Icon must be set AFTER QApplication is created
- Setting on QApplication affects all windows created afterward
- Individual windows can override with their own `setWindowIcon()`
- On Windows, the taskbar icon may come from the executable's embedded resource

## Why Qt Icon Might Not Be Showing

### Potential Issues:

1. **Timing:** Icon hooks may fire before the window is fully realized
   - QApplication.__init__ hook fires, but window isn't created yet
   - Window gets created later without the icon

2. **Multiple QApplication instances:** rendercanvas may create its own QApplication before your hooks install

3. **Window handle not ready:** `window.windowHandle()` may return None if window isn't shown yet

4. **Icon path not found:** Package assets may not be correctly resolved in installed vs development mode

5. **Native widget ordering:** For Qt, the window needs to be "realized" (have a native handle) before the icon properly applies

### Debugging Steps:

```python
# Add this after creating the figure to debug:
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

app = QApplication.instance()
print(f"QApplication exists: {app is not None}")
print(f"App icon null: {app.windowIcon().isNull() if app else 'N/A'}")

for w in app.topLevelWidgets():
    print(f"Window: {w.windowTitle()}")
    print(f"  Icon null: {w.windowIcon().isNull()}")
    print(f"  Has handle: {w.windowHandle() is not None}")
```

## Git History of Icon Changes

Key commits in mbo_utilities:

| Commit | Description |
|--------|-------------|
| `3e536ce` | First Qt icon setup in `graphics/run_gui.py` |
| `8bb04ef` | Fix icon and install scripts |
| `be92328` | Metadata search, keybinds, icon fix - adds assets to wheel |
| `0353a8c` | Add white icon to docs |

The `refact-array` branch has additional monkey-patching in `_setup.py` that's not in master.

## Recommended Fix Strategy

### Option A: Ensure Qt icon is set at the right time

```python
def _set_qt_icon_on_all_windows():
    """Set icon on all Qt windows, including those already created."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon

    app = QApplication.instance()
    if not app:
        return

    icon_path = get_package_assets_path() / "app_settings" / "icon.png"
    if not icon_path.exists():
        return

    icon = QIcon(str(icon_path))

    # Set on application (affects new windows)
    app.setWindowIcon(icon)

    # Set on all existing top-level widgets
    for widget in app.topLevelWidgets():
        widget.setWindowIcon(icon)
        # Force update if window has native handle
        if widget.windowHandle():
            widget.windowHandle().setIcon(icon)

# Call this AFTER all windows are created and shown
```

### Option B: Use a QTimer to delay icon setting

```python
from PySide6.QtCore import QTimer

def delayed_icon_set():
    QTimer.singleShot(100, _set_qt_icon_on_all_windows)
```

### Option C: Use Qt event filter

```python
class IconEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QWidget):
            if obj.isWindow():
                obj.setWindowIcon(get_icon())
        return False

app.installEventFilter(IconEventFilter())
```

## Summary

| Library | Icon Support | Status |
|---------|--------------|--------|
| hello_imgui | GLFW: Yes, SDL: Yes, Qt: No | Works for pure ImGui windows |
| rendercanvas | None | Must set externally |
| fastplotlib | None | Relies on rendercanvas |
| mbo_utilities | Qt: Multiple hooks | Timing-dependent, may not work |
| PySide6 | Full | Works if called correctly |

The GLFW icon works because hello_imgui handles it at the C++ level before Python code runs. The Qt icon is problematic because:
1. rendercanvas doesn't set it
2. fastplotlib doesn't set it
3. mbo_utilities hooks may fire at the wrong time
4. The window may not have a native handle when the icon is set
