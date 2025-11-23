"""
Graphics module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p) until actually needed.
"""

import os
import sys
import importlib.util

# Force rendercanvas to use Qt backend if PySide6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
# Note: rendercanvas.qt requires PySide6 to be IMPORTED, not just available
if importlib.util.find_spec("PySide6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PySide6  # noqa: F401 - Must be imported before rendercanvas.qt can load

# Configure wgpu instance to skip OpenGL backend and avoid EGL warnings
# This must be done before any wgpu instance is created (before enumerate_adapters/request_adapter)
# Only applies to native platforms (not pyodide/emscripten)
if sys.platform != "emscripten":
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        # Use Vulkan on Linux, DX12 on Windows, Metal on macOS - skip GL to avoid EGL errors
        if sys.platform == "win32":
            set_instance_extras(backends=["Vulkan", "DX12"])
        elif sys.platform == "darwin":
            set_instance_extras(backends=["Metal"])
        else:
            # Linux - Vulkan only, skip GL/EGL
            set_instance_extras(backends=["Vulkan"])
    except ImportError:
        pass  # wgpu not installed or older version without extras

__all__ = [
    "PreviewDataWidget",
    "run_gui",
    "BaseImageProcessor",
    "RasterScanProcessor",
    "MboImageProcessor",  # Alias for backwards compatibility
    "MultiSessionImageProcessor",
]


def __getattr__(name):
    """Lazy import heavy graphics modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    elif name == "PreviewDataWidget":
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    elif name == "BaseImageProcessor":
        from ._processors import BaseImageProcessor
        return BaseImageProcessor
    elif name == "RasterScanProcessor":
        from ._processors import RasterScanProcessor
        return RasterScanProcessor
    elif name == "MboImageProcessor":
        # Backwards compatibility alias
        from ._processors import MboImageProcessor
        return MboImageProcessor
    elif name == "MultiSessionImageProcessor":
        from ._processors import MultiSessionImageProcessor
        return MultiSessionImageProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
