"""
Graphics module with lazy imports to avoid loading heavy dependencies
(torch, cupy, suite2p, wgpu, imgui_bundle) until actually needed.

The CLI entry point (mbo command) imports this module, so we must keep
top-level imports minimal for fast startup of light operations like
--download-notebook and --check-install.
"""

import os
import sys
import importlib.util

__all__ = [
    "PreviewDataWidget",
    "run_gui",
    "BaseImageProcessor",
    "RasterScanProcessor",
    "MboImageProcessor",  # Alias for backwards compatibility
    "MultiSessionImageProcessor",
]

# Track if wgpu has been configured (to avoid duplicate setup)
_wgpu_configured = False


def _configure_wgpu_backend():
    """Configure wgpu backend settings. Called lazily before first GUI use."""
    global _wgpu_configured
    if _wgpu_configured:
        return

    # Force rendercanvas to use Qt backend if PySide6 is available
    # This must happen BEFORE importing fastplotlib to avoid glfw selection
    if importlib.util.find_spec("PySide6") is not None:
        os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
        import PySide6  # noqa: F401 - Must be imported before rendercanvas.qt can load

    # Configure wgpu instance to skip OpenGL backend and avoid EGL warnings
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

    _wgpu_configured = True


def __getattr__(name):
    """Lazy import heavy graphics modules only when accessed."""
    if name == "run_gui":
        from .run_gui import run_gui
        return run_gui
    elif name == "PreviewDataWidget":
        _configure_wgpu_backend()  # Configure before GUI imports
        from .imgui import PreviewDataWidget
        return PreviewDataWidget
    elif name == "BaseImageProcessor":
        _configure_wgpu_backend()  # Configure before fastplotlib imports
        from ._processors import BaseImageProcessor
        return BaseImageProcessor
    elif name == "RasterScanProcessor":
        _configure_wgpu_backend()
        from ._processors import RasterScanProcessor
        return RasterScanProcessor
    elif name == "MboImageProcessor":
        # Backwards compatibility alias
        _configure_wgpu_backend()
        from ._processors import MboImageProcessor
        return MboImageProcessor
    elif name == "MultiSessionImageProcessor":
        _configure_wgpu_backend()
        from ._processors import MultiSessionImageProcessor
        return MultiSessionImageProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
