import importlib.util

__all__ = [
    "HAS_SUITE2P",
    "HAS_SUITE3D",
    "HAS_CUPY",
    "HAS_TORCH",
    "HAS_RASTERMAP",
    "HAS_IMGUI",
    "HAS_FASTPLOTLIB",
    "HAS_PYSIDE6",
]


def _check_import(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    return importlib.util.find_spec(module_name) is not None


# Suite2p pipeline (lbm_suite2p_python is the main entry point)
HAS_SUITE2P: bool = _check_import("lbm_suite2p_python")

# Suite3D volumetric registration (requires both suite3d AND cupy for GPU)
HAS_SUITE3D: bool = _check_import("suite3d") and _check_import("cupy")

# CuPy for GPU acceleration (used by suite3d and cusignal)
HAS_CUPY: bool = _check_import("cupy")

# PyTorch for neural network operations
HAS_TORCH: bool = _check_import("torch")

# Rastermap for dimensionality reduction
HAS_RASTERMAP: bool = _check_import("rastermap")

# gui dependencies
# ImGui Bundle - core GUI framework
HAS_IMGUI: bool = _check_import("imgui_bundle")

# FastPlotLib - visualization
HAS_FASTPLOTLIB: bool = _check_import("fastplotlib")

# PySide6 - optional Qt backend (forces Qt over glfw if available)
HAS_PYSIDE6: bool = _check_import("PySide6")
