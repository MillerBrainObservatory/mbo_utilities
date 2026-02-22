"""
Analysis tools for mbo_utilities.

This package uses lazy imports to minimize startup time. Heavy dependencies
like lbm_suite2p_python are only loaded when actually needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mbo_utilities.analysis.phasecorr import (
        bidir_phasecorr as bidir_phasecorr,
        ALL_PHASECORR_METHODS as ALL_PHASECORR_METHODS,
        TWO_DIM_PHASECORR_METHODS as TWO_DIM_PHASECORR_METHODS,
        THREE_DIM_PHASECORR_METHODS as THREE_DIM_PHASECORR_METHODS,
    )
    from mbo_utilities.analysis.scanphase import (
        ScanPhaseAnalyzer as ScanPhaseAnalyzer,
        ScanPhaseResults as ScanPhaseResults,
        analyze_scanphase as analyze_scanphase,
        run_scanphase_analysis as run_scanphase_analysis,
    )
    from mbo_utilities.analysis.metrics import (
        snr_roi as snr_roi,
        mean_row_misalignment as mean_row_misalignment,
        temporal_corr as temporal_corr,
        sharpness_metric as sharpness_metric,
        avg_sharpness as avg_sharpness,
        frame_correlations as frame_correlations,
    )
    from mbo_utilities.analysis.fiber_activity import (
        run as run_fiber_activity,
        FiberActivityResult as FiberActivityResult,
    )

__all__ = [
    "ALL_PHASECORR_METHODS",
    "THREE_DIM_PHASECORR_METHODS",
    "TWO_DIM_PHASECORR_METHODS",
    # scanphase
    "ScanPhaseAnalyzer",
    "ScanPhaseResults",
    # helpers
    "_patch_qt_checkbox",
    "analyze_scanphase",
    "avg_sharpness",
    # phasecorr
    "bidir_phasecorr",
    "detect_format",
    "ensure_cellpose_format",
    "export_suite2p_for_cellpose",
    "frame_correlations",
    "get_results",
    "import_cellpose_to_suite2p",
    "load_cellpose_results",
    "masks_to_stat",
    "mean_row_misalignment",
    "open_cellpose_gui",
    "run_scanphase_analysis",
    "save_cellpose_comparison",
    # cellpose/lbm_suite2p_python (optional)
    "save_cellpose_results",
    "sharpness_metric",
    # fiber_activity
    "FiberActivityResult",
    "run_fiber_activity",
    # metrics
    "snr_roi",
    "stat_to_masks",
    "temporal_corr",
]

# Lazy import mapping: name -> (module, attr)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # phasecorr
    "bidir_phasecorr": (".phasecorr", "bidir_phasecorr"),
    "ALL_PHASECORR_METHODS": (".phasecorr", "ALL_PHASECORR_METHODS"),
    "TWO_DIM_PHASECORR_METHODS": (".phasecorr", "TWO_DIM_PHASECORR_METHODS"),
    "THREE_DIM_PHASECORR_METHODS": (".phasecorr", "THREE_DIM_PHASECORR_METHODS"),
    # scanphase
    "ScanPhaseAnalyzer": (".scanphase", "ScanPhaseAnalyzer"),
    "ScanPhaseResults": (".scanphase", "ScanPhaseResults"),
    "analyze_scanphase": (".scanphase", "analyze_scanphase"),
    "run_scanphase_analysis": (".scanphase", "run_scanphase_analysis"),
    # fiber_activity
    "run_fiber_activity": (".fiber_activity", "run"),
    "FiberActivityResult": (".fiber_activity", "FiberActivityResult"),
    # metrics
    "snr_roi": (".metrics", "snr_roi"),
    "mean_row_misalignment": (".metrics", "mean_row_misalignment"),
    "temporal_corr": (".metrics", "temporal_corr"),
    "sharpness_metric": (".metrics", "sharpness_metric"),
    "avg_sharpness": (".metrics", "avg_sharpness"),
    "frame_correlations": (".metrics", "frame_correlations"),
}

# lbm_suite2p_python imports (optional dependency)
_LSP_IMPORTS: dict[str, tuple[str, str]] = {
    "save_cellpose_results": ("lbm_suite2p_python.cellpose", "save_gui_results"),
    "load_cellpose_results": ("lbm_suite2p_python.cellpose", "load_seg_file"),
    "_open_in_gui": ("lbm_suite2p_python.cellpose", "open_in_gui"),
    "masks_to_stat": ("lbm_suite2p_python.cellpose", "masks_to_stat"),
    "stat_to_masks": ("lbm_suite2p_python.cellpose", "stat_to_masks"),
    "save_cellpose_comparison": ("lbm_suite2p_python.cellpose", "save_comparison"),
    "export_suite2p_for_cellpose": ("lbm_suite2p_python.conversion", "export_for_gui"),
    "import_cellpose_to_suite2p": ("lbm_suite2p_python.conversion", "import_from_gui"),
    "get_results": ("lbm_suite2p_python.conversion", "get_results"),
    "ensure_cellpose_format": ("lbm_suite2p_python.conversion", "ensure_cellpose_format"),
    "detect_format": ("lbm_suite2p_python.conversion", "detect_format"),
}

# Cache for loaded objects
_loaded: dict[str, object] = {}


def _patch_qt_checkbox():
    """Patch QCheckBox for Qt5/Qt6 compatibility with cellpose."""
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, "checkStateChanged"):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass


def open_cellpose_gui(*args, **kwargs):
    """Open cellpose GUI with Qt compatibility patch applied."""
    _patch_qt_checkbox()
    _open_in_gui = __getattr__("_open_in_gui")
    return _open_in_gui(*args, **kwargs)


def __getattr__(name: str) -> object:
    if name in _loaded:
        return _loaded[name]

    # Local submodule imports
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_name, package="mbo_utilities.analysis")
        obj = getattr(module, attr_name)
        _loaded[name] = obj
        return obj

    # lbm_suite2p_python imports (optional)
    if name in _LSP_IMPORTS:
        module_name, attr_name = _LSP_IMPORTS[name]
        try:
            from importlib import import_module
            module = import_module(module_name)
            obj = getattr(module, attr_name)
            _loaded[name] = obj
            return obj
        except ImportError:
            raise AttributeError(
                f"'{name}' requires lbm_suite2p_python which is not installed"
            ) from None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
