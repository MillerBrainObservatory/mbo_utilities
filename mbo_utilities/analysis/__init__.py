"""
Analysis tools for mbo_utilities.
"""

from mbo_utilities.analysis.phasecorr import (
    bidir_phasecorr,
    ALL_PHASECORR_METHODS,
    TWO_DIM_PHASECORR_METHODS,
    THREE_DIM_PHASECORR_METHODS,
)

from mbo_utilities.analysis.scanphase import (
    ScanPhaseAnalyzer,
    ScanPhaseResults,
    analyze_scanphase,
    run_scanphase_analysis,
)

from mbo_utilities.analysis.metrics import (
    snr_roi,
    mean_row_misalignment,
    temporal_corr,
    sharpness_metric,
    avg_sharpness,
    frame_correlations,
)

__all__ = [
    # phasecorr
    "bidir_phasecorr",
    "ALL_PHASECORR_METHODS",
    "TWO_DIM_PHASECORR_METHODS",
    "THREE_DIM_PHASECORR_METHODS",
    # scanphase
    "ScanPhaseAnalyzer",
    "ScanPhaseResults",
    "analyze_scanphase",
    "run_scanphase_analysis",
    # metrics
    "snr_roi",
    "mean_row_misalignment",
    "temporal_corr",
    "sharpness_metric",
    "avg_sharpness",
    "frame_correlations",
]


def _patch_qt_checkbox():
    """patch QCheckBox for Qt5/Qt6 compatibility with cellpose."""
    try:
        from qtpy.QtWidgets import QCheckBox
        if not hasattr(QCheckBox, 'checkStateChanged'):
            QCheckBox.checkStateChanged = QCheckBox.stateChanged
    except ImportError:
        pass


# cellpose functions are now in lbm_suite2p_python
# import with optional dependency handling
try:
    from lbm_suite2p_python.cellpose import (
        save_gui_results as save_cellpose_results,
        load_seg_file as load_cellpose_results,
        open_in_gui as _open_in_gui,
        masks_to_stat,
        stat_to_masks,
        save_comparison as save_cellpose_comparison,
    )

    def open_cellpose_gui(*args, **kwargs):
        """open cellpose GUI with Qt compatibility patch applied."""
        _patch_qt_checkbox()
        return _open_in_gui(*args, **kwargs)

    __all__.extend([
        "save_cellpose_results",
        "load_cellpose_results",
        "open_cellpose_gui",
        "masks_to_stat",
        "stat_to_masks",
        "save_cellpose_comparison",
    ])
except ImportError:
    pass  # lbm_suite2p_python not installed
