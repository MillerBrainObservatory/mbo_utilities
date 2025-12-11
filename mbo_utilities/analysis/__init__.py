"""
Analysis tools for mbo_utilities.
"""

from mbo_utilities.analysis.scanphase import (
    ScanPhaseAnalyzer,
    ScanPhaseResults,
    analyze_scanphase,
    run_scanphase_analysis,
)

__all__ = [
    "ScanPhaseAnalyzer",
    "ScanPhaseResults",
    "analyze_scanphase",
    "run_scanphase_analysis",
]

# cellpose functions are now in lbm_suite2p_python
# import with optional dependency handling
try:
    from lbm_suite2p_python.cellpose import (
        save_gui_results as save_cellpose_results,
        load_seg_file as load_cellpose_results,
        open_in_gui as open_cellpose_gui,
        masks_to_stat,
        stat_to_masks,
        save_comparison as save_cellpose_comparison,
    )
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
