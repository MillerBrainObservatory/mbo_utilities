"""
cellpose utilities - moved to lbm_suite2p_python.

these functions are now available in lbm_suite2p_python.cellpose.
this module re-exports them for backwards compatibility.
"""

try:
    from lbm_suite2p_python.cellpose import (
        save_gui_results as save_results,
        load_seg_file as load_results,
        open_in_gui,
        masks_to_stat,
        stat_to_masks,
        save_comparison,
    )

    __all__ = [
        "save_results",
        "load_results",
        "open_in_gui",
        "masks_to_stat",
        "stat_to_masks",
        "save_comparison",
    ]
except ImportError:
    raise ImportError(
        "cellpose utilities require lbm_suite2p_python. "
        "install with: pip install lbm-suite2p-python"
    )
