"""
cellpose utilities - moved to lbm_suite2p_python.

these functions are now available in lbm_suite2p_python.cellpose.
this module re-exports them for backwards compatibility.
"""

try:
    from lbm_suite2p_python.cellpose import (
        load_seg_file as load_results,
    )
    from lbm_suite2p_python.cellpose import (
        masks_to_stat,
        open_in_gui,
        save_comparison,
        stat_to_masks,
    )
    from lbm_suite2p_python.cellpose import (
        save_gui_results as save_results,
    )
    from lbm_suite2p_python.conversion import (
        export_for_gui,
        import_from_gui,
    )

    __all__ = [
        "export_for_gui",
        "import_from_gui",
        "load_results",
        "masks_to_stat",
        "open_in_gui",
        "save_comparison",
        "save_results",
        "stat_to_masks",
    ]
except ImportError:
    raise ImportError(
        "cellpose utilities require lbm_suite2p_python. "
        "install with: pip install lbm-suite2p-python"
    )
