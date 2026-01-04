"""
mbo_db - dataset management and discovery for MBO imaging pipelines.

provides tools to scan, index, and browse datasets produced by
mbo_utilities and lbm_suite2p_python pipelines.

usage:
    # cli
    mbo db scan /path/to/data      # scan and index directory
    mbo db                         # launch GUI browser
    mbo db list                    # list indexed datasets
    mbo db info <dataset>          # show dataset details

    # python api
    from mbo_utilities.db import scan_directory, get_datasets, launch_browser

    # scan a directory
    scan_directory("/path/to/data")

    # get all indexed datasets
    datasets = get_datasets()

    # launch the browser GUI
    launch_browser()
"""

__all__ = [
    # models
    "Dataset",
    "DatasetLink",
    "add_tag",
    "get_dataset",
    "get_datasets",
    # database operations
    "get_db_path",
    "init_db",
    # gui
    "launch_browser",
    "link_datasets",
    "remove_tag",
    "scan_directory",
]


def __getattr__(name):
    """Lazy import to avoid loading sqlite/gui until needed."""
    if name in ("get_db_path", "init_db", "scan_directory", "get_datasets",
                "get_dataset", "link_datasets", "add_tag", "remove_tag"):
        from . import database
        return getattr(database, name)

    if name in ("Dataset", "DatasetLink"):
        from . import models
        return getattr(models, name)

    if name == "launch_browser":
        from . import gui
        return gui.launch_browser

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
