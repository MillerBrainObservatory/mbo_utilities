from .file_io import (
    get_files,
    stack_from_files,
    read_scan,
    ScanMultiROIReordered,
)
from .metadata import is_raw_scanimage, get_metadata
from .gui.widgets import run_gui

from . import _version
__version__ = _version.get_versions()['version']

__all__ = [
    'get_files',
    'stack_from_files',
    'read_scan',
    'run_gui',
    'ScanMultiROIReordered',
]
