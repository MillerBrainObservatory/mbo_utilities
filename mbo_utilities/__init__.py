from .lcp_io import get_metadata
# from .assembly import *

from . import _version
__version__ = _version.get_versions()['version']

__all__ = ['get_metadata']
