try:
    from .masknmf import load_from_dir
    HAS_MASKNMF = True
except ImportError:
    load_from_dir = None
    HAS_MASKNMF = False

try:
    from .suite2p import *
    from suite2p import BinaryFile
    HAS_SUITE2P = True
except ImportError:
    suite2p = None
    HAS_SUITE2P = False

__all__ = [
    "load_from_dir",
    "HAS_MASKNMF",
    "HAS_SUITE2P",
]