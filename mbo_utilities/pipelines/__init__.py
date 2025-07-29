try:
    import masknmf
    from .masknmf import load_from_dir
    HAS_MASKNMF = True
except ImportError:
    masknmf = None
    load_from_dir = None
    HAS_MASKNMF = False

try:
    import torch

    MBO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HAS_TORCH = True
except ImportError:
    MBO_DEVICE = "cpu"
    HAS_TORCH = False

try:
    from suite2p.io import BinaryFile

    HAS_SUITE2P = True
except ImportError:
    suite2p = None
    HAS_SUITE2P = False

__all__ = [
    "HAS_MASKNMF",
    "HAS_SUITE2P",
    "HAS_TORCH",
    "MBO_DEVICE",
    "load_from_dir",
]
