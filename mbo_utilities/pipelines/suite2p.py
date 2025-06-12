try:
    import suite2p
    HAS_SUITE2P = True
except ImportError:
    suite2p = None
    HAS_SUITE2P = False