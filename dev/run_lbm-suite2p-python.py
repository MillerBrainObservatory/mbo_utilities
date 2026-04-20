"""run lbm_suite2p_python on raw scanimage data, all zplanes,
with suite3d axial z-registration enabled via the new `register_z` kwarg.

the axial-registration wiring now lives inside lsp.pipeline() — the caller
just flips `register_z=True` and pipeline handles suite3d reuse / invocation
and ops wiring internally.
"""

from lbm_suite2p_python import pipeline


pipeline("D:/demo/raw", save_path="D:/demo/bugfixing", register_z=True)
