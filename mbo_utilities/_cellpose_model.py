"""Race-free provisioning of the cellpose model used by suite2p detection.

suite2p anatomical ROI detection loads the default "cpsam" checkpoint through
cellpose. When the model is not cached yet and the pipeline fans out to N
parallel plane workers, every worker downloads to the same path at once; the
overlapping writes corrupt the file and torch.load fails with a
PytorchStreamReader/miniz error. ensure_cellpose_model() downloads and
validates the file once in the parent process before fan-out.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path


def _log(logger: logging.Logger | None, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def _is_valid_checkpoint(path: Path) -> bool:
    """True if path is a readable torch zip checkpoint (full CRC check)."""
    try:
        if not path.exists() or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:
        return False


def ensure_cellpose_model(
    force: bool = False, logger: logging.Logger | None = None
) -> str | None:
    """Download and validate the cellpose "cpsam" model once, serially.

    Returns the model path, or None if cellpose is unavailable or the file
    could not be made valid. A ``cpsam.verified`` marker caches the
    validation so steady-state calls only stat two files.
    """
    try:
        from cellpose import models
    except Exception as e:
        _log(logger, f"cellpose not available, skipping model prefetch: {e}")
        return None

    path = Path(models.MODEL_DIR) / "cpsam"
    marker = path.with_name("cpsam.verified")

    # fast path: a validation marker newer than the model file means a prior
    # call already CRC-checked this exact file.
    if not force and path.exists() and marker.exists():
        try:
            if marker.stat().st_mtime >= path.stat().st_mtime:
                return str(path)
        except OSError:
            pass

    valid = _is_valid_checkpoint(path) if path.exists() else False

    if force or not valid:
        if path.exists():
            reason = "forced refresh" if force and valid else "corrupt"
            _log(logger, f"cellpose model {reason}; re-downloading cpsam")
            # the cache call only downloads when the file is absent, so remove
            # the bad/forced one first.
            try:
                path.unlink()
            except OSError:
                pass
        _log(logger, "Pre-fetching cellpose model (cpsam, ~1.15 GB)...")
        try:
            # cellpose 4.x renamed cache_CPSAM_model_path() -> cache_model_path(backbone)
            if hasattr(models, "cache_model_path"):
                models.cache_model_path("cpsam")
            else:
                models.cache_CPSAM_model_path()
        except Exception as e:
            _log(logger, f"cellpose model download failed: {e}")
            return None
        valid = _is_valid_checkpoint(path)

    if valid:
        try:
            marker.write_text("ok", encoding="utf-8")
        except OSError:
            pass
        return str(path)

    _log(logger, "cellpose model still invalid after download")
    return None
