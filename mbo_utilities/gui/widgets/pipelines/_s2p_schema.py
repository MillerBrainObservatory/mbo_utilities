"""
Schema lookup for upstream `suite2p.parameters.SETTINGS`.

Single source of truth for parameter metadata (default, type, min/max,
description, gui_name). The mbo dataclass `Suite2pSettings` flattens the
upstream nested dict and renames a handful of fields for ergonomics; this
module maps each mbo field name back to its upstream schema entry so we
can:
  - reset a value to the upstream default
  - format a tooltip from the upstream description
  - tell whether the current value matches the upstream default
  - color-code modified fields in the GUI

If suite2p adds or renames a parameter, this module is the only place
that needs to change. When the parameter exists upstream but is renamed
in mbo, the mapping handles that translation. When the parameter is
mbo-only (no upstream equivalent) it simply isn't in the map.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

# suite2p loads out-of-process: a daemon at module import either reads a
# disk cache (~/.mbo/cache/s2p_settings_<version>.json) or spawns a one-
# shot subprocess that imports suite2p and dumps SETTINGS/DB to that file.
# This avoids importing suite2p (and its torch/numba/scipy/sklearn chain)
# in the GUI interpreter, which previously froze the imgui loop 1-3 s on
# the first Run-tab click and even longer when attempted from a thread.
# Cache key includes the installed suite2p version so upgrades invalidate.
_SETTINGS: dict | None = None
_DB: dict | None = None
_LOAD_LOCK = threading.Lock()
_LOAD_EVENT = threading.Event()
_LOAD_THREAD: threading.Thread | None = None
# bounded wait when callers hit _ensure_loaded before the daemon finishes.
# longer than typical subprocess cold-start (~3 s) but short enough that a
# truly broken environment surfaces as "no schema" instead of hanging UI.
_LOAD_TIMEOUT_S = 10.0


def _cache_dir() -> Path:
    override = os.environ.get("MBO_CACHE_DIR")
    return Path(override) if override else Path.home() / ".mbo" / "cache"


def _suite2p_version() -> str:
    try:
        return importlib.metadata.version("suite2p")
    except Exception:
        return "unknown"


def _cache_path() -> Path:
    return _cache_dir() / f"s2p_settings_{_suite2p_version()}.json"


def _load_cache_file(path: Path) -> tuple[dict, dict] | None:
    try:
        with path.open() as f:
            data = json.load(f)
        s = data.get("SETTINGS")
        d = data.get("DB")
        if isinstance(s, dict) and isinstance(d, dict):
            return s, d
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    return None


# Source for the subprocess. Encodes type objects as their __name__
# (consumers only need a display string via _format_type) and falls
# back to repr() for any other non-JSON-serializable value.
_EXTRACTOR_SRC = r"""
import json, sys
def encode(o):
    if isinstance(o, type):
        return o.__name__
    if isinstance(o, dict):
        return {k: encode(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [encode(v) for v in o]
    try:
        json.dumps(o)
        return o
    except TypeError:
        return repr(o)
try:
    from suite2p import parameters as p
    json.dump({"SETTINGS": encode(p.SETTINGS), "DB": encode(p.DB)}, sys.stdout)
except Exception as e:
    sys.stderr.write(repr(e))
    sys.exit(2)
"""


def _build_cache_via_subprocess() -> tuple[dict, dict] | None:
    """Spawn a child that imports suite2p and dumps SETTINGS+DB as JSON.

    Writes the result to the on-disk cache before returning so the next
    process launch hits the fast path. Returns ``None`` on any failure
    (suite2p not installed, subprocess crash, timeout) — callers degrade
    to ``_SETTINGS is None`` which downstream code already handles.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _EXTRACTOR_SRC],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    settings = data.get("SETTINGS")
    db = data.get("DB")
    if not isinstance(settings, dict) or not isinstance(db, dict):
        return None
    cache_path = _cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump({"SETTINGS": settings, "DB": db}, f)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return settings, db


def _load_or_build() -> None:
    """Daemon body: populate _SETTINGS/_DB from cache, else from subprocess."""
    global _SETTINGS, _DB
    try:
        cached = _load_cache_file(_cache_path())
        if cached is None:
            cached = _build_cache_via_subprocess()
        if cached is not None:
            with _LOAD_LOCK:
                _SETTINGS, _DB = cached
    finally:
        _LOAD_EVENT.set()


def _start_loader() -> None:
    global _LOAD_THREAD
    if _LOAD_THREAD is not None:
        return
    with _LOAD_LOCK:
        if _LOAD_THREAD is not None:
            return
        t = threading.Thread(
            target=_load_or_build, daemon=True, name="s2p-schema-loader",
        )
        _LOAD_THREAD = t
    t.start()


# Kick off loader on module import. Daemon thread + subprocess means the
# main interpreter never imports suite2p; the GUI paints normally while
# the child process spins up.
_start_loader()


def _ensure_loaded() -> None:
    """Block (bounded) until the loader daemon finishes."""
    if _SETTINGS is not None:
        return
    if _LOAD_THREAD is None:
        _start_loader()
    _LOAD_EVENT.wait(timeout=_LOAD_TIMEOUT_S)


def warm_up_suite2p_schema() -> None:
    """Block (bounded) until the schema is loaded.

    Public alias for the load contract. Used by file-load paths that
    need ``is_default`` / ``get_default`` to return real values rather
    than the pre-load "everything is default" fallback (see ``is_default``).
    """
    _ensure_loaded()


def _resolve(root: dict, path: tuple[str, ...]) -> dict | None:
    """Walk path through a parameter root (SETTINGS or DB), returning the spec dict."""
    cur: Any = root
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, dict) and "default" in cur else None


def _to_py(v: Any) -> Any:
    """Convert numpy scalars / arrays to Python equivalents.

    Values loaded from .npy come back as numpy types: 0-d arrays for
    scalars (e.g. tau as np.float64), N-d arrays for sequences (e.g.
    diameter as np.array([12., 12.])). The rest of this module — and
    every downstream consumer — assumes Python types: `value == default`,
    `value in (None, "", [])`, `bool(value)`, etc. all break on
    multi-element numpy arrays with `truth value of an array with more
    than one element is ambiguous`. Normalising at the boundary here
    keeps the rest of the module simple and bug-free.

    No-op for already-Python values; cheap to call defensively.
    """
    # numpy scalar wrapped as a 0-d array
    if hasattr(v, "shape") and v.shape == () and hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    # multi-element numpy array → tuple of Python scalars
    if type(v).__name__ == "ndarray":
        try:
            return tuple(x.item() if hasattr(x, "item") else x for x in v.tolist())
        except Exception:
            return v
    # numpy scalar that isn't a 0-d array (e.g. np.float64) — has .item()
    if hasattr(v, "item") and v.__class__.__module__ == "numpy":
        try:
            return v.item()
        except Exception:
            return v
    return v


# (mbo_field) -> (upstream path, optional list/tuple index)
# - upstream path is a tuple of keys into SETTINGS (or DB) ending at a
#   parameter spec dict (the one with "default", "type", etc.)
# - index is set when the upstream spec is a list/tuple value (diameter,
#   block_size) and the mbo field stores a single component.
_MBO_TO_S2P: dict[str, tuple[tuple[str, ...], int | None]] = {
    # top-level
    "torch_device": (("torch_device",), None),
    "tau": (("tau",), None),
    "fs": (("fs",), None),
    "diameter_y": (("diameter",), 0),
    "diameter_x": (("diameter",), 1),
    # run
    "do_registration": (("run", "do_registration"), None),
    "do_regmetrics": (("run", "do_regmetrics"), None),
    "do_detection": (("run", "do_detection"), None),
    "do_deconvolution": (("run", "do_deconvolution"), None),
    "multiplane_parallel": (("run", "multiplane_parallel"), None),
    # io
    "combined": (("io", "combined"), None),
    "save_mat": (("io", "save_mat"), None),
    "save_NWB": (("io", "save_NWB"), None),
    "save_ops_orig": (("io", "save_ops_orig"), None),
    "delete_bin": (("io", "delete_bin"), None),
    "move_bin": (("io", "move_bin"), None),
    # registration (renames: reg_batch_size, block_size_y/x, align_by_chan)
    "align_by_chan": (("registration", "align_by_chan2"), None),
    "nimg_init": (("registration", "nimg_init"), None),
    "maxregshift": (("registration", "maxregshift"), None),
    "do_bidiphase": (("registration", "do_bidiphase"), None),
    "bidiphase": (("registration", "bidiphase"), None),
    "reg_batch_size": (("registration", "batch_size"), None),
    "nonrigid": (("registration", "nonrigid"), None),
    "maxregshiftNR": (("registration", "maxregshiftNR"), None),
    "block_size_y": (("registration", "block_size"), 0),
    "block_size_x": (("registration", "block_size"), 1),
    "smooth_sigma_time": (("registration", "smooth_sigma_time"), None),
    "smooth_sigma": (("registration", "smooth_sigma"), None),
    "spatial_taper": (("registration", "spatial_taper"), None),
    "th_badframes": (("registration", "th_badframes"), None),
    "norm_frames": (("registration", "norm_frames"), None),
    "snr_thresh": (("registration", "snr_thresh"), None),
    "subpixel": (("registration", "subpixel"), None),
    "two_step_registration": (("registration", "two_step_registration"), None),
    "reg_tif": (("registration", "reg_tif"), None),
    "reg_tif_chan2": (("registration", "reg_tif_chan2"), None),
    # detection (renames: det_block_size_y/x)
    "algorithm": (("detection", "algorithm"), None),
    "denoise": (("detection", "denoise"), None),
    "det_block_size_y": (("detection", "block_size"), 0),
    "det_block_size_x": (("detection", "block_size"), 1),
    "nbins": (("detection", "nbins"), None),
    "bin_size": (("detection", "bin_size"), None),
    "highpass_time": (("detection", "highpass_time"), None),
    "threshold_scaling": (("detection", "threshold_scaling"), None),
    "npix_norm_min": (("detection", "npix_norm_min"), None),
    "npix_norm_max": (("detection", "npix_norm_max"), None),
    "max_overlap": (("detection", "max_overlap"), None),
    "soma_crop": (("detection", "soma_crop"), None),
    "chan2_threshold": (("detection", "chan2_threshold"), None),
    "cellpose_chan2": (("detection", "cellpose_chan2"), None),
    # detection.sparsery_settings
    "highpass_neuropil": (("detection", "sparsery_settings", "highpass_neuropil"), None),
    "max_ROIs": (("detection", "sparsery_settings", "max_ROIs"), None),
    "spatial_scale": (("detection", "sparsery_settings", "spatial_scale"), None),
    "active_percentile": (("detection", "sparsery_settings", "active_percentile"), None),
    # detection.sourcery_settings
    "connected": (("detection", "sourcery_settings", "connected"), None),
    "max_iterations": (("detection", "sourcery_settings", "max_iterations"), None),
    "smooth_masks": (("detection", "sourcery_settings", "smooth_masks"), None),
    # detection.cellpose_settings
    "cellpose_model": (("detection", "cellpose_settings", "cellpose_model"), None),
    "cellpose_img": (("detection", "cellpose_settings", "img"), None),
    "highpass_spatial": (("detection", "cellpose_settings", "highpass_spatial"), None),
    "flow_threshold": (("detection", "cellpose_settings", "flow_threshold"), None),
    "cellprob_threshold": (("detection", "cellpose_settings", "cellprob_threshold"), None),
    # classification
    "classifier_path": (("classification", "classifier_path"), None),
    "use_builtin_classifier": (("classification", "use_builtin_classifier"), None),
    "preclassify": (("classification", "preclassify"), None),
    # extraction (rename: extract_batch_size)
    "snr_threshold": (("extraction", "snr_threshold"), None),
    "extract_batch_size": (("extraction", "batch_size"), None),
    "neuropil_extract": (("extraction", "neuropil_extract"), None),
    "neuropil_coefficient": (("extraction", "neuropil_coefficient"), None),
    "inner_neuropil_radius": (("extraction", "inner_neuropil_radius"), None),
    "min_neuropil_pixels": (("extraction", "min_neuropil_pixels"), None),
    "lam_percentile": (("extraction", "lam_percentile"), None),
    "allow_overlap": (("extraction", "allow_overlap"), None),
    "circular_neuropil": (("extraction", "circular_neuropil"), None),
    # dcnv_preprocess
    "baseline": (("dcnv_preprocess", "baseline"), None),
    "win_baseline": (("dcnv_preprocess", "win_baseline"), None),
    "sig_baseline": (("dcnv_preprocess", "sig_baseline"), None),
    "prctile_baseline": (("dcnv_preprocess", "prctile_baseline"), None),
}


# fields in mbo's dataclass that have NO upstream equivalent. listed
# here so callers can quickly tell whether a field is "mbo-only" (in
# which case there's no schema, no default to compare against).
# Suite2pDB fields that ARE upstream-defined but live on `Suite2pDB`
# (not `Suite2pSettings`). Schema lookups check this dict in addition to
# `_MBO_TO_S2P` so the GUI can apply the same `_hi()` tracking and Reset
# button to db fields.
_MBO_DB_TO_S2P: dict[str, tuple[tuple[str, ...], int | None]] = {
    "keep_movie_raw": (("keep_movie_raw",), None),
}


MBO_ONLY_FIELDS: set[str] = {
    # cellpose_niter resolves to cellpose_settings.params.niter at to_dict
    # time, so it has no direct schema entry. Always rendered with the
    # mbo-only color tint to flag it as not-suite2p-canonical.
    "cellpose_niter",
    # NOTE: `anatomical_only` and `sparse_mode` were removed in favor of the
    # `algorithm` selector + the explicit `cellpose_img` string. Legacy ops
    # files using the old keys are remapped on load by _FLAT_TO_MBO.
    # NOTE: `spatial_hp_detect` was the legacy upstream key; it was renamed
    # to `highpass_neuropil` (detection.sparsery_settings.highpass_neuropil)
    # in suite2p's restructure. Old flat ops.npy still use the legacy name —
    # see _FLAT_TO_MBO below for the remap.
}


def _entry_for(mbo_field: str) -> tuple[dict, tuple, int | None] | None:
    """Resolve a mbo field to (root_dict, path, idx) — checks settings and db."""
    _ensure_loaded()
    if mbo_field in _MBO_TO_S2P:
        path, idx = _MBO_TO_S2P[mbo_field]
        return _SETTINGS, path, idx
    if mbo_field in _MBO_DB_TO_S2P:
        path, idx = _MBO_DB_TO_S2P[mbo_field]
        return _DB, path, idx
    return None


def get_param_info(mbo_field: str) -> dict | None:
    """Return the upstream parameter spec for a mbo field, or None if mbo-only."""
    entry = _entry_for(mbo_field)
    if entry is None:
        return None
    root, path, _ = entry
    return _resolve(root, path)


def get_default(mbo_field: str) -> Any:
    """Return upstream default for a mbo field, translating renamed/typed mbo fields.

    Returns None for mbo-only fields with no upstream equivalent.
    """
    entry = _entry_for(mbo_field)
    if entry is None:
        return None
    root, path, idx = entry
    spec = _resolve(root, path)
    if spec is None:
        return None
    default = spec.get("default")
    # list/tuple unpacking (diameter, block_size)
    if idx is not None and isinstance(default, (list, tuple)):
        if idx < len(default):
            return default[idx]
        return None
    # align_by_chan: upstream stores `align_by_chan2` as bool;
    # mbo stores `align_by_chan` as int 1/2.
    if mbo_field == "align_by_chan":
        return 2 if default else 1
    return default




def is_default(mbo_field: str, value: Any) -> bool:
    """Whether the given value matches upstream's default. False for mbo-only fields.

    Uses a float-tolerant comparison: imgui.input_float uses C++ `float`
    (32-bit) internally, so a Python double like 1.15 gets silently
    truncated to 1.149999976... after the first frame. A strict `==` check
    would mark every fractional default as "modified" the moment its widget
    rendered, even with no user edit. Tolerate a relative diff at float32
    precision (~7 decimal digits) to keep the modified-color signal honest.
    """
    if mbo_field in MBO_ONLY_FIELDS:
        return False  # no upstream default to compare against
    # Schema loads in a background daemon (subprocess on cold cache,
    # disk read on warm). While it's pending, report "matches default"
    # so the Run tab isn't a sea of orange for the few seconds it takes
    # on cold launch. Subsequent draws color-code accurately.
    if _SETTINGS is None:
        return True
    default = get_default(mbo_field)
    # normalize at the boundary — numpy types leak through ops.npy
    # round-trips and break `value in (...)` / `value == default` with
    # `truth value of an array with more than one element is ambiguous`.
    value = _to_py(value)
    default = _to_py(default)
    # treat any equality / membership exception as "not default" so a
    # surprising type combination doesn't crash the gui — the user
    # sees the field as modified, which is the safer bias (visually
    # flags it for review).
    try:
        if default is None and value in (None, "", [], ()):
            return True
    except Exception:
        pass
    try:
        if value == default:
            return True
    except Exception:
        pass
    # numeric comparison with float32 tolerance, but never compare bool to
    # number (Python's `True == 1` would otherwise sneak past).
    if (
        isinstance(value, (int, float))
        and isinstance(default, (int, float))
        and not isinstance(value, bool)
        and not isinstance(default, bool)
    ):
        import math
        try:
            return math.isclose(
                float(value), float(default), rel_tol=1e-6, abs_tol=1e-9
            )
        except Exception:
            return False
    return False


def _format_type(t: Any) -> str:
    if t is None:
        return "?"
    return getattr(t, "__name__", str(t))


def format_tooltip(mbo_field: str, extra: str = "") -> str:
    """Build a tooltip string from upstream parameter metadata.

    Layout:
        <description>

        Default: <default>
        Type:    <type>
        Range:   [<min>, <max>]      (omitted if both are None)
        <extra>                      (optional caller-provided notes)
    """
    info = get_param_info(mbo_field)
    if info is None:
        # mbo-only field — caller can still pass `extra` and it's all we have
        return extra
    parts: list[str] = []
    desc = info.get("description")
    if desc:
        parts.append(desc.strip())
    default = info.get("default")
    parts.append(f"Default: {default!r}")
    parts.append(f"Type:    {_format_type(info.get('type'))}")
    mn, mx = info.get("min"), info.get("max")
    if mn is not None or mx is not None:
        parts.append(f"Range:   [{mn}, {mx}]")
    if extra:
        parts.append("")
        parts.append(extra.strip())
    return "\n".join(parts)


def all_mapped_fields() -> list[str]:
    """List every mbo field that has an upstream schema entry (settings + db)."""
    return sorted(set(_MBO_TO_S2P.keys()) | set(_MBO_DB_TO_S2P.keys()))


# =============================================================================
# Loading parameters from ops.npy / settings.npy
# =============================================================================
#
# Two on-disk formats coexist:
#   1. flat ops.npy   — pre-v1 suite2p (and current LBM-Suite2p-Python output).
#                       all keys live at the top level; many use legacy spellings
#                       (`nbinned`, `roidetect`, `chan2_thres`, `neucoeff`, …).
#   2. structured     — suite2p v1.0.0+ settings.npy (the shape produced by
#                       `Suite2pSettings.to_dict()`); nested under run/io/
#                       registration/detection/classification/extraction/
#                       dcnv_preprocess.
#
# `from_ops(d)` auto-detects which format `d` is and returns
# `{mbo_field: value}` for every mbo field it could resolve. Mbo-only
# fields (anatomical_only, sparse_mode, cellpose_niter) are loaded when
# present. Legacy upstream keys (e.g. spatial_hp_detect → highpass_neuropil)
# are remapped to their modern names by _FLAT_TO_MBO.


# Flat-key → mbo-field map. Includes legacy spellings (left-side aliases).
# When the value is a 2-tuple of strings the source value is a list/tuple
# that gets unpacked into both fields. When it's a (str, callable) tuple
# the callable transforms the value before assignment.
_FLAT_TO_MBO: dict[str, Any] = {
    # top-level
    "torch_device": "torch_device",
    "tau": "tau",
    "fs": "fs",
    "diameter": ("diameter_y", "diameter_x"),  # int → both, [y,x] → unpack
    # run
    "do_registration": "do_registration",
    "do_regmetrics": "do_regmetrics",
    "do_detection": "do_detection",
    "roidetect": ("do_detection", lambda v: 1 if v else 0),  # legacy bool
    "do_deconvolution": "do_deconvolution",
    "spikedetect": "do_deconvolution",  # legacy bool
    "multiplane_parallel": "multiplane_parallel",
    # io
    "combined": "combined",
    "save_mat": "save_mat",
    "save_NWB": "save_NWB",
    "save_ops_orig": "save_ops_orig",
    "delete_bin": "delete_bin",
    "move_bin": "move_bin",
    # registration
    "align_by_chan": "align_by_chan",
    "align_by_chan2": ("align_by_chan", lambda v: 2 if v else 1),
    "nimg_init": "nimg_init",
    "maxregshift": "maxregshift",
    "do_bidiphase": "do_bidiphase",
    "bidiphase": "bidiphase",
    # registration batch size — flat ops "batch_size" is the legacy name.
    # extraction batch size now has its own distinct flat key
    # ("extract_batch_size") per lsp's _SECTION_FLAT_RENAMES — historically
    # they conflated to the same flat slot, so older ops files (pre-fix)
    # only set "batch_size" and we silently routed to reg.
    "batch_size": "reg_batch_size",
    "extract_batch_size": "extract_batch_size",
    "nonrigid": "nonrigid",
    "maxregshiftNR": "maxregshiftNR",
    # registration block_size keeps the legacy flat "block_size" name.
    # detection block_size now lives at "det_block_size" per lsp's
    # _SECTION_FLAT_RENAMES (was silently dropped from flat ops before
    # the fix because both sections wrote to the same slot).
    "block_size": ("block_size_y", "block_size_x"),
    "det_block_size": ("det_block_size_y", "det_block_size_x"),
    "smooth_sigma_time": "smooth_sigma_time",
    "smooth_sigma": "smooth_sigma",
    "spatial_taper": "spatial_taper",
    "th_badframes": "th_badframes",
    "norm_frames": "norm_frames",
    "snr_thresh": "snr_thresh",
    "subpixel": "subpixel",
    "two_step_registration": "two_step_registration",
    "reg_tif": "reg_tif",
    "reg_tif_chan2": "reg_tif_chan2",
    # db (also lives at top-level of ops/db.npy — Suite2pDB field, not Settings)
    "keep_movie_raw": "keep_movie_raw",
    # detection
    "algorithm": "algorithm",
    "denoise": "denoise",
    "nbins": "nbins",
    "nbinned": "nbins",  # legacy
    "bin_size": "bin_size",
    "highpass_time": "highpass_time",
    "high_pass": "highpass_time",  # legacy
    "threshold_scaling": "threshold_scaling",
    "npix_norm_min": "npix_norm_min",
    "npix_norm_max": "npix_norm_max",
    "max_overlap": "max_overlap",
    "soma_crop": "soma_crop",
    "chan2_threshold": "chan2_threshold",
    "chan2_thres": "chan2_threshold",  # legacy
    "cellpose_chan2": "cellpose_chan2",
    # legacy upstream key: spatial_hp_detect was renamed to highpass_neuropil
    # in the suite2p restructure. Map old flat ops to the new field.
    "spatial_hp_detect": "highpass_neuropil",
    # NOTE: legacy `anatomical_only` (int 0-4) and `sparse_mode` (bool) are
    # NOT in this dict — the joint decision (`algorithm` + `cellpose_img`)
    # lives in `from_flat` itself so the priority rule "anatomical_only > 0
    # forces cellpose, regardless of sparse_mode" is preserved.
    # detection.sparsery_settings
    "highpass_neuropil": "highpass_neuropil",
    "max_ROIs": "max_ROIs",
    "spatial_scale": "spatial_scale",
    "active_percentile": "active_percentile",
    # detection.sourcery_settings
    "connected": "connected",
    "max_iterations": "max_iterations",
    "smooth_masks": "smooth_masks",
    # detection.cellpose_settings
    "cellpose_model": "cellpose_model",
    "pretrained_model": "cellpose_model",  # legacy
    "highpass_spatial": "highpass_spatial",
    "spatial_hp_cp": "highpass_spatial",  # legacy
    "flow_threshold": "flow_threshold",
    "cellprob_threshold": "cellprob_threshold",
    # classification
    "classifier_path": "classifier_path",
    "use_builtin_classifier": "use_builtin_classifier",
    "preclassify": "preclassify",
    # extraction
    "snr_threshold": "snr_threshold",
    "neuropil_extract": "neuropil_extract",
    "neuropil_coefficient": "neuropil_coefficient",
    "neucoeff": "neuropil_coefficient",  # legacy
    "inner_neuropil_radius": "inner_neuropil_radius",
    "min_neuropil_pixels": "min_neuropil_pixels",
    "lam_percentile": "lam_percentile",
    "allow_overlap": "allow_overlap",
    "circular_neuropil": "circular_neuropil",
    # dcnv_preprocess
    "baseline": "baseline",
    "win_baseline": "win_baseline",
    "sig_baseline": "sig_baseline",
    "prctile_baseline": "prctile_baseline",
    # mbo-only post-processing knobs (MboSuite2pExtras). lsp persists
    # these as top-level ops keys after the post-processing block runs,
    # outside the suite2p settings schema (so settings.npy stays a
    # record of the suite2p stages only). Listed here so flat-ops
    # loaders pick them up; the hydrator routes them to s2p_extras.
    "dff_window_size": "dff_window_size",
    "dff_percentile": "dff_percentile",
    "dff_smooth_window": "dff_smooth_window",
    "correct_neuropil": "correct_neuropil",
    "accept_all_cells": "accept_all_cells",
    "save_json": "save_json",
    # cell_filters / rastermap_kwargs are persisted by lsp as nested
    # structures (list[dict] / nested dict) rather than scalars; the
    # decompose-back-to-GUI-state step is handled separately so they're
    # intentionally not in this scalar map.
}


_STRUCTURED_TOP_KEYS = {
    "run", "io", "registration", "detection",
    "classification", "extraction", "dcnv_preprocess",
}


def _is_structured(d: dict) -> bool:
    """Heuristic: structured settings have at least one nested-group key at top."""
    return any(k in d and isinstance(d[k], dict) for k in _STRUCTURED_TOP_KEYS)


def from_structured(settings: dict) -> dict[str, Any]:
    """Read a v1.0.0+ structured settings dict into {mbo_field: value}.

    Walks `_MBO_TO_S2P`; for every path that resolves in the input dict,
    yields the corresponding mbo field. Also pulls cellpose_settings.params.niter
    into `cellpose_niter`. Mbo-only fields with no upstream path are skipped.
    """
    out: dict[str, Any] = {}
    for mbo_field, (path, idx) in _MBO_TO_S2P.items():
        cur: Any = settings
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if not ok:
            continue
        # normalize numpy types at the boundary so downstream code only
        # ever sees Python scalars / tuples (`is_default`, type-coercion
        # in the gui hydrator, etc. all break on numpy multi-element
        # arrays — `truth value ambiguous`).
        cur = _to_py(cur)
        if idx is not None:
            # accept any sequence here (numpy arrays now arrive as
            # tuples after _to_py, but also handle list/tuple defensively).
            if isinstance(cur, (list, tuple)) and idx < len(cur):
                out[mbo_field] = _to_py(cur[idx])
            continue
        if mbo_field == "align_by_chan":
            out[mbo_field] = 2 if cur else 1
            continue
        out[mbo_field] = cur

    # cellpose_settings.params is a dict with optional `niter`; mbo stores
    # it as a top-level int (0 = "let cellpose decide").
    cp = settings.get("detection", {}).get("cellpose_settings", {}) if isinstance(settings.get("detection"), dict) else {}
    params = cp.get("params") if isinstance(cp, dict) else None
    if isinstance(params, dict) and "niter" in params:
        out["cellpose_niter"] = int(_to_py(params["niter"]))
    return out


def from_flat(ops: dict) -> dict[str, Any]:
    """Read a flat ops.npy-style dict into {mbo_field: value}.

    Handles legacy aliases (`roidetect`, `nbinned`, `chan2_thres`, …),
    list/single-int unpacks for `diameter` and `block_size`, and the
    joint legacy `anatomical_only` + `sparse_mode` → `algorithm` +
    `cellpose_img` translation (see post-pass below).
    """
    out: dict[str, Any] = {}
    for flat_key, target in _FLAT_TO_MBO.items():
        if flat_key not in ops:
            continue
        # normalize numpy at the boundary — diameter, block_size,
        # det_block_size land here as np.ndarray when ops.npy was
        # produced via numpy. _to_py turns 0-d arrays into scalars and
        # multi-element arrays into tuples so the unpack and downstream
        # comparisons work correctly.
        v = _to_py(ops[flat_key])
        # (mbo_field, callable) transform
        if isinstance(target, tuple) and len(target) == 2 and callable(target[1]):
            out[target[0]] = target[1](v)
            continue
        # (y_field, x_field) list unpack — covers diameter, block_size,
        # det_block_size. v is now guaranteed to be a Python tuple/list
        # if it was a numpy array, so the isinstance check works.
        if isinstance(target, tuple):
            yf, xf = target
            if isinstance(v, (list, tuple)):
                if len(v) >= 1:
                    out[yf] = _to_py(v[0])
                if len(v) >= 2:
                    out[xf] = _to_py(v[1])
                else:
                    out[xf] = _to_py(v[0])
            else:
                out[yf] = v
                out[xf] = v
            continue
        # plain string target
        out[target] = v

    # Legacy algorithm derivation: anatomical_only > 0 forces cellpose
    # (regardless of sparse_mode); otherwise sparse_mode picks
    # sparsery/sourcery. Mirrors the historical mbo `_derived_algorithm`
    # logic so old ops files map onto the new (`algorithm` + `cellpose_img`)
    # interface without ambiguity.
    if "anatomical_only" in ops or "sparse_mode" in ops:
        ana = ops.get("anatomical_only", 0) or 0
        try:
            ana_i = int(ana)
        except (TypeError, ValueError):
            ana_i = 0
        if ana_i > 0:
            out["algorithm"] = "cellpose"
            out["cellpose_img"] = {
                1: "max_proj / meanImg",
                2: "meanImg",
                3: "max_proj",  # upstream removed enhanced_mean_img
                4: "max_proj",
            }.get(ana_i, "max_proj / meanImg")
        elif "sparse_mode" in ops:
            out["algorithm"] = "sparsery" if ops["sparse_mode"] else "sourcery"

    # lsp post-processing: cell_filters list-of-dicts → GUI's per-criterion
    # (enabled, value) pairs. lsp run_lsp persists this list to ops.npy
    # with one entry per active filter; mirrors build_cell_filters() in
    # mbo_utilities (settings.py) so a saved run round-trips back into
    # the same checkbox state.
    cell_filters = ops.get("cell_filters")
    if isinstance(cell_filters, (list, tuple)):
        for entry in cell_filters:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if name == "max_diameter":
                if "min_diameter_um" in entry and entry["min_diameter_um"]:
                    out["min_diameter_um_enabled"] = True
                    out["min_diameter_um"] = float(_to_py(entry["min_diameter_um"]))
                if "max_diameter_um" in entry and entry["max_diameter_um"]:
                    out["max_diameter_um_enabled"] = True
                    out["max_diameter_um"] = float(_to_py(entry["max_diameter_um"]))
            elif name == "negative_baseline":
                out["baseline_filter_enabled"] = True
                out["baseline_reject_negative_F0"] = True
            elif name == "min_baseline_abs":
                out["baseline_filter_enabled"] = True
                out["baseline_min_F0_abs_enabled"] = True
                if "min_F0_abs" in entry and entry["min_F0_abs"] is not None:
                    out["baseline_min_F0_abs"] = float(_to_py(entry["min_F0_abs"]))
            elif name == "min_baseline_rel":
                out["baseline_filter_enabled"] = True
                out["baseline_min_F0_rel_enabled"] = True
                if "min_F0_rel" in entry and entry["min_F0_rel"] is not None:
                    out["baseline_min_F0_rel"] = float(_to_py(entry["min_F0_rel"]))

    # lsp post-processing: rastermap_kwargs dict → GUI's planar/volumetric
    # toggles + per-mode overrides. Presence of "planar" / "volumetric"
    # sub-dicts is the per-mode enable signal (mirrors lsp's unified api).
    rm_kw = ops.get("rastermap_kwargs")
    if isinstance(rm_kw, dict):
        any_mode = False
        planar = rm_kw.get("planar")
        if isinstance(planar, dict):
            out["rastermap_planar"] = True
            any_mode = True
            if "n_clusters" in planar and planar["n_clusters"] is not None:
                out["rastermap_planar_n_clusters"] = int(_to_py(planar["n_clusters"]))
            if "n_PCs" in planar and planar["n_PCs"] is not None:
                out["rastermap_planar_n_pcs"] = int(_to_py(planar["n_PCs"]))
            if "locality" in planar and planar["locality"] is not None:
                out["rastermap_planar_locality"] = float(_to_py(planar["locality"]))
        volumetric = rm_kw.get("volumetric")
        if isinstance(volumetric, dict):
            out["rastermap_volumetric"] = True
            any_mode = True
            if "n_clusters" in volumetric and volumetric["n_clusters"] is not None:
                out["rastermap_volumetric_n_clusters"] = int(_to_py(volumetric["n_clusters"]))
            if "n_PCs" in volumetric and volumetric["n_PCs"] is not None:
                out["rastermap_volumetric_n_pcs"] = int(_to_py(volumetric["n_PCs"]))
        if any_mode:
            # Skip=0, Run=1, Force=2. We don't know force-vs-run from a
            # saved run, so default to Run when either mode is enabled.
            out["rastermap_mode"] = 1

    return out


def from_ops(d: dict) -> dict[str, Any]:
    """Auto-detect format and return {mbo_field: value} from an ops/settings dict."""
    return from_structured(d) if _is_structured(d) else from_flat(d)


def from_npy_file(path) -> dict[str, Any]:
    """Load a flat ops.npy or structured settings.npy and decode into mbo fields.

    Accepts a pickled-dict .npy (the standard suite2p output shape).
    """
    import numpy as np
    arr = np.load(str(path), allow_pickle=True)
    d = arr.item() if hasattr(arr, "item") and arr.ndim == 0 else arr
    if not isinstance(d, dict):
        raise ValueError(
            f"expected dict-like content in {path}, got {type(d).__name__}"
        )
    return from_ops(d)
