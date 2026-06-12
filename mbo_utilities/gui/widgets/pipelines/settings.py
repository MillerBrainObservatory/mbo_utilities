import json
import os
import pathlib
import threading
from pathlib import Path
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from imgui_bundle import (
    imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
    hello_imgui,
    icons_fontawesome_6 as fa,
)

from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    draw_boxed_label,
    set_tooltip,
    settings_row_with_popup,
    text_wrapped_cell,
    tooltip_marks_right,
    _popup_states,
)
from mbo_utilities.gui.widgets.pipelines._s2p_schema import is_default as _is_default


# light orange for parameters whose value differs from upstream suite2p
# default. picked to read as "modified" without competing with the title
# color (which is also orange-ish) — slightly redder, lower brightness.
_MODIFIED_COLOR = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)

# light green for mbo-only parameters that have no upstream suite2p
# equivalent (cellpose_niter, accept_all_cells, dff_*). Always rendered
# in this color regardless of value, to flag "this isn't a vanilla
# suite2p knob".
_MBO_ONLY_COLOR = imgui.ImVec4(0.55, 0.85, 0.55, 1.0)

# orangish-yellow — reserved for the MAIN suite2p section header
# ("Suite2p Main Settings"). Matches the column-header color so users
# can tell "this group is vanilla suite2p" at a glance.
_S2P_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)

# light blue — used for sub-section titles within any column (Cell
# Filters, dF/F, Rastermap, Sparsery, Sourcery, Cellpose, Shared,
# Classification, Denoise Block, Channel 2). Distinct from
# _S2P_TITLE_COLOR (main header) and _MBO_ONLY_COLOR (mbo-only group).
_SUBSECTION_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)

# parallel-processing load-status colors. safe = dimmed/default (no
# explicit color); yellow = fits in logical but exceeds physical cores;
# red = exceeds logical cores (~10x slowdown on SVD-heavy registration).
_LOAD_WARN_COLOR = imgui.ImVec4(1.00, 0.85, 0.30, 1.0)
_LOAD_BAD_COLOR = imgui.ImVec4(1.00, 0.40, 0.40, 1.0)

def _has_gpu_torch() -> bool:
    """True if the installed PyTorch is a GPU build (CUDA / ROCm / Intel XPU),
    or macOS (MPS).

    Reads ``torch/version.py`` — the build's ``cuda``/``hip``/``xpu`` fields,
    which are ``None`` on a CPU-only wheel — WITHOUT importing torch (an
    import is multi-second and would freeze the GUI thread). The version
    *string* alone is ambiguous (a plain '2.12.0' can be either build), so
    the accelerator fields are the reliable signal. ~0.3 ms; reflects what a
    fresh worker process will actually load. Only the definitive CPU-only
    case warns; anything indeterminate does not.
    """
    import sys

    if sys.platform == "darwin":
        return True  # MPS path on macOS
    try:
        import importlib.util
        import re
        from pathlib import Path

        spec = importlib.util.find_spec("torch")
        text = (Path(spec.origin).parent / "version.py").read_text()
    except Exception:
        return True  # can't determine — don't nag
    for field in ("cuda", "hip", "xpu"):
        m = re.search(rf"^{field}\b[^=\n]*=\s*(.+?)\s*$", text, re.M)
        if m and m.group(1).strip() not in ("None", ""):
            return True
    return False  # all accelerator fields None -> CPU-only build


def _detect_physical_cores() -> int:
    """Best-effort physical-core count. Falls back to os.cpu_count() // 2
    assuming HT/SMT is enabled (the common case on dev machines)."""
    try:
        import psutil  # type: ignore
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    cpu = os.cpu_count() or 4
    return max(1, cpu // 2)


def _default_workers() -> int:
    """Sensible default workers count. Aim for workers*2 (threads default)
    <= physical_cores so a fresh-install user lands in the green zone."""
    physical = _detect_physical_cores()
    return max(1, min(4, physical // 2))


def _parallel_load_status(workers: int, threads_per_worker: int) -> tuple:
    """Return (color_or_None, tooltip) for current workers/threads_per_worker.

    color is None when safe (caller renders the marker dimmed). Yellow when
    load fits in logical but not physical cores; red when it exceeds logical.
    """
    physical = _detect_physical_cores()
    logical = os.cpu_count() or physical
    eff_workers = workers if workers > 0 else max(1, physical // 2)
    eff_threads = threads_per_worker if threads_per_worker > 0 else 1
    load = eff_workers * eff_threads

    if load <= physical:
        return None, f"{load} threads on {physical} cores"
    if load <= logical:
        return _LOAD_WARN_COLOR, f"{load} threads > {physical} physical (borderline)"
    return _LOAD_BAD_COLOR, f"{load} threads > {logical} logical (oversubscribed)"


def _parallel_profiles() -> dict[str, tuple[int, int]]:
    """Profile name -> (workers, threads_per_worker), sized to this host.

    Parallel profiles favor plane parallelism (more workers, one thread
    each); Sequential runs one plane on every physical core; Max
    oversubscribes with logical cores. Drives the "Profile" selector,
    which fills the workers / threads fields in one click.
    """
    physical = _detect_physical_cores()
    logical = os.cpu_count() or physical
    return {
        "Sequential": (1, max(1, physical)),
        "Low": (max(1, physical // 4), 1),
        "Medium": (max(1, physical // 2), 1),
        "High": (max(1, physical), 1),
        "Max": (max(1, logical), 1),
    }


_PROFILE_HELP = (
    "workers = parallel plane processes\n"
    "threads/worker = BLAS/OMP threads within one plane\n"
    "load = workers x threads; green when load <= physical cores\n"
    "\n"
    "- Low/Medium/High: cores go to workers, threads=1\n"
    "  (maximize planes running at once)\n"
    "- Sequential: 1 worker x all cores (one plane at a time)\n"
    "- Max: workers = logical cores, so load > physical\n"
    "\n"
    "Best when planes >= workers. If planes < workers the extra\n"
    "workers sit idle - prefer fewer workers, more threads/worker."
)


# "Look at these first" — fields whose label is rendered in a bold font
# inside a thin rounded box in the pipeline-settings popup, so users know
# which knobs typically matter most. Add / remove freely; the emphasis is
# applied automatically by `_emp_label` inside `_draw_section_suite2p_content`
# whenever its `field` argument is in this set.
#
# The bold font (Roboto-Bold.ttf) is loaded in `gui/widgets/preview_data.py`
# and stashed on the parent widget as `self._bold_font`. If that's None
# (font file missing) the box is still drawn but with the regular font.
_IMPORTANT_FIELDS: set[str] = {
    "algorithm",
    "tau",
    "threshold_scaling",
    "two_step_registration",
    "cellprob_threshold",
    "flow_threshold",
    "cellpose_img",
    "highpass_time",
}


@contextmanager
def _hi(field: str, value):
    """Push _MODIFIED_COLOR for widget text iff `value` differs from upstream
    default for `field`. No-op for mbo-only fields (they have no default to
    compare against). Use as a context manager around any imgui widget call:

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("tau", self.s2p.tau):
            _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
        set_tooltip(...)

    Wrap only the widget — keep set_next_item_width above and set_tooltip
    below the `with` so the (?) marker stays at the default color.
    """
    pushed = not _is_default(field, value)
    if pushed:
        imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
    try:
        yield
    finally:
        if pushed:
            imgui.pop_style_color()


@contextmanager
def _mbo():
    """
    Push _MBO_ONLY_COLOR around a widget.
    """
    imgui.push_style_color(imgui.Col_.text, _MBO_ONLY_COLOR)
    try:
        yield
    finally:
        imgui.pop_style_color()


@contextmanager
def _ghost_button():
    """
    Ghost button: a faint white surface tint at rest with a slightly stronger overlay on hover/active.
    """
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(1, 1, 1, 0.05))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1, 1, 1, 0.10))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(1, 1, 1, 0.16))
    try:
        yield
    finally:
        imgui.pop_style_color(3)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities._writers import _convert_paths_to_strings

# lazy availability check - avoid heavy import at module load
_HAS_LSP: bool | None = None


def _check_lsp_available() -> bool:
    """check if lbm_suite2p_python is available.

    Caches a positive result for the session (find_spec is cheap but a
    True answer can't change once we've seen it), but ALWAYS re-probes
    after a negative or unknown answer — that way `pip install
    lbm_suite2p_python` mid-session is picked up on the next click
    instead of leaving the user stuck behind a stale False cache.
    """
    global _HAS_LSP
    if _HAS_LSP is True:
        return True
    import importlib.util
    found = importlib.util.find_spec("lbm_suite2p_python") is not None
    if found:
        _HAS_LSP = True
    return found


# Rastermap install probe — cheap synchronous find_spec, cached like
# _check_lsp_available. find_spec does not import rastermap, so it's safe
# to call every frame. A positive result sticks for the session; a
# negative answer re-probes so a mid-session `pip install rastermap` is
# picked up on the next frame.
_HAS_RASTERMAP: bool | None = None


def _check_rastermap_available() -> bool:
    """True if `rastermap` is importable. Caches a positive result."""
    global _HAS_RASTERMAP
    if _HAS_RASTERMAP is True:
        return True
    import importlib.util
    found = importlib.util.find_spec("rastermap") is not None
    if found:
        _HAS_RASTERMAP = True
    return found


USER_PIPELINES = ["suite2p"]


def build_cell_filters(
    min_um_enabled: bool,
    min_um: float,
    max_um_enabled: bool,
    max_um: float,
    *,
    baseline_filter_enabled: bool = False,
    baseline_reject_negative_F0: bool = False,
    baseline_min_F0_abs_enabled: bool = False,
    baseline_min_F0_abs: float = 0.0,
    baseline_min_F0_rel_enabled: bool = False,
    baseline_min_F0_rel: float = 0.0,
    correct_neuropil: bool = True,
) -> list[dict]:
    """Compose lsp's `cell_filters` list from the GUI's paired checkbox+value fields.

    Emits one entry per active criterion. lsp runs filters in apply order
    (each one writes its own 14<letter>_filter_<name>.png) so the order
    here is the order the user sees in the output dir.

    Possible entries:
      - `{"name": "max_diameter", ...}` — diameter bounds.
      - `{"name": "negative_baseline", "correct_neuropil": ...}` — drop ROIs
        whose rolling F0 dips below zero anywhere.
      - `{"name": "min_baseline_abs", "correct_neuropil": ..., "min_F0_abs":
        ...}` — drop ROIs whose median rolling F0 is below an absolute
        photon-count floor.
      - `{"name": "min_baseline_rel", "correct_neuropil": ..., "min_F0_rel":
        ...}` — drop ROIs whose minimum rolling F0 is below this fraction
        of median(F_raw).

    The dff window/percentile that each baseline filter scores against is
    resolved at the lsp layer from ops/pipeline kwargs (same values feeding
    the dF/F plot), so we don't thread those per-filter here.
    """
    out: list[dict] = []

    diameter: dict = {}
    if min_um_enabled and min_um > 0:
        diameter["min_diameter_um"] = float(min_um)
    if max_um_enabled and max_um > 0:
        diameter["max_diameter_um"] = float(max_um)
    if diameter:
        diameter["name"] = "max_diameter"
        out.append(diameter)

    if baseline_filter_enabled:
        cn = bool(correct_neuropil)
        if baseline_reject_negative_F0:
            out.append({"name": "negative_baseline", "correct_neuropil": cn})
        if baseline_min_F0_abs_enabled and baseline_min_F0_abs > 0:
            out.append({
                "name": "min_baseline_abs",
                "correct_neuropil": cn,
                "min_F0_abs": float(baseline_min_F0_abs),
            })
        if baseline_min_F0_rel_enabled and baseline_min_F0_rel > 0:
            out.append({
                "name": "min_baseline_rel",
                "correct_neuropil": cn,
                "min_F0_rel": float(baseline_min_F0_rel),
            })

    return out


def _build_planar_sub(extras) -> dict:
    """Compose the planar sub-dict for lsp's rastermap_kwargs.

    Sentinel values (0 for ints, -1 for locality) signal "omit — use
    lsp's cell-count-aware default for this field". An empty dict is
    valid and means "use defaults for ALL params" — still on.
    """
    sub: dict = {}
    if extras.rastermap_planar_n_clusters > 0:
        sub["n_clusters"] = int(extras.rastermap_planar_n_clusters)
    if extras.rastermap_planar_n_pcs > 0:
        sub["n_PCs"] = int(extras.rastermap_planar_n_pcs)
    if extras.rastermap_planar_locality >= 0:
        sub["locality"] = float(extras.rastermap_planar_locality)
    return sub


def _build_volumetric_sub(extras) -> dict:
    """Compose the volumetric sub-dict for lsp's rastermap_kwargs."""
    sub: dict = {}
    if extras.rastermap_volumetric_n_clusters > 0:
        sub["n_clusters"] = int(extras.rastermap_volumetric_n_clusters)
    if extras.rastermap_volumetric_n_pcs > 0:
        sub["n_PCs"] = int(extras.rastermap_volumetric_n_pcs)
    return sub


def build_rastermap_kwargs(extras) -> dict | None:
    """Compose lsp's unified rastermap_kwargs for `pipeline()`.

    Returns:
      None — when rastermap_mode is Skip, OR when neither sub-mode is
             enabled (lsp treats None as "both off").
      {"planar": {...}}                   — planar only.
      {"volumetric": {...}}               — volumetric only.
      {"planar": {...}, "volumetric": ...} — both.
    """
    if extras.rastermap_mode == 0:
        return None
    out: dict = {}
    if extras.rastermap_planar:
        out["planar"] = _build_planar_sub(extras)
    if extras.rastermap_volumetric:
        out["volumetric"] = _build_volumetric_sub(extras)
    return out or None


def build_planar_rastermap_kwargs(extras) -> dict | None:
    """Compose the planar-only kwargs for `run_plane()`.

    `run_plane` doesn't have a volumetric mode, so we hand it just the
    planar sub-dict (or None to disable). Empty dict = "on, use defaults".
    """
    if extras.rastermap_mode == 0 or not extras.rastermap_planar:
        return None
    return _build_planar_sub(extras)


def collect_modified_params(
    s2p, s2p_db, s2p_extras=None
) -> list[tuple[str, object, object, str]]:
    """Return modified parameters across both pipelines.

    Each tuple is `(field, current, default, source)` where `source` is
    `"s2p"` for upstream-mapped fields (Suite2pSettings / Suite2pDB) and
    `"lsp"` for MboSuite2pExtras fields (lbm_suite2p_python kwargs). s2p
    defaults come from `suite2p.parameters.SETTINGS` via the schema map;
    lsp defaults come from the dataclass field spec.

    Sorted: s2p group first (alphabetical), then lsp group (alphabetical).
    """
    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
        _MBO_TO_S2P,
        _MBO_DB_TO_S2P,
        get_default,
        is_default,
    )

    s2p_rows: list[tuple[str, object, object, str]] = []
    for field in _MBO_TO_S2P:
        if not hasattr(s2p, field):
            continue
        cur = getattr(s2p, field)
        if not is_default(field, cur):
            s2p_rows.append((field, cur, get_default(field), "s2p"))
    for field in _MBO_DB_TO_S2P:
        if not hasattr(s2p_db, field):
            continue
        cur = getattr(s2p_db, field)
        if not is_default(field, cur):
            s2p_rows.append((field, cur, get_default(field), "s2p"))
    s2p_rows.sort(key=lambda t: t[0])

    lsp_rows: list[tuple[str, object, object, str]] = []
    if s2p_extras is not None:
        import dataclasses as _dc
        for ef in _dc.fields(s2p_extras):
            if ef.default is _dc.MISSING:
                continue
            cur = getattr(s2p_extras, ef.name)
            if cur != ef.default:
                lsp_rows.append((ef.name, cur, ef.default, "lsp"))
        lsp_rows.sort(key=lambda t: t[0])

    return s2p_rows + lsp_rows


def draw_suite2p_settings_panel(
    settings: "Suite2pSettings",
    input_width: int = 120,
    show_header: bool = False,
    show_footer: bool = False,
    header_text: str = "",
    footer_text: str = "",
    readonly: bool = False,
) -> "Suite2pSettings":
    """
    Draw a reusable Suite2p settings panel.

    This function renders the Suite2p configuration UI and can be used in:
    - The main PreviewDataWidget pipeline tab (via draw_section_suite2p)
    - Standalone documentation screenshots
    - Any imgui context where Suite2p settings need to be displayed

    Parameters
    ----------
    settings : Suite2pSettings
        The settings dataclass to render and modify.
    input_width : int
        Width for input fields in pixels.
    show_header : bool
        Whether to show a header explanation text.
    show_footer : bool
        Whether to show a footer tip text.
    header_text : str
        Custom header text. If empty, uses default.
    footer_text : str
        Custom footer text. If empty, uses default.
    readonly : bool
        If True, inputs are display-only (no modification).

    Returns
    -------
    Suite2pSettings
        The (potentially modified) settings.
    """
    if show_header:
        text = header_text or (
            "Suite2p pipeline parameters for calcium imaging analysis. "
            "These defaults are optimized for LBM (Light Beads Microscopy) datasets."
        )
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
        imgui.text_colored(imgui.ImVec4(0.7, 0.85, 1.0, 1.0), text)
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    # Main Settings section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Main Settings:")
    imgui.dummy(imgui.ImVec2(0, 4))

    imgui.text("  tau")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.tau:.1f}")
    else:
        _, settings.tau = imgui.input_float(
            "##tau_panel", settings.tau, 0.1, 0.5, "%.1f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Indicator timescale (s)")

    imgui.text("  frames_include")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.frames_include}")
    else:
        _, settings.frames_include = imgui.input_int(
            "##frames_panel", settings.frames_include
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "-1 = all frames")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # Registration section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Registration:")
    imgui.dummy(imgui.ImVec2(0, 4))

    # do_registration is int 0/1/2 (skip/run/force) in upstream
    _reg_labels = ["Skip", "Run", "Force"]
    if readonly:
        _reg_label = _reg_labels[settings.do_registration] if 0 <= settings.do_registration <= 2 else "?"
        imgui.text(f"  do_registration = {_reg_label}")
    else:
        imgui.set_next_item_width(hello_imgui.em_size(8))
        _reg_idx = settings.do_registration if 0 <= settings.do_registration <= 2 else 1
        _changed, _new_idx = imgui.combo("do_registration##panel", _reg_idx, _reg_labels)
        if _changed:
            settings.do_registration = _new_idx
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Run motion correction")

    if readonly:
        imgui.text(f"  [{'x' if settings.nonrigid else ' '}] nonrigid")
    else:
        _, settings.nonrigid = imgui.checkbox("nonrigid##panel", settings.nonrigid)
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Non-rigid registration")

    imgui.text("  maxregshift")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.maxregshift:.2f}")
    else:
        _, settings.maxregshift = imgui.input_float(
            "##maxreg_panel", settings.maxregshift, 0.01, 0.1, "%.2f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Max shift (fraction)")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # ROI Detection section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "ROI Detection:")
    imgui.dummy(imgui.ImVec2(0, 4))

    # do_detection is int 0/1/2 (skip/run/force) — mirrors do_registration
    _det_labels = ["Skip", "Run", "Force"]
    if readonly:
        _det_label = _det_labels[settings.do_detection] if 0 <= settings.do_detection <= 2 else "?"
        imgui.text(f"  do_detection = {_det_label}")
    else:
        imgui.set_next_item_width(hello_imgui.em_size(8))
        _det_idx = settings.do_detection if 0 <= settings.do_detection <= 2 else 1
        _changed, _new_idx = imgui.combo("do_detection##panel", _det_idx, _det_labels)
        if _changed:
            settings.do_detection = _new_idx
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Run cell detection")

    _algo_options = ["cellpose", "sparsery", "sourcery"]
    if readonly:
        imgui.text(f"  algorithm = {settings.algorithm}")
    else:
        if settings.algorithm not in _algo_options:
            settings.algorithm = "sparsery"
        _algo_idx = _algo_options.index(settings.algorithm)
        imgui.set_next_item_width(hello_imgui.em_size(10))
        _changed, _new_idx = imgui.combo("algorithm##panel", _algo_idx, _algo_options)
        if _changed:
            settings.algorithm = _algo_options[_new_idx]
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Detection algorithm")

    imgui.text("  diameter")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.diameter_y:.1f}, {settings.diameter_x:.1f}")
    else:
        _, settings.diameter_y = imgui.input_float(
            "##diam_y_panel", settings.diameter_y, 0.5, 2.0, "%.1f"
        )
        imgui.same_line()
        _, settings.diameter_x = imgui.input_float(
            "##diam_x_panel", settings.diameter_x, 0.5, 2.0, "%.1f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Cell diameter (dy, dx) px")

    imgui.text("  threshold_scaling")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.threshold_scaling:.1f}")
    else:
        _, settings.threshold_scaling = imgui.input_float(
            "##thresh_panel", settings.threshold_scaling, 0.1, 0.5, "%.1f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Higher = fewer ROIs")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # Signal Extraction section
    imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Signal Extraction:")
    imgui.dummy(imgui.ImVec2(0, 4))

    if readonly:
        imgui.text(f"  [{'x' if settings.neuropil_extract else ' '}] neuropil_extract")
    else:
        _, settings.neuropil_extract = imgui.checkbox(
            "neuropil_extract##panel", settings.neuropil_extract
        )
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Extract neuropil signal")

    imgui.text("  neuropil_coefficient")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.neuropil_coefficient:.2f}")
    else:
        _, settings.neuropil_coefficient = imgui.input_float(
            "##neuc_panel", settings.neuropil_coefficient, 0.05, 0.1, "%.2f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Neuropil coefficient")

    if readonly:
        imgui.text(f"  [{'x' if settings.do_deconvolution else ' '}] do_deconvolution")
    else:
        _, settings.do_deconvolution = imgui.checkbox(
            "do_deconvolution##panel", settings.do_deconvolution
        )
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Run spike deconvolution")

    if show_footer:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        text = footer_text or (
            "Tip: For LBM data, tau=1.3 and diameter=4 are good starting points. "
            "Increase threshold_scaling if detecting too many false ROIs."
        )
        imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), text)
        imgui.pop_text_wrap_pos()

    return settings


@dataclass
class Suite2pSettings:
    """
    Suite2p pipeline processing settings — mirrors upstream
    `suite2p.default_settings()`.

    Attribute names match upstream canonical names. The dataclass is kept
    flat for GUI ergonomics; `to_dict()` produces the upstream-shaped
    nested dict (run / io / registration / detection / ... ).

    Paths, plane counts, and I/O config live on `Suite2pDB`. Mbo-only
    helper fields (dff_*, accept_all_cells, etc.) live on
    `MboSuite2pExtras`.
    """

    # top-level — defaults match suite2p.parameters.SETTINGS exactly.
    # Don't bake in LBM tunings here; users start from upstream defaults
    # and tweak from there (the schema helper at _s2p_schema.py is the
    # single source of truth for what "default" means).
    torch_device: str = "cuda"  # upstream default
    tau: float = 1.0  # upstream default
    fs: float = 10.0  # upstream default
    diameter_y: float = 12.0  # upstream default; was 4.0 (LBM tuning)
    diameter_x: float = 12.0  # upstream default; was 4.0 (LBM tuning)
    # Cell-detection algorithm. Drives which sub-section of detection
    # settings is honored at runtime.
    algorithm: str = "sparsery"  # upstream default; one of "cellpose"/"sparsery"/"sourcery"

    # run section
    do_registration: int = 1  # 0=skip, 1=run, 2=force
    do_regmetrics: bool = False  # PC reg-quality metrics: ~40s/plane on >=1500 frames
    do_detection: int = 1  # 0=skip, 1=run, 2=force (was fork's "roidetect")
    do_deconvolution: bool = True  # was fork's "spikedetect"
    multiplane_parallel: bool = False

    # io section
    combined: bool = True
    save_mat: bool = False
    save_NWB: bool = False  # upstream naming
    save_ops_orig: bool = True  # drives whether ops.npy is written per plane
    delete_bin: bool = False
    move_bin: bool = False

    # registration section
    align_by_chan: int = 1  # user-facing 1/2; serialized as bool align_by_chan2
    nimg_init: int = 400  # upstream default (fork was 300)
    maxregshift: float = 0.1
    do_bidiphase: bool = False
    bidiphase: float = 0.0
    reg_batch_size: int = 100  # upstream registration.batch_size (fork conflated with extraction)
    nonrigid: bool = True
    maxregshiftNR: int = 5
    block_size_y: int = 128
    block_size_x: int = 128
    smooth_sigma_time: float = 0.0
    smooth_sigma: float = 1.15
    spatial_taper: float = 3.45  # upstream default; 1P override lives on MboSuite2pExtras
    th_badframes: float = 1.0
    norm_frames: bool = True
    snr_thresh: float = 1.2
    subpixel: int = 10
    two_step_registration: bool = False
    reg_tif: bool = False
    reg_tif_chan2: bool = False

    # detection section
    # NOTE: legacy `sparse_mode` (bool) and `anatomical_only` (int 0-4) were
    # removed in favor of `algorithm` (the canonical selector) and
    # `cellpose_img` (the literal upstream string). Old flat ops files using
    # those legacy keys are remapped on load by _s2p_schema._FLAT_TO_MBO.
    denoise: bool = False
    det_block_size_y: int = 64
    det_block_size_x: int = 64
    nbins: int = 5000  # was fork's "nbinned"
    bin_size: int | None = None  # upstream defaults to tau*fs when None
    highpass_time: int = 100  # was fork's "high_pass"
    threshold_scaling: float = 1.0
    npix_norm_min: float = 0.0
    npix_norm_max: float = 100.0
    max_overlap: float = 0.75
    soma_crop: bool = True
    chan2_threshold: float = 0.25  # was fork's "chan2_thres" (default differs)
    cellpose_chan2: bool = False

    # detection.sparsery_settings
    highpass_neuropil: int = 25
    max_ROIs: int = 5000
    spatial_scale: int = 0  # upstream default 0=auto
    active_percentile: float = 0.0

    # detection.sourcery_settings
    connected: bool = True
    max_iterations: int = 20
    smooth_masks: bool = False  # upstream default False (fork was True)
    # NOTE: legacy `spatial_hp_detect` was renamed to `highpass_neuropil`
    # (under detection.sparsery_settings) in the upstream restructure.
    # Old flat ops.npy files with `spatial_hp_detect` are remapped on load
    # by _s2p_schema._FLAT_TO_MBO.

    # detection.cellpose_settings
    cellpose_model: str = "cpsam"
    # cellpose image source — one of upstream's three allowed strings:
    # "max_proj / meanImg" | "meanImg" | "max_proj"
    cellpose_img: str = "max_proj / meanImg"
    highpass_spatial: float = 0.0  # upstream default
    flow_threshold: float = 0.0  # LBM default: flow check disabled (suite2p ships 0.4)
    cellprob_threshold: float = -4.0  # LBM default: more permissive than suite2p's 0.0
    # cellpose iterations (upstream forwards via `params` dict → model.eval(niter=...))
    # 0 = let cellpose pick based on diameter (default). Bump for dense / small cells.
    cellpose_niter: int = 0

    # classification section (upstream settings keys)
    classifier_path: str | None = None  # upstream default; was "" (mbo legacy)
    use_builtin_classifier: bool = False
    preclassify: float = 0.0
    # `accept_all_cells` is NOT an upstream settings key; it's an
    # lbm_suite2p_python.run_plane kwarg — moved to MboSuite2pExtras.

    # extraction section
    snr_threshold: float = 0.0
    extract_batch_size: int = 500
    neuropil_extract: bool = True
    neuropil_coefficient: float = 0.7  # was fork's "neucoeff"
    inner_neuropil_radius: int = 2
    min_neuropil_pixels: int = 350
    lam_percentile: float = 50.0  # upstream type is float; was int 50 (caused input_int crash on Reset)
    allow_overlap: bool = False
    circular_neuropil: bool = False

    # dcnv_preprocess section
    baseline: str = "maximin"  # "maximin" | "prctile" | "constant" (upstream spellings)
    win_baseline: float = 60.0
    sig_baseline: float = 10.0
    prctile_baseline: float = 8.0

    def to_dict(self) -> dict:
        """Return the upstream-shaped nested settings dict."""
        return {
            "torch_device": self.torch_device,
            "tau": self.tau,
            "fs": self.fs,
            "diameter": [float(self.diameter_y), float(self.diameter_x)],
            "run": {
                "do_registration": int(self.do_registration),
                "do_regmetrics": self.do_regmetrics,
                # upstream's settings expects bool; the int 0/1/2 is collapsed
                # to the 0-vs-nonzero distinction here. force-rerun (=2) is
                # signaled via the force_detect kwarg derived at call time.
                "do_detection": bool(self.do_detection),
                "do_deconvolution": self.do_deconvolution,
                "multiplane_parallel": self.multiplane_parallel,
            },
            "io": {
                "combined": self.combined,
                "save_mat": self.save_mat,
                "save_NWB": self.save_NWB,
                "save_ops_orig": self.save_ops_orig,
                "delete_bin": self.delete_bin,
                "move_bin": self.move_bin,
            },
            "registration": {
                "align_by_chan2": bool(self.align_by_chan == 2),
                "nimg_init": self.nimg_init,
                "maxregshift": self.maxregshift,
                "do_bidiphase": self.do_bidiphase,
                "bidiphase": self.bidiphase,
                "batch_size": self.reg_batch_size,
                "nonrigid": self.nonrigid,
                "maxregshiftNR": self.maxregshiftNR,
                "block_size": (int(self.block_size_y), int(self.block_size_x)),
                "smooth_sigma_time": self.smooth_sigma_time,
                "smooth_sigma": self.smooth_sigma,
                "spatial_taper": self.spatial_taper,
                "th_badframes": self.th_badframes,
                "norm_frames": self.norm_frames,
                "snr_thresh": self.snr_thresh,
                "subpixel": self.subpixel,
                "two_step_registration": self.two_step_registration,
                "reg_tif": self.reg_tif,
                "reg_tif_chan2": self.reg_tif_chan2,
            },
            "detection": {
                "algorithm": self.algorithm,
                "denoise": self.denoise,
                "block_size": (int(self.det_block_size_y), int(self.det_block_size_x)),
                "nbins": self.nbins,
                "bin_size": self.bin_size,
                "highpass_time": self.highpass_time,
                "threshold_scaling": self.threshold_scaling,
                "npix_norm_min": self.npix_norm_min,
                "npix_norm_max": self.npix_norm_max,
                "max_overlap": self.max_overlap,
                "soma_crop": self.soma_crop,
                "chan2_threshold": self.chan2_threshold,
                "cellpose_chan2": self.cellpose_chan2,
                "sparsery_settings": {
                    "highpass_neuropil": self.highpass_neuropil,
                    "max_ROIs": self.max_ROIs,
                    "spatial_scale": self.spatial_scale,
                    "active_percentile": self.active_percentile,
                },
                "sourcery_settings": {
                    "connected": self.connected,
                    "max_iterations": self.max_iterations,
                    "smooth_masks": self.smooth_masks,
                },
                "cellpose_settings": {
                    "cellpose_model": self.cellpose_model,
                    "img": self.cellpose_img,
                    "highpass_spatial": self.highpass_spatial,
                    "flow_threshold": self.flow_threshold,
                    "cellprob_threshold": self.cellprob_threshold,
                    # `params` is unpacked into cellpose's model.eval() via
                    # `**params`. only emit it when the user picked a niter
                    # — passing an empty dict is fine, but we want None to
                    # remain None so cellpose's own default kicks in.
                    "params": (
                        {"niter": int(self.cellpose_niter)}
                        if self.cellpose_niter and self.cellpose_niter > 0
                        else None
                    ),
                },
            },
            "classification": {
                "classifier_path": self.classifier_path,
                "use_builtin_classifier": self.use_builtin_classifier,
                "preclassify": self.preclassify,
            },
            "extraction": {
                "snr_threshold": self.snr_threshold,
                "batch_size": self.extract_batch_size,
                "neuropil_extract": self.neuropil_extract,
                "neuropil_coefficient": self.neuropil_coefficient,
                "inner_neuropil_radius": self.inner_neuropil_radius,
                "min_neuropil_pixels": self.min_neuropil_pixels,
                "lam_percentile": self.lam_percentile,
                "allow_overlap": self.allow_overlap,
                "circular_neuropil": self.circular_neuropil,
            },
            "dcnv_preprocess": {
                "baseline": self.baseline,
                "win_baseline": self.win_baseline,
                "sig_baseline": self.sig_baseline,
                "prctile_baseline": self.prctile_baseline,
            },
        }

    def to_file(self, filepath):
        """Save settings to settings.npy (upstream-shaped nested dict)."""
        np.save(filepath, _convert_paths_to_strings(self.to_dict()), allow_pickle=True)

    def update_from_ops(self, ops) -> list[str]:
        """Merge values from a flat ops.npy dict OR structured settings.npy dict
        OR a path to either, into this Suite2pSettings instance.

        Auto-detects flat vs structured via _s2p_schema.from_ops. Only writes
        to fields that already exist on the dataclass. Returns the names of
        fields whose value actually changed.
        """
        from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
            from_ops as _decode_ops,
            from_npy_file as _decode_file,
        )
        if isinstance(ops, (str, pathlib.Path)):
            mapping = _decode_file(ops)
        else:
            mapping = _decode_ops(ops)
        changed: list[str] = []
        for field, value in mapping.items():
            if not hasattr(self, field):
                continue
            current = getattr(self, field)
            # coerce numeric types to match the field's current type — keeps
            # imgui input_int / input_float widgets happy when an old ops.npy
            # contains loose types (e.g. lam_percentile written as int when
            # the current dataclass field is float, or vice versa).
            if value is not None and current is not None:
                ct = type(current)
                try:
                    if ct is bool and not isinstance(value, bool):
                        value = bool(value)
                    elif ct is int and isinstance(value, bool):
                        # bool → int (handles upstream bools being assigned
                        # to mbo's tri-state ints, e.g. do_detection)
                        value = int(value)
                    elif ct is int and isinstance(value, float):
                        value = int(value)
                    elif ct is float and isinstance(value, int) and not isinstance(value, bool):
                        value = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
            if current != value:
                setattr(self, field, value)
                changed.append(field)
        return changed

    @classmethod
    def from_ops(cls, ops) -> "Suite2pSettings":
        """Build a fresh Suite2pSettings populated from a flat ops.npy /
        structured settings.npy / path to either. Unmentioned fields keep
        their dataclass defaults."""
        instance = cls()
        instance.update_from_ops(ops)
        return instance


@dataclass
class Suite2pDB:
    """
    Suite2p input/output database — mirrors upstream `suite2p.default_db()`.
    These are paths, plane counts, and I/O flags that identify *what* to
    process (not *how*).
    """

    data_path: list | None = None  # list of directories containing input files
    look_one_level_down: bool = False
    input_format: str = "tif"  # "tif" | "h5" | "nwb" | "bruker" | "movie" | "dcimg"
    keep_movie_raw: bool = False
    nplanes: int = 1
    nrois: int = 1
    nchannels: int = 1
    swap_order: bool = False
    functional_chan: int = 1  # moved here from settings
    subfolders: list | None = None
    save_path0: str = ""
    fast_disk: str = ""
    save_folder: str = "suite2p"
    h5py_key: str = "data"
    nwb_driver: str = ""
    nwb_series: str = ""
    force_sktiff: bool = False
    bruker_bidirectional: bool = False

    def __post_init__(self):
        if self.data_path is None:
            self.data_path = []
        if self.subfolders is None:
            self.subfolders = []

    def to_dict(self) -> dict:
        return {
            "data_path": list(self.data_path or []),
            "look_one_level_down": self.look_one_level_down,
            "input_format": self.input_format,
            "keep_movie_raw": self.keep_movie_raw,
            "nplanes": self.nplanes,
            "nrois": self.nrois,
            "nchannels": self.nchannels,
            "swap_order": self.swap_order,
            "functional_chan": self.functional_chan,
            "subfolders": list(self.subfolders or []),
            "save_path0": self.save_path0,
            "fast_disk": self.fast_disk,
            "save_folder": self.save_folder,
            "h5py_key": self.h5py_key,
            "nwb_driver": self.nwb_driver,
            "nwb_series": self.nwb_series,
            "force_sktiff": self.force_sktiff,
            "bruker_bidirectional": self.bruker_bidirectional,
        }

    def to_file(self, filepath):
        """Save db to db.npy (upstream-shaped flat dict)."""
        np.save(filepath, _convert_paths_to_strings(self.to_dict()), allow_pickle=True)


@dataclass
class MboSuite2pExtras:
    """
    Fork/mbo-specific fields that have no upstream home. Kept on a
    dedicated object so `Suite2pSettings` can stay in lockstep with
    upstream's schema.

    These are consumed by `lbm_suite2p_python.run_plane`'s own kwargs or
    by mbo's pre/post-processing (dF/F, target_timepoints clipping).
    """

    # lbm_suite2p_python.run_plane kwargs that are MBO-only (no upstream
    # equivalent). The `keep_raw`/`keep_reg` kwargs are still passed to
    # run_plane at runtime, but they're derived from upstream fields:
    #   keep_raw <- self.s2p_db.keep_movie_raw
    #   keep_reg <- not self.s2p.delete_bin
    # so we don't store duplicate state here.
    #
    # force_reg / force_detect are also derived at call time from
    # `Suite2pSettings.do_registration == 2` and
    # `Suite2pSettings.do_detection == 2` (the Skip/Run/Force radios are
    # the single source of truth).

    # when True, lbm's run_plane keeps every detected ROI after upstream
    # classification (mirrors the `accept_all_cells` kwarg in lsp.pipeline).
    accept_all_cells: bool = False
    # normalization written to norm_traces.npy (lsp.pipeline(norm_method=...)).
    # "dff" = rolling-percentile ΔF/F (dff_window_size / dff_percentile apply);
    # "zscore" = per-ROI (F - mean) / std. dff_smooth_window applies to both.
    norm_method: str = "dff"
    dff_window_size: int = 300
    dff_percentile: int = 20
    dff_smooth_window: int = 0

    # save a human-readable ops.json sibling next to ops.npy via
    # lbm_suite2p_python.pipeline(save_json=...). useful for inspecting
    # what suite2p actually ran with. mbo-only (lsp run-plane kwarg, not
    # a suite2p settings field).
    save_json: bool = False

    # cell_filters knobs — paired (enabled, value). compose into the lsp
    # `cell_filters` list at run time. value 0 keeps the spinner enabled
    # but is treated as disabled by build_cell_filters() — both signals
    # collapse to "ignore this bound".
    min_diameter_um_enabled: bool = False
    min_diameter_um: float = 0.0
    max_diameter_um_enabled: bool = False
    max_diameter_um: float = 0.0

    # baseline cell filter (lsp's `baseline` filter — rejects ROIs whose
    # F0 baseline is negative, sub-photon-floor, or sub-fraction-of-median).
    # baseline scoring inherits the dF/F window/percentile and the top-level
    # correct_neuropil toggle so the filter sees the same trace the dF/F
    # plot does.
    # always-on master flag — UI doesn't expose it; build_cell_filters()
    # consults the per-criterion enables below. Default True so a fresh
    # dataclass matches the runtime contract; loaded settings.npy values
    # round-trip unchanged.
    baseline_filter_enabled: bool = True
    baseline_reject_negative_F0: bool = False
    baseline_min_F0_abs_enabled: bool = False
    baseline_min_F0_abs: float = 1.0
    baseline_min_F0_rel_enabled: bool = False
    baseline_min_F0_rel: float = 0.0

    # top-level neuropil-correction toggle (lsp pipeline / run_plane /
    # run_volume kwarg). when False, the normalized trace is computed from
    # raw F instead of F - neuropil_coef * Fneu, and trace-quality metrics
    # skip neuropil subtraction. mirrors `--no-correct-neuropil` on the lsp
    # cli. defaults off in the GUI.
    correct_neuropil: bool = False

    # rastermap stage gate (Skip=0, Run=1, Force=2 — matches the
    # pipeline-settings popup convention for tri-state stage radios).
    # Planar / volumetric flags below are only honored when this is
    # non-zero. Force is treated identically to Run by the underlying
    # lsp api (it has its own cache-detection); the radio is a UX echo
    # of the other stage gates, not a separate runtime mode.
    rastermap_mode: int = 0

    # rastermap mode toggles. Forwarded as `rastermap_kwargs={"planar":
    # {...}, "volumetric": {...}}` — presence of each key is the on/off
    # signal per lsp's unified api. Sub-params with sentinel values (0 /
    # -1) get omitted from the sub-dict so lsp's cell-count-aware defaults
    # apply to those fields only.
    rastermap_planar: bool = False
    rastermap_volumetric: bool = False

    # planar Rastermap() overrides. 0 / -1 = "omit, use lsp default".
    # lsp defaults derive from cell count: n_clusters = 100 if >=200 else
    # None; n_PCs = min(128, n_cells-1); locality = 0.0 if >=200 else 0.1.
    rastermap_planar_n_clusters: int = 0
    rastermap_planar_n_pcs: int = 0
    rastermap_planar_locality: float = -1.0

    # volumetric Rastermap() overrides. lsp's defaults: n_clusters=40
    # (caller-passed), n_PCs = min(200, n_cells-1).
    rastermap_volumetric_n_clusters: int = 40
    rastermap_volumetric_n_pcs: int = 0

    # mbo-side timepoint / plane selection
    target_timepoints: int = -1
    frames_include: int = -1

    # parallel processing — forwarded to lsp.pipeline(workers=...,
    # threads_per_worker=..., skip_volumetric=...). workers=1 keeps the
    # sequential path; 0 or negative means auto = min(num_planes,
    # cpu_count//2, 8). threads_per_worker caps BLAS / OMP / numba /
    # torch threads inside each worker so workers*threads doesn't
    # oversubscribe the CPU (the SVD-heavy registration step balloons
    # ~10x without this cap). 0 or negative = library defaults.
    # skip_volumetric drops merge_mrois/volume_stats/volume plots after
    # the per-plane loop — useful when farming planes across machines.
    workers: int = field(default_factory=_default_workers)
    threads_per_worker: int = 2
    skip_volumetric: bool = False

    # gui-only display
    aspect: float = 1.0

    # NOTE: removed dead fields (none read by suite2p>=v1 or lbm pipeline):
    #   force_refImg, pad_fft  — pre-v1 reg flags, gone from upstream
    #   do_1Preg, spatial_hp_reg, pre_smooth  — old 1P-specific keys; modern
    #     advice is to bump smooth_sigma / spatial_taper directly
    #   report_time           — never consumed; suite2p logs plane_times always
    #   keep_raw, keep_reg    — supplanted by db.keep_movie_raw / io.delete_bin;
    #     the run path now derives them from those upstream fields

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()  # type: ignore
        }


def draw_tab_process(self):
    """Draws the pipeline selection and configuration section."""
    if not hasattr(self, "_current_pipeline"):
        self._current_pipeline = USER_PIPELINES[0]
    if not hasattr(self, "_install_error"):
        self._install_error = False
    if not hasattr(self, "_show_red_text"):
        self._show_red_text = False
    if not hasattr(self, "_show_green_text"):
        self._show_green_text = False
    if not hasattr(self, "_show_install_button"):
        self._show_install_button = False

    if self._current_pipeline == "suite2p":
        draw_section_suite2p(self)
    elif self._current_pipeline == "masknmf":
        imgui.text("MaskNMF pipeline not yet implemented.")


def _init_s2p_selection_state(self):
    """Initialize selection state for suite2p run dialog."""
    from mbo_utilities.arrays.features._slicing import parse_timepoint_selection

    # get data dimensions
    max_frames = 1000
    num_planes = 1
    num_channels = 1
    try:
        if hasattr(self, "image_widget") and self.image_widget.data:
            data = self.image_widget.data[0]
            max_frames = data.shape[0]
            if hasattr(data, "num_planes"):
                num_planes = data.num_planes
            elif data.ndim == 4:
                num_planes = data.shape[1]

            # fallback: if multiple files loaded but each is single-plane natively,
            # treat the list of files as a volumetric stack (used when reloading Suite2p output)
            if isinstance(self.fpath, (list, tuple)) and len(self.fpath) > 1:
                if num_planes == 1:
                    num_planes = len(self.fpath)

            if hasattr(data, "num_color_channels"):
                num_channels = data.num_color_channels
            elif hasattr(data, "_num_color_channels"):
                num_channels = data._num_color_channels
            # fallback: detect channels from dims/shape for 5D data
            if num_channels <= 1 and data.ndim == 5:
                from mbo_utilities.arrays.features import get_dims
                dims = get_dims(data)
                if dims is not None and len(dims) >= 5 and dims[1] == "C":
                    num_channels = data.shape[1]
    except Exception:
        pass

    # track file to reset state on file change
    current_fpath = self.fpath[0] if isinstance(self.fpath, list) else self.fpath
    current_fpath = str(current_fpath) if current_fpath else ""
    file_changed = False
    if not hasattr(self, "_s2p_last_fpath"):
        self._s2p_last_fpath = current_fpath
    elif self._s2p_last_fpath != current_fpath:
        file_changed = True
        self._s2p_last_fpath = current_fpath

    # initialize or reset selection state
    if file_changed or not hasattr(self, "_s2p_tp_selection"):
        self._s2p_tp_selection = f"1:{max_frames}"
        self._s2p_tp_error = ""
        self._s2p_tp_parsed = None
        self._s2p_last_max_tp = max_frames
        self._s2p_z_start = 1
        self._s2p_z_stop = num_planes
        self._s2p_z_step = 1
        self._s2p_last_num_planes = num_planes
        self._s2p_c_start = 1
        self._s2p_c_stop = num_channels
        self._s2p_c_step = 1
        self._s2p_last_num_channels = num_channels

        # Auto-fill Sampling Rate (Hz) from source metadata on file change.
        # User-edited metadata wins over source (mirrors task_suite2p's
        # resolution order). Only overwrites when a valid fs is found so a
        # user value entered in the GUI is preserved when loading files
        # without metadata fs.
        try:
            from mbo_utilities.metadata import get_param
            src_fs = None
            if hasattr(self, "image_widget") and self.image_widget.data:
                mdata = getattr(self.image_widget.data[0], "metadata", {}) or {}
                src_fs = get_param(mdata, "fs")
            # prefer user-edited fs from the metadata editor if present
            custom = getattr(self, "_custom_metadata", None) or {}
            if "fs" in custom and custom["fs"] is not None:
                src_fs = custom["fs"]
            if src_fs is not None and self.s2p is not None:
                try:
                    self.s2p.fs = float(src_fs)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

    # check if max changed
    if not hasattr(self, "_s2p_last_max_tp"):
        self._s2p_last_max_tp = max_frames
    elif self._s2p_last_max_tp != max_frames:
        self._s2p_last_max_tp = max_frames
        self._s2p_tp_selection = f"1:{max_frames}"
        self._s2p_tp_parsed = None
        self._s2p_tp_error = ""

    # z-plane state
    if not hasattr(self, "_s2p_z_start"):
        self._s2p_z_start = 1
    if not hasattr(self, "_s2p_z_stop"):
        self._s2p_z_stop = num_planes
    if not hasattr(self, "_s2p_z_step"):
        self._s2p_z_step = 1
    if not hasattr(self, "_s2p_last_num_planes"):
        self._s2p_last_num_planes = num_planes
    elif self._s2p_last_num_planes != num_planes:
        self._s2p_last_num_planes = num_planes
        self._s2p_z_start = 1
        self._s2p_z_stop = num_planes
        self._s2p_z_step = 1

    # channel state
    if not hasattr(self, "_s2p_c_start"):
        self._s2p_c_start = 1
    if not hasattr(self, "_s2p_c_stop"):
        self._s2p_c_stop = num_channels
    if not hasattr(self, "_s2p_c_step"):
        self._s2p_c_step = 1
    if not hasattr(self, "_s2p_last_num_channels"):
        self._s2p_last_num_channels = num_channels
    elif self._s2p_last_num_channels != num_channels:
        self._s2p_last_num_channels = num_channels
        self._s2p_c_start = 1
        self._s2p_c_stop = num_channels
        self._s2p_c_step = 1

    # always ensure _selected_planes stays purely synced to the current z slicing logic
    if num_planes > 1:
        self._selected_planes = set(range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step))
    else:
        self._selected_planes = {1}

    # parse timepoint selection if needed
    if self._s2p_tp_parsed is None and not self._s2p_tp_error:
        try:
            self._s2p_tp_parsed = parse_timepoint_selection(self._s2p_tp_selection, max_frames)
        except ValueError as e:
            self._s2p_tp_error = str(e)

    return max_frames, num_planes, num_channels


def _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels=1):
    """Draw frame / plane / channel selection counts as one line per field
    (matches the Current dataset info layout — non-colored, separate rows)."""
    if self._s2p_tp_parsed:
        n_frames = self._s2p_tp_parsed.count
    else:
        n_frames = max_frames

    n_planes = len(range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step))
    n_channels = len(range(self._s2p_c_start, self._s2p_c_stop + 1, self._s2p_c_step))

    imgui.text(f"Frames: {n_frames}")
    if num_planes > 1:
        imgui.text(f"Planes: {n_planes}")
    if num_channels > 1:
        imgui.text(f"Channels: {n_channels}")


def _draw_s2p_slicing_popup(self):
    """Slicing popup — pick which timepoints, z-planes, and channels to process."""
    from mbo_utilities.arrays.features._slicing import parse_timepoint_selection  # noqa: F401

    if getattr(self, "_s2p_slicing_open", False):
        imgui.open_popup("Frames & Planes##s2p_slice")
        self._s2p_slicing_open = False

    imgui.set_next_window_size(imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Frames & Planes##s2p_slice"):
        max_frames = getattr(self, "_s2p_last_max_tp", 1000)
        num_planes = getattr(self, "_s2p_last_num_planes", 1)
        num_channels = getattr(self, "_s2p_last_num_channels", 1)

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Frames & Planes")
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
            imgui.text_unformatted(
                "Format: start:stop or start:stop:step\n"
                "Exclude: 1:100,50:60 = 1-100 excluding 50-60"
            )
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        imgui.dummy(imgui.ImVec2(0, 4))

        tp_label, z_label, c_label = resolve_dim_labels(self)
        tp_parsed, *_rest = draw_selection_table(
            self,
            max_frames,
            num_planes,
            tp_attr="_s2p_tp",
            z_attr="_s2p_z",
            id_suffix="_s2p",
            num_channels=num_channels,
            c_attr="_s2p_c",
            tp_label=tp_label,
            z_label=z_label,
            c_label=c_label,
        )

        if tp_parsed:
            # only record a positive count when the user has actually narrowed
            # the selection. opening the popup parses the default full-range
            # selection (1:max_frames) which would otherwise overwrite -1
            # ("all timepoints") with max_frames and falsely flag this field
            # as modified in the run-tab summary.
            self.s2p_extras.target_timepoints = (
                tp_parsed.count if tp_parsed.count < max_frames else -1
            )

        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _draw_data_options_content(self):
    """Draw data options content showing settings that affect Suite2p processing."""

    INPUT_WIDTH = 100
    has_phase_support = getattr(self, "has_raster_scan_support", False)
    nz = getattr(self, "nz", 1)

    is_raw = getattr(self, "is_mbo_scan", False)
    has_z_reg = nz > 1 and is_raw

    has_any_options = has_phase_support or has_z_reg

    # ensure s2p phase attributes exist with defaults
    if not hasattr(self, "_s2p_fix_phase"):
        self._s2p_fix_phase = True
    if not hasattr(self, "_s2p_use_fft"):
        self._s2p_use_fft = True

    # scan-phase correction section
    if has_phase_support:
        imgui.text("Scan-Phase Correction")
        imgui.spacing()

        # fix phase (uses separate _s2p_* settings, defaults True for s2p runs)
        phase_changed, phase_value = imgui.checkbox("Fix Phase", self._s2p_fix_phase)
        set_tooltip("Apply bidirectional scan-phase correction to output data")
        if phase_changed:
            self._s2p_fix_phase = phase_value

        # fft subpixel (uses separate _s2p_* settings, defaults True for s2p runs)
        fft_changed, fft_value = imgui.checkbox("Sub-Pixel (FFT)", self._s2p_use_fft)
        set_tooltip("Use FFT-based sub-pixel registration (slower but more accurate)")
        if fft_changed:
            self._s2p_use_fft = fft_value

        # border exclusion
        imgui.set_next_item_width(INPUT_WIDTH)
        border_changed, border_val = imgui.input_int("Border (px)", self.border, step=1)
        set_tooltip("Pixels to exclude from edges when computing phase offset")
        if border_changed:
            self.border = max(0, border_val)

        # max offset
        imgui.set_next_item_width(INPUT_WIDTH)
        max_offset_changed, max_offset_val = imgui.input_int("Max Offset", self.max_offset, step=1)
        set_tooltip("Maximum allowed pixel shift for phase correction")
        if max_offset_changed:
            self.max_offset = max(1, max_offset_val)

        # show current offset values
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Current offsets:")
        for i, ofs in enumerate(self.current_offset):
            imgui.text(f"  Array {i + 1}: {ofs:.3f} px")

    # axial z-registration - only for raw scanimage data
    if has_z_reg:
        if has_phase_support:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        imgui.text("Axial Registration")
        imgui.spacing()

        if has_z_reg:
            reg_z_val = getattr(self, "_register_z", False)
            reg_changed, reg_value = imgui.checkbox("Register Z-Planes Axially", reg_z_val)
            set_tooltip(
                "Compute per-plane rigid shifts via phase correlation and\n"
                "apply them on write. Corrects z-drift between adjacent planes.\n"
                "Uses GPU (cupy) if available, otherwise CPU."
            )
            if reg_changed:
                self._register_z = reg_value

            if self._register_z:
                if not hasattr(self, "_axial_max_frames"):
                    self._axial_max_frames = 200
                if not hasattr(self, "_axial_max_reg_xy"):
                    self._axial_max_reg_xy = 30

                imgui.indent(16)
                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max frames", self._axial_max_frames, 50, 100
                )
                if _changed:
                    self._axial_max_frames = max(10, _val)
                set_tooltip(
                    "Frames subsampled (evenly spaced) for the time-mean used\n"
                    "to compute plane shifts. 200 is usually plenty."
                )

                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max shift (px)", self._axial_max_reg_xy, 10, 50
                )
                if _changed:
                    self._axial_max_reg_xy = max(1, _val)
                set_tooltip(
                    "Max shift search radius in pixels. Default 30."
                )
                imgui.unindent(16)
        else:
            imgui.begin_disabled()
            imgui.checkbox("Register Z-Planes Axially", False)
            imgui.end_disabled()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Requires multi-plane raw data",
            )

    if not has_any_options:
        imgui.text_disabled("No available options for this data type.")


# soft red used to flag missing-but-required scalar metadata fields
# (fs / dx / dy / dz) in the Current dataset section.
_MISSING_COLOR = imgui.ImVec4(1.0, 0.45, 0.45, 1.0)


def _format_size(size_bytes: int | None) -> str:
    """Render a byte count as human-friendly string."""
    if size_bytes is None or size_bytes < 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    n = float(size_bytes)
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def _compute_dataset_size_sync(filenames: list) -> int:
    """Recursive sum of bytes for ``filenames``.

    Handles directory-format containers (``.zarr``, ``.ome.zarr``,
    suite2p plane dirs) by walking their contents — ``Path.stat()`` on
    a directory only reports the dir-entry size, not the tree.
    """
    from stat import S_ISDIR
    total = 0
    for f in filenames:
        try:
            p = Path(f)
            st = p.stat()
            if S_ISDIR(st.st_mode):
                for child in p.rglob("*"):
                    try:
                        cst = child.stat()
                        if not S_ISDIR(cst.st_mode):
                            total += cst.st_size
                    except OSError:
                        continue
            else:
                total += st.st_size
        except OSError:
            continue
    return total


def _dataset_size_disk_cache_path(key: tuple[str, ...]) -> Path:
    """Per-dataset cache path; hashed to keep the filename short."""
    import hashlib
    h = hashlib.blake2b(
        "\n".join(key).encode("utf-8", errors="replace"), digest_size=16
    ).hexdigest()
    override = os.environ.get("MBO_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        from mbo_utilities.preferences import get_mbo_dirs
        base = get_mbo_dirs()["cache"]
    return base / f"dataset_size_{h}.json"


def _dataset_size_load_disk(key: tuple[str, ...]) -> int | None:
    p = _dataset_size_disk_cache_path(key)
    try:
        with p.open() as f:
            data = json.load(f)
        # Reject stale entries where the saved filename set differs
        # from the requested one — hash collisions are astronomically
        # unlikely but the explicit key check is essentially free.
        if data.get("key") == list(key) and isinstance(data.get("size"), int):
            return int(data["size"])
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _dataset_size_save_disk(key: tuple[str, ...], size: int) -> None:
    p = _dataset_size_disk_cache_path(key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump({"key": list(key), "size": int(size)}, f)
        os.replace(tmp, p)
    except OSError:
        pass


def _dataset_size_bytes(self, filenames: list) -> int | None:
    """Total bytes across ``filenames``, with two-level cache + async fill.

    Lookup order:
      1. in-memory cache on ``self`` (key = filenames tuple)
      2. on-disk cache at ``~/.mbo/cache/dataset_size_<hash>.json``
      3. cold — return ``None`` immediately and spawn a daemon to
         compute the size, populate both caches, and let the next
         frame's call hit (1). UI renders ``_format_size(None) → "—"``
         until the daemon finishes.

    Input files are treated as read-only for the lifetime of a dataset
    open, so once we have a size we never recompute it.
    """
    if not filenames:
        return None
    key = tuple(str(f) for f in filenames)
    cache = getattr(self, "_dataset_size_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    disk = _dataset_size_load_disk(key)
    if disk is not None:
        self._dataset_size_cache = (key, disk)
        return disk
    pending = getattr(self, "_dataset_size_pending_key", None)
    if pending == key:
        return None
    self._dataset_size_pending_key = key

    def _worker(_key: tuple[str, ...], _files: list) -> None:
        size = _compute_dataset_size_sync(_files)
        _dataset_size_save_disk(_key, size)
        self._dataset_size_cache = (_key, size)
        if getattr(self, "_dataset_size_pending_key", None) == _key:
            self._dataset_size_pending_key = None

    threading.Thread(
        target=_worker, args=(key, list(filenames)),
        daemon=True, name="dataset-size-walker",
    ).start()
    return None


def _truncate_to_width(text: str, max_width: float) -> str:
    """Front-truncate `text` (drop chars from the start, replace with …)
    so its rendered width fits within `max_width`. Keeps the tail of the
    string intact, which is what you want for paths (filename stays
    visible). Returns the original string when it already fits."""
    if max_width <= 0:
        return ""
    if imgui.calc_text_size(text).x <= max_width:
        return text
    ellipsis = "…"
    if imgui.calc_text_size(ellipsis).x >= max_width:
        return ""
    # binary search for the largest tail that still fits with leading …
    lo, hi = 0, len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = ellipsis + text[-mid:] if mid > 0 else ellipsis
        if imgui.calc_text_size(cand).x <= max_width:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _draw_md_field(label: str, value, unit: str = "") -> None:
    """Render `label: value unit` on its own line. If value is None,
    render in red and attach a 'press Shift+M' tooltip."""
    if value is None:
        imgui.text_colored(_MISSING_COLOR, f"{label}: —")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"{label} not detected. Press Shift+M to set metadata."
            )
        return
    if isinstance(value, float):
        text = f"{label}: {value:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{label}: {value}"
    if unit:
        text += f" {unit}"
    imgui.text(text)


def _draw_dataset_files_popup(
    filenames: list,
    frames_per_file: list[int] | None = None,
    sizer: PopupAutoSize | None = None,
) -> None:
    """Modal popup listing each file in concatenation order.

    When `frames_per_file` is provided and length-matches `filenames`, a
    "Frames" column is added and the header shows the total frame count.
    The header row carries a ghost-styled icon button that copies the
    listing to the clipboard as JSON.
    """
    imgui.set_next_window_size(
        imgui.ImVec2(700, 450), imgui.Cond_.first_use_ever
    )
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(420, 240), imgui.ImVec2(1600, 1200)
    )
    _flags = (
        sizer.flags(imgui.WindowFlags_.no_saved_settings)
        if sizer is not None
        else imgui.WindowFlags_.no_saved_settings
    )
    opened = imgui.begin_popup_modal(
        "Dataset files##current_dataset_files_popup",
        flags=_flags,
    )[0]
    if not opened:
        return

    has_frames = (
        isinstance(frames_per_file, (list, tuple))
        and len(frames_per_file) == len(filenames)
    )

    # header: count (+ total frames when known), with the copy icon
    # snapped to the right edge.
    if has_frames:
        imgui.text(
            f"{len(filenames)} files (concatenation order, "
            f"{sum(frames_per_file)} total frames):"
        )
    else:
        imgui.text(f"{len(filenames)} files (concatenation order):")
    imgui.same_line()
    avail = imgui.get_content_region_avail().x
    btn_w = imgui.get_frame_height()
    if avail > btn_w + 4:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - btn_w - 2)
    with _ghost_button():
        if imgui.button(
            f"{fa.ICON_FA_COPY}##dataset_files_copy",
            imgui.ImVec2(btn_w, btn_w),
        ):
            payload: dict = {
                "num_files": len(filenames),
                "file_paths": [str(f) for f in filenames],
            }
            if has_frames:
                payload["frames_per_file"] = list(frames_per_file)
                payload["num_timepoints"] = sum(frames_per_file)
            imgui.set_clipboard_text(json.dumps(payload, indent=2))
    set_tooltip(
        "Copy file list as JSON (file_paths, frames_per_file, totals).",
        show_mark=False,
    )

    imgui.separator()

    _reserved = imgui.get_frame_height_with_spacing() + 8
    if imgui.begin_child(
        "##current_dataset_files_list",
        imgui.ImVec2(-1, -_reserved),
        imgui.ChildFlags_.borders,
    ):
        n_cols = 3 if has_frames else 2
        if imgui.begin_table(
            "##dataset_files_table",
            n_cols,
            imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h
            | imgui.TableFlags_.sizing_stretch_prop,
        ):
            imgui.table_setup_column(
                "#", imgui.TableColumnFlags_.width_fixed, 40
            )
            imgui.table_setup_column("File")
            if has_frames:
                imgui.table_setup_column(
                    "Frames", imgui.TableColumnFlags_.width_fixed, 80
                )
            imgui.table_headers_row()

            for i, f in enumerate(filenames):
                imgui.table_next_row()
                imgui.table_set_column_index(0)
                imgui.text(f"{i + 1}")
                imgui.table_set_column_index(1)
                imgui.text_wrapped(str(f))
                if has_frames:
                    imgui.table_set_column_index(2)
                    imgui.text(f"{frames_per_file[i]}")

            imgui.end_table()
        imgui.end_child()

    imgui.separator()
    if imgui.button("Close", imgui.ImVec2(80, 0)):
        imgui.close_current_popup()
    imgui.end_popup()


def _draw_current_dataset_section(self) -> None:
    """Draw the 'Current dataset' info block at the top of the run tab.

    Shows source path, a Files (N) button that opens a popup with the
    concatenation order, the array shape, total bytes on disk, plane /
    frame / channel counts, frame rate, and dx / dy / dz. Missing scalar
    metadata fields are tinted red with a Shift+M hint.
    """
    # local import — get_param lives in metadata, not worth a top-level import
    from mbo_utilities.metadata import get_param

    arr = None
    try:
        arr = self.image_widget.data[0]
    except (IndexError, AttributeError):
        pass

    imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
    imgui.spacing()
    imgui.spacing()
    if arr is None:
        imgui.text_disabled("(no file loaded)")
        imgui.spacing()
        imgui.spacing()
        return

    # path — front-truncate so the filename stays visible and we never
    # blow past the right edge of the side panel.
    if isinstance(self.fpath, (list, tuple)):
        path_str = str(self.fpath[0]) if self.fpath else ""
    else:
        path_str = str(self.fpath) if self.fpath else ""
    avail_x = imgui.get_content_region_avail().x
    display_path = _truncate_to_width(path_str or "(in-memory)", avail_x)
    imgui.text_unformatted(display_path)
    if display_path != (path_str or "(in-memory)") and imgui.is_item_hovered():
        imgui.set_tooltip(path_str)

    # filenames in concat order
    filenames = list(getattr(arr, "filenames", []) or [])
    if not filenames:
        if isinstance(self.fpath, (list, tuple)):
            filenames = list(self.fpath)
        elif self.fpath:
            filenames = [self.fpath]

    # metadata; user-overrides win over array.metadata (mirrors task_suite2p)
    md = dict(getattr(arr, "metadata", {}) or {})
    custom = getattr(self, "_custom_metadata", None) or {}

    def _resolve(key):
        v = custom.get(key)
        if v is not None:
            return v
        return get_param(md, key)

    fs = _resolve("fs")
    dx = _resolve("dx")
    dy = _resolve("dy")
    dz = _resolve("dz")

    # dimensions
    shape = tuple(arr.shape)
    try:
        from mbo_utilities.arrays.features import get_dims as _get_dims
        _dims = _get_dims(arr)
    except Exception:
        _dims = None

    size_bytes = _dataset_size_bytes(self, filenames)

    # files button on its own line; size on disk on the next.
    # avoiding same_line() here so neither piece can spill past the right edge.
    n_files = len(filenames)
    # lazy-init position-only sizer (auto_resize=False — body contains a
    # negative-fill scrollable child that would collapse under
    # always_auto_resize).
    _files_sizer = getattr(self, "_files_popup_sizer", None)
    if _files_sizer is None:
        _files_sizer = PopupAutoSize(
            "Dataset files##current_dataset_files_popup",
            auto_resize=False,
        )
        self._files_popup_sizer = _files_sizer
    if imgui.button(f"Files ({n_files})##current_dataset_files"):
        _files_sizer.before_open()
        imgui.open_popup("Dataset files##current_dataset_files_popup")
    imgui.text(f"Size on disk: {_format_size(size_bytes)}")

    # surface frames_per_file from metadata when present and length-aligned;
    # otherwise the popup falls back to the 2-column (#, File) layout.
    fpf = md.get("frames_per_file")
    if not (isinstance(fpf, (list, tuple)) and len(fpf) == n_files):
        fpf = None
    _draw_dataset_files_popup(filenames, fpf, sizer=_files_sizer)

    # shape with bracketed dim labels (e.g. "1024 × 4 × 256 × 256 [T,Z,Y,X]").
    # push wrap_pos so long shape strings wrap rather than clip on narrow panels.
    shape_text = " × ".join(str(s) for s in shape)
    if _dims and len(_dims) == len(shape):
        shape_text = f"{shape_text} [{','.join(_dims)}]"
    imgui.push_text_wrap_pos(0.0)
    try:
        imgui.text(f"Shape: {shape_text}")
    finally:
        imgui.pop_text_wrap_pos()

    # frame rate / pixel size — one field per line so the inline-`same_line`
    # chain can't overflow, and per-field red highlight stays intact.
    _draw_md_field("Frame rate", fs, "Hz")
    _draw_md_field("dx", dx, "µm")
    _draw_md_field("dy", dy, "µm")
    _draw_md_field("dz", dz, "µm")

    imgui.spacing()
    imgui.spacing()


def draw_section_suite2p(self):
    """Draw Suite2p configuration UI with button-based popups."""
    imgui.spacing()

    # consistent input width
    INPUT_WIDTH = 80

    # set proper padding and spacing using context manager for safety
    with imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)):
        with imgui_ctx.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2)):
            _draw_section_suite2p_content(self)


def _draw_section_suite2p_content(self):
    """inner content for suite2p section (called within style context)."""
    INPUT_WIDTH = 80

    # Local `_hi` shadows the module-level one to drop the font-push
    # behavior. Emphasis (bold + box) now lives in `_emp_label` so only
    # the parameter name (e.g. "Tau (s)") gets emphasized — not the
    # input value (e.g. "1.00"). Call sites for fields in
    # _IMPORTANT_FIELDS use the pattern:
    #
    #   imgui.set_next_item_width(INPUT_WIDTH)
    #   with _hi("tau", self.s2p.tau):
    #       _, self.s2p.tau = imgui.input_float("##tau", self.s2p.tau)
    #   imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    #   _emp_label("tau", self.s2p.tau, "Tau (s)")
    #   set_tooltip(...)
    #
    # `_hi` still wraps the widget so the modified-color tint applies to
    # the value; `_emp_label` separately tints + emphasizes the label.
    _bold_font = getattr(self, "_bold_font", None)

    @contextmanager
    def _hi(field: str, value):  # noqa: F811 — intentional local shadow
        push_color = not _is_default(field, value)
        if push_color:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        try:
            yield
        finally:
            if push_color:
                imgui.pop_style_color()

    @contextmanager
    def _hi_extras(name: str, value):
        """Like _hi but for MboSuite2pExtras fields — compares against the
        dataclass default. Tints the widget orange when modified.

        Handles both `default=` and `default_factory=` field specs; the
        latter is resolved by invoking the factory once.
        """
        from dataclasses import MISSING

        try:
            f = MboSuite2pExtras.__dataclass_fields__[name]
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:
                try:
                    default = f.default_factory()
                except Exception:
                    default = None
            else:
                default = None
        except KeyError:
            default = None
        push_color = default is not None and value != default
        if push_color:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        try:
            yield
        finally:
            if push_color:
                imgui.pop_style_color()

    def _emp_label(field: str, value, text: str) -> None:
        """Render an external label for a widget — wrapped in a thin box
        with a bold font when `field` is in _IMPORTANT_FIELDS, plain text
        otherwise. Modified-color tint applies when `value` differs from
        the upstream default. Pair with `imgui.<widget>("##field", ...)`
        + `imgui.same_line(0, item_inner_spacing.x)` to replicate imgui's
        default inline-label placement.
        """
        push_color = not _is_default(field, value)
        is_important = field in _IMPORTANT_FIELDS
        if push_color:
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
        if is_important:
            draw_boxed_label(text, font=_bold_font)
        else:
            imgui.text(text)
        if push_color:
            imgui.pop_style_color()

    # self.s2p is a lazy property: returns None when HAS_SUITE2P is False
    # or when the lazy import failed. bail early so the later rows
    # (`self.s2p.do_registration`, etc.) don't raise AttributeError
    # mid-table and leave imgui's table scope open.
    if self.s2p is None:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "Suite2p is not installed."
        )
        imgui.text("Install with:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install mbo_utilities"
        )
        return

    # initialize selection state (timepoints, z-planes, channels)
    max_frames, num_planes, num_channels = _init_s2p_selection_state(self)

    # get output path
    s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")
    has_save_path = bool(s2p_path)

    # button-width constants:
    #   _RUN_W: full-width green Run Suite2p button at the bottom
    #   _BTN_W: small "Open" / "Set slice" entry buttons next to titles
    _RUN_W = 220
    _BTN_W = 90

    # === CURRENT DATASET ===
    _draw_current_dataset_section(self)

    # === OUTPUT FOLDER ===
    # Section title + small folder-icon Browse button + full-width
    # editable path. The text input below is fully editable — typing or
    # pasting works the same as picking via the dialog. Both write to
    # self._s2p_outdir.
    #
    # NOTE: the "Metadata" button was removed — the metadata editor is
    # now accessible only via File menu → "Set Metadata" or Shift+M.
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.text_colored(_SUBSECTION_COLOR, "Select output data folder")
    set_tooltip(
        "Where Suite2p output is written. Each run creates a `suite2p` "
        "subfolder here with one `plane*` directory per z-plane. Type or "
        "paste a path, or click the folder icon to browse.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##s2p_browse"):
        default_dir = s2p_path or str(
            get_last_dir("suite2p_output") or pathlib.Path().home()
        )
        self._s2p_folder_dialog = pfd.select_folder(
            "Select Suite2p output folder", default_dir
        )
    imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
    # editable path field — fills the rest of the row
    imgui.set_next_item_width(-1)
    _path_changed, _new_path = imgui.input_text(
        "##s2p_outdir", self._s2p_outdir or ""
    )
    if _path_changed:
        self._s2p_outdir = _new_path
        s2p_path = _new_path
        has_save_path = bool(_new_path)

    # consume async folder-dialog result on the frame it becomes ready
    if (
        getattr(self, "_s2p_folder_dialog", None) is not None
        and self._s2p_folder_dialog.ready()
    ):
        result = self._s2p_folder_dialog.result()
        if result:
            self._s2p_outdir = str(result)
            set_last_dir("suite2p_output", result)
        self._s2p_folder_dialog = None

    imgui.spacing()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # === DATA SLICING ===
    imgui.text_colored(_SUBSECTION_COLOR, "Data slicing")
    set_tooltip(
        "Restrict which timepoints, z-planes, and channels feed the run. "
        "Click Set slice to define ranges using start:stop[:step] syntax — "
        "e.g. 1:1000:2 selects the first 1000 frames in steps of 2. The "
        "summary line below the button shows the resulting counts.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button("Set slice", imgui.ImVec2(_BTN_W, 0)):
        self._s2p_slicing_open = True
    # frame / plane / channel preview on its own line below the button
    _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels)

    _draw_s2p_slicing_popup(self)

    imgui.spacing()
    imgui.spacing()

    # NOTE: the Run Suite2p button moved to underneath the Pipeline
    # Settings / Data Options buttons (after the closures are defined and
    # the popups have been declared). The closures need `_hi`, `_BTN_W`,
    # and `self.s2p` to all be in scope, which they already are.

    # --- Main settings (closures, drawn inside the unified pipeline-settings
    # popup as two separate row-1 columns) ---
    #
    # LBM-Suite2p-Python Settings — knobs honored by the lbm pipeline
    # (forwarded to lbm_suite2p_python.pipeline / plot_zplane_figures).
    # Sits as the leftmost column in row 1 of the pipeline-settings popup.
    def draw_lsp_settings():
        _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y

        # title is green so individual widgets don't need _mbo() tinting;
        # the title alone signals "this group is mbo/lsp-specific".
        imgui.begin_child("##lsp_box", imgui.ImVec2(-1, 0), _box_flags)
        imgui.text_colored(_MBO_ONLY_COLOR, "LBM-Suite2p-Python Settings")
        imgui.spacing()

        # Top-level lsp knobs (no subsection header — they belong directly
        # under the section title). Run-in-background lives here because
        # it controls how the lsp pipeline is launched, not a suite2p
        # setting.
        if not hasattr(self, "_s2p_background"):
            self._s2p_background = True
        _, self._s2p_background = imgui.checkbox(
            "Run in background", self._s2p_background
        )
        set_tooltip(
            "Run as a separate process that continues after closing the GUI."
        )
        with _hi_extras("accept_all_cells", self.s2p_extras.accept_all_cells):
            _, self.s2p_extras.accept_all_cells = imgui.checkbox(
                "Accept all cells", self.s2p_extras.accept_all_cells
            )
        set_tooltip(
            "Keep every detected ROI regardless of classifier output. "
            "Forwarded to lbm_suite2p_python.pipeline(accept_all_cells=...)."
        )

        imgui.spacing()
        imgui.spacing()

        # --- Parallel processing ---
        # Forwarded to lsp.pipeline(workers=..., skip_volumetric=...).
        # Each worker is a separate Python process; per-plane outputs go
        # to disjoint subdirectories so no inter-worker contention on
        # disk. Cellpose on GPU may OOM with workers > 1 — drop workers
        # or switch cellpose to CPU when that happens.
        imgui.text_colored(_SUBSECTION_COLOR, "Parallel processing")
        # (?) load indicator: dimmed when safe, yellow when borderline,
        # red when oversubscribed (load > logical cores).
        imgui.same_line()
        _load_color, _load_tooltip = _parallel_load_status(
            self.s2p_extras.workers, self.s2p_extras.threads_per_worker
        )
        if _load_color is None:
            imgui.text_disabled("(?)")
        else:
            imgui.text_colored(_load_color, "(?)")
        if imgui.is_item_hovered():
            imgui.set_tooltip(_load_tooltip)

        # Profile presets size workers x threads/worker to this machine in
        # one click; the active profile is derived from the current values,
        # so editing either field below flips the selector to Custom.
        _profiles = _parallel_profiles()
        _profile_names = list(_profiles) + ["Custom"]
        _cur_wt = (self.s2p_extras.workers, self.s2p_extras.threads_per_worker)
        _active = next((n for n, v in _profiles.items() if v == _cur_wt), "Custom")
        imgui.set_next_item_width(INPUT_WIDTH)
        _pchanged, _pidx = imgui.combo(
            "Profile", _profile_names.index(_active), _profile_names
        )
        if _pchanged and _profile_names[_pidx] in _profiles:
            self.s2p_extras.workers, self.s2p_extras.threads_per_worker = (
                _profiles[_profile_names[_pidx]]
            )
        set_tooltip(
            "Sizes workers x threads/worker to this machine. Sequential = "
            "one plane on all cores; Low/Medium/High/Max = more parallel "
            "planes. Custom = manual values below."
        )
        # help behind a popup button — keeps the dense text out of the way
        # but on-screen (imgui clamps popups to the viewport, unlike a tall
        # tooltip that clips off the popup edge).
        if imgui.small_button("Profile help##profile_help_btn"):
            imgui.open_popup("profile_help_popup")
        if imgui.begin_popup("profile_help_popup"):
            imgui.text(_PROFILE_HELP)
            imgui.end_popup()

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("workers", self.s2p_extras.workers):
            _, self.s2p_extras.workers = imgui.input_int(
                "Workers", self.s2p_extras.workers
            )
        set_tooltip("Zplane worker processes. 1 = sequential, 0 = auto.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("threads_per_worker", self.s2p_extras.threads_per_worker):
            _, self.s2p_extras.threads_per_worker = imgui.input_int(
                "Threads/worker", self.s2p_extras.threads_per_worker
            )
        set_tooltip("BLAS / OMP / torch threads per worker. 0 = library default.")
        with _hi_extras("skip_volumetric", self.s2p_extras.skip_volumetric):
            _, self.s2p_extras.skip_volumetric = imgui.checkbox(
                "Skip volumetric figures", self.s2p_extras.skip_volumetric
            )
        set_tooltip("Skip merge, volume_stats, and volume plots.")

        imgui.spacing()
        imgui.spacing()

        # --- Cell Filters ---
        # Each bound is a (checkbox, value) pair. value=0 OR checkbox-off
        # both disable that bound. When the checkbox is off the spinner is
        # greyed out via begin_disabled. build_cell_filters() at run time
        # composes these into lsp's `cell_filters=[{"name": "max_diameter",
        # "min_diameter_um": ..., "max_diameter_um": ...}]` shape.
        imgui.text_colored(_SUBSECTION_COLOR, "Cell Filters")
        for _attr, _label, _tip in (
            (
                "min_diameter_um",
                "Min diameter (um)",
                "Drop ROIs whose fitted diameter is BELOW this in microns.",
            ),
            (
                "max_diameter_um",
                "Max diameter (um)",
                "Drop ROIs whose fitted diameter is ABOVE this in microns.",
            ),
        ):
            _enabled_attr = f"{_attr}_enabled"
            _enabled = getattr(self.s2p_extras, _enabled_attr)
            with _hi_extras(_enabled_attr, _enabled):
                _changed_e, _enabled = imgui.checkbox(f"##{_attr}_en", _enabled)
            if _changed_e:
                setattr(self.s2p_extras, _enabled_attr, _enabled)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            imgui.begin_disabled(not _enabled)
            imgui.set_next_item_width(INPUT_WIDTH)
            _val = getattr(self.s2p_extras, _attr)
            with _hi_extras(_attr, _val):
                _changed_v, _new_v = imgui.input_float(
                    f"{_label}##{_attr}", _val, 0.0, 0.0, "%.1f"
                )
            imgui.end_disabled()
            if _changed_v:
                _new_v = max(0.0, _new_v)
                setattr(self.s2p_extras, _attr, _new_v)
                # auto-uncheck when the user types 0 — both 0 and off mean
                # "ignore this bound", so keep the two signals consistent.
                if _new_v == 0.0:
                    setattr(self.s2p_extras, _enabled_attr, False)
            set_tooltip(_tip)

        # baseline filter — section title (matches "Cell Filters" style).
        # Three independent rejection criteria below; each has its own
        # enable flag, no master gate. Inherits dff window/percentile
        # and correct_neuropil at run time so the filter scores the same
        # trace as the dF/F plot.
        imgui.spacing()
        imgui.spacing()
        imgui.text_colored(_SUBSECTION_COLOR, "Baseline Filter")

        with _hi_extras("baseline_reject_negative_F0", self.s2p_extras.baseline_reject_negative_F0):
            _, self.s2p_extras.baseline_reject_negative_F0 = imgui.checkbox(
                "Reject negative F0", self.s2p_extras.baseline_reject_negative_F0
            )
        set_tooltip(
            "Drop cells whose baseline ever goes below zero — strongest "
            "single predictor of catastrophic dF/F outliers."
        )

        for _attr, _label, _fmt, _tip in (
            (
                "baseline_min_F0_abs",
                "Min F0",
                "%.2f",
                "Drop cells whose median baseline falls below this absolute "
                "fluorescence value. Default 1.0; typical 0.5-5.0.",
            ),
        ):
            _enabled_attr = f"{_attr}_enabled"
            _enabled = getattr(self.s2p_extras, _enabled_attr)
            with _hi_extras(_enabled_attr, _enabled):
                _changed_e, _enabled = imgui.checkbox(f"##{_attr}_en", _enabled)
            if _changed_e:
                setattr(self.s2p_extras, _enabled_attr, _enabled)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            imgui.begin_disabled(not _enabled)
            imgui.set_next_item_width(INPUT_WIDTH)
            _val = getattr(self.s2p_extras, _attr)
            with _hi_extras(_attr, _val):
                _changed_v, _new_v = imgui.input_float(
                    f"{_label}##{_attr}", _val, 0.0, 0.0, _fmt
                )
            imgui.end_disabled()
            if _changed_v:
                _new_v = max(0.0, _new_v)
                setattr(self.s2p_extras, _attr, _new_v)
                if _new_v == 0.0:
                    setattr(self.s2p_extras, _enabled_attr, False)
            set_tooltip(_tip)

        imgui.spacing()
        imgui.spacing()

        # --- Trace Normalization ---
        # mbo-only post-processing (not in upstream suite2p). Forwarded to
        # lsp.pipeline(norm_method=..., dff_window_size=..., dff_percentile=...,
        # dff_smooth_window=...). norm_method picks dF/F vs z-score for
        # norm_traces.npy; window/percentile only apply to dF/F.
        imgui.text_colored(_SUBSECTION_COLOR, "Trace Normalization")
        _norm_methods = ["dff", "zscore"]
        _norm_labels = ["dF/F0", "Z-score"]
        _norm_idx = (
            _norm_methods.index(self.s2p_extras.norm_method)
            if self.s2p_extras.norm_method in _norm_methods else 0
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("norm_method", self.s2p_extras.norm_method):
            _nm_changed, _nm_idx = imgui.combo(
                "Method##norm_method", _norm_idx, _norm_labels
            )
            if _nm_changed:
                self.s2p_extras.norm_method = _norm_methods[_nm_idx]
        set_tooltip(
            "Trace saved to norm_traces.npy.\n\n"
            "dF/F0 = (F - F0) / F0, F0 = rolling-percentile baseline.\n"
            "  Relative fluorescence change; default choice for transient "
            "detection and comparing event sizes within a cell.\n\n"
            "Z-score = (F - mean) / std, per ROI over time.\n"
            "  Unitless, centered at 0; use to compare activity patterns "
            "across cells with different brightness or noise."
        )
        _is_zscore = self.s2p_extras.norm_method == "zscore"
        with _hi_extras("correct_neuropil", self.s2p_extras.correct_neuropil):
            _, self.s2p_extras.correct_neuropil = imgui.checkbox(
                "Correct neuropil", self.s2p_extras.correct_neuropil
            )
        set_tooltip(
            "Use F - coef*Fneu in the normalized trace (default on). Off uses "
            "raw F. Shared with the baseline cell filter so both score the "
            "same trace."
        )
        imgui.begin_disabled(_is_zscore)
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("dff_window_size", self.s2p_extras.dff_window_size):
            _, self.s2p_extras.dff_window_size = imgui.input_int(
                "Window", self.s2p_extras.dff_window_size
            )
        set_tooltip("Frames for rolling percentile baseline (dF/F only, default: 300).")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("dff_percentile", self.s2p_extras.dff_percentile):
            _, self.s2p_extras.dff_percentile = imgui.input_int(
                "Percentile", self.s2p_extras.dff_percentile
            )
        set_tooltip("Percentile for baseline F0 estimation (dF/F only, default: 20).")
        imgui.end_disabled()
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi_extras("dff_smooth_window", self.s2p_extras.dff_smooth_window):
            _, self.s2p_extras.dff_smooth_window = imgui.input_int(
                "Smooth", self.s2p_extras.dff_smooth_window
            )
        set_tooltip("Smooth the normalized trace with a rolling window (0 = disabled).")

        imgui.spacing()
        imgui.spacing()

        # --- Rastermap ---
        # Tri-state stage gate (Skip/Run/Force) on the title row, matching
        # the pipeline-step radio convention used for Registration / ROI
        # Detection. When Run/Force, the Planar and Volumetric checkboxes
        # become editable; their sub-params only render when the parent
        # checkbox is on. build_rastermap_kwargs() composes the unified
        # lsp dict from this state at run time.
        # When rastermap isn't installed, force Skip and clear the
        # sub-modes so the stage can't run; the title goes red and the
        # Skip/Run/Force radios + Planar/Volumetric checkboxes grey out.
        _rm_available = _check_rastermap_available()
        if not _rm_available:
            self.s2p_extras.rastermap_mode = 0
            self.s2p_extras.rastermap_planar = False
            self.s2p_extras.rastermap_volumetric = False

        _rm_title_color = (
            _SUBSECTION_COLOR if _rm_available
            else imgui.ImVec4(0.95, 0.4, 0.4, 1.0)
        )
        imgui.text_colored(_rm_title_color, "Rastermap")
        if not _rm_available:
            set_tooltip(
                "Rastermap not installed. Install with `pip install rastermap`.",
                show_mark=False,
            )
        imgui.same_line()
        _rm_mode = self.s2p_extras.rastermap_mode
        imgui.begin_disabled(not _rm_available)
        if imgui.radio_button("Skip##rastermap_mode", _rm_mode == 0):
            # transitioning to Skip greys out the Planar/Volumetric
            # checkboxes below — also clear them so they don't pop back
            # on if the user later flips Run/Force on again.
            self.s2p_extras.rastermap_mode = 0
            self.s2p_extras.rastermap_planar = False
            self.s2p_extras.rastermap_volumetric = False
        imgui.same_line()
        if imgui.radio_button("Run##rastermap_mode", _rm_mode == 1):
            self.s2p_extras.rastermap_mode = 1
        imgui.same_line()
        if imgui.radio_button("Force##rastermap_mode", _rm_mode == 2):
            self.s2p_extras.rastermap_mode = 2
        imgui.end_disabled()

        _rm_active = self.s2p_extras.rastermap_mode > 0

        # cache-reuse hint as a (?) at the END of the title row so it
        # never displaces the Skip/Run/Force radios in narrow panels.
        set_tooltip(
            "Sorts cells so similar activity is adjacent in the figure "
            "(bands = ensembles, diagonals = sequences). "
            "Run reuses cached model.npy when params match; Force always recomputes."
        )

        imgui.begin_disabled(not _rm_active)

        # Planar (per-plane) — runs inside lsp's plot_zplane_figures
        # after each plane finishes detection. Writes 12_rastermap.png +
        # rastermap_model.npy in each plane dir.
        with _hi_extras("rastermap_planar", self.s2p_extras.rastermap_planar):
            _, self.s2p_extras.rastermap_planar = imgui.checkbox(
                "Planar (per-plane)", self.s2p_extras.rastermap_planar
            )
        set_tooltip(
            "One rastermap per plane (12_rastermap.png each). "
            "Requires `rastermap` installed."
        )
        if self.s2p_extras.rastermap_planar:
            imgui.indent(16)
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi_extras(
                "rastermap_planar_n_clusters",
                self.s2p_extras.rastermap_planar_n_clusters,
            ):
                _, self.s2p_extras.rastermap_planar_n_clusters = imgui.input_int(
                    "n_clusters##rm_planar",
                    self.s2p_extras.rastermap_planar_n_clusters,
                )
            set_tooltip(
                "Superclusters; each row averages ~n_cells/n_clusters neurons. "
                "Default 0 = adaptive (None for <200 cells, 100 otherwise)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi_extras(
                "rastermap_planar_n_pcs",
                self.s2p_extras.rastermap_planar_n_pcs,
            ):
                _, self.s2p_extras.rastermap_planar_n_pcs = imgui.input_int(
                    "n_PCs##rm_planar",
                    self.s2p_extras.rastermap_planar_n_pcs,
                )
            set_tooltip(
                "Principal components for similarity. Default 0 = min(128, "
                "n_cells - 1). Lower (16-32) for <50 cells; rarely needs tuning."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi_extras(
                "rastermap_planar_locality",
                self.s2p_extras.rastermap_planar_locality,
            ):
                _, self.s2p_extras.rastermap_planar_locality = imgui.input_float(
                    "locality##rm_planar",
                    self.s2p_extras.rastermap_planar_locality,
                    0.0, 0.0, "%.2f",
                )
            set_tooltip(
                "How time-local the sort is. 0 = global match anywhere, "
                "1 = strict same-time match. Range 0.0-1.0; "
                "default -1 = adaptive (0.0 for >=200 cells, 0.1 otherwise)."
            )
            imgui.unindent(16)

        # Volumetric — runs once per volume after run_volume() finishes
        # all planes. Calls plot_3d_rastermap_clusters and writes
        # rastermap_3d.png alongside the volume plots.
        with _hi_extras("rastermap_volumetric", self.s2p_extras.rastermap_volumetric):
            _, self.s2p_extras.rastermap_volumetric = imgui.checkbox(
                "Volumetric", self.s2p_extras.rastermap_volumetric
            )
        set_tooltip(
            "One rastermap pooling cells across all planes "
            "(rastermap_3d.png). Each row can now span multiple z-planes."
        )
        if self.s2p_extras.rastermap_volumetric:
            imgui.indent(16)
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi_extras(
                "rastermap_volumetric_n_clusters",
                self.s2p_extras.rastermap_volumetric_n_clusters,
            ):
                _, self.s2p_extras.rastermap_volumetric_n_clusters = imgui.input_int(
                    "n_clusters##rm_vol",
                    self.s2p_extras.rastermap_volumetric_n_clusters,
                )
            set_tooltip(
                "Superclusters across the whole volume. Default 40; "
                "drop to 10-20 for <200 cells, raise to 60-100 for >2000."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi_extras(
                "rastermap_volumetric_n_pcs",
                self.s2p_extras.rastermap_volumetric_n_pcs,
            ):
                _, self.s2p_extras.rastermap_volumetric_n_pcs = imgui.input_int(
                    "n_PCs##rm_vol",
                    self.s2p_extras.rastermap_volumetric_n_pcs,
                )
            set_tooltip(
                "Principal components for similarity. Default 0 = min(200, "
                "n_cells - 1). Only lower with <100 total cells."
            )
            imgui.unindent(16)

        imgui.end_disabled()
        imgui.end_child()  # close LBM box

    # Suite2p Main Settings — vanilla suite2p top-level params
    # (torch_device, tau, fs). Sits to the right of LBM-Suite2p-Python
    # in row 1 of the pipeline-settings popup, above Registration.
    def draw_main_settings():
        _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        imgui.begin_child("##s2p_main_box", imgui.ImVec2(-1, 0), _box_flags)
        imgui.text_colored(_S2P_TITLE_COLOR, "Suite2p Main Settings")
        imgui.spacing()

        # torch device (top-level upstream setting)
        device_options = ["cuda", "cpu", "mps"]
        current_device_idx = (
            device_options.index(self.s2p.torch_device)
            if self.s2p.torch_device in device_options
            else 0
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("torch_device", self.s2p.torch_device):
            device_changed, selected_device_idx = imgui.combo(
                "Torch Device", current_device_idx, device_options
            )
        if device_changed:
            self.s2p.torch_device = device_options[selected_device_idx]
        set_tooltip(
            "GPU device for registration / detection / extraction / dcnv.\n"
            "'cuda' falls back to 'cpu' at runtime if allocation fails."
        )

        imgui.spacing()
        imgui.spacing()

        # tau / fs
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("tau", self.s2p.tau):
            _, self.s2p.tau = imgui.input_float("##tau", self.s2p.tau)
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label("tau", self.s2p.tau, "Tau (s)")
        set_tooltip(
            "Calcium indicator decay timescale in seconds.\n"
            "GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("fs", self.s2p.fs):
            _, self.s2p.fs = imgui.input_float("Sampling Rate (Hz)", self.s2p.fs)
        set_tooltip("Per-plane sampling rate in Hz; drives baseline window sizing.")

        imgui.end_child()  # close Suite2p Main box

    # --- Registration ---
    # body only — Skip/Run/Force is rendered inline on the column title by
    # the popup loop (drives self.s2p.do_registration 0/1/2; force position
    # → force_reg=True at run time).
    #
    # Field ordering follows suite2p.parameters.SETTINGS["registration"]:
    # align_by_chan2, nimg_init, maxregshift, do_bidiphase, bidiphase,
    # batch_size, nonrigid (tree), block_size (tree), smooth_sigma_*,
    # spatial_taper, th_badframes, norm_frames, snr_thresh (tree),
    # subpixel, two_step_registration, reg_tif, reg_tif_chan2.
    def draw_registration_settings():
        # Output binaries — both upstream suite2p options:
        #   keep_movie_raw (db)  → keep data_raw.bin  (unregistered movie)
        #   delete_bin     (io)  → delete data.bin    (registered movie)
        # "Keep Registered Binary" is the inverse-display of delete_bin so
        # the two checkboxes read symmetrically. Tracked with _hi() against
        # upstream defaults (keep_movie_raw=False, delete_bin=False).
        with _hi("keep_movie_raw", self.s2p_db.keep_movie_raw):
            _, self.s2p_db.keep_movie_raw = imgui.checkbox(
                "Keep Raw Binary", self.s2p_db.keep_movie_raw
            )
        set_tooltip(
            "Keep data_raw.bin (unregistered) after processing. "
            "Suite2p db.keep_movie_raw — default False."
        )
        _keep_reg = not self.s2p.delete_bin
        with _hi("delete_bin", self.s2p.delete_bin):
            _kr_changed, _new_keep_reg = imgui.checkbox(
                "Keep Registered Binary", _keep_reg
            )
        if _kr_changed:
            self.s2p.delete_bin = not _new_keep_reg
        set_tooltip(
            "Keep data.bin (registered) after processing. Inverse of "
            "suite2p io.delete_bin — Keep Registered = True maps to "
            "delete_bin=False (the upstream default)."
        )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("align_by_chan", self.s2p.align_by_chan):
            _, self.s2p.align_by_chan = imgui.input_int(
                "Align by chan", self.s2p.align_by_chan
            )
        set_tooltip("Channel index used for alignment (1-based). "
                    "Serialized to settings['registration']['align_by_chan2'] = (align_by_chan == 2).")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("nimg_init", self.s2p.nimg_init):
            _, self.s2p.nimg_init = imgui.input_int("# refImg frames", self.s2p.nimg_init)
        set_tooltip("Number of subsampled frames used to build the reference image.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("maxregshift", self.s2p.maxregshift):
            _, self.s2p.maxregshift = imgui.input_float(
                "Max reg shift", self.s2p.maxregshift
            )
        set_tooltip("Max allowed registration shift, as a fraction of frame max(width, height).")

        # bidirectional phase offset (2P only). The user-provided offset
        # is mutually exclusive with auto-compute: when do_bidiphase is on
        # suite2p ignores the manual value, so we grey the input out to
        # signal that and prevent stale edits.
        with _hi("do_bidiphase", self.s2p.do_bidiphase):
            _, self.s2p.do_bidiphase = imgui.checkbox(
                "Compute bidiphase", self.s2p.do_bidiphase
            )
        set_tooltip(
            "Auto-compute the bidirectional scan-phase offset and apply it "
            "to all frames (2P only)."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        imgui.begin_disabled(self.s2p.do_bidiphase)
        with _hi("bidiphase", self.s2p.bidiphase):
            _, self.s2p.bidiphase = imgui.input_float(
                "Bidiphase offset", self.s2p.bidiphase, 0.0, 0.0, "%.2f"
            )
        imgui.end_disabled()
        set_tooltip(
            "User-provided bidirectional phase offset; applied to all frames "
            "when 'Compute bidiphase' is off."
        )

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("reg_batch_size", self.s2p.reg_batch_size):
            _, self.s2p.reg_batch_size = imgui.input_int("# frames/batch", self.s2p.reg_batch_size)
        set_tooltip("Frames per registration batch — reduce if running out of GPU memory.")

        # nonrigid block sits here in suite2p's source order (between
        # batch_size and smooth_sigma_time). kept as a tree for ergonomics
        # but follows the upstream key order: nonrigid, maxregshiftNR,
        # block_size, then snr_thresh (which is also a nonrigid knob).
        if imgui.tree_node("Non-rigid Registration"):
            with _hi("nonrigid", self.s2p.nonrigid):
                _, self.s2p.nonrigid = imgui.checkbox("Use nonrigid", self.s2p.nonrigid)
            set_tooltip(
                "Split FOV into blocks and compute registration offsets per block."
            )

            imgui.begin_disabled(not self.s2p.nonrigid)

            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("maxregshiftNR", self.s2p.maxregshiftNR):
                _, self.s2p.maxregshiftNR = imgui.input_int(
                    "NR max shift", self.s2p.maxregshiftNR
                )
            set_tooltip("Max pixel shift of block relative to rigid shift.")

            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("block_size_y", self.s2p.block_size_y):
                _, self.s2p.block_size_y = imgui.input_int(
                    "Block height", self.s2p.block_size_y
                )
            set_tooltip(
                "Block height for non-rigid registration (keep multiple of 2/3/5)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("block_size_x", self.s2p.block_size_x):
                _, self.s2p.block_size_x = imgui.input_int(
                    "Block width", self.s2p.block_size_x
                )
            set_tooltip(
                "Block width for non-rigid registration (keep multiple of 2/3/5)."
            )

            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("snr_thresh", self.s2p.snr_thresh):
                _, self.s2p.snr_thresh = imgui.input_float(
                    "NR SNR thresh", self.s2p.snr_thresh
                )
            set_tooltip(
                "If a nonrigid block falls below this, it gets smoothed until above. "
                "1.0 = no smoothing; 1.5 recommended for 1P."
            )

            imgui.end_disabled()
            imgui.tree_pop()

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("smooth_sigma_time", self.s2p.smooth_sigma_time):
            _, self.s2p.smooth_sigma_time = imgui.input_float(
                "Time smoothing", self.s2p.smooth_sigma_time
            )
        set_tooltip("Gaussian smoothing in time (frames) before computing shifts — useful for low-SNR.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("smooth_sigma", self.s2p.smooth_sigma):
            _, self.s2p.smooth_sigma = imgui.input_float(
                "XY smoothing", self.s2p.smooth_sigma
            )
        set_tooltip(
            "Gaussian smoothing sigma (pixels) in XY before registration.\n"
            "  ~1.0 — 2-photon recordings (default 1.15)\n"
            "  3-5  — 1-photon recordings"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("spatial_taper", self.s2p.spatial_taper):
            _, self.s2p.spatial_taper = imgui.input_float(
                "Edge taper", self.s2p.spatial_taper
            )
        set_tooltip(
            "Edge tapering width in pixels (zero out edges before FFT) — "
            "important for vignetted windows. Bump higher for 1-photon "
            "recordings. Upstream default: 3.45."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("th_badframes", self.s2p.th_badframes):
            _, self.s2p.th_badframes = imgui.input_float(
                "Bad frame thresh", self.s2p.th_badframes
            )
        set_tooltip("Threshold for excluding low-quality frames when cropping — smaller excludes more.")
        with _hi("norm_frames", self.s2p.norm_frames):
            _, self.s2p.norm_frames = imgui.checkbox(
                "Normalize frames", self.s2p.norm_frames
            )
        set_tooltip("Normalize frames when detecting shifts.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("subpixel", self.s2p.subpixel):
            _, self.s2p.subpixel = imgui.input_int("Subpixel reg", self.s2p.subpixel)
        set_tooltip("Subpixel precision level (1/subpixel step).")
        with _hi("two_step_registration", self.s2p.two_step_registration):
            _, self.s2p.two_step_registration = imgui.checkbox(
                "##two_step_registration", self.s2p.two_step_registration
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label(
            "two_step_registration",
            self.s2p.two_step_registration,
            "Run reg twice",
        )
        set_tooltip("Perform registration twice for low-SNR data (set keep_movie_raw=True).")
        # NOTE: reg_tif / reg_tif_chan2 widgets removed — same reason as
        # save_mat / save_NWB above. The flags reach settings.registration
        # correctly, but in practice the export step depends on the inner
        # registration_wrapper writing to tif_root_align, which isn't
        # exercised by lbm's run_plane in the current configuration. Dataclass
        # fields remain so old ops round-trips. Re-add when lbm honors them.
        with _hi("do_regmetrics", self.s2p.do_regmetrics):
            _, self.s2p.do_regmetrics = imgui.checkbox(
                "Compute reg metrics", self.s2p.do_regmetrics
            )
        set_tooltip("Compute registration QC metrics (requires ≥1500 frames).")

        # NOTE: the old "1-Photon Registration" tree (do_1Preg, spatial_hp_reg,
        # pre_smooth) was deleted along with the dataclass fields — those
        # keys were dead in modern suite2p. Upstream's guidance for 1-photon
        # data is to raise `smooth_sigma` (3-5) and `spatial_taper` (larger)
        # directly — both already exposed in the main Registration body.

    # --- ROI Detection ---
    # body only — Skip/Run/Force is rendered inline on the column title by
    # the popup loop (drives self.s2p.do_detection 0/1/2; force position
    # → force_detect=True at run time).
    #
    # Layout: shared settings on top (apply to every algorithm), then the
    # algorithm selector, then the matching algorithm-specific block —
    # only the selected branch (cellpose / sparsery / sourcery) renders.
    def draw_roi_detection_settings():
        # shared settings — ordered to match upstream detection top-level
        # source order (denoise/block_size, nbins, bin_size, highpass_time,
        # npix_norm_min/max, max_overlap, soma_crop).
        imgui.text_colored(_SUBSECTION_COLOR, "Shared")
        imgui.spacing()

        # denoise gates the two block_size inputs. per detect.py:185 in
        # suite2p v1.0+: `if settings.get("denoise", False): mov =
        # pca_denoise(mov, block_size=settings["block_size"], ...)`. when
        # denoise=False (upstream default), block_size is unused, so the
        # Y/X inputs grey out.
        with _hi("denoise", self.s2p.denoise):
            _, self.s2p.denoise = imgui.checkbox(
                "Denoise", self.s2p.denoise
            )
        set_tooltip(
            "Run PCA denoising on the binned movie before cell detection. "
            "Suite2p detection.denoise — default False. The block size below "
            "is only used when this is on."
        )
        imgui.begin_disabled(not self.s2p.denoise)
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("det_block_size_y", self.s2p.det_block_size_y):
            _, self.s2p.det_block_size_y = imgui.input_int(
                "Denoise block Y", self.s2p.det_block_size_y
            )
        set_tooltip(
            "Block height for PCA denoising (settings['detection']['block_size'][0])."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("det_block_size_x", self.s2p.det_block_size_x):
            _, self.s2p.det_block_size_x = imgui.input_int(
                "Denoise block X", self.s2p.det_block_size_x
            )
        set_tooltip(
            "Block width for PCA denoising (settings['detection']['block_size'][1])."
        )
        imgui.end_disabled()

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("nbins", self.s2p.nbins):
            _, self.s2p.nbins = imgui.input_int("Max binned frames", self.s2p.nbins)
        set_tooltip("Max binned frames for cell detection (reduce if RAM-limited).")

        # bin_size: None signals upstream to use tau*fs at runtime.
        bin_auto = self.s2p.bin_size is None
        auto_changed, new_bin_auto = imgui.checkbox(
            "Auto bin size (tau*fs)", bin_auto
        )
        set_tooltip(
            "When checked, suite2p picks the cell-detection bin size as "
            "tau*fs at runtime. Uncheck to set an explicit value."
        )
        if auto_changed:
            if new_bin_auto:
                self.s2p.bin_size = None
            else:
                self.s2p.bin_size = max(
                    1, int(round(self.s2p.tau * self.s2p.fs))
                )
        imgui.begin_disabled(self.s2p.bin_size is None)
        imgui.set_next_item_width(INPUT_WIDTH)
        _bin_display = (
            self.s2p.bin_size if self.s2p.bin_size is not None else 0
        )
        with _hi("bin_size", self.s2p.bin_size):
            _bin_changed, _bin_new = imgui.input_int("Bin size", _bin_display)
        if _bin_changed and self.s2p.bin_size is not None:
            self.s2p.bin_size = max(1, _bin_new)
        set_tooltip("settings['detection']['bin_size'] in frames.")
        imgui.end_disabled()

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("highpass_time", self.s2p.highpass_time):
            _, self.s2p.highpass_time = imgui.input_int(
                "##highpass_time", self.s2p.highpass_time
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label("highpass_time", self.s2p.highpass_time, "Highpass time")
        set_tooltip(
            "Running mean subtraction window across bins for temporal high-pass."
        )

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("npix_norm_min", self.s2p.npix_norm_min):
            _, self.s2p.npix_norm_min = imgui.input_float(
                "Min npix norm", self.s2p.npix_norm_min, 0.0, 0.0, "%.2f"
            )
        set_tooltip(
            "Minimum npix_norm filter for ROIs (per-ROI npix normalized "
            "by the highest-variance ROIs' mean npix). Default 0.0."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("npix_norm_max", self.s2p.npix_norm_max):
            _, self.s2p.npix_norm_max = imgui.input_float(
                "Max npix norm", self.s2p.npix_norm_max, 0.0, 0.0, "%.2f"
            )
        set_tooltip("Maximum npix_norm filter for ROIs. Default 100.0.")

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("max_overlap", self.s2p.max_overlap):
            _, self.s2p.max_overlap = imgui.input_float(
                "Max overlap", self.s2p.max_overlap
            )
        set_tooltip("ROIs with more overlap than this fraction are discarded.")

        with _hi("soma_crop", self.s2p.soma_crop):
            _, self.s2p.soma_crop = imgui.checkbox("Soma crop", self.s2p.soma_crop)
        set_tooltip("Crop dendrites from ROI to determine npix_norm and compactness.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # algorithm selector — picks which branch below renders.
        algo_options = ["cellpose", "sparsery", "sourcery"]
        if self.s2p.algorithm not in algo_options:
            self.s2p.algorithm = "cellpose"
        cur_algo_idx = algo_options.index(self.s2p.algorithm)
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("algorithm", self.s2p.algorithm):
            algo_changed, new_algo_idx = imgui.combo(
                "##algorithm", cur_algo_idx, algo_options
            )
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        _emp_label("algorithm", self.s2p.algorithm, "Algorithm")
        if algo_changed:
            self.s2p.algorithm = algo_options[new_algo_idx]
        set_tooltip(
            "Cell-detection algorithm.\n"
            "  cellpose: anatomical detection via deep learning (LBM default)\n"
            "  sparsery: functional detection, sparse activity (upstream default)\n"
            "  sourcery: functional detection, legacy"
        )

        # Warn once (modal) when cellpose is selected without GPU PyTorch:
        # detection silently falls back to CPU, which is far slower.
        if self.s2p.algorithm == "cellpose" and not _has_gpu_torch():
            if not getattr(self, "_cellpose_cpu_warned", False):
                imgui.open_popup("No GPU for cellpose")
                self._cellpose_cpu_warned = True
        else:
            self._cellpose_cpu_warned = False

        imgui.set_next_window_pos(
            imgui.get_main_viewport().get_center(),
            imgui.Cond_.appearing,
            pivot=imgui.ImVec2(0.5, 0.5),
        )
        imgui.set_next_window_size(
            imgui.ImVec2(hello_imgui.em_size(26), 0.0), imgui.Cond_.appearing
        )
        _cp_opened, _ = imgui.begin_popup_modal(
            "No GPU for cellpose", flags=imgui.WindowFlags_.no_saved_settings
        )
        if _cp_opened:
            imgui.text_colored(_LOAD_BAD_COLOR, "No GPU PyTorch detected.")
            imgui.text("Cellpose will run very slowly on the CPU.")
            imgui.spacing()
            _torch_cmd = (
                "uv pip uninstall torch torchvision\n\n"
                "uv pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu126"
            )
            imgui.text_wrapped(
                "You need a version of PyTorch with GPU capabilities. See:"
            )
            _torch_url = "https://pytorch.org/get-started/locally/"
            if imgui.text_link(_torch_url):
                import webbrowser
                webbrowser.open(_torch_url)
            imgui.text_wrapped(
                "for the command that matches your operating system and "
                "CUDA version. For example:"
            )
            imgui.spacing()
            imgui.push_style_color(imgui.Col_.text, _SUBSECTION_COLOR)
            imgui.text_wrapped(_torch_cmd)
            imgui.pop_style_color()
            with _ghost_button():
                if imgui.small_button(f"{fa.ICON_FA_COPY}##torch_gpu_copy"):
                    imgui.set_clipboard_text(_torch_cmd)
            set_tooltip("Copy commands to clipboard", show_mark=False)
            imgui.spacing()
            if imgui.button("OK", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        imgui.spacing()

        if self.s2p.algorithm == "cellpose":
            imgui.text_colored(
                _SUBSECTION_COLOR, "Cellpose"
            )
            imgui.spacing()

            # follows upstream cellpose_settings order:
            # cellpose_model, img, highpass_spatial, flow_threshold,
            # cellprob_threshold. diameter is top-level in upstream but
            # cellpose-specific in practice, so it sits with this branch.
            cellpose_model_buf = self.s2p.cellpose_model or ""
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("cellpose_model", self.s2p.cellpose_model):
                _changed, _new_model = imgui.input_text(
                    "Cellpose model", cellpose_model_buf
                )
            if _changed:
                self.s2p.cellpose_model = _new_model
            set_tooltip(
                "Cellpose model name (e.g. 'cpsam', 'cyto3'). Forwarded to "
                "settings['detection']['cellpose_settings']['cellpose_model']."
            )

            # cellpose image source — direct upstream string
            img_options = ["max_proj / meanImg", "meanImg", "max_proj"]
            if self.s2p.cellpose_img not in img_options:
                self.s2p.cellpose_img = img_options[0]
            cur_img_idx = img_options.index(self.s2p.cellpose_img)
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("cellpose_img", self.s2p.cellpose_img):
                img_changed, new_img_idx = imgui.combo(
                    "##cellpose_img", cur_img_idx, img_options
                )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            _emp_label("cellpose_img", self.s2p.cellpose_img, "Cellpose image")
            if img_changed:
                self.s2p.cellpose_img = img_options[new_img_idx]
            set_tooltip(
                "Image cellpose runs detection on. Upstream allowed values:\n"
                "  max_proj / meanImg  (default; combined)\n"
                "  meanImg             (mean image only)\n"
                "  max_proj            (max projection only)"
            )

            # diameter Y/X with a Y/X lock toggle
            if not hasattr(self, "_s2p_diameter_lock"):
                self._s2p_diameter_lock = True
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("diameter_y", self.s2p.diameter_y):
                dy_changed, dy = imgui.input_float(
                    "Diameter Y", self.s2p.diameter_y
                )
            set_tooltip(
                "Expected cell diameter (Y axis) in pixels. Passed to Cellpose.\n"
                "Upstream default: 12."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("diameter_x", self.s2p.diameter_x):
                dx_changed, dx = imgui.input_float(
                    "Diameter X", self.s2p.diameter_x
                )
            set_tooltip(
                "Expected cell diameter (X axis) in pixels. Passed to Cellpose.\n"
                "Upstream default: 12."
            )
            _, self._s2p_diameter_lock = imgui.checkbox(
                "Lock Y/X", self._s2p_diameter_lock
            )
            set_tooltip(
                "Keep diameter_y and diameter_x in sync when editing either."
            )
            if dy_changed:
                self.s2p.diameter_y = dy
                if self._s2p_diameter_lock:
                    self.s2p.diameter_x = dy
            if dx_changed:
                self.s2p.diameter_x = dx
                if self._s2p_diameter_lock:
                    self.s2p.diameter_y = dx

            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("highpass_spatial", self.s2p.highpass_spatial):
                _, self.s2p.highpass_spatial = imgui.input_float(
                    "Highpass spatial", self.s2p.highpass_spatial
                )
            set_tooltip(
                "Spatial high-pass filtering before Cellpose, as a multiple "
                "of diameter. 0.5 = LBM default."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("flow_threshold", self.s2p.flow_threshold):
                _, self.s2p.flow_threshold = imgui.input_float(
                    "##flow_threshold", self.s2p.flow_threshold
                )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            _emp_label(
                "flow_threshold", self.s2p.flow_threshold, "Flow threshold"
            )
            set_tooltip(
                "Max allowed flow error per mask. LBM default: 0 (flow checking disabled).\n"
                "INCREASE (suite2p default 0.4) if missing masks.\n"
                "DECREASE if cellpose returns ill-shaped masks."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("cellprob_threshold", self.s2p.cellprob_threshold):
                _, self.s2p.cellprob_threshold = imgui.input_float(
                    "##cellprob_threshold", self.s2p.cellprob_threshold
                )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            _emp_label(
                "cellprob_threshold",
                self.s2p.cellprob_threshold,
                "Cellprob threshold",
            )
            set_tooltip(
                "Cell probability threshold for Cellpose. LBM default: -4.\n\n"
                "DECREASE if cellpose is missing masks (or masks too small).\n"
                "INCREASE (suite2p default 0.0) if too many masks / false positives."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _mbo():
                _, self.s2p.cellpose_niter = imgui.input_int(
                    "Cellpose niter", self.s2p.cellpose_niter
                )
            set_tooltip(
                "Cellpose iteration count (model.eval(niter=...)).\n"
                "0 = let cellpose pick based on diameter (default).\n"
                "Bump (e.g. 200) for dense or small cells.\n"
                "MBO-only — composes into cellpose_settings.params at run time."
            )

        elif self.s2p.algorithm == "sparsery":
            imgui.text_colored(
                _SUBSECTION_COLOR, "Sparsery"
            )
            imgui.spacing()

            # functional_chan (mbo extra) + threshold_scaling at top, then
            # upstream sparsery_settings order: highpass_neuropil, max_ROIs,
            # spatial_scale, active_percentile.
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p_db.functional_chan = imgui.input_int(
                "Functional chan", self.s2p_db.functional_chan
            )
            set_tooltip("Channel used for functional ROI extraction (1-based).")
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("threshold_scaling", self.s2p.threshold_scaling):
                _, self.s2p.threshold_scaling = imgui.input_float(
                    "##threshold_scaling", self.s2p.threshold_scaling
                )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            _emp_label(
                "threshold_scaling",
                self.s2p.threshold_scaling,
                "Threshold scaling",
            )
            set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("highpass_neuropil", self.s2p.highpass_neuropil):
                _, self.s2p.highpass_neuropil = imgui.input_int(
                    "Highpass neuropil", self.s2p.highpass_neuropil
                )
            set_tooltip(
                "Highpass filter in pixels on the binned movie to subtract neuropil."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("max_ROIs", self.s2p.max_ROIs):
                _, self.s2p.max_ROIs = imgui.input_int(
                    "Max ROIs", self.s2p.max_ROIs
                )
            set_tooltip("Hard cap on detected ROIs (sparsery only).")
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("spatial_scale", self.s2p.spatial_scale):
                _, self.s2p.spatial_scale = imgui.input_int(
                    "Spatial scale", self.s2p.spatial_scale
                )
            set_tooltip(
                "ROI size scale: 0=auto, 1=6-pixel cells (LBM default), "
                "2=medium, 3=large, 4=very large."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("active_percentile", self.s2p.active_percentile):
                _, self.s2p.active_percentile = imgui.input_float(
                    "Active percentile", self.s2p.active_percentile,
                    0.0, 0.0, "%.2f",
                )
            set_tooltip(
                "Percentile of active pixels in the movie used for thresholding."
            )

        elif self.s2p.algorithm == "sourcery":
            imgui.text_colored(
                _SUBSECTION_COLOR, "Sourcery"
            )
            imgui.spacing()

            # functional_chan (mbo extra) + diameter (top-level upstream,
            # used by sourcery & cellpose) + threshold_scaling at top, then
            # upstream sourcery_settings order: connected, max_iterations,
            # smooth_masks.
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p_db.functional_chan = imgui.input_int(
                "Functional chan", self.s2p_db.functional_chan
            )
            set_tooltip("Channel used for functional ROI extraction (1-based).")

            # diameter Y/X (settings.diameter[0/1]) — used by sourcery and
            # cellpose per upstream's parameters.py.
            if not hasattr(self, "_s2p_diameter_lock"):
                self._s2p_diameter_lock = True
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("diameter_y", self.s2p.diameter_y):
                dy_changed_s, dy_s = imgui.input_float(
                    "Diameter Y", self.s2p.diameter_y
                )
            set_tooltip(
                "Expected cell diameter (Y axis) in pixels. Used by sourcery "
                "and cellpose. Upstream default: 12."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("diameter_x", self.s2p.diameter_x):
                dx_changed_s, dx_s = imgui.input_float(
                    "Diameter X", self.s2p.diameter_x
                )
            set_tooltip(
                "Expected cell diameter (X axis) in pixels. Used by sourcery "
                "and cellpose. Upstream default: 12."
            )
            _, self._s2p_diameter_lock = imgui.checkbox(
                "Lock Y/X##sourcery", self._s2p_diameter_lock
            )
            set_tooltip(
                "Keep diameter_y and diameter_x in sync when editing either."
            )
            if dy_changed_s:
                self.s2p.diameter_y = dy_s
                if self._s2p_diameter_lock:
                    self.s2p.diameter_x = dy_s
            if dx_changed_s:
                self.s2p.diameter_x = dx_s
                if self._s2p_diameter_lock:
                    self.s2p.diameter_y = dx_s

            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("threshold_scaling", self.s2p.threshold_scaling):
                _, self.s2p.threshold_scaling = imgui.input_float(
                    "##threshold_scaling", self.s2p.threshold_scaling
                )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            _emp_label(
                "threshold_scaling",
                self.s2p.threshold_scaling,
                "Threshold scaling",
            )
            set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
            with _hi("connected", self.s2p.connected):
                _, self.s2p.connected = imgui.checkbox(
                    "Connected", self.s2p.connected
                )
            set_tooltip("Keep ROIs fully connected (set False for dendrites).")
            imgui.set_next_item_width(INPUT_WIDTH)
            with _hi("max_iterations", self.s2p.max_iterations):
                _, self.s2p.max_iterations = imgui.input_int(
                    "Max iterations", self.s2p.max_iterations
                )
            set_tooltip("Maximum number of ROI-detection iterations.")
            with _hi("smooth_masks", self.s2p.smooth_masks):
                _, self.s2p.smooth_masks = imgui.checkbox(
                    "Smooth masks", self.s2p.smooth_masks
                )
            set_tooltip("Smooth masks in the final ROI detection pass (sourcery).")

        # NOTE: removed the "Channel 2" sub-section (Chan2 Detection Threshold
        # and Cellpose chan2 checkbox). Their dataclass fields stay so
        # to_dict() keeps emitting them at upstream defaults; resurface
        # widgets here if a multi-channel workflow ever needs them again.

    # --- Signal Extraction ---
    # body only — Skip/Run is rendered inline on the column title by the
    # popup loop (drives bool self.s2p.neuropil_extract). suite2p / lbm
    # don't expose a force-extract kwarg, so no Force option here.
    def draw_signal_extraction_settings():
        with _hi("neuropil_extract", self.s2p.neuropil_extract):
            _, self.s2p.neuropil_extract = imgui.checkbox(
                "Extract Neuropil", self.s2p.neuropil_extract
            )
        set_tooltip(
            "Compute Fneu from neuropil masks. Default: True.\n\n"
            "WARNING: setting this to False crashes upstream suite2p — "
            "create_masks returns neuropil_masks=None but extract_traces "
            "doesn't handle None and raises 'NoneType is not iterable'. "
            "Bug is in suite2p/extraction/extract.py:60. Keep this on "
            "until upstream fixes it."
        )
        with _hi("allow_overlap", self.s2p.allow_overlap):
            _, self.s2p.allow_overlap = imgui.checkbox(
                "Allow Overlap", self.s2p.allow_overlap
            )
        set_tooltip("Allow overlapping ROI pixels during extraction.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("min_neuropil_pixels", self.s2p.min_neuropil_pixels):
            _, self.s2p.min_neuropil_pixels = imgui.input_int(
                "Min Neuropil Pixels", self.s2p.min_neuropil_pixels
            )
        set_tooltip("Minimum neuropil pixels per ROI.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("inner_neuropil_radius", self.s2p.inner_neuropil_radius):
            _, self.s2p.inner_neuropil_radius = imgui.input_int(
                "Inner Neuropil Radius", self.s2p.inner_neuropil_radius
            )
        set_tooltip("Pixels to exclude between ROI and neuropil region.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("lam_percentile", self.s2p.lam_percentile):
            _, self.s2p.lam_percentile = imgui.input_float(
                "Lambda Percentile", self.s2p.lam_percentile, 0.0, 0.0, "%.2f"
            )
        set_tooltip("Percentile of Lambda used for neuropil exclusion.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("snr_threshold", self.s2p.snr_threshold):
            _, self.s2p.snr_threshold = imgui.input_float(
                "SNR Threshold", self.s2p.snr_threshold, 0.0, 0.0, "%.2f"
            )
        set_tooltip("settings['extraction']['snr_threshold']. Default 0.0.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("extract_batch_size", self.s2p.extract_batch_size):
            _, self.s2p.extract_batch_size = imgui.input_int(
                "Extract Batch Size", self.s2p.extract_batch_size
            )
        set_tooltip(
            "Frames per extraction batch (settings['extraction']['batch_size']). "
            "Default 500."
        )

    # --- Deconvolution ---
    # body only — Skip/Run is rendered inline on the column title by the
    # popup loop (drives bool self.s2p.do_deconvolution). suite2p / lbm
    # don't expose a force-deconv kwarg, so no Force option here.
    def draw_deconvolution_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("neuropil_coefficient", self.s2p.neuropil_coefficient):
            _, self.s2p.neuropil_coefficient = imgui.input_float(
                "Neuropil coefficient", self.s2p.neuropil_coefficient
            )
        set_tooltip(
            "Neuropil coefficient for all ROIs (F_corrected = F - coeff * F_neu)."
        )

        # Baseline method as combo box. Upstream spelling is "prctile"
        # (not "constant_percentile"); the lbm_suite2p_python db_settings
        # helper round-trips it to the fork's "constant_prctile" literal
        # at pipeline entry so the fork's dcnv.preprocess still branches.
        baseline_options = ["maximin", "constant", "prctile"]
        # accept legacy values silently; normalize to upstream spelling
        _legacy_map = {
            "constant_percentile": "prctile",
            "constant_prctile": "prctile",
        }
        if self.s2p.baseline in _legacy_map:
            self.s2p.baseline = _legacy_map[self.s2p.baseline]
        current_baseline_idx = (
            baseline_options.index(self.s2p.baseline)
            if self.s2p.baseline in baseline_options
            else 0
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("baseline", self.s2p.baseline):
            baseline_changed, selected_baseline_idx = imgui.combo(
                "Baseline method", current_baseline_idx, baseline_options
            )
        if baseline_changed:
            self.s2p.baseline = baseline_options[selected_baseline_idx]
        set_tooltip(
            "maximin: moving baseline with min/max filters.\n"
            "constant: minimum of Gaussian-filtered trace.\n"
            "prctile: percentile of trace (controlled by 'Baseline percentile')."
        )

        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("win_baseline", self.s2p.win_baseline):
            _, self.s2p.win_baseline = imgui.input_float(
                "Baseline window (s)", self.s2p.win_baseline
            )
        set_tooltip("Window for maximin filter in seconds.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("sig_baseline", self.s2p.sig_baseline):
            _, self.s2p.sig_baseline = imgui.input_float(
                "Baseline sigma (s)", self.s2p.sig_baseline
            )
        set_tooltip("Gaussian filter width in seconds for baseline computation.")
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("prctile_baseline", self.s2p.prctile_baseline):
            _, self.s2p.prctile_baseline = imgui.input_float(
                "Baseline percentile", self.s2p.prctile_baseline
            )
        set_tooltip("Percentile of trace for 'prctile' baseline method.")

    # --- Classification ---
    # body only — no skip/run gate (classification always runs alongside
    # detection upstream; do_detection gates the whole detection→classify
    # block, do_deconvolution gates only OASIS).
    def draw_classification_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        with _hi("preclassify", self.s2p.preclassify):
            _, self.s2p.preclassify = imgui.input_float(
                "Preclassify threshold", self.s2p.preclassify
            )
        set_tooltip(
            "Probability threshold to apply classifier before extraction. "
            "0 = skip pre-classification."
        )

        with _hi("use_builtin_classifier", self.s2p.use_builtin_classifier):
            _, self.s2p.use_builtin_classifier = imgui.checkbox(
                "Use built-in classifier", self.s2p.use_builtin_classifier
            )
        set_tooltip(
            "Forces the shipped suite2p classifier. Only takes effect when "
            "Classifier Path below is empty AND you have a saved "
            "~/.suite2p/classifiers/classifier_user.npy — otherwise suite2p "
            "would pick that user file; this forces the shipped one instead. "
            "Classification runs either way."
        )

        # NOTE: "Accept all cells" moved to the LBM-Suite2p-Python Settings
        # section in the Main column (it's a lsp-pipeline kwarg, not a
        # suite2p classification field — placement matches its real home).

        imgui.spacing()
        imgui.text("Classifier path:")
        path_display = self.s2p.classifier_path if self.s2p.classifier_path else "(none)"
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x - 80)
        imgui.text(path_display)
        imgui.pop_text_wrap_pos()
        imgui.same_line()
        if imgui.button("Browse##classifier"):
            default_dir = (
                str(Path(self.s2p.classifier_path).parent)
                if self.s2p.classifier_path
                else str(Path.home())
            )
            self._classifier_dialog = pfd.open_file(
                "Select classifier file",
                default_dir,
                ["Classifier files", "*.npy *.pkl *.pickle", "All files", "*"],
            )
        set_tooltip("Select custom classifier file (e.g. .npy or .pkl)")

        # handle deferred file-dialog result
        if getattr(self, "_classifier_dialog", None) is not None:
            if self._classifier_dialog.ready():
                result = self._classifier_dialog.result()
                if result:
                    self.s2p.classifier_path = (
                        result[0] if isinstance(result, list) else result
                    )
                self._classifier_dialog = None

    # === ENTRY BUTTONS ===
    # Each section: light-blue title (matching the Pipeline Settings
    # popup convention) + small Open button below it. Data Options is
    # data-specific (scan-phase, axial reg); Pipeline Settings opens the
    # unified pipeline-step parameters popup. The popup itself is
    # declared further down once the per-column closures are in scope.
    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    imgui.text_colored(_SUBSECTION_COLOR, "Data-specific options")
    set_tooltip(
        "Toggles for raw scan-phase correction (bidirectional alignment, "
        "FFT sub-pixel) and axial registration. Available only for raw "
        "MBO scans with multiple z-planes.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button("Open##data_options_btn", imgui.ImVec2(_BTN_W, 0)):
        _popup_states["data_options"] = True
        _do_sizer = getattr(self, "_data_options_sizer", None)
        if _do_sizer is None:
            _do_sizer = PopupAutoSize(
                "Data Options##data_options", auto_resize=False
            )
            self._data_options_sizer = _do_sizer
        _do_sizer.before_open()
        imgui.open_popup("Data Options##data_options")

    imgui.spacing()
    _has_phase_support = getattr(self, "has_raster_scan_support", False)
    _nz = getattr(self, "nz", 1)
    _is_raw = getattr(self, "is_mbo_scan", False)
    _has_z_reg = _nz > 1 and _is_raw
    if _has_phase_support or _has_z_reg:
        imgui.indent(12)
        if _has_phase_support:
            imgui.text(f"Fix Phase: {getattr(self, '_s2p_fix_phase', True)}")
            imgui.text(f"Sub-Pixel (FFT): {getattr(self, '_s2p_use_fft', True)}")
            imgui.text(f"Border (px): {getattr(self, 'border', 0)}")
            imgui.text(f"Max Offset: {getattr(self, 'max_offset', 0)}")
        if _has_z_reg:
            _reg_z = getattr(self, "_register_z", False)
            imgui.text(f"Register Z-Planes: {_reg_z}")
            if _reg_z:
                imgui.text(f"Max frames: {getattr(self, '_axial_max_frames', 200)}")
                imgui.text(f"Max shift (px): {getattr(self, '_axial_max_reg_xy', 30)}")
        imgui.unindent(12)
    else:
        imgui.text_disabled("No options available for this data")

    imgui.spacing()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # parameters and settings + modified parameters share the same group —
    # no separator between them.
    imgui.text_colored(_SUBSECTION_COLOR, "Parameters and settings")
    set_tooltip(
        "All Suite2p and LBM-Suite2p-Python knobs in one popup: Main, "
        "Registration, ROI Detection, Signal Extraction, Deconvolution, "
        "and Classification. Modified parameters appear below in this "
        "panel for a quick at-a-glance summary.",
        align="right",
    )
    imgui.spacing()
    imgui.spacing()
    if imgui.button("Open##pipe_settings_btn", imgui.ImVec2(_BTN_W, 0)):
        _popup_states["pipeline_settings"] = True
        self._pipe_settings_just_opened = True
        # NOTE: before_open() is called in the popup body below, not
        # here. The modified-params preview between this button and the
        # popup body uses begin_child(), which internally calls Begin()
        # and consumes the buffered next-window-pos — so calling
        # before_open() here loses the top-anchor position. Calling it
        # right before begin_popup_modal() guarantees consumption order.
        imgui.open_popup("Pipeline Settings##pipeline_settings_popup")

    imgui.spacing()

    # === MODIFIED PARAMETERS PREVIEW ===
    # Shows fields whose current value differs from the upstream suite2p
    # default. Driven by collect_modified_params() which iterates the
    # _MBO_TO_S2P / _MBO_DB_TO_S2P maps and uses _is_default() for the
    # diff check. Useful right after a settings.npy hydration so the
    # user can see at a glance what's changed.
    _mods = collect_modified_params(self.s2p, self.s2p_db, self.s2p_extras)
    _n_mods = len(_mods)
    imgui.text(f"Modified parameters ({_n_mods})")
    if _mods:
        # copy-all icon next to the title — emits a Python dict literal
        # with each field's default + source pipeline annotated as an
        # inline comment so the snippet is both pasteable and reviewable.
        # source markers (s2p / lsp) match the in-table param-name color:
        # s2p → suite2p settings dict; lsp → lbm_suite2p_python pipeline kwargs.
        imgui.same_line()
        with _ghost_button():
            if imgui.small_button(f"{fa.ICON_FA_COPY}##mod_params_copy_all"):
                _lines = ["{"]
                for _field, _cur, _def, _src in _mods:
                    _lines.append(
                        f"    {_field!r}: {_cur!r},  # {_src}, default: {_def!r}"
                    )
                _lines.append("}")
                imgui.set_clipboard_text("\n".join(_lines))
        set_tooltip(
            "Copy all modified params as a Python dict. Each line tags its "
            "source pipeline (s2p = suite2p settings, lsp = lbm_suite2p_python "
            "kwarg) and the default value, as inline comments.",
            align="right",
        )

        # bounded scroll region — caps height so this section never
        # eats the Run Suite2p button's space when many fields differ.
        _mod_h = min(180, imgui.get_frame_height_with_spacing() * (_n_mods + 2))
        if imgui.begin_child(
            "##mod_params",
            imgui.ImVec2(-1, _mod_h),
            imgui.ChildFlags_.borders,
        ):
            _table_flags = (
                imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.sizing_stretch_prop
            )
            if imgui.begin_table("##mod_params_tbl", 3, _table_flags):
                imgui.table_setup_column(
                    "Parameter", imgui.TableColumnFlags_.width_stretch, 4.0
                )
                imgui.table_setup_column(
                    "Current", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_setup_column(
                    "Default", imgui.TableColumnFlags_.width_stretch, 2.5
                )
                imgui.table_headers_row()
                for _field, _cur, _def, _src in _mods:
                    _cur_s = (
                        f"{_cur:.3g}" if isinstance(_cur, float) else str(_cur)
                    )
                    _def_s = (
                        f"{_def:.3g}" if isinstance(_def, float) else str(_def)
                    )
                    # name color signals pipeline source: s2p = yellow
                    # (matches the suite2p column titles), lsp = green
                    # (matches the LBM-Suite2p-Python section header).
                    _name_color = (
                        _S2P_TITLE_COLOR if _src == "s2p" else _MBO_ONLY_COLOR
                    )
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    text_wrapped_cell(_field, _name_color)
                    imgui.table_set_column_index(1)
                    text_wrapped_cell(_cur_s)
                    imgui.table_set_column_index(2)
                    text_wrapped_cell(
                        _def_s,
                        imgui.get_style_color_vec4(imgui.Col_.text_disabled),
                    )
                imgui.end_table()
        imgui.end_child()
    else:
        imgui.text_disabled("Run pipeline to view parameters")

    imgui.spacing()
    imgui.spacing()

    # === RUN SUITE2P BUTTON ===
    # Wider button than the entry "Open" buttons so it reads as the
    # primary action. Disabled until an output path is set; green style
    # signals "this kicks off the work". Centered horizontally.
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0))

    if not has_save_path:
        imgui.begin_disabled()

    _run_avail = imgui.get_content_region_avail().x
    if _run_avail > _RUN_W:
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + (_run_avail - _RUN_W) * 0.5
        )
    button_clicked = imgui.button("Run Suite2p", imgui.ImVec2(_RUN_W, 0))

    if not has_save_path:
        imgui.end_disabled()

    imgui.pop_style_color(3)

    if not has_save_path and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(
            "Set the output folder above (folder button or paste a path)"
        )

    if button_clicked and has_save_path:
        run_process(self)

    if self._install_error:
        if self._show_red_text:
            imgui.text_colored(imgui.ImVec4(1.0, 0.0, 0.0, 1.0), "Error: lbm_suite2p_python is not installed.")
        if self._show_green_text:
            imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.0, 1.0), "lbm_suite2p_python install success.")
        if self._show_install_button and imgui.button("Install", imgui.ImVec2(_BTN_W, 0)):
            import subprocess
            self.logger.log("info", "Installing lbm_suite2p_python...")
            try:
                subprocess.check_call(["pip", "install", "lbm_suite2p_python"])
                self.logger.log("info", "Installation complete.")
                self._install_error = False
                self._show_red_text = False
                self._show_green_text = True
                self._show_install_button = False
            except Exception as e:
                self.logger.log("error", f"Installation failed: {e}")

    # === Unified Pipeline Settings popup ===
    # two-row layout. row 1 = Main / Registration / ROI Detection (always
    # expanded). row 2 = Classification / Extraction / Deconvolution
    # (collapsing-header per column, collapsed by default — these stages
    # are wired up by the run-time pipeline and rarely need adjusting).
    # each row's columns flex-share that row's full width every frame so
    # resizing the popup re-flows them. the popup's min-size constraint
    # is clamped to the viewport so it can never demand a width larger
    # than the screen (otherwise imgui flickers between min-size and
    # screen-clamp).
    # section schema: (title, draw_fn, mode_kind, mode_attr)
    #   mode_kind:
    #     None = no inline control (Main has no skip/run concept)
    #     "tri"  = Skip/Run/Force, drives an int 0/1/2 on self.s2p
    #     "bool" = Skip/Run only, drives a bool on self.s2p (no Force —
    #              suite2p / lbm don't expose force_extract or force_deconv)
    #   mode_attr: attribute name on self.s2p
    # mode_kind notes:
    # - Main: no skip — global params, not a stage gate.
    # - Registration / ROI Detection: tri-state via int 0/1/2 (force_reg /
    #   force_detect derived at run time).
    # - Classification: NO skip — runs alongside detection upstream;
    #   do_detection gates the whole detect→classify block.
    # - Extraction: NO skip — upstream pipeline_s2p calls
    #   extraction_wrapper unconditionally. neuropil_extract is a knob
    #   inside the body (not a stage gate); setting it False also crashes
    #   upstream's extract_traces (their bug, not ours).
    # - Deconvolution: bool — do_deconvolution gates the OASIS step
    #   cleanly (pipeline_s2p falls through to spks=zeros when False).
    # Registration column stacks two self-titled bordered boxes:
    #   Box 1: Suite2p Main Settings (torch device, tau, fs)
    #   Box 2: Registration (with inline Skip/Run/Force radios on the
    #          title row, mirroring the popup-level title controls used
    #          by non-self-titled columns)
    # The Registration body is wrapped in begin_disabled when do_registration
    # is Skip (0), matching the disable-on-skip behavior of the popup loop.
    def draw_main_above_registration():
        draw_main_settings()

        imgui.spacing()
        imgui.spacing()

        _box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        imgui.begin_child("##s2p_reg_box", imgui.ImVec2(-1, 0), _box_flags)
        imgui.text_colored(_S2P_TITLE_COLOR, "Registration")
        _cur_reg = self.s2p.do_registration
        imgui.same_line()
        if imgui.radio_button("Skip##do_registration_inner", _cur_reg == 0):
            self.s2p.do_registration = 0
        imgui.same_line()
        if imgui.radio_button("Run##do_registration_inner", _cur_reg == 1):
            self.s2p.do_registration = 1
        imgui.same_line()
        if imgui.radio_button("Force##do_registration_inner", _cur_reg == 2):
            self.s2p.do_registration = 2
        imgui.separator()
        imgui.spacing()
        _reg_skip = self.s2p.do_registration == 0
        imgui.begin_disabled(_reg_skip)
        draw_registration_settings()
        imgui.end_disabled()
        imgui.end_child()

    rows = [
        [
            ("LBM-Suite2p-Python", draw_lsp_settings, None, None),
            ("Registration", draw_main_above_registration, "tri", "do_registration"),
            ("ROI Detection", draw_roi_detection_settings, "tri", "do_detection"),
        ],
        [
            ("Classification", draw_classification_settings, None, None),
            ("Extraction", draw_signal_extraction_settings, None, None),
            ("Deconvolution", draw_deconvolution_settings, "bool", "do_deconvolution"),
        ],
    ]
    n_rows = len(rows)

    # worst-case (longest) labels in each column — used to compute the
    # minimum width that fits the input/checkbox + label + a right-aligned
    # (?) marker without clipping. Add new long labels here if you add
    # widgets that push past these. The Registration column now also
    # holds the Suite2p Main box; its "Sampling Rate (Hz)" label is the
    # widest input across the combined contents.
    _WORST_INPUT_LABEL = {
        "LBM-Suite2p-Python": "Max diameter (um)",
        "Registration": "Sampling Rate (Hz)",
        "ROI Detection": "Chan2 Detection Threshold",
        "Classification": "Preclassify threshold",
        "Extraction": "Inner Neuropil Radius",
        "Deconvolution": "Neuropil coefficient",
    }
    _WORST_CHECKBOX_LABEL = {
        "LBM-Suite2p-Python": "Skip volumetric figures",
        "Registration": "Export Registered TIFF",
        "ROI Detection": "Auto bin size (tau*fs)",
        "Classification": "Use built-in classifier",
        "Extraction": "Extract Neuropil",
        "Deconvolution": "",
    }
    # title-line content varies per mode; account for "Title  Skip Run Force"
    _TITLE_RADIO_LABELS = {
        "tri": "Skip Run Force",
        "bool": "Skip Run",
        None: "",
    }
    # columns whose draw_fn already renders its own title(s) — the popup
    # loop skips the column header + separator for these so the label
    # doesn't appear twice, and also skips the inline Skip/Run/Force
    # radios + disabled-on-skip body wrap (the function manages both).
    # LBM-Suite2p-Python renders the green-titled lsp box; Registration
    # stacks Suite2p Main + Registration as two inner bordered boxes.
    _SELF_TITLED_COLUMNS = {"LBM-Suite2p-Python", "Registration"}

    viewport = imgui.get_main_viewport()
    style = imgui.get_style()
    spacing_x = style.item_spacing.x
    window_pad_x = style.window_padding.x
    frame_pad_x = style.frame_padding.x
    qm_w = imgui.calc_text_size("(?)").x
    cb_w = imgui.get_frame_height()  # checkbox box (square frame)

    def _calc_col_natural_w(title: str, mode_kind) -> float:
        """Natural minimum width of a column to fit its widest content
        plus right-aligned (?) marker, without clipping."""
        # widest body row: max(input row, checkbox row)
        inp_label_w = imgui.calc_text_size(_WORST_INPUT_LABEL[title]).x
        cb_label_w = imgui.calc_text_size(_WORST_CHECKBOX_LABEL[title]).x
        body_inp = INPUT_WIDTH + spacing_x + inp_label_w + spacing_x + qm_w
        body_cb = cb_w + spacing_x + cb_label_w + spacing_x + qm_w
        body = max(body_inp, body_cb)
        # title row: "Title  Skip Run Force" (radio buttons)
        title_w = imgui.calc_text_size(title).x
        radios = _TITLE_RADIO_LABELS[mode_kind]
        if radios:
            # radio button = circle + item_inner_spacing + label, per radio,
            # then item_spacing between radios.
            radio_labels = radios.split()
            n_radios = len(radio_labels)
            inner_sp = imgui.get_style().item_inner_spacing.x
            radio_widgets_w = sum(
                cb_w + inner_sp + imgui.calc_text_size(lbl).x
                for lbl in radio_labels
            )
            title_row_w = (
                title_w
                + spacing_x
                + radio_widgets_w
                + spacing_x * max(0, n_radios - 1)  # gaps between radios
            )
        else:
            title_row_w = title_w
        # account for child window's inner padding (borders + frame_padding)
        inner_pad = 2 * frame_pad_x
        # extra breathing room (~a few mm) so right-aligned (?) tooltip
        # markers never clip against the column's right edge.
        _tooltip_pad = 34
        natural = max(body, title_row_w) + inner_pad + 8 + _tooltip_pad
        # Self-titled columns wrap their content in a SECOND nested
        # bordered child (LBM -> ##lsp_box; Registration -> ##s2p_main_box
        # and ##s2p_reg_box). That extra box layer eats another round of
        # border + frame_padding from the inner usable width, so without
        # this bump the "Force" radio and right-aligned (?) markers clip
        # against the inner box's right edge.
        if title in _SELF_TITLED_COLUMNS:
            natural += inner_pad + 8
        return natural

    # natural column widths per row — used both for the static popup
    # width below and for the per-column flex-share inside the row loop.
    col_naturals_per_row = [
        [_calc_col_natural_w(t, mk) for (t, _, mk, _) in row]
        for row in rows
    ]

    # static popup width — derived from the widest row's natural column
    # widths + inter-column spacing + window padding. fixed across frames
    # because column natural widths don't depend on row 2 expansion state.
    row_natural_totals = [
        sum(nat) + spacing_x * (len(nat) - 1)
        for nat in col_naturals_per_row
    ]
    _content_w_static = max(row_natural_totals) + 2 * window_pad_x + 16
    _content_w_static = min(_content_w_static, viewport.size.x * 0.98)

    _just_opened = getattr(self, "_pipe_settings_just_opened", False)
    _vp_size = viewport.size

    # Sizing policy: always_auto_resize lets imgui resize the popup
    # every frame to fit its content — expanding a row-2 collapsing
    # header grows the popup, collapsing shrinks it back, no scrollbar.
    # Lazy-init so reopening across sessions still works if the user
    # closed the popup via the close button (sizer instance survives).
    _pipe_sizer: PopupAutoSize = getattr(self, "_pipe_settings_sizer", None)
    if _pipe_sizer is None:
        _pipe_sizer = PopupAutoSize(
            "Pipeline Settings##pipeline_settings_popup"
        )
        self._pipe_settings_sizer = _pipe_sizer
    # Position is buffered every frame; Cond_.appearing means only the
    # frame the popup transitions hidden→visible actually applies it.
    # Must be called RIGHT before begin_popup_modal so no intervening
    # Begin/BeginChild consumes the buffered next-window-pos.
    _pipe_sizer.before_open()

    # Cap the popup at the viewport so a fully-expanded popup on a
    # small screen still fits. Min width keeps narrow columns readable;
    # auto-resize will respect both bounds.
    imgui.set_next_window_size_constraints(
        imgui.ImVec2(min(_content_w_static, _vp_size.x * 0.98), 200.0),
        imgui.ImVec2(_vp_size.x * 0.98, _vp_size.y * 0.98),
    )

    _popup_flags = _pipe_sizer.flags(imgui.WindowFlags_.no_saved_settings)

    opened, visible = imgui.begin_popup_modal(
        "Pipeline Settings##pipeline_settings_popup",
        p_open=True,
        flags=_popup_flags,
    )
    if opened:
        try:
            if not visible:
                _popup_states["pipeline_settings"] = False
                imgui.close_current_popup()
            else:
                # Render rows directly into the popup (no fixed-height
                # scroll child). This lets the separator + button row
                # sit immediately below the row 2 boxes; any extra
                # vertical space the user opens up by resizing the
                # popup falls below the buttons instead of between
                # the boxes and the separator.
                avail = imgui.get_content_region_avail()

                col_idx = 0
                for row_i, row in enumerate(rows):
                    is_row2 = (row_i == n_rows - 1)
                    cols_in_row = len(row)
                    nat_widths = col_naturals_per_row[row_i]
                    nat_sum = sum(nat_widths)
                    avail_for_cols = (
                        avail.x - spacing_x * (cols_in_row - 1)
                    )
                    # stretch each column proportionally if there's slack;
                    # never below natural (which is what fits content).
                    if avail_for_cols > nat_sum:
                        scale = avail_for_cols / nat_sum
                    else:
                        scale = 1.0
                    for ci, (title, draw_fn, mode_kind, mode_attr) in enumerate(row):
                        if ci > 0:
                            imgui.same_line()
                        col_w = max(1.0, nat_widths[ci] * scale)
                        child_size = imgui.ImVec2(col_w, 0)
                        child_flags = (
                            imgui.ChildFlags_.borders
                            | imgui.ChildFlags_.auto_resize_y
                        )
                        if imgui.begin_child(
                            f"##pipe_col_{col_idx}",
                            child_size,
                            child_flags,
                        ):
                            # row 2 columns render as collapsing headers
                            # (collapsed by default each open). row 1
                            # uses the original title-text path so it's
                            # always visible.
                            # columns whose draw_fn renders its own header
                            # (e.g. one that splits into multiple internally-
                            # labeled boxes) skip the popup-level title +
                            # separator so the label doesn't appear twice.
                            self_titled = title in _SELF_TITLED_COLUMNS
                            if is_row2:
                                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                                # let the Skip/Run gate radios drawn (via
                                # same_line) on top of this full-width header
                                # receive clicks — without this the header
                                # captures them and the radios do nothing.
                                imgui.set_next_item_allow_overlap()
                                imgui.push_style_color(
                                    imgui.Col_.text, imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
                                )
                                expanded = imgui.collapsing_header(
                                    f"{title}##pipe_col_hdr_{col_idx}"
                                )
                                imgui.pop_style_color()
                            elif self_titled:
                                expanded = True
                            else:
                                # title + inline Skip/Run[/Force] on the same line
                                imgui.text_colored(
                                    imgui.ImVec4(1.0, 0.85, 0.4, 1.0), title
                                )
                                expanded = True
                            skip_active = False
                            # self-titled columns render their own radios
                            # and manage their own disabled state inside
                            # the draw_fn; skip the popup-level radios so
                            # the controls don't appear twice.
                            if mode_kind == "tri" and not self_titled:
                                cur = getattr(self.s2p, mode_attr)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Skip##{mode_attr}_title", cur == 0
                                ):
                                    setattr(self.s2p, mode_attr, 0)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Run##{mode_attr}_title", cur == 1
                                ):
                                    setattr(self.s2p, mode_attr, 1)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Force##{mode_attr}_title", cur == 2
                                ):
                                    setattr(self.s2p, mode_attr, 2)
                                skip_active = (
                                    getattr(self.s2p, mode_attr) == 0
                                )
                            elif mode_kind == "bool" and not self_titled:
                                cur = bool(getattr(self.s2p, mode_attr))
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Skip##{mode_attr}_title", not cur
                                ):
                                    setattr(self.s2p, mode_attr, False)
                                imgui.same_line()
                                if imgui.radio_button(
                                    f"Run##{mode_attr}_title", cur
                                ):
                                    setattr(self.s2p, mode_attr, True)
                                skip_active = not bool(
                                    getattr(self.s2p, mode_attr)
                                )

                            if expanded:
                                if not is_row2 and not self_titled:
                                    imgui.separator()
                                imgui.spacing()
                                imgui.begin_disabled(skip_active)
                                # right-align (?) markers across the whole body
                                # so all tooltips line up at the column's right
                                # edge regardless of label length.
                                with tooltip_marks_right():
                                    try:
                                        draw_fn()
                                    except Exception as e:
                                        imgui.text_colored(
                                            imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                                            f"Error: {e}",
                                        )
                                imgui.end_disabled()
                        imgui.end_child()
                        col_idx += 1

                imgui.spacing()
                imgui.separator()
                # Bottom button row: Run Suite2p (green, left) | Defaults
                # (suite2p-yellow, middle) | Close (red, right). Run and
                # Defaults share one width — sized to the wider of the two
                # labels so neither button text overflows; Close stays
                # compact and right-aligned regardless of popup width.
                _reset_label = "Defaults"
                _frame_pad_x = imgui.get_style().frame_padding.x
                _btn_w = imgui.calc_text_size("Run Suite2p").x + 2 * _frame_pad_x
                _close_w = 60
                _pad_x = imgui.get_style().window_padding.x

                # Run Suite2p (green) — kicks off the same run_process used
                # by the column-level button, then dismisses the popup.
                # Disabled when no output path is set, matching the
                # column button.
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0))
                if not has_save_path:
                    imgui.begin_disabled()
                _run_clicked = imgui.button("Run Suite2p##pipeline_settings_run", imgui.ImVec2(_btn_w, 0))
                if not has_save_path:
                    imgui.end_disabled()
                imgui.pop_style_color(3)
                if not has_save_path and imgui.is_item_hovered(
                    imgui.HoveredFlags_.allow_when_disabled
                ):
                    imgui.set_tooltip(
                        "Set output path via Browse or by typing in the path field"
                    )
                if _run_clicked and has_save_path:
                    run_process(self)
                    _popup_states["pipeline_settings"] = False
                    imgui.close_current_popup()

                imgui.same_line()

                # Defaults — suite2p-yellow. Resets BOTH upstream suite2p
                # fields (Suite2pSettings + Suite2pDB; values pulled live
                # from suite2p.parameters.SETTINGS via the schema helper)
                # AND mbo-only fields on MboSuite2pExtras (defaults from
                # the dataclass spec). Upstream bumps a default → the
                # button picks it up automatically.
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.55, 0.45, 0.15, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.70, 0.60, 0.22, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.45, 0.38, 0.12, 1.0))
                if imgui.button(_reset_label, imgui.ImVec2(_btn_w, 0)):
                    from mbo_utilities.gui.widgets.pipelines._s2p_schema import (
                        all_mapped_fields as _all_fields,
                        get_default as _get_default,
                    )
                    for _f in _all_fields():
                        # field may live on Suite2pSettings OR Suite2pDB
                        # (e.g. keep_movie_raw is on the db).
                        if hasattr(self.s2p, _f):
                            _target = self.s2p
                        elif hasattr(self.s2p_db, _f):
                            _target = self.s2p_db
                        else:
                            continue
                        _new = _get_default(_f)
                        _cur = getattr(_target, _f)
                        # coerce upstream default to the current field's
                        # type so widgets (input_int/input_float) don't get
                        # an unexpected type at the next frame.
                        if _new is not None and _cur is not None:
                            _ct = type(_cur)
                            try:
                                if _ct is bool and not isinstance(_new, bool):
                                    _new = bool(_new)
                                elif _ct is int and isinstance(_new, bool):
                                    # bool → int (e.g. do_detection: upstream
                                    # is bool True, mbo stores int 0/1/2)
                                    _new = int(_new)
                                elif _ct is int and isinstance(_new, float):
                                    _new = int(_new)
                                elif _ct is float and isinstance(_new, int) and not isinstance(_new, bool):
                                    _new = float(_new)
                            except (TypeError, ValueError, OverflowError):
                                continue
                        setattr(_target, _f, _new)
                    # mbo-only fields — reset every MboSuite2pExtras field
                    # to its dataclass default. fields with no default (e.g.
                    # those defined via field(default_factory=...)) get a
                    # MISSING sentinel; skip those rather than resetting to
                    # an arbitrary value.
                    import dataclasses as _dc
                    for _ef in _dc.fields(self.s2p_extras):
                        if _ef.default is _dc.MISSING:
                            continue
                        setattr(self.s2p_extras, _ef.name, _ef.default)
                imgui.pop_style_color(3)
                set_tooltip(
                    "Reset every parameter in this popup to its default. "
                    "Suite2p fields pull from suite2p.parameters.SETTINGS; "
                    "LBM-Suite2p-Python fields reset to their dataclass "
                    "defaults. Both groups are tinted orange when modified.",
                    show_mark=False,
                )

                # Legend — small button next to Defaults that opens an
                # explanatory popup describing the color / box conventions
                # used in the parameter columns. samples the same constants
                # and draw_boxed_label helper so the legend tracks palette
                # tweaks automatically.
                imgui.same_line()
                if imgui.button("Legend##pipe_legend", imgui.ImVec2(_btn_w, 0)):
                    imgui.open_popup("Legend##pipe_legend_popup")
                if imgui.begin_popup("Legend##pipe_legend_popup"):
                    imgui.text_colored(_S2P_TITLE_COLOR, "Suite2p parameter")
                    imgui.text_colored(
                        _MBO_ONLY_COLOR, "LBM-Suite2p-Python parameter"
                    )
                    imgui.text_colored(_MODIFIED_COLOR, "Modified from default")
                    draw_boxed_label("Important parameter")
                    imgui.end_popup()

                # Close (red) — right-aligned. `set_cursor_pos_x` snaps
                # to (window_right_edge - button_w - window_padding) so
                # the button hugs the right side regardless of popup width.
                imgui.same_line()
                imgui.set_cursor_pos_x(
                    imgui.get_window_width() - _close_w - _pad_x
                )
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.55, 0.13, 0.13, 1.0))
                imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.65, 0.18, 0.18, 1.0))
                imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.45, 0.10, 0.10, 1.0))
                if imgui.button("Close##pipeline_settings", imgui.ImVec2(_close_w, 0)):
                    _popup_states["pipeline_settings"] = False
                    imgui.close_current_popup()
                imgui.pop_style_color(3)

                if _just_opened:
                    self._pipe_settings_just_opened = False
        finally:
            imgui.end_popup()

    # === Data Options popup (kept separate from pipeline steps) ===
    imgui.set_next_window_size(imgui.ImVec2(350, 0), imgui.Cond_.first_use_ever)
    opened, visible = imgui.begin_popup_modal(
        "Data Options##data_options",
        p_open=True,
        flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
    )
    if opened:
        try:
            if not visible:
                _popup_states["data_options"] = False
                imgui.close_current_popup()
            else:
                _draw_data_options_content(self)
                imgui.spacing()
                imgui.separator()
                if imgui.button("Close##data_opts", imgui.ImVec2(80, 0)):
                    _popup_states["data_options"] = False
                    imgui.close_current_popup()
        finally:
            imgui.end_popup()


def _build_channel_dirname(self, channel: int) -> str:
    """build dimension-tagged dirname for a single channel run."""
    from mbo_utilities.arrays.features import DimensionTag, TAG_REGISTRY

    tags = []
    # channel first (varies across sibling dirs)
    tags.append(DimensionTag(TAG_REGISTRY["C"], start=channel, stop=None, step=1))
    # z-planes
    z_start = getattr(self, "_s2p_z_start", 1)
    z_stop = getattr(self, "_s2p_z_stop", 1)
    z_step = getattr(self, "_s2p_z_step", 1)
    tags.append(DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=z_stop, step=z_step))
    # timepoints last (same across sibling dirs)
    tp = self._s2p_tp_parsed
    if tp and tp.final_indices:
        tp_start = tp.final_indices[0] + 1
        tp_stop = tp.final_indices[-1] + 1
        tags.append(DimensionTag(TAG_REGISTRY["T"], start=tp_start, stop=tp_stop, step=1))
    return "_".join(tag.to_string() for tag in tags)


def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline != "suite2p":
        if self._current_pipeline == "masknmf":
            self.logger.info("Running MaskNMF pipeline (not yet implemented).")
        else:
            self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
        return

    self.logger.debug(f"suite2p settings: {self.s2p}")
    if not _check_lsp_available():
        # Only emit the warning once until the user does something to retry —
        # mashing Run on a missing install otherwise spams the log on every
        # click. The flag is cleared when the install probe finally succeeds.
        if not getattr(self, "_lsp_missing_warned", False):
            self.logger.warning(
                "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
                "`uv pip install lbm_suite2p_python`",
            )
            self._lsp_missing_warned = True
        self._install_error = True
        return
    self._lsp_missing_warned = False

    if not self._install_error:
        # Get selected planes (1-indexed)
        selected_planes = getattr(self, "_selected_planes", None)
        if not selected_planes:
            # Fallback to current plane
            from mbo_utilities.arrays.features import find_slider_name
            names = self.image_widget._slider_dim_names or ()
            z_name = find_slider_name(names, "z")
            try:
                current_z = self.image_widget.indices[z_name] if z_name else 0
            except (IndexError, KeyError):
                current_z = 0
            selected_planes = {current_z + 1}

        self.logger.info(
            f"Running Suite2p pipeline on {len(selected_planes)} plane(s)..."
        )

        # Check if running in background subprocess
        use_background = getattr(self, "_s2p_background", True)

        if use_background:
            # Use ProcessManager to spawn detached subprocesses
            from mbo_utilities.gui.widgets.process_manager import get_process_manager

            pm = get_process_manager()

            # self.fpath is the array's canonical source_path — a single
            # Path that imread can use to reconstruct the full volume.
            input_path = str(self.fpath) if self.fpath else ""

            # get output path
            s2p_path = getattr(self, "_s2p_outdir", "") or getattr(
                self, "_saveas_outdir", ""
            )

            # determine roi
            num_rois = (
                len(self.image_widget.graphics)
                if hasattr(self.image_widget, "graphics")
                else 1
            )
            roi = 1 if num_rois > 1 else None

            # determine selected channels
            c_start = getattr(self, "_s2p_c_start", 1)
            c_stop = getattr(self, "_s2p_c_stop", 1)
            c_step = getattr(self, "_s2p_c_step", 1)
            selected_channels = list(range(c_start, c_stop + 1, c_step))
            multi_channel = len(selected_channels) > 1
            # always pass channel for multi-channel source data (5D needs _ChannelView)
            has_channels = getattr(self, "_s2p_last_num_channels", 1) > 1

            plane_list = sorted(selected_planes)

            for channel in selected_channels:
                # per-channel output subdir when multiple channels selected
                if multi_channel:
                    output_dir = str(Path(s2p_path) / _build_channel_dirname(self, channel))
                else:
                    output_dir = s2p_path

                worker_args = {
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "planes": plane_list,
                    "roi": roi,
                    "num_timepoints": self.s2p_extras.target_timepoints,
                    "settings": self.s2p.to_dict(),
                    "db": self.s2p_db.to_dict(),
                    "fix_phase": self._s2p_fix_phase,
                    "use_fft": self._s2p_use_fft,
                    "channel": channel if (multi_channel or has_channels) else None,
                    "custom_metadata": dict(getattr(self, "_custom_metadata", {})),
                    "tp_indices": (
                        list(self._s2p_tp_parsed.final_indices)
                        if getattr(self, "_s2p_tp_parsed", None) is not None
                        else None
                    ),
                    "selected_planes_0based": [p - 1 for p in plane_list],
                    "register_z": getattr(self, "_register_z", False),
                    "max_frames": getattr(self, "_axial_max_frames", 200),
                    "max_reg_xy": getattr(self, "_axial_max_reg_xy", 30),
                    "s2p_settings": {
                        "keep_raw": self.s2p_db.keep_movie_raw,
                        "keep_reg": (not self.s2p.delete_bin),
                        "force_reg": (self.s2p.do_registration == 2),
                        "force_detect": (self.s2p.do_detection == 2),
                        "accept_all_cells": self.s2p_extras.accept_all_cells,
                        "norm_method": self.s2p_extras.norm_method,
                        "dff_window_size": self.s2p_extras.dff_window_size,
                        "dff_percentile": self.s2p_extras.dff_percentile,
                        "dff_smooth_window": self.s2p_extras.dff_smooth_window,
                        "save_json": self.s2p_extras.save_json,
                        "correct_neuropil": self.s2p_extras.correct_neuropil,
                        "cell_filters": build_cell_filters(
                            self.s2p_extras.min_diameter_um_enabled,
                            self.s2p_extras.min_diameter_um,
                            self.s2p_extras.max_diameter_um_enabled,
                            self.s2p_extras.max_diameter_um,
                            baseline_filter_enabled=self.s2p_extras.baseline_filter_enabled,
                            baseline_reject_negative_F0=self.s2p_extras.baseline_reject_negative_F0,
                            baseline_min_F0_abs_enabled=self.s2p_extras.baseline_min_F0_abs_enabled,
                            baseline_min_F0_abs=self.s2p_extras.baseline_min_F0_abs,
                            baseline_min_F0_rel_enabled=self.s2p_extras.baseline_min_F0_rel_enabled,
                            baseline_min_F0_rel=self.s2p_extras.baseline_min_F0_rel,
                            correct_neuropil=self.s2p_extras.correct_neuropil,
                        ),
                        "rastermap_kwargs": build_rastermap_kwargs(
                            self.s2p_extras
                        ),
                        "force_rastermap": (self.s2p_extras.rastermap_mode == 2),
                        "workers": self.s2p_extras.workers,
                        "threads_per_worker": self.s2p_extras.threads_per_worker,
                        "skip_volumetric": self.s2p_extras.skip_volumetric,
                    },
                }

                if len(plane_list) == 1:
                    description = f"Suite2p plane{plane_list[0]:02d}"
                else:
                    description = f"Suite2p: {len(plane_list)} plane(s)"
                if multi_channel or has_channels:
                    description += f" ch{channel}"
                if roi:
                    description += f" ROI {roi}"

                pid = pm.spawn(
                    task_type="suite2p",
                    args=worker_args,
                    description=description,
                    output_path=output_dir,
                )

                if not pid:
                    self.logger.error(
                        f"Failed to start background process for {description}"
                    )
        else:
            # Use daemon threads (original behavior)
            # determine selected channels
            c_start = getattr(self, "_s2p_c_start", 1)
            c_stop = getattr(self, "_s2p_c_stop", 1)
            c_step = getattr(self, "_s2p_c_step", 1)
            selected_channels = list(range(c_start, c_stop + 1, c_step))
            multi_channel = len(selected_channels) > 1
            has_channels = getattr(self, "_s2p_last_num_channels", 1) > 1

            s2p_path = getattr(self, "_s2p_outdir", "") or getattr(
                self, "_saveas_outdir", ""
            )

            # Pre-extract shared state (upstream-shaped dicts for settings/db)
            s2p_settings_dict = self.s2p.to_dict()
            s2p_db_dict = self.s2p_db.to_dict()
            target_timepoints = self.s2p_extras.target_timepoints
            # keep_raw / keep_reg derived from upstream fields, not stored
            # as duplicates on MboSuite2pExtras.
            keep_raw = self.s2p_db.keep_movie_raw
            keep_reg = (not self.s2p.delete_bin)
            # force_reg / force_detect are derived from the Skip/Run/Force
            # radios — Force == 2.
            force_reg = (self.s2p.do_registration == 2)
            force_detect = (self.s2p.do_detection == 2)
            accept_all_cells = self.s2p_extras.accept_all_cells
            norm_method = self.s2p_extras.norm_method
            dff_window_size = self.s2p_extras.dff_window_size
            dff_percentile = self.s2p_extras.dff_percentile
            dff_smooth_window = self.s2p_extras.dff_smooth_window
            save_json = self.s2p_extras.save_json
            correct_neuropil = self.s2p_extras.correct_neuropil
            cell_filters = build_cell_filters(
                self.s2p_extras.min_diameter_um_enabled,
                self.s2p_extras.min_diameter_um,
                self.s2p_extras.max_diameter_um_enabled,
                self.s2p_extras.max_diameter_um,
                baseline_filter_enabled=self.s2p_extras.baseline_filter_enabled,
                baseline_reject_negative_F0=self.s2p_extras.baseline_reject_negative_F0,
                baseline_min_F0_abs_enabled=self.s2p_extras.baseline_min_F0_abs_enabled,
                baseline_min_F0_abs=self.s2p_extras.baseline_min_F0_abs,
                baseline_min_F0_rel_enabled=self.s2p_extras.baseline_min_F0_rel_enabled,
                baseline_min_F0_rel=self.s2p_extras.baseline_min_F0_rel,
                correct_neuropil=correct_neuropil,
            )
            # daemon-thread path uses run_plane (planar only). pipeline
            # also accepts the unified dict but we don't have a
            # volumetric loop here.
            planar_rastermap_kwargs = build_planar_rastermap_kwargs(
                self.s2p_extras
            )
            # Force == 2 → drop cached model.npy in plane_dir before run_plane
            # so lsp's plot_zplane_figures recomputes; Run reuses the cached
            # model when shape matches.
            force_rastermap = (self.s2p_extras.rastermap_mode == 2)
            fix_phase = getattr(self, "_s2p_fix_phase", False)
            use_fft = getattr(self, "_s2p_use_fft", False)

            if not s2p_path:
                from mbo_utilities.preferences import get_mbo_dirs
                from mbo_utilities.file_io import get_last_savedir_path
                last_savedir = get_last_savedir_path()
                s2p_path = str(Path(last_savedir) if last_savedir else get_mbo_dirs()["data"])

            fpath_str = str(self.fpath) if self.fpath else ""

            # Snapshot the user's selections so the worker can rebuild
            # the OutputMetadata reactive layer (fs/dz reactively scaled
            # by the timepoint and z-plane stride). Without these, the
            # worker would fall back to source values and the resulting
            # ops.npy reports the wrong fs/dz when the user struds the
            # selection.
            tp_indices = (
                list(self._s2p_tp_parsed.final_indices)
                if getattr(self, "_s2p_tp_parsed", None) is not None
                else None
            )
            selected_planes_0based = [p - 1 for p in sorted(selected_planes)]

            # Build list of configuration dicts for each job to completely decouple GUI state
            jobs = []
            for i, _arr in enumerate(self.image_widget.data):
                for channel in selected_channels:
                    if multi_channel:
                        base_out = Path(s2p_path) / _build_channel_dirname(self, channel)
                    else:
                        base_out = Path(s2p_path)

                    for z_plane in sorted(selected_planes):
                        current_z = z_plane - 1

                        if len(self.image_widget.graphics) > 1:
                            plane_dir = base_out / f"plane{z_plane:02d}_roi{i + 1:02d}"
                            roi = i + 1
                        else:
                            plane_dir = base_out / f"plane{z_plane:02d}_stitched"
                            roi = None

                        import copy as _copy
                        config = {
                            "arr": _arr,
                            "arr_idx": i,
                            "z_plane": current_z,
                            "plane": z_plane,
                            "channel": channel if (multi_channel or has_channels) else None,
                            "base_out": base_out,
                            "plane_dir": plane_dir,
                            "roi": roi,
                            "s2p_settings_dict": _copy.deepcopy(s2p_settings_dict),
                            "s2p_db_dict": _copy.deepcopy(s2p_db_dict),
                            "target_timepoints": target_timepoints,
                            "fpath": fpath_str,
                            "keep_raw": keep_raw,
                            "keep_reg": keep_reg,
                            "force_reg": force_reg,
                            "force_detect": force_detect,
                            "accept_all_cells": accept_all_cells,
                            "norm_method": norm_method,
                            "dff_window_size": dff_window_size,
                            "dff_percentile": dff_percentile,
                            "dff_smooth_window": dff_smooth_window,
                            "save_json": save_json,
                            "correct_neuropil": correct_neuropil,
                            "cell_filters": cell_filters,
                            "planar_rastermap_kwargs": planar_rastermap_kwargs,
                            "force_rastermap": force_rastermap,
                            "fix_phase": fix_phase,
                            "use_fft": use_fft,
                            # User-set metadata (e.g. dz from the metadata
                            # editor) lives on parent._custom_metadata, NOT
                            # on arr.metadata. Snapshot it here so the
                            # worker can merge it before computing voxel
                            # size — otherwise the user's z_step is
                            # silently dropped and the source-file dz
                            # (often None for LBM, 1.0 default otherwise)
                            # ends up in ops.npy.
                            "custom_metadata": dict(getattr(self, "_custom_metadata", {})),
                            # User's timepoint selection — 0-based final
                            # indices from TimeSelection. None means all.
                            "tp_indices": tp_indices,
                            # FULL list of all selected planes (0-based)
                            # — needed by OutputMetadata to compute the
                            # z-step factor reactively, even though each
                            # job processes only one plane at a time.
                            "selected_planes_0based": selected_planes_0based,
                            "logger": self.logger
                        }

                        jobs.append(config)

            def run_all_planes_sequential():
                for job_idx, config in enumerate(jobs):
                    desc = f"plane {config['plane']}"
                    if config['channel'] is not None:
                        desc += f" ch{config['channel']}"
                    self.logger.info(
                        f"Processing {desc} ({job_idx + 1}/{len(jobs)})..."
                    )
                    try:
                        _run_plane_worker_thread(config)
                    except Exception as e:
                        self.logger.exception(
                            f"Error processing {desc}: {e}"
                        )
                self.logger.info("Suite2p processing complete.")

            threading.Thread(target=run_all_planes_sequential, daemon=True).start()


def _run_plane_worker_thread(config):
    """
    Decoupled pure worker function for processing planes on a thread.
    Takes a pre-configured dict of snapshot variables to prevent GUI data racing.
    """
    if not _check_lsp_available():
        if config["logger"]:
            config["logger"].error("lbm_suite2p_python is not installed.")
        return

    arr = config["arr"]
    arr_idx = config["arr_idx"]
    current_z = config["z_plane"]
    plane = config["plane"]
    channel = config["channel"]
    base_out = config["base_out"]
    plane_dir = config["plane_dir"]
    roi = config["roi"]

    if not base_out.exists():
        base_out.mkdir(parents=True, exist_ok=True)

    ops_path = plane_dir / "ops.npy"
    lazy_mdata = getattr(arr, "metadata", {}).copy()

    # Merge GUI-set custom metadata (e.g. dz from the metadata editor)
    # into lazy_mdata BEFORE computing voxel size, so the OutputMetadata
    # reactive layer sees the user's value rather than the source file's.
    custom_metadata = config.get("custom_metadata") or {}
    if custom_metadata:
        lazy_mdata.update(custom_metadata)

    Lx = arr.shape[-1]
    Ly = arr.shape[-2]

    from mbo_utilities.metadata import OutputMetadata, get_param

    # Build the output metadata via the reactive layer. fs scales by
    # the timepoint stride, dz scales by the z-plane stride, and every
    # alias (num_timepoints/T/nt/...) gets updated consistently.
    # Anything done by hand here would just re-introduce the kind of
    # alias-drift bugs we just spent a week chasing.
    tp_indices = config.get("tp_indices")
    selected_planes_0based = config.get("selected_planes_0based")

    selections = {}
    if tp_indices is not None:
        selections["T"] = list(tp_indices)
    if selected_planes_0based is not None:
        selections["Z"] = list(selected_planes_0based)

    # source shape/dims must be the uniform 5D TCZYX, so use _shape5d()
    # (arr.shape is the natural rank for BinArray / a 4D _ChannelView).
    source_shape = tuple(arr._shape5d()) if hasattr(arr, "_shape5d") else None
    source_dims = ("T", "C", "Z", "Y", "X")

    # Log raw source values BEFORE scaling — critical diagnostic when
    # the user reports a wrong fs/dz in the output.
    raw_src_fs = get_param(lazy_mdata, "fs")
    raw_src_dz = get_param(lazy_mdata, "dz")
    config["logger"].info(
        f"_run_plane_worker_thread: source fs={raw_src_fs}, dz={raw_src_dz}, "
        f"plane={plane}"
    )

    if raw_src_fs is None:
        config["logger"].warning(
            f"_run_plane_worker_thread: source metadata for plane {plane} has "
            f"NO fs field — reactive scaling cannot compute the output rate. "
            f"ops.npy fs will fall through to lbm_suite2p_python's default "
            f"(10 Hz). To fix: set fs via the metadata editor before running, "
            f"or fix the source TIFF metadata."
        )

    out_meta = OutputMetadata(
        source=lazy_mdata,
        source_shape=source_shape,
        source_dims=source_dims,
        selections=selections,
    )

    # to_dict() gives reactively-scaled fs/dz/dx/dy and consistent
    # timepoint aliases. Add only the per-plane bookkeeping on top.
    md = out_meta.to_dict()

    # Strip fs/dz keys if they're None — otherwise `defaults.update(md)`
    # below would clobber the lbm default with None, which then leaks
    # into ops.npy as either None or write_ops's hardcoded fs=10
    # fallback. Removing the key surfaces the missing-data signal
    # explicitly to the user (they get None in ops.npy, not a fake 10).
    for key in ("fs", "dz"):
        if key in md and md[key] is None:
            md.pop(key)

    md["process_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    md["original_file"] = config["fpath"]
    md["roi_index"] = arr_idx
    md["z_index"] = current_z
    md["plane"] = plane
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["ops_path"] = str(ops_path)
    md["save_path"] = str(plane_dir)
    md["raw_file"] = str((plane_dir / "data_raw.bin").resolve())

    config["logger"].info(
        f"_run_plane_worker_thread: applied reactive metadata -> "
        f"fs={md.get('fs')}, dz={md.get('dz')} "
        f"(t-stride from {len(tp_indices) if tp_indices else 0} indices, "
        f"z-stride from {len(selected_planes_0based) if selected_planes_0based else 0} planes)"
    )

    # Build upstream-shaped (settings, db) dicts from the GUI dataclasses.
    # settings is nested; db is flat. The lbm_suite2p_python patch flattens
    # these to a legacy ops dict at run_plane entry and writes db.npy /
    # settings.npy siblings for every ops.npy write.
    settings_dict = config["s2p_settings_dict"]
    db_dict = config["s2p_db_dict"]

    # Merge reactive metadata (fs/dz/dx/dy plus per-plane bookkeeping)
    # into the right halves. Keys that map to upstream go into the
    # appropriate dict; anything unknown falls through via extra_ops.
    extra_ops: dict = {}
    _settings_top = {"fs", "tau", "diameter"}
    _db_keys = {
        "nplanes", "nchannels", "data_path", "save_path0", "fast_disk",
        "save_folder", "h5py_key", "functional_chan", "nrois",
    }
    for key, value in md.items():
        if key == "fs":
            settings_dict["fs"] = value
        elif key in _settings_top:
            settings_dict[key] = value
        elif key in _db_keys:
            db_dict[key] = value
        else:
            # voxel size (dx/dy/dz), timepoint aliases, bookkeeping
            # (process_timestamp, original_file, roi_index, Ly, Lx,
            # ops_path, save_path, raw_file) — upstream schema has no
            # home so park on the flat ops dict via lbm's merge helper.
            extra_ops[key] = value

    if channel is not None:
        db_dict["functional_chan"] = 1
        # align_by_chan2 False (we're aligning by channel 1)
        settings_dict.setdefault("registration", {})["align_by_chan2"] = False

    from mbo_utilities.writer import imwrite

    plane_dir.mkdir(parents=True, exist_ok=True)

    # Pass `frames=` (1-based) to imwrite when the user has a stride
    # selection. Without this, only the COUNT was honored — the actual
    # stride was silently dropped and ops.npy ended up with raw source
    # fs even though the writer thought it was truncating.
    if tp_indices:
        frames_arg = [i + 1 for i in tp_indices]
    else:
        frames_arg = None

    # imwrite still expects a flat metadata dict. Flatten our (db, settings)
    # pair so the bin-writer's ops header stays consistent.
    try:
        from lbm_suite2p_python.db_settings import db_settings_to_ops
        metadata_flat = db_settings_to_ops(db_dict, settings_dict)
    except ImportError:
        metadata_flat = {**db_dict, **extra_ops}
    metadata_flat.update(extra_ops)

    imwrite(
        arr,
        plane_dir,
        ext=".bin",
        overwrite=True,
        register_z=False,
        planes=plane,
        channels=[channel] if channel is not None else None,
        output_name="data_raw.bin",
        roi=roi,
        metadata=metadata_flat,
        frames=frames_arg,
        num_frames=config["target_timepoints"] if frames_arg is None else None,
        fix_phase=config["fix_phase"],
        use_fft=config["use_fft"],
    )

    from lbm_suite2p_python import run_plane

    raw_file = plane_dir / "data_raw.bin"

    # Write db.npy / settings.npy up-front (the lbm patch also writes them
    # on every ops.npy save, but producing them BEFORE run_plane starts
    # gives the user something to inspect even if the pipeline crashes).
    try:
        import numpy as _np
        _np.save(plane_dir / "db.npy", db_dict, allow_pickle=True)
        _np.save(plane_dir / "settings.npy", settings_dict, allow_pickle=True)
    except Exception as _e:
        config["logger"].warning(f"Could not pre-write db.npy/settings.npy: {_e}")

    config["logger"].info(
        f"Suite2p run — do_detection: {settings_dict.get('run', {}).get('do_detection')}, "
        f"force_detect: {config['force_detect']}"
    )
    config["logger"].info(
        f"Detection params — algorithm: {settings_dict.get('detection', {}).get('algorithm')}, "
        f"diameter: {settings_dict.get('diameter')}"
    )

    # Rastermap Force → drop the cached model so lsp recomputes. lsp's
    # plot_zplane_figures already deletes the rastermap PNG every run,
    # but reuses model.npy when its isort length matches n_accepted.
    if config.get("force_rastermap") and config.get("planar_rastermap_kwargs") is not None:
        cached_model = plane_dir / "model.npy"
        if cached_model.is_file():
            try:
                cached_model.unlink()
                config["logger"].info(
                    f"Force rastermap: removed cached {cached_model.name}"
                )
            except OSError as _e:
                config["logger"].warning(
                    f"Force rastermap: could not remove {cached_model}: {_e}"
                )

    try:
        run_plane(
            input_data=raw_file,
            save_path=plane_dir,
            db=db_dict,
            settings=settings_dict,
            ops=extra_ops if extra_ops else None,
            keep_raw=config["keep_raw"],
            keep_reg=config["keep_reg"],
            force_reg=config["force_reg"],
            force_detect=config["force_detect"],
            accept_all_cells=config.get("accept_all_cells", False),
            norm_method=config.get("norm_method", "dff"),
            dff_window_size=config["dff_window_size"],
            dff_percentile=config["dff_percentile"],
            dff_smooth_window=config["dff_smooth_window"] if config["dff_smooth_window"] > 0 else None,
            save_json=config.get("save_json", False),
            correct_neuropil=config.get("correct_neuropil", True),
            cell_filters=config.get("cell_filters") or None,
            # run_plane's contract: rastermap_kwargs=None → off, anything
            # else → on (empty dict = use lsp's built-in defaults).
            rastermap_kwargs=config.get("planar_rastermap_kwargs"),
        )
        config["logger"].info(
            f"Suite2p processing complete for plane {current_z}, roi {arr_idx}. Results in {plane_dir}"
        )

        stat_file = plane_dir / "stat.npy"
        if stat_file.exists():
            config["logger"].info("Detection succeeded - stat.npy created")
        else:
            config["logger"].warning(
                "Detection did not run - stat.npy not found. Check Suite2p output logs."
            )
    except Exception as e:
        config["logger"].exception(
            f"Suite2p processing failed for plane {current_z}, roi {arr_idx}: {e}"
        )

