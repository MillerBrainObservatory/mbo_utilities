"""Isoview pipeline widget (Consolidate / Correct / Fuse).

Three modes share one widget, gated by the loaded ``IsoviewArray.kind``
and the presence of the optional ``isoview`` package:

- **Consolidate** — collapse a corrected or fused output tree into one
  OME-Zarr group via :func:`mbo_utilities.arrays.isoview.consolidate_isoview`.
  Always available; in-tree, no extra dependency.
- **Correct** — runs the IsoView ``correct_stack`` pipeline against raw
  ``.stack`` data (pixel correction + segmentation + per-camera output).
  Requires ``pip install isoview``.
- **Fuse** — runs the IsoView ``multi_fuse`` pipeline against a corrected
  tree (camera-pair registration + blending). Requires ``pip install isoview``.

Side-panel layout matches Suite2p: dataset summary + output path + mode
picker + small Settings button + a big green Run button. All other
parameters live in a popup formatted like Suite2p's Pipeline Settings.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from imgui_bundle import hello_imgui, icons_fontawesome_6 as fa, imgui

from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    draw_boxed_label,
    selected_button_style,
    set_tooltip,
    text_wrapped_cell,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _dataset_size_bytes,
    _draw_dataset_files_popup,
    _draw_md_field,
    _format_size,
    _truncate_to_width,
)


# Style palette matched to Suite2p's settings panel — colors picked
# from mbo_utilities/gui/widgets/pipelines/settings.py so the two
# pipelines look like they belong to the same family.
_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)         # section title (yellow)
_SUBSECTION_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)   # subtitle (light blue)

# Button sizes — same scheme as Suite2p (_RUN_W primary, _BTN_W secondary).
_RUN_W = 220
_BTN_W = 90

# Soft cap on the Workers input. Real limit is RAM (each worker holds
# a full 3D camera-pair volume), not cores — cap kept below the typical
# logical-CPU count so a fat-finger 32 can't slip in.
_MAX_WORKERS = 16

# Rough per-worker RAM as a multiple of one raw camera volume. Correction
# promotes uint16->float32 and holds the stack + a corrected copy + segmentation
# buffers, peaking ~5x one raw volume; fusion holds two camera volumes + two
# masks + the fused output (~6x, also used by the fuse worker-count RAM cap
# below).
_CORRECT_RAM_PER_WORKER_X = 5.0
_FUSE_RAM_PER_WORKER_X = 6.0


def _iso_volume_gb(arr: Any) -> float:
    """One camera-volume size in GiB from an input array's last 3 dims (Z,Y,X)."""
    try:
        shp = tuple(int(s) for s in arr.shape)
        if len(shp) < 3:
            return 0.0
        itemsize = int(getattr(arr.dtype, "itemsize", 2))
        return (shp[-3] * shp[-2] * shp[-1] * itemsize) / (1024 ** 3)
    except Exception:
        return 0.0


def _default_workers() -> int:
    """Conservative default: half the logical CPUs, capped at 4."""
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu // 2))


def _input_w() -> float:
    """Default width for numeric inputs in the popup."""
    return hello_imgui.em_size(8)


# Compressor choices per output format. zarr uses chunked-codec names
# io.write_volume maps to numcodecs; tif uses TIFF codec names. Fiji/
# Bio-Formats can't read zstd-compressed TIFFs, so deflate leads the tif
# list and is the default applied when the format switches to tif.
_ZARR_CODECS = ["zstd", "gzip", "blosc-zstd", "blosc-lz4", "none"]
_TIFF_CODECS = ["deflate", "zstd", "lzw", "lzma", "none"]


def _codecs_for_format(fmt: str) -> list[str]:
    return _TIFF_CODECS if fmt == "tif" else _ZARR_CODECS


def _default_codec_for_format(fmt: str) -> str:
    return "deflate" if fmt == "tif" else "zstd"


# Worst-case label across all isoview sub-section boxes — used to size
# columns so the right-aligned (?) tooltip marker never clips the right
# edge. Update if a longer label is added.
_WORST_ROW_LABEL = "Compression level"


def _natural_col_w() -> float:
    """Min width per sub-section box: widest input row + (?) + paddings."""
    style = imgui.get_style()
    spacing_x = style.item_spacing.x
    frame_pad_x = style.frame_padding.x
    qm_w = imgui.calc_text_size("(?)").x
    label_w = imgui.calc_text_size(_WORST_ROW_LABEL).x
    body = _input_w() + spacing_x + label_w + spacing_x + qm_w
    inner_pad = 2 * frame_pad_x + 4  # child border + inner frame padding
    return body + inner_pad + 8  # safety margin so (?) doesn't kiss the edge


def _hint(text: str) -> None:
    """Wrapped subtitle/help text in the same blue Suite2p uses."""
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(_SUBSECTION_COLOR, text)
    imgui.pop_text_wrap_pos()


def _text_disabled_wrapped(text: str) -> None:
    """``text_disabled`` that wraps at the current content region edge.
    Plain ``text_disabled`` doesn't wrap, so long captions like the
    Spark JDK warning clip at narrow column widths.
    """
    imgui.push_text_wrap_pos(0.0)
    imgui.text_disabled(text)
    imgui.pop_text_wrap_pos()


@contextmanager
def _green_button():
    """Push the Suite2p green-Run-button color scheme (idle/hover/active)."""
    imgui.push_style_color(imgui.Col_.button,         imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
    imgui.push_style_color(imgui.Col_.button_active,  imgui.ImVec4(0.10, 0.45, 0.10, 1.0))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


_MODE_CONSOLIDATE = "Consolidate"
_MODE_CORRECT = "Correct"
_MODE_FUSE = "Fuse"
_MODE_STITCHER = "BigStitcher XML"

_RUN_LABELS = {
    _MODE_CONSOLIDATE: "Run consolidation",
    _MODE_CORRECT: "Run correct_stack",
    _MODE_FUSE: "Run multi_fuse",
    _MODE_STITCHER: "Export XML",
}

# Forced output-dir prefix per pipeline step. The Output-options field is the
# optional *variant*: empty -> "<raw><prefix>", text -> "<raw><prefix>_<text>"
# (a leading underscore is added if missing). Mirrors isoview's
# config.derive_output_name. Consolidate is excluded (it names a .zarr file).
_MODE_PREFIX = {
    _MODE_CORRECT: ".corrected",
    _MODE_FUSE: ".fused",
    _MODE_STITCHER: ".stitcher",
}

# VW00 orientation profiles for the BigStitcher export. A profile seeds the
# editable rotations + flips in the Orientation box; the resulting ops are
# forwarded to generate_bigstitcher_xml(orientation=...) and baked onto the
# VW00 (angle-0) view only. "None" = no transform (default). Normal/Rotated
# are starting seeds you refine in BigStitcher. Rotations are
# (sign, axis, magnitude-degrees); flips are axes.
_STITCHER_ORIENT_PROFILES: dict[str, dict] = {
    "none": {"rotations": [], "flips": []},
    "default": {"rotations": [("-", "X", 90), ("-", "Z", 90)], "flips": []},
    "rotated": {"rotations": [("-", "X", 90)], "flips": []},
}

# Default per-camera reorientation onto CM00 for a Normal acquisition, as
# editable rotation/flip UI state (matches isoview _CAMERA_TO_CM00_NORMAL:
# CM01 = flip X, CM02 = +90 about Y, CM03 = +90 about Y then flip Z). CM00 is
# the reference (no transform). Pre-selected when raw/.corrected (per-camera).
_CM_ALIGN_DEFAULT: dict[int, dict] = {
    1: {"rotations": [], "flips": ["X"]},
    2: {"rotations": [{"sign": "+", "axis": "Y", "deg": 90}], "flips": []},
    3: {"rotations": [{"sign": "+", "axis": "Y", "deg": 90}], "flips": ["Z"]},
}

# Per-camera CM->CM00 for a Rotated acquisition (every camera mounted 90deg on
# its side). All three verified in the align widget: CM01 = flip Y, CM02 =
# rot X -90, CM03 = rot X +90 then flip Y (the latter equals rot X -90 + flip Z
# as a transform — same matrix, different op decomposition).
_CM_ALIGN_DEFAULT_ROTATED: dict[int, dict] = {
    1: {"rotations": [], "flips": ["Y"]},
    2: {"rotations": [{"sign": "-", "axis": "X", "deg": 90}], "flips": []},
    3: {"rotations": [{"sign": "+", "axis": "X", "deg": 90}], "flips": ["Y"]},
}


def _norm_variant(text: str) -> str:
    """Normalize a user variant: '' -> '' ; 'x' or '_x' -> '_x'."""
    s = (text or "").lstrip("_")
    return f"_{s}" if s else ""

# "Look at these first" — labels rendered in a thin rounded box (bold
# when self.parent._bold_font is available) inside the Parameters popup.
# Mirrors Suite2p's _IMPORTANT_FIELDS treatment.
_ISOVIEW_IMPORTANT_FIELDS: set[str] = {
    # Correct (raw -> correct_stack)
    "segment_threshold",
    # Fuse (corrected -> multi_fuse)
    "blending_method",
    "blending_range",
    "transition_plane",
    "front_flag",
}


def _unwrap_array(arr: Any) -> Any:
    """Peel display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`)
    so we can isinstance-check the underlying IsoviewArray.
    """
    seen: set[int] = set()
    while True:
        if id(arr) in seen:
            return arr
        seen.add(id(arr))
        inner = getattr(arr, "_arr", None)
        if inner is None or inner is arr:
            return arr
        arr = inner


def _isoview_pkg_available() -> bool:
    """True iff the upstream ``isoview`` package is importable.

    Checked via find_spec so the test stays cheap. Required by the
    Correct and Fuse modes; Consolidate works without it.
    """
    return importlib.util.find_spec("isoview") is not None


def maybe_spawn_raw_projections(parent: Any) -> None:
    """Precompute raw XY projections in a background worker when a raw
    isoview dataset is loaded, so the segmentation / dead-pixel previews
    have images to show.

    Idempotent per (session, dataset): spawns at most once per raw dir and
    not while a worker for that dir is already running. The worker itself
    skips projections already on disk, so re-runs are cheap. Does not need
    the upstream ``isoview`` package.
    """
    iw = getattr(parent, "image_widget", None)
    if iw is None or not getattr(iw, "data", None):
        return
    arr = _unwrap_array(iw.data[0])
    from mbo_utilities.arrays.isoview import IsoviewArray
    if not isinstance(arr, IsoviewArray) or arr.kind != "raw":
        return
    raw_dir = str(arr.scan_root)

    spawned = getattr(parent, "_iso_raw_proj_spawned", None)
    if spawned is None:
        spawned = parent._iso_raw_proj_spawned = set()
    if raw_dir in spawned:
        return

    from mbo_utilities.gui.widgets.process_manager import get_process_manager
    pm = get_process_manager()
    for p in pm.get_all():
        if (
            p.task_type == "isoview_raw_projections"
            and (p.args or {}).get("raw_dir") == raw_dir
            and p.is_alive()
        ):
            spawned.add(raw_dir)
            return

    pid = pm.spawn(
        task_type="isoview_raw_projections",
        args={"raw_dir": raw_dir},
        description=f"Raw projections: {Path(raw_dir).name}",
        output_path=raw_dir,
    )
    if pid:
        spawned.add(raw_dir)
        if hasattr(parent, "logger"):
            parent.logger.info(f"raw projections worker spawned PID {pid}")


def maybe_refresh_raw_projections(parent: Any) -> None:
    """Rebuild the preview widget list once a background raw-projections
    worker finishes, so the Projections widget appears without reloading.

    The widget list is built only on dataset load; a worker that writes
    projections afterwards leaves :class:`ProjectionsViewer` excluded
    until something refreshes. Runs once per raw dir.
    """
    spawned = getattr(parent, "_iso_raw_proj_spawned", None)
    if not spawned:
        return
    refreshed = getattr(parent, "_iso_raw_proj_refreshed", None)
    if refreshed is None:
        refreshed = parent._iso_raw_proj_refreshed = set()
    pending = spawned - refreshed
    if not pending:
        return

    from mbo_utilities.gui.widgets.process_manager import get_process_manager
    pm = get_process_manager()
    procs = pm.get_all()
    for raw_dir in list(pending):
        alive = any(
            p.task_type == "isoview_raw_projections"
            and (p.args or {}).get("raw_dir") == raw_dir
            and p.is_alive()
            for p in procs
        )
        if alive:
            continue
        refreshed.add(raw_dir)
        if hasattr(parent, "_refresh_widgets"):
            parent._refresh_widgets()


def _available_modes(arr: Any) -> list[str]:
    """Modes the widget can offer for ``arr``.

    Filtered by both ``kind`` and whether the upstream ``isoview``
    package is installed (Correct/Fuse need it; Consolidate doesn't).
    """
    if arr is None or not hasattr(arr, "kind"):
        return []
    modes: list[str] = []
    has_pkg = _isoview_pkg_available()
    if arr.kind == "raw":
        if has_pkg:
            modes.append(_MODE_CORRECT)
            modes.append(_MODE_STITCHER)
    elif arr.kind == "corrected":
        if has_pkg:
            modes.append(_MODE_FUSE)
            modes.append(_MODE_STITCHER)
        modes.append(_MODE_CONSOLIDATE)
    elif arr.kind == "fused":
        if has_pkg:
            modes.append(_MODE_STITCHER)
        modes.append(_MODE_CONSOLIDATE)
    return modes


class IsoviewPipelineWidget(PipelineWidget):
    """Run IsoView pipelines (correct / fuse / consolidate)."""

    name = "Isoview"
    install_command = "uv pip install isoview"
    # The widget itself is always available — Consolidate runs in-tree.
    # The Correct/Fuse modes self-gate on isoview package presence.
    is_available = True

    @classmethod
    def applies_to(cls, arr: Any) -> bool:
        if arr is None:
            return False
        from mbo_utilities.arrays.isoview import IsoviewArray
        underlying = _unwrap_array(arr)
        return isinstance(underlying, IsoviewArray) and underlying.kind in (
            "raw", "corrected", "fused",
        )

    def __init__(self, parent: Any):
        super().__init__(parent)

        # mode state (persisted across redraws within session)
        self._selected_mode: str | None = None
        self._initialized_for_scan_root: Path | None = None
        self._last_status: str = ""

        # Shared output knobs (used by Correct and Fuse; Consolidate has
        # its own _consolidate_* set because its output is a single .zarr
        # file, not a directory tree).
        self._output_dir: str = ""
        self._output_format: str = "zarr"
        self._compression: str = "zstd"
        self._compression_level: int = 3
        self._overwrite: bool = False
        self._workers: int = _default_workers()
        self._pyramid: bool = True
        self._pyramid_max_layers: int = 4

        # Consolidate-mode state
        self._consolidate_output_path: str = ""
        self._consolidate_pyramid: bool = True
        self._consolidate_pyramid_max_layers: int = 4
        self._consolidate_compressor: str = "zstd"
        self._consolidate_compression_level: int = 3
        self._consolidate_overwrite: bool = False

        # BigStitcher-export state. The destination is derived
        # (sibling of the raw root, suffix=".stitcher") so we only show
        # the suffix + a filename preview.
        self._stitcher_output_path: str = ""
        self._stitcher_suffix: str = ""

        # Lay out tiles using acquisition stage positions (stage_x/y/z from
        # each SPM's XML) as a per-tile Translation. Off by default — tiles
        # land at the world origin and you'd run BigStitcher's "Move Tiles
        # To Regular Grid" / registration instead.
        self._stitcher_bake_tile_positions: bool = True

        # Reverse each tile's content in Z (camera scans -Z, so stored planes
        # run opposite to the baked Z stride). On: adjacent z-blocks join
        # contiguously (first plane of the lower block meets the last of the
        # upper). Forwarded to generate_bigstitcher_xml(reverse_z=...).
        self._stitcher_reverse_z: bool = True

        # Rotated only: rotate content upright to match the tile grid. Tilts the
        # dataset vs world axes (BDV view-rotation swings it); off keeps cameras
        # in CM00's native frame, orient the whole dataset in BigStitcher.
        self._stitcher_upright: bool = True

        # Link the existing .corrected zarrs in the dataset.xml instead of
        # writing a converted copy (no conversion step). On by default; uncheck
        # to extract a converted copy. Forwarded to
        # generate_bigstitcher_xml(link_existing=...).
        self._stitcher_link_existing: bool = True

        # Per-camera export from .corrected (comma-separated CM indices, e.g.
        # "0,2"). Empty = export fused VW00/VW90 pairs. Forwarded to
        # generate_bigstitcher_xml(cameras=...).
        self._stitcher_cameras: str = ""

        # BDV zarr store version: 2 (gzip, classic chunks) or 3 (sharded +
        # zstd, for the new BigStitcher ZarrV3 reader). Forwarded to
        # generate_bigstitcher_xml(zarr_version=...).
        self._stitcher_zarr_version: int = 2

        # Per-target reorientation seeds for the BigStitcher export. Targets:
        # "VW00"/"VW90" (view-level) and "CM00".."CM03" (per-camera overrides,
        # for the .corrected per-camera export). Each value is
        # {"rotations": [{sign,axis,deg}...], "flips": [axis...]}, composed to
        # op-lists and forwarded as orientation / orientation_vw90 /
        # camera_orientations.
        self._stitcher_orient: dict[str, dict] = {}
        self._stitcher_orient_target: str = "VW00"
        # Global orientation profile (acquisition mounting), seeded from the
        # XML camera_orientation and cascaded to every target. Per-target
        # rotations/flips below override it.
        self._stitcher_orient_profile: str = "default"

        # Per-step output suffix shared by correct/fuse/stitcher dirs.
        # Empty string yields the canonical names (.corrected, .fused,
        # .stitcher); a non-empty value is appended as `_<suffix>`.
        self._fuse_output_suffix: str = ""

        # Consolidate-mode filename suffix (relative to scan_root.name).
        # Re-derived per dataset in :meth:`_ensure_defaults` so it
        # reflects the kind that was loaded.
        self._consolidate_suffix: str = "_isoview-corrected.zarr"

        # Correct-mode state
        self._correct_output_suffix: str = ""
        self._correct_segment_mode: int = 1
        self._correct_apply_seg_mask: bool = False
        self._correct_do_tenengrad: bool = False
        self._correct_background_percentile: float = 5.0
        self._correct_median_kernel_size: int = 3
        self._correct_median_kernel_enabled: bool = True
        self._correct_mask_percentile: float = 1.0
        self._correct_subsample_factor: int = 100
        self._correct_gauss_kernel: int = 5
        self._correct_gauss_sigma: float = 2.0
        self._correct_segment_threshold: float = 0.4
        self._correct_splitting: int = 10

        # Fuse-mode state. The fields below are per-view (one entry per
        # camera-pair / view ID in ``arr.metadata.camera_view_map``).
        # ``_fuse_view_params`` is a dict-of-dicts keyed by view ID;
        # ``_fuse_active_view`` tracks which view the editor controls
        # are currently bound to. Both initialized lazily in
        # :meth:`_ensure_defaults` once the dataset's view list is known.
        self._fuse_output_suffix: str = ""  # appended to <method> folder name
        self._fuse_view_params: dict[int, dict] = {}
        self._fuse_view_ids: list[int] = []
        self._fuse_active_view: int | None = None

        # Fuse background subtraction — dataset-wide (scalar ProcessingConfig
        # fields, not per-view). "pooled" = one floor from both cameras
        # (MATLAB dataType=0); "per_camera" = each camera's own floor
        # (MATLAB dataType=1).
        self._fuse_subtract_background: bool = True
        self._fuse_background_mode: str = "pooled"
        self._fuse_background_percentile: float = 5.0

        # Microscope override (all modes). -1 = auto from XML.
        self._mic_overrides_enabled: bool = False
        self._mic_pixel_spacing_z: float = -1.0
        self._mic_objective_mag: float = -1.0
        self._mic_pixel_spacing_camera: float = -1.0

        # Popup visibility — set True to open on next frame.
        self._show_settings_popup: bool = False
        self._popup_just_opened: bool = False
        # Crop-window handoff: when the user clicks Edit inside the
        # Parameters popup we close the popup and open the floating
        # crop window. Reopen the popup when the crop window closes.
        self._reopen_settings_after_crop: bool = False
        self._crop_was_open: bool = False

        # Data slicing (timepoints). Initialized on first redraw per dataset
        # via _init_iso_selection_state. Attribute names follow the suite2p
        # convention used by `draw_selection_table` (_iso_tp_*).
        self._iso_slicing_open: bool = False
        self._iso_last_max_tp: int = 0
        self._iso_last_fpath: str = ""

        # camera (C-axis) subset for correction. None = all cameras.
        self._iso_selected_cameras: set[int] | None = None

        # Files popup lazy state
        self._iso_files_sizer: PopupAutoSize | None = None
        self._iso_slicing_sizer: PopupAutoSize | None = None

    def _get_array(self) -> Any | None:
        iw = getattr(self.parent, "image_widget", None)
        if iw is None or not iw.data:
            return None
        return _unwrap_array(iw.data[0])

    @staticmethod
    def _default_fuse_view_params() -> dict:
        """Per-view defaults that seed the Fuse editor for a new dataset.

        Mirrors the prior scalar defaults so behavior is unchanged when
        a user never touches the View selector.
        """
        return {
            "blending_method": "geometric",
            "blending_range": 4,
            "transition_plane": -1,  # -1 = None (center) in the UI
            "front_flag": 1,
            "flip_z": False,
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": 0,
            "search_x_start": -50,
            "search_x_stop": 50,
            "search_x_step": 10,
            "search_y_start": -50,
            "search_y_stop": 50,
            "search_y_step": 10,
        }

    def _fv(self, name: str):
        """Read ``name`` from the active view's Fuse params dict."""
        view = self._fuse_active_view
        if view is None or view not in self._fuse_view_params:
            return self._default_fuse_view_params()[name]
        return self._fuse_view_params[view][name]

    def _set_fv(self, name: str, value) -> None:
        """Write ``name`` into the active view's Fuse params dict."""
        view = self._fuse_active_view
        if view is None:
            return
        params = self._fuse_view_params.setdefault(
            view, self._default_fuse_view_params(),
        )
        params[name] = value

    def _ensure_defaults(self, arr: Any) -> None:
        """Compute per-dataset defaults once. Re-runs when the loaded
        IsoviewArray's scan_root changes (different dataset opened).
        """
        sr = Path(arr.scan_root)
        if self._initialized_for_scan_root == sr:
            return
        self._initialized_for_scan_root = sr

        # Seed per-view Fuse params from the array's camera_view_map.
        # Fallback view set {0, 90} matches isoview's default (VW00/VW90).
        meta = getattr(arr, "metadata", None) or {}
        cv = meta.get("camera_view_map") or {0: 0, 1: 0, 2: 90, 3: 90}
        view_ids = sorted({int(v) for v in cv.values()}) or [0, 90]
        self._fuse_view_ids = view_ids
        self._fuse_view_params = {
            vid: self._default_fuse_view_params() for vid in view_ids
        }
        if (
            self._fuse_active_view is None
            or self._fuse_active_view not in view_ids
        ):
            self._fuse_active_view = view_ids[0]

        # Default consolidate output: drop a .zarr sibling next to the
        # input scan_root. Suffix is kind-aware so the result for a
        # corrected SPM00 lands at .corrected/SPM00_isoview-corrected.zarr
        # and a fused method dir lands at <method>_isoview-fused.zarr.
        self._consolidate_suffix = f"_isoview-{arr.kind}.zarr"
        self._consolidate_output_path = str(
            sr.parent / f"{sr.name}{self._consolidate_suffix}"
        )

        # BigStitcher destination is derived live in _current_output_path
        # from the loaded tree + suffix box (no raw-root dependency), so
        # nothing to seed here.

        # Correct / Fuse output_dir defaults:
        #   raw       → leave EMPTY so isoview's ProcessingConfig derives
        #               output_dir from input_dir + .corrected[_<suffix>].
        #               Pre-filling here freezes a stale ".corrected"
        #               path that overrides the user's suffix at submit
        #               time (isoview honors output_dir when set and
        #               ignores output_suffix).
        #   corrected → .corrected/ root (parent of scan_root). The
        #               fuse task reads from here; it doesn't write a
        #               new corrected tree, so no suffix conflict.
        if arr.kind == "corrected":
            self._output_dir = str(sr.parent)
        else:
            self._output_dir = ""

        # Seed the global BigStitcher orientation profile from the acquisition
        # mounting (XML camera_orientation): Normal -> default, Rotated ->
        # rotated. Cascades to every target; per-target edits override.
        self._apply_orient_profile_all(self._metadata_orient_profile(arr))

        # auto-select the most natural mode for this kind
        modes = _available_modes(arr)
        if self._selected_mode not in modes:
            if arr.kind == "corrected" and _MODE_FUSE in modes:
                self._selected_mode = _MODE_FUSE
            elif modes:
                self._selected_mode = modes[0]
            else:
                self._selected_mode = None

    def _output_path_attr(self) -> str:
        """Which instance attribute holds the active mode's output path."""
        if self._selected_mode == _MODE_CONSOLIDATE:
            return "_consolidate_output_path"
        if self._selected_mode == _MODE_STITCHER:
            return "_stitcher_output_path"
        return "_output_dir"

    @staticmethod
    def _iso_input_tree(arr: Any) -> "Path | None":
        """The .corrected*/.fused* tree the array was loaded from.

        Matches custom suffixes (e.g. ``.fused_v2``) by accepting a
        ``.corrected``/``.fused`` token followed by end-of-name or ``_``.
        """
        scan_root = Path(arr.scan_root)
        for ancestor in (scan_root, *scan_root.parents):
            n = ancestor.name
            for suf in (".corrected", ".fused"):
                idx = n.find(suf)
                if idx >= 0:
                    rest = n[idx + len(suf):]
                    if rest == "" or rest.startswith("_"):
                        return ancestor
        return None

    @staticmethod
    def _iso_raw_stem(tree: "Path") -> str:
        """Raw dataset stem of a tree dir, e.g. ``zebrafish.fused_v2`` -> ``zebrafish``."""
        from mbo_utilities.arrays.isoview.array import (
            _CORRECTED_TAIL_RE, _FUSED_TAIL_RE,
        )
        stem = tree.name
        for rx in (_CORRECTED_TAIL_RE, _FUSED_TAIL_RE):
            m = rx.search(stem)
            if m:
                return stem[: m.start()]
        return stem

    @staticmethod
    def _iso_tree_suffix(tree: "Path") -> str:
        """Variant suffix carried by a tree dir, e.g. ``zebrafish.fused_v2`` -> ``v2``."""
        n = tree.name
        for pre in (".corrected", ".fused"):
            idx = n.find(pre)
            if idx >= 0:
                return n[idx + len(pre):].lstrip("_")
        return ""

    def _iso_stitcher_dest(self, arr: Any) -> "Path | None":
        """BigStitcher output dir = ``<rawstem>.stitcher[_<variant>]``, where
        the variant is the Output-options field (empty -> bare ``.stitcher``).
        Matches isoview's ``derive_output_name(raw, ".stitcher", variant)``.
        Works without the raw root."""
        # raw datasets export straight from the acquisition root, which has
        # no .corrected/.fused ancestor for _iso_input_tree to match.
        if getattr(arr, "kind", None) == "raw":
            tree = Path(arr.scan_root)
        else:
            tree = self._iso_input_tree(arr)
        if tree is None:
            return None
        stem = self._iso_raw_stem(tree)
        return tree.parent / f"{stem}.stitcher{_norm_variant(self._stitcher_suffix)}"

    def _current_output_path(self) -> str:
        # STITCHER has no user path field — its destination is always
        # derived live from the input tree + suffix box, so editing the
        # suffix updates the path and corrected-only datasets (raw root
        # deleted) still resolve a destination.
        if self._selected_mode == _MODE_STITCHER:
            arr = self._get_array()
            dest = self._iso_stitcher_dest(arr) if arr is not None else None
            return str(dest) if dest is not None else ""
        explicit = getattr(self, self._output_path_attr(), "") or ""
        if explicit:
            return explicit
        # CORRECT mode against raw data intentionally leaves _output_dir
        # empty so isoview's ProcessingConfig derives the output from
        # input_dir + .corrected[_<output_suffix>]. Mirror that derivation
        # here so the Run-tab gate doesn't block on the empty string.
        if self._selected_mode == _MODE_CORRECT:
            arr = self._get_array()
            sr = getattr(arr, "scan_root", None) if arr is not None else None
            if sr is not None:
                tail = ".corrected" + _norm_variant(self._correct_output_suffix)
                return f"{Path(sr).resolve()}{tail}"
        return ""

    def _spawn(self, task_type: str, args: dict, description: str,
               output_path: str | None) -> None:
        from mbo_utilities.gui.widgets.process_manager import get_process_manager
        pm = get_process_manager()
        pid = pm.spawn(
            task_type=task_type, args=args, description=description,
            output_path=output_path,
        )
        if pid:
            self._last_status = f"Started (PID {pid})"
            if hasattr(self.parent, "logger"):
                self.parent.logger.info(
                    f"{task_type} spawned PID {pid}"
                )
        else:
            self._last_status = "Failed to spawn worker."

    def _draw_settings_popup(self, arr: Any) -> None:
        """Full pipeline-settings popup, formatted like Suite2p's.

        Layout: two rows of bordered child boxes, each row sized to the
        widest natural column. Row contents depend on the active mode:

        - Consolidate: Output options + Microscope overrides.
        - Correct: Output options + Segmentation + Pixel correction
          (advanced) + Microscope overrides.
        - Fuse: Output options + Fusion + Registration search +
          Microscope overrides.

        Edits are live; the popup just hides controls until needed.
        """
        if not self._show_settings_popup:
            return

        # cache one camera-volume size for the worker RAM-estimate lines (the
        # IO boxes drawn below don't receive ``arr``).
        self._iso_vol_gb = _iso_volume_gb(arr)

        popup_title = "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
        viewport = imgui.get_main_viewport()
        _vp_size = viewport.size

        # Lazy-init the auto-size policy. before_open() is called every
        # frame the popup is potentially drawn — it buffers a
        # set_next_window_pos with Cond_.appearing, which is a no-op on
        # all frames except the one where the popup transitions from
        # hidden to visible.
        if getattr(self, "_iso_settings_sizer", None) is None:
            self._iso_settings_sizer = PopupAutoSize(popup_title)
        self._iso_settings_sizer.before_open()

        if self._popup_just_opened:
            self._popup_just_opened = False

        # Anchor min width to fit MAX_COLS sub-section boxes at their
        # natural width (widest input row + (?) marker + padding). Without
        # this, the popup could shrink narrow enough to clip the
        # right-aligned (?) markers at the sub-section edges. Auto-resize
        # respects both bounds.
        _MAX_COLS = 3  # max columns across all isoview modes
        style = imgui.get_style()
        spacing_x = style.item_spacing.x
        window_pad_x = style.window_padding.x
        col_floor = _natural_col_w()
        min_popup_w = (
            _MAX_COLS * col_floor
            + spacing_x * (_MAX_COLS - 1)
            + 2 * window_pad_x
            + 16  # safety
        )
        min_popup_w = min(min_popup_w, _vp_size.x * 0.98)
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(min_popup_w, 300.0),
            imgui.ImVec2(_vp_size.x * 0.98, _vp_size.y * 0.98),
        )

        opened, visible = imgui.begin_popup_modal(
            popup_title,
            p_open=True,
            flags=self._iso_settings_sizer.flags(
                imgui.WindowFlags_.no_saved_settings
            ),
        )
        if not opened:
            return
        try:
            if not visible:
                self._show_settings_popup = False
                imgui.close_current_popup()
                return

            mode = self._selected_mode
            imgui.text_colored(
                _SUBSECTION_COLOR,
                f"Mode: {mode}    Dataset: {Path(arr.scan_root).name}",
            )
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if mode == _MODE_CONSOLIDATE:
                self._draw_consolidate_popup_rows()
            elif mode == _MODE_CORRECT:
                self._draw_correct_popup_rows()
            elif mode == _MODE_FUSE:
                self._draw_fuse_popup_rows()
            elif mode == _MODE_STITCHER:
                self._draw_stitcher_popup_rows()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            if imgui.button("Close##iso_settings_close",
                            imgui.ImVec2(_BTN_W, 0)):
                self._show_settings_popup = False
                imgui.close_current_popup()
        finally:
            imgui.end_popup()

    def _draw_popup_columns(
        self, columns: "list[tuple[str, Any] | tuple[str, Any, bool]]",
    ) -> None:
        """Render a row of equal-width bordered child boxes.

        ``columns`` is a list of ``(title, draw_fn)`` or
        ``(title, draw_fn, collapsing)``. Each box gets a yellow
        ``_TITLE_COLOR`` header, matching Suite2p's pipeline popup. When
        ``collapsing=True`` the title renders as a collapsing header
        (collapsed by default each open) — same idiom Suite2p uses for
        its row-2 advanced columns. Boxes auto-resize their height;
        widths split the row evenly.
        """
        if not columns:
            return
        n = len(columns)
        avail_x = imgui.get_content_region_avail().x
        style = imgui.get_style()
        spacing_x = style.item_spacing.x
        col_w = max(_natural_col_w(), (avail_x - spacing_x * (n - 1)) / n)
        box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        for i, col in enumerate(columns):
            if len(col) == 3:
                title, draw_fn, collapsing = col
            else:
                title, draw_fn = col
                collapsing = False
            if i > 0:
                imgui.same_line()
            imgui.begin_child(
                f"##iso_col_{title}",
                imgui.ImVec2(col_w, 0),
                box_flags,
            )
            try:
                if collapsing:
                    imgui.set_next_item_open(False, imgui.Cond_.appearing)
                    imgui.push_style_color(imgui.Col_.text, _TITLE_COLOR)
                    expanded = imgui.collapsing_header(
                        f"{title}##iso_col_hdr_{title}"
                    )
                    imgui.pop_style_color()
                    if expanded:
                        imgui.spacing()
                        draw_fn()
                else:
                    imgui.text_colored(_TITLE_COLOR, title)
                    imgui.spacing()
                    draw_fn()
            finally:
                imgui.end_child()

    def _draw_consolidate_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("I/O options", self._draw_consolidate_io_box),
            ("Codec", self._draw_consolidate_codec_box),
            ("Acquisition (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_correct_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("I/O options", self._draw_correct_io_box),
            ("Segmentation", self._draw_correct_segmentation_box),
            ("Dead pixel correction", self._draw_correct_advanced_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Acquisition (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_fuse_popup_rows(self) -> None:
        self._draw_fuse_view_selector()
        self._draw_popup_columns([
            ("I/O options", self._draw_fuse_io_box),
            ("Fusion", self._draw_fuse_blending_box),
            ("View transforms", self._draw_fuse_transforms_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Background", self._draw_fuse_background_box),
            ("Registration search", self._draw_fuse_search_box, True),
            ("Acquisition (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_fuse_view_selector(self) -> None:
        """Per-view editor switcher.

        Only the boxes tagged "editing VW##" follow this combo — Fusion,
        View transforms, and Registration search. I/O options and
        Microscope overrides are dataset-wide.
        """
        if not self._fuse_view_ids:
            return
        labels = [f"VW{v:02d}" for v in self._fuse_view_ids]
        try:
            idx = self._fuse_view_ids.index(self._fuse_active_view)
        except (ValueError, TypeError):
            idx = 0
            self._fuse_active_view = self._fuse_view_ids[0]
        imgui.text_colored(_SUBSECTION_COLOR, "Per-view edits target:")
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(_input_w())
        with tooltip_marks_right():
            changed, new_idx = imgui.combo("##fuse_view_selector", idx, labels)
            if changed:
                self._fuse_active_view = self._fuse_view_ids[new_idx]
            set_tooltip(
                "Each view (camera pair) carries its own blending, "
                "transforms, and registration-search params.\n"
                "Follows this combo:  Fusion · View transforms · "
                "Registration search.\n"
                "I/O options + Microscope are dataset-wide.\n"
                "The active view's blending method names the fused "
                "output sub-folder."
            )
        imgui.spacing()

    def _draw_view_scope_badge(self, scope: str) -> None:
        """One-line scope hint at the top of a Fuse-mode column box.

        ``scope`` is ``"per_view"`` (binds to the View selector),
        ``"all_views"`` (edits all views in one panel), or ``"shared"``
        (dataset-wide, unaffected by the selector).
        """
        if scope == "per_view":
            view = self._fuse_active_view
            text = (
                f"editing VW{view:02d}" if view is not None
                else "per-view (no view loaded)"
            )
            color = _SUBSECTION_COLOR
        elif scope == "all_views":
            text = "edits all views"
            color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
        else:
            text = "shared across views"
            color = imgui.ImVec4(0.7, 0.7, 0.7, 1.0)
        imgui.text_colored(color, text)
        imgui.spacing()

    def _draw_stitcher_popup_rows(self) -> None:
        """BigStitcher Parameters popup, laid out in Suite2p-style boxes."""
        self._draw_popup_columns([
            ("Orientation", self._draw_stitcher_orientation_box),
            ("View transforms", self._draw_stitcher_transforms_box),
            ("Acquisition (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _orient_toggle(self, label: str, active: bool, width: float = 0.0) -> bool:
        """Highlighted selection button; returns True when clicked."""
        with selected_button_style(active):
            return imgui.button(label, imgui.ImVec2(width, 0))

    def _orient_target_state(self, target: "str | None" = None) -> dict:
        """Editable rotations/flips for one orientation target (lazy-init)."""
        target = target or self._stitcher_orient_target
        return self._stitcher_orient.setdefault(
            target, {"rotations": [], "flips": []}
        )

    def _parse_stitcher_cameras(self) -> list:
        """Camera indices from the Cameras field (e.g. '0,2' -> [0, 2])."""
        return [
            int(c) for c in self._stitcher_cameras.replace(",", " ").split()
            if c.strip().isdigit()
        ]

    def _is_per_camera_export(self) -> bool:
        """True when the export is per-camera (raw, corrected with a Cameras
        filter, or corrected linked in place) rather than fused VW00/VW90."""
        kind = getattr(self._get_array(), "kind", None)
        if kind == "raw":
            return True
        if kind == "corrected":
            return bool(self._parse_stitcher_cameras()) or self._stitcher_link_existing
        return False

    def _orient_targets(self) -> list:
        """Orientation targets. Per-camera (raw/.corrected): the views
        themselves (VW00/VW90/VW180/VW270, or the Cameras filter). Fused: the
        fused views."""
        if self._is_per_camera_export():
            from mbo_utilities.arrays.isoview.array import camera_view_label
            cams = self._parse_stitcher_cameras()
            if not cams:
                cams = [0, 1, 2, 3]  # raw exports all cameras by default
            return [camera_view_label(c) for c in sorted(set(cams))]
        return ["VW00", "VW90"]

    def _metadata_orient_profile(self, arr: Any) -> str:
        """Default orientation profile from the XML camera_orientation:
        Normal -> 'default', Rotated -> 'rotated'. When the field is absent,
        per-camera falls back to the Normal CM->CM00 alignment (prior
        behaviour) and fused to 'none'."""
        meta = getattr(arr, "metadata", None) or {}
        co = str(meta.get("camera_orientation") or "").strip().lower()
        if not co:
            for v in (getattr(arr, "camera_metadata", None) or {}).values():
                c = str((v or {}).get("camera_orientation") or "").strip().lower()
                if c:
                    co = c
                    break
        if co == "normal":
            return "default"
        if co == "rotated":
            return "rotated"
        return "default" if self._is_per_camera_export() else "none"

    def _apply_orient_profile_all(self, name: str) -> None:
        """Seed every orientation target from the global profile.

        Per-camera export: Normal and Rotated each apply their verified
        CM->CM00 alignment to every camera (CM00 = reference); None clears
        them. Fused export: the profile seeds VW00 (the profiles are
        VW00-defined); VW90 keeps its own seed.
        """
        self._stitcher_orient_profile = name
        if self._is_per_camera_export():
            from mbo_utilities.arrays.isoview.array import camera_view_label
            table = {
                "default": _CM_ALIGN_DEFAULT,
                "rotated": _CM_ALIGN_DEFAULT_ROTATED,
            }.get(name)
            for cam in range(4):
                seed = table.get(cam) if table else None
                st = self._orient_target_state(camera_view_label(cam))
                st["rotations"] = (
                    [dict(r) for r in seed["rotations"]] if seed else []
                )
                st["flips"] = list(seed["flips"]) if seed else []
        else:
            self._load_orient_profile(name, "VW00")
            # VW90 has no fused profile (set manually). Clear it so a prior
            # per-camera seed (e.g. rot X-90 on camera 2) can't leak into the
            # fused export's orientation_vw90 and tilt it out of plane.
            vw90 = self._orient_target_state("VW90")
            vw90["rotations"], vw90["flips"] = [], []

    def _toggle_button_row(self, id_prefix, items, is_active, on_click) -> None:
        """Selectable toggle buttons that wrap to the box width (no clipping).

        ``items`` is a sequence of ``(key, label)``. Buttons flow left to
        right and wrap to the next line when the next one would overrun the
        box's content region, so a row never clips its last entry.
        """
        style = imgui.get_style()
        right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
        for i, (key, label) in enumerate(items):
            w = imgui.calc_text_size(label).x + 2 * style.frame_padding.x
            if i > 0:
                nxt = imgui.get_item_rect_max().x + style.item_spacing.x + w
                if nxt <= right:
                    imgui.same_line()
            if self._orient_toggle(f"{label}##{id_prefix}_{key}", is_active(key)):
                on_click(key)

    def _toggle_flip(self, st: dict, axis: str) -> None:
        if axis in st["flips"]:
            st["flips"].remove(axis)
        else:
            st["flips"].append(axis)

    def _load_orient_profile(self, name: str, target: "str | None" = None) -> None:
        """Seed the target's editable rotations + flips from a profile."""
        prof = _STITCHER_ORIENT_PROFILES.get(name, {"rotations": [], "flips": []})
        st = self._orient_target_state(target)
        st["rotations"] = [
            {"sign": s, "axis": a, "deg": int(d)} for (s, a, d) in prof["rotations"]
        ]
        st["flips"] = list(prof["flips"])

    def _compose_orientation_ops(self, target: "str | None" = None) -> list:
        """Orientation op list for one target.

        Rotations (row order, innermost first) then flips. Rotation ->
        ["rot", axis, signed degrees]; flip -> ["flip", axis].
        """
        st = self._orient_target_state(target)
        ops: list = []
        for rot in st["rotations"]:
            deg = int(rot["deg"])
            if rot["sign"] == "-":
                deg = -deg
            ops.append(["rot", rot["axis"], deg])
        for axis in st["flips"]:
            ops.append(["flip", axis])
        return ops

    def _draw_stitcher_orientation_box(self) -> None:
        """Reorientation for the export.

        Pick a global Profile (acquisition mounting) that cascades to every
        target, then optionally select a target and fine-tune its rotations
        + flips. Baked as an affine; BigStitcher interpolates it.
        """
        per_camera = self._is_per_camera_export()
        targets = self._orient_targets()
        if self._stitcher_orient_target not in targets:
            self._stitcher_orient_target = targets[0] if targets else "VW00"

        # PROFILE — global, above Target: it cascades to every target (seeded
        # from the XML camera_orientation). Per-target edits below override it.
        imgui.text("Profile")
        self._toggle_button_row(
            "orient_profile",
            (("none", "None"), ("default", "Normal"), ("rotated", "Rotated")),
            lambda name: self._stitcher_orient_profile == name,
            lambda name: self._apply_orient_profile_all(name),
        )
        if per_camera:
            _hint("Normal/Rotated align VW90/180/270 onto VW00; None clears "
                  "them. Applies to every view.")
        else:
            _hint("Seeds VW00; set VW90's ~90 below. Applies to the views.")
        imgui.spacing()

        # TARGET — which view/camera the rotations + flips below edit.
        imgui.text("Target")
        self._toggle_button_row(
            "orient_target",
            tuple((t, t) for t in targets),
            lambda t: self._stitcher_orient_target == t,
            lambda t: setattr(self, "_stitcher_orient_target", t),
        )
        imgui.spacing()

        cur = self._stitcher_orient_target
        st = self._orient_target_state(cur)

        # Rotations — editable list seeded by the profile.
        imgui.text("Rotations")
        axes = ["X", "Y", "Z"]
        degs = ["90", "180", "270"]
        remove_idx = -1
        for i, rot in enumerate(st["rotations"]):
            imgui.push_id(i)
            if imgui.button(
                f"{rot['sign']}##sign", imgui.ImVec2(hello_imgui.em_size(2.2), 0)
            ):
                rot["sign"] = "-" if rot["sign"] == "+" else "+"
            imgui.same_line()
            imgui.set_next_item_width(hello_imgui.em_size(3.6))
            ai = axes.index(rot["axis"]) if rot["axis"] in axes else 0
            changed, ai = imgui.combo("##axis", ai, axes)
            if changed:
                rot["axis"] = axes[ai]
            imgui.same_line()
            imgui.set_next_item_width(hello_imgui.em_size(5.0))
            cur_deg = str(int(rot["deg"]))
            di = degs.index(cur_deg) if cur_deg in degs else 0
            changed, di = imgui.combo("##deg", di, degs)
            if changed:
                rot["deg"] = int(degs[di])
            imgui.same_line()
            imgui.text("deg")
            imgui.same_line()
            if imgui.small_button("x##rm"):
                remove_idx = i
            imgui.pop_id()
        if remove_idx >= 0:
            del st["rotations"][remove_idx]
        if imgui.small_button("+ add##orient_add_rot"):
            st["rotations"].append({"sign": "-", "axis": "X", "deg": 90})
        imgui.same_line()
        if imgui.small_button("clear##orient_clear_rot"):
            st["rotations"] = []
        imgui.spacing()

        # Flip — Horizontal/Vertical/Depth toggle = flip X/Y/Z.
        imgui.text("Flip")
        self._toggle_button_row(
            "orient_flip",
            (("X", "Horizontal (X)"), ("Y", "Vertical (Y)"), ("Z", "Depth (Z)")),
            lambda axis: axis in st["flips"],
            lambda axis: self._toggle_flip(st, axis),
        )
        imgui.spacing()

        # Live summary of every target that has a transform.
        any_set = False
        for t in targets:
            ops = self._compose_orientation_ops(t)
            if not ops:
                continue
            any_set = True
            parts = []
            for op in ops:
                parts.append(
                    f"{op[2]:+d} {op[1]}" if op[0] == "rot" else f"flip {op[1]}"
                )
            imgui.text_disabled(f"{t}: " + ", ".join(parts))
        if not any_set:
            imgui.text_disabled("no transforms")

    def _draw_stitcher_transforms_box(self) -> None:
        with tooltip_marks_right():
            _, self._stitcher_bake_tile_positions = imgui.checkbox(
                "Bake tile positions",
                self._stitcher_bake_tile_positions,
            )
            set_tooltip(
                "Lay tiles on the grid from the zarr metadata "
                "(stride offset, else stage). Off: tiles land at the origin."
            )

            _, self._stitcher_reverse_z = imgui.checkbox(
                "Reverse Z", self._stitcher_reverse_z,
            )
            set_tooltip(
                "Join adjacent z-blocks contiguously (camera scans -Z)."
            )

            _, self._stitcher_upright = imgui.checkbox(
                "Upright (rotated)", self._stitcher_upright,
            )
            set_tooltip(
                "Rotated only: rotate content upright to match the tile grid. "
                "Off keeps cameras in CM00's native frame (no tilt under view "
                "rotation; orient in BigStitcher)."
            )

            ch_link, self._stitcher_link_existing = imgui.checkbox(
                "Link existing (.corrected)", self._stitcher_link_existing,
            )
            if ch_link:
                # toggling link flips per-camera <-> fused, which switches the
                # orientation targets (per-camera tables vs the VW00 profile).
                # Re-seed so a fused VW00 profile can't leak onto cam0.
                self._apply_orient_profile_all(self._stitcher_orient_profile)
            set_tooltip(
                "Reference the .corrected zarrs in place, no conversion. "
                "Much faster; .corrected only."
            )

            imgui.set_next_item_width(_input_w())
            prev_pc = self._is_per_camera_export()
            ch_cam, self._stitcher_cameras = imgui.input_text(
                "Cameras", self._stitcher_cameras,
            )
            if ch_cam and self._is_per_camera_export() != prev_pc:
                # entering/leaving per-camera switches the orientation targets;
                # re-seed so the fused VW00 profile can't leak onto cam0.
                self._apply_orient_profile_all(self._stitcher_orient_profile)
            set_tooltip(
                "Per-camera export from .corrected, e.g. 0,2. "
                "Empty: fused VW00/VW90."
            )

            zarr_versions = ["v2", "v3"]
            zv_idx = 0 if self._stitcher_zarr_version == 2 else 1
            imgui.set_next_item_width(_input_w())
            changed, zv_idx = imgui.combo("Zarr version", zv_idx, zarr_versions)
            if changed:
                self._stitcher_zarr_version = 2 if zv_idx == 0 else 3
            set_tooltip(
                "v2: classic chunks + gzip. v3: sharded + zstd, for the "
                "new BigStitcher ZarrV3 reader."
            )

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int("Workers", self._workers, 1, 2)
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel volume writers (threads). RAM-bound: each holds "
                "one full camera volume + its pyramid."
            )

    def _draw_consolidate_io_box(self) -> None:
        """Consolidate I/O options for the Parameters popup."""
        with tooltip_marks_right():
            _, self._consolidate_pyramid = imgui.checkbox(
                "Pyramid", self._consolidate_pyramid,
            )
            set_tooltip(
                "Build the OME-NGFF pyramid for fast zoomed-out viewing. "
                "Adds write time + disk."
            )

            if self._consolidate_pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._consolidate_pyramid_max_layers = imgui.input_int(
                    "Pyramid levels",
                    self._consolidate_pyramid_max_layers, 1, 1,
                )
                set_tooltip(
                    "Max additional pyramid levels beyond /0.\n"
                    "Stops early when Y or X would fall below 64 voxels."
                )

    def _draw_consolidate_codec_box(self) -> None:
        with tooltip_marks_right():
            compressors = [
                "zstd", "gzip", "blosc-zstd", "blosc-lz4", "none",
            ]
            try:
                idx = compressors.index(self._consolidate_compressor)
            except ValueError:
                idx = 0
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Compressor", idx, compressors)
            if changed:
                self._consolidate_compressor = compressors[new_idx]
            set_tooltip(
                "Inner-chunk codec. zstd = best size/speed; "
                "none = fastest, biggest."
            )

            imgui.set_next_item_width(_input_w())
            _, self._consolidate_compression_level = imgui.input_int(
                "Level", self._consolidate_compression_level, 1, 1,
            )
            set_tooltip(
                "0–9. Higher = smaller files, slower writes; "
                "gains taper past ~5."
            )

    def _draw_codec_controls(self) -> None:
        """Compressor + level for the active output format (zarr or tif).

        klb carries its own codec, so nothing is shown for it. Called from
        inside a ``tooltip_marks_right()`` block.
        """
        if self._output_format not in ("zarr", "tif"):
            return
        compressors = _codecs_for_format(self._output_format)
        if self._compression not in compressors:
            self._compression = compressors[0]
        cidx = compressors.index(self._compression)
        imgui.set_next_item_width(_input_w())
        c_changed, c_new = imgui.combo("Compressor", cidx, compressors)
        if c_changed:
            self._compression = compressors[c_new]
        if self._output_format == "tif":
            set_tooltip(
                "TIFF codec. deflate default; Fiji can't read zstd TIFFs."
            )
        else:
            set_tooltip(
                "Inner-chunk codec. zstd = best size/speed; "
                "none = fastest writes, biggest files."
            )

        imgui.set_next_item_width(_input_w())
        _, self._compression_level = imgui.input_int(
            "Level", self._compression_level, 1, 1,
        )
        set_tooltip(
            "0–9. Higher = smaller files, slower writes; gains taper past ~5."
        )

    def _draw_correct_io_box(self) -> None:
        """Correct-mode I/O options for the Parameters popup. The
        ``output_suffix`` lives in the Run-tab Output section now;
        only the codec / worker / pyramid / overwrite knobs stay here.
        """
        with tooltip_marks_right():
            formats = ["zarr", "tif", "klb"]
            try:
                idx = formats.index(self._output_format)
            except ValueError:
                idx = 0
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Format", idx, formats)
            if changed:
                self._output_format = formats[new_idx]
                self._compression = _default_codec_for_format(
                    self._output_format
                )
            set_tooltip(
                "Container per (timepoint, camera). zarr = chunked + "
                "pyramids; tif/klb for legacy tools."
            )

            self._draw_codec_controls()

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers", self._workers, 1, 2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel timepoint/tile workers. RAM-bound, not "
                "CPU-bound: each holds a full 3D volume.\n"
                f"Start 2–4 and watch RAM (soft cap {_MAX_WORKERS})."
            )
            self._draw_worker_ram_estimate(_CORRECT_RAM_PER_WORKER_X)

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Downsampled levels for fast zoomed-out viewing "
                "(zarr / OME-TIFF). Adds write time + disk."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels", self._pyramid_max_layers, 1, 1,
                )
                set_tooltip(
                    "Extra downsample levels beyond full-res. More = "
                    "smoother zoom-out + more disk; stops when an axis "
                    "< 64 px."
                )

    def _draw_correct_segmentation_box(self) -> None:
        with tooltip_marks_right():
            seg_labels = ["none", "generate + save masks"]
            imgui.set_next_item_width(_input_w())
            s_changed, s_new = imgui.combo(
                "Segment mode", self._correct_segment_mode, seg_labels,
            )
            if s_changed:
                self._correct_segment_mode = s_new
            set_tooltip(
                "  0  off — skip segmentation entirely\n"
                "  1  build a binary foreground mask per (t, c) and "
                "save alongside the volume."
            )

            _, self._correct_apply_seg_mask = imgui.checkbox(
                "Apply mask to volume", self._correct_apply_seg_mask,
            )
            set_tooltip(
                "Zero background pixels in the saved volume (irreversible).\n"
                "Off keeps full intensities; the mask is saved separately."
            )

            _, self._correct_do_tenengrad = imgui.checkbox(
                "Tenengrad diagnostic", self._correct_do_tenengrad,
            )
            set_tooltip(
                "Extra per-Z sharpness plot per camera pair (small added "
                "time).\nUse it to set transition_plane / blending_range "
                "for Fuse."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_kernel = imgui.input_int(
                "Gaussian kernel", self._correct_gauss_kernel, 1, 1,
            )
            set_tooltip(
                "Pre-blur window (px) before thresholding.\n"
                "Larger = smoother masks, fewer specks, a bit slower."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_sigma = imgui.input_float(
                "Gaussian sigma", self._correct_gauss_sigma, 0.1, 1.0, "%.2f",
            )
            set_tooltip(
                "Pre-blur strength (px).\n"
                "Higher = smoother foreground; too high merges nearby objects."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_segment_threshold = imgui.input_float(
                "##segment_threshold", self._correct_segment_threshold,
                0.01, 0.1, "%.3f",
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("segment_threshold", "Threshold")
            set_tooltip(
                "Foreground cutoff on the smoothed image (0–1).\n"
                "Lower keeps more as signal; raise if background leaks in."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_mask_percentile = imgui.input_float(
                "Mask percentile", self._correct_mask_percentile,
                0.1, 1.0, "%.2f",
            )
            set_tooltip(
                "Baseline percentile for the mask (0–100).\n"
                "Higher = stricter foreground (drops dim signal)."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_splitting = imgui.input_int(
                "Splitting", self._correct_splitting, 1, 5,
            )
            set_tooltip(
                "Y-axis slabs for the segmentation filter.\n"
                "Raise to cut peak RAM (10→20 roughly halves the filter's "
                "working buffer), ~5–10% slower from slab overlap."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_subsample_factor = imgui.input_int(
                "Subsample factor", self._correct_subsample_factor, 1, 10,
            )
            set_tooltip(
                "Percentile sampling stride — uses every Nth voxel "
                "(100 ≈ 1%).\nHigher = faster, slightly noisier. Shared by "
                "segmentation + background."
            )

    def _draw_correct_advanced_box(self) -> None:
        with tooltip_marks_right():
            _, self._correct_median_kernel_enabled = imgui.checkbox(
                "Median filter", self._correct_median_kernel_enabled,
            )
            set_tooltip(
                "Replace hot/dead pixels with a per-plane median.\n"
                "Off if the camera is already corrected."
            )
            if self._correct_median_kernel_enabled:
                imgui.set_next_item_width(_input_w())
                _, self._correct_median_kernel_size = imgui.input_int(
                    "Kernel (N×N)", self._correct_median_kernel_size, 1, 1,
                )
                set_tooltip(
                    "Median window (px). Larger removes bigger clusters "
                    "but softens detail and is slower."
                )

            imgui.set_next_item_width(_input_w())
            _, self._correct_background_percentile = imgui.input_float(
                "Background pct",
                self._correct_background_percentile,
                0.5, 5.0, "%.2f",
            )
            set_tooltip(
                "Percentile for the camera background floor (0–100).\n"
                "Raise if a dark haze remains; too high clips dim signal."
            )

    def _draw_fuse_io_box(self) -> None:
        """Fuse-mode I/O options for the Parameters popup.

        Dataset-wide: format / codec / workers / pyramid. The output-folder
        variant lives in the Run-tab Output options block.
        """
        self._draw_view_scope_badge("shared")
        with tooltip_marks_right():
            formats = ["zarr", "tif", "klb"]
            try:
                idx = formats.index(self._output_format)
            except ValueError:
                idx = 0
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("Format", idx, formats)
            if changed:
                self._output_format = formats[new_idx]
                self._compression = _default_codec_for_format(
                    self._output_format
                )
            set_tooltip(
                "Container per (timepoint, camera). zarr = chunked + "
                "pyramids; tif/klb for legacy tools."
            )

            self._draw_codec_controls()

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers", self._workers, 1, 2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel workers. RAM-bound: each holds both camera "
                "volumes + the output; adaptive blending uses more.\n"
                f"Start 2–4 (soft cap {_MAX_WORKERS})."
            )
            self._draw_worker_ram_estimate(_FUSE_RAM_PER_WORKER_X)

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Downsampled levels for fast zoomed-out viewing "
                "(zarr / OME-TIFF). Adds write time + disk."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels", self._pyramid_max_layers, 1, 1,
                )
                set_tooltip(
                    "Extra downsample levels beyond full-res. More = "
                    "smoother zoom-out + more disk; stops when an axis "
                    "< 64 px."
                )

    def _draw_fuse_blending_box(self) -> None:
        self._draw_view_scope_badge("per_view")
        with tooltip_marks_right():
            methods = ["geometric", "adaptive", "average"]
            try:
                idx = methods.index(self._fv("blending_method"))
            except ValueError:
                idx = 0
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("##blending_method", idx, methods)
            if changed:
                self._set_fv("blending_method", methods[new_idx])
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_method", "Blending method")
            set_tooltip(
                "  geometric  fixed transition plane (fastest)\n"
                "  adaptive   per-XY tenengrad blend (sharper, more "
                "RAM/time)\n"
                "  average    plain mean (no blending)."
            )

            imgui.set_next_item_width(_input_w())
            _, new_br = imgui.input_int(
                "##blending_range", int(self._fv("blending_range")), 1, 5,
            )
            self._set_fv("blending_range", new_br)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_range", "Blending range")
            set_tooltip(
                "Z-plane width of the blend seam (geometric).\n"
                "Wider = smoother seam, more mixing of both cameras."
            )

            imgui.set_next_item_width(_input_w())
            _, new_tp = imgui.input_int(
                "##transition_plane", int(self._fv("transition_plane")), 1, 5,
            )
            self._set_fv("transition_plane", new_tp)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("transition_plane", "Transition plane")
            set_tooltip(
                "Z-index where the cameras hand off (geometric).\n"
                "−1 = volume center; read it off the Tenengrad plot."
            )

            imgui.set_next_item_width(_input_w())
            _, new_ff = imgui.input_int(
                "##front_flag", int(self._fv("front_flag")), 1, 1,
            )
            self._set_fv("front_flag", new_ff)
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("front_flag", "Front camera")
            set_tooltip(
                "Which camera carries the sharper signal at low Z.\n"
                "  1 = reference camera (cam0)\n"
                "  2 = transformed camera (cam1)"
            )

    def _draw_fuse_background_box(self) -> None:
        self._draw_view_scope_badge("shared")
        with tooltip_marks_right():
            _, self._fuse_subtract_background = imgui.checkbox(
                "Subtract background", self._fuse_subtract_background,
            )
            set_tooltip(
                "Subtract an estimated background floor from both cameras "
                "before blending.\nOff: keep raw intensities."
            )

            if self._fuse_subtract_background:
                modes = ["pooled", "per_camera"]
                try:
                    idx = modes.index(self._fuse_background_mode)
                except ValueError:
                    idx = 0
                imgui.set_next_item_width(_input_w())
                changed, new_idx = imgui.combo("Mode", idx, modes)
                if changed:
                    self._fuse_background_mode = modes[new_idx]
                set_tooltip(
                    "  pooled      one floor from both cameras\n"
                    "  per_camera  each camera subtracts its own floor — "
                    "removes the seam when the two cameras sit on "
                    "different baselines."
                )

                imgui.set_next_item_width(_input_w())
                _, self._fuse_background_percentile = imgui.input_float(
                    "Background pct", self._fuse_background_percentile,
                    0.5, 5.0, "%.2f",
                )
                set_tooltip(
                    "Percentile for the background floor (0–100).\n"
                    "Raise if a dark haze remains; too high clips dim signal."
                )

    def _draw_fuse_transforms_box(self) -> None:
        self._draw_view_scope_badge("per_view")
        with tooltip_marks_right():
            _, new_fz = imgui.checkbox("Flip Z", bool(self._fv("flip_z")))
            self._set_fv("flip_z", new_fz)
            set_tooltip(
                "Flip the second camera along Z before fusion.\n"
                "Use for opposing camera pairs."
            )

            _, new_fh = imgui.checkbox(
                "Flip horizontal", bool(self._fv("flip_horizontal")),
            )
            self._set_fv("flip_horizontal", new_fh)
            set_tooltip("Flip the second camera horizontally before fusion.")

            _, new_fv = imgui.checkbox(
                "Flip vertical", bool(self._fv("flip_vertical")),
            )
            self._set_fv("flip_vertical", new_fv)
            set_tooltip("Flip the second camera vertically before fusion.")

    def _draw_fuse_search_box(self) -> None:
        self._draw_view_scope_badge("per_view")
        with tooltip_marks_right():
            imgui.text_colored(_SUBSECTION_COLOR, "X offsets")
            for label, key in (
                ("Start##sx", "search_x_start"),
                ("Stop##sx", "search_x_stop"),
                ("Step##sx", "search_x_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label, int(self._fv(key)), 1, 10,
                )
                self._set_fv(key, new_val)
                set_tooltip(
                    "Pixel offset grid searched along X to align the pair "
                    "(start, stop, step).\nWider range or smaller step = "
                    "more robust, slower (cost ∝ range/step)."
                )

            imgui.spacing()
            imgui.text_colored(_SUBSECTION_COLOR, "Y offsets")
            for label, key in (
                ("Start##sy", "search_y_start"),
                ("Stop##sy", "search_y_stop"),
                ("Step##sy", "search_y_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label, int(self._fv(key)), 1, 10,
                )
                self._set_fv(key, new_val)
                set_tooltip(
                    "Pixel offset grid searched along Y to align the pair "
                    "(start, stop, step).\nWider range or smaller step = "
                    "more robust, slower (cost ∝ range/step)."
                )

    def _draw_microscope_overrides_box(self) -> None:
        with tooltip_marks_right():
            _, self._mic_overrides_enabled = imgui.checkbox(
                "Override XML defaults", self._mic_overrides_enabled,
            )
            set_tooltip(
                "When off, values are read from the dataset XML.\n"
                "Enable to override one or more values below."
            )
            imgui.spacing()

            if not self._mic_overrides_enabled:
                imgui.begin_disabled()
            try:
                _hint("−1 = auto from XML")
                imgui.spacing()

                imgui.set_next_item_width(_input_w())
                _, self._mic_pixel_spacing_z = imgui.input_float(
                    "Z spacing (µm)", self._mic_pixel_spacing_z,
                    0.01, 0.1, "%.3f",
                )
                set_tooltip("Physical Z-step in micrometers.")

                imgui.set_next_item_width(_input_w())
                _, self._mic_objective_mag = imgui.input_float(
                    "Objective mag", self._mic_objective_mag,
                    0.5, 5.0, "%.2f",
                )
                set_tooltip("Detection objective magnification (e.g. 16.0).")

                imgui.set_next_item_width(_input_w())
                _, self._mic_pixel_spacing_camera = imgui.input_float(
                    "Camera pixel (µm)", self._mic_pixel_spacing_camera,
                    0.1, 1.0, "%.2f",
                )
                set_tooltip(
                    "Physical camera pixel size in micrometers "
                    "(e.g. 6.5 for the C11440-22C)."
                )
            finally:
                if not self._mic_overrides_enabled:
                    imgui.end_disabled()

    def _microscope_kwargs(self) -> dict:
        """Pack microscope overrides as a kwargs dict for the task args.

        Sentinels of ``-1`` map to ``None`` (auto from XML); only set
        keys the user explicitly overrode.
        """
        out: dict = {}
        if not self._mic_overrides_enabled:
            return out
        if self._mic_pixel_spacing_z >= 0:
            out["pixel_spacing_z"] = float(self._mic_pixel_spacing_z)
        if self._mic_objective_mag > 0:
            out["detection_objective_mag"] = float(self._mic_objective_mag)
        if self._mic_pixel_spacing_camera > 0:
            out["pixel_spacing_camera"] = float(self._mic_pixel_spacing_camera)
        return out

    def _submit_active(self, arr: Any) -> None:
        if self._selected_mode == _MODE_CONSOLIDATE:
            self._submit_consolidate(arr)
        elif self._selected_mode == _MODE_CORRECT:
            self._submit_correct(arr)
        elif self._selected_mode == _MODE_FUSE:
            self._submit_fuse(arr)
        elif self._selected_mode == _MODE_STITCHER:
            self._submit_stitcher(arr)

    def _submit_consolidate(self, arr: Any) -> None:
        # Derive the output .zarr path from the current suffix so a
        # mid-session suffix edit on the Output tab takes effect without
        # the user re-opening the dataset.
        sr = Path(arr.scan_root)
        out_path = str(sr.parent / f"{sr.name}{self._consolidate_suffix}")
        self._consolidate_output_path = out_path
        if not out_path or not (self._consolidate_suffix or "").strip():
            self._last_status = "Suffix is required."
            return
        args = {
            "input_path": str(arr.scan_root),
            "output_path": out_path,
            "kind": arr.kind,
            "overwrite": self._consolidate_overwrite,
            "pyramid": self._consolidate_pyramid,
            "pyramid_max_layers": self._consolidate_pyramid_max_layers,
            "compressor": self._consolidate_compressor,
            "compression_level": self._consolidate_compression_level,
        }
        self._spawn(
            "isoview", args,
            description=f"Isoview consolidate ({arr.kind}) → "
                        f"{Path(out_path).name}",
            output_path=out_path,
        )

    def _tile_spm_key(self, arr: Any, ti: Any) -> "str | int":
        """The fused ``spm_key`` for tile index ``ti``.

        Mirrors isoview's tiled naming: the ``specimen_name`` grid token
        (e.g. ``"TL000"``) when it parses as a grid position, else the
        integer specimen id (isoview formats that as ``SPM##``).
        """
        from mbo_utilities.arrays.isoview.array import _parse_tile_grid_position
        tm = (getattr(arr, "tile_metadata", None) or {}).get(ti) or {}
        name = tm.get("specimen_name")
        if name and _parse_tile_grid_position(str(name)) is not None:
            return str(name)
        spc = tm.get("specimen", ti)
        try:
            return int(spc)
        except (TypeError, ValueError):
            return ti

    def _submit_stitcher(self, arr: Any) -> None:
        """Spawn ``generate_bigstitcher_xml`` against the loaded tree.

        Prefers the ``.corrected*`` or ``.fused*`` tree the array was
        loaded from — upstream ProcessingConfig now accepts those
        directly, so the raw acquisition root is no longer required.
        Falls back to a sibling raw root only for legacy datasets that
        predate the embedded ``ome.isoview`` metadata.

        Forwards ``specimens`` and ``timepoints`` from the loaded array
        so ProcessingConfig skips its auto-detect step entirely.
        """
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root

        scan_root = Path(arr.scan_root)
        input_dir: Path | None = None
        # raw datasets export straight from the raw acquisition root.
        if getattr(arr, "kind", None) == "raw":
            input_dir = scan_root
        # Walk up from scan_root looking for a .corrected* or .fused* dir.
        # arr.scan_root for kind="fused" points at the method dir (e.g.
        # geometric) under <root>.fused/; for "corrected" it's the SPM##.
        for ancestor in () if input_dir is not None else (scan_root, *scan_root.parents):
            n = ancestor.name
            for suf in (".corrected", ".fused"):
                idx = n.find(suf)
                if idx < 0:
                    continue
                rest = n[idx + len(suf):]
                if rest == "" or rest.startswith("_"):
                    input_dir = ancestor
                    break
            if input_dir is not None:
                break

        # legacy fallback: sibling raw root for pre-isoview-meta data
        if input_dir is None:
            input_dir = _sibling_raw_root(arr)
        if input_dir is None or not input_dir.is_dir():
            self._last_status = (
                "Cannot locate a corrected or fused tree to export from."
            )
            return

        meta = arr.metadata if hasattr(arr, "metadata") else {}
        specimen = meta.get("specimen", 0)

        # Old multi-method fused layout: scan_root IS the method dir
        # (…/geometric, …/adaptive), so pass its name. New single-method
        # layout: scan_root is the .fused root (no method dir) → None, and
        # isoview resolves the one method present. Corrected input also → None.
        from mbo_utilities.arrays.isoview.array import _is_method_dir
        method = None
        if getattr(arr, "kind", None) == "fused":
            sr = Path(arr.scan_root)
            if _is_method_dir(sr):
                method = sr.name

        # output_suffix must match the loaded tree's variant: isoview
        # derives config.fused_dir from input_dir.name stripped + ".fused"
        # + output_suffix, so passing the wrong (or empty) suffix makes it
        # scan bare ".fused" instead of e.g. ".fused_v2" and find nothing.
        # stitcher_suffix names the output dir independently (so it can be
        # ".stitcher_default" while reading from ".fused").
        cameras = self._parse_stitcher_cameras()
        # raw datasets export per-camera with all cameras oriented onto CM00;
        # corrected/fused keep reading the fused tree (unchanged).
        source = "raw" if getattr(arr, "kind", None) == "raw" else "fused"
        # link mode references the .corrected per-camera zarrs in place, so it
        # uses the corrected per-camera discovery (not the fused tree).
        if self._stitcher_link_existing and getattr(arr, "kind", None) == "corrected":
            source = "corrected"
        per_camera = self._is_per_camera_export()
        cam_orient = None
        orient_to_cm00 = True
        if per_camera:
            # GUI owns every exported camera's orientation (VW90/180/270
            # pre-aligned to VW00, VW00 = none). Orientation state is keyed by
            # VW label; the export's camera_orientations dict stays keyed by the
            # camera index (the .stack file identity).
            from mbo_utilities.arrays.isoview.array import camera_view_label
            export_cams = cameras if cameras else [0, 1, 2, 3]
            cam_orient = {
                str(c): self._compose_orientation_ops(camera_view_label(c))
                for c in export_cams
            }
            orient_to_cm00 = False
        # Data-slicing → tile filter. For tiled trees the slicing T-axis is
        # the tile (SPM) axis, so map the selected positions to the fused
        # spm_key (specimen_name grid token, e.g. "TL000", else "SPM##") and
        # pass them as included_tiles so the export skips the rest. A
        # full-range selection returns None → export all tiles. Non-tiled
        # trees have no tile/timepoint filter in isoview's export.
        included_tiles = None
        if bool(getattr(arr, "is_tiled", False)):
            sel = self._selected_timepoints()
            if sel is not None:
                tps = list(getattr(arr, "_timepoints", []) or [])
                included_tiles = [
                    self._tile_spm_key(arr, tps[i])
                    for i in sel if 0 <= i < len(tps)
                ] or None
        args = {
            "input_path": str(input_dir),
            "method": method,
            "output_suffix": self._iso_tree_suffix(input_dir),
            "stitcher_suffix": self._stitcher_suffix,
            "bake_tile_positions": self._stitcher_bake_tile_positions,
            "orientation": self._compose_orientation_ops("VW00"),
            "orientation_vw90": self._compose_orientation_ops("VW90"),
            "camera_orientations": cam_orient,
            "overwrite": self._overwrite,
            "specimens": [int(specimen)],
            "timepoints": list(getattr(arr, "_timepoints", []) or []),
            "included_tiles": included_tiles,
            "cameras": cameras or None,
            "source": source,
            "orient_to_cm00": orient_to_cm00,
            "reverse_z": self._stitcher_reverse_z,
            "upright": self._stitcher_upright,
            "link_existing": self._stitcher_link_existing,
            "zarr_version": self._stitcher_zarr_version,
            "workers": self._workers,
        }
        args.update(self._microscope_kwargs())
        self._spawn(
            "isoview_bigstitcher", args,
            description=f"BigStitcher XML: {input_dir.name}",
            output_path=self._current_output_path() or str(input_dir),
        )

    def _submit_correct(self, arr: Any) -> None:
        median_kernel = None
        if self._correct_median_kernel_enabled:
            k = max(1, int(self._correct_median_kernel_size))
            median_kernel = [k, k]
        resolved_out = self._current_output_path()
        args = {
            "input_path": str(arr.scan_root),
            "output_dir": resolved_out or None,
            "output_suffix": self._correct_output_suffix,
            "output_format": self._output_format,
            "compression": (
                None if self._compression == "none" else self._compression
            ),
            "compression_level": self._compression_level,
            "overwrite": self._overwrite,
            "workers": self._workers,
            "pyramid": self._pyramid,
            "pyramid_max_layers": self._pyramid_max_layers,
            "segment_mode": self._correct_segment_mode,
            "apply_segmentation_mask": self._correct_apply_seg_mask,
            "do_tenengrad": self._correct_do_tenengrad,
            "background_percentile": self._correct_background_percentile,
            "median_kernel": median_kernel,
            "mask_percentile": self._correct_mask_percentile,
            "subsample_factor": self._correct_subsample_factor,
            "gauss_kernel": self._correct_gauss_kernel,
            "gauss_sigma": self._correct_gauss_sigma,
            "segment_threshold": self._correct_segment_threshold,
            "splitting": self._correct_splitting,
        }
        sel = self._selected_timepoints()
        if sel is not None:
            if bool(getattr(arr, "is_tiled", False)):
                # tiled: the slicing timeline IS the spatial-tile (SPM) axis, and
                # isoview slices tiled data by `specimens`, not `timepoints`. map
                # the selected positions to SPM indices so "process 1 tile" works.
                tiles = list(getattr(arr, "_timepoints", []) or [])
                args["specimens"] = [tiles[i] for i in sel if 0 <= i < len(tiles)] or sel
            else:
                args["timepoints"] = sel
        cams = self._selected_cameras(arr)
        if cams is not None:
            args["cameras"] = cams
        args.update(self._microscope_kwargs())
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        args.update(crop_state.to_config_args(arr))
        self._spawn(
            "isoview_correct", args,
            description=f"correct_stack: {Path(arr.scan_root).name}",
            output_path=resolved_out or str(arr.scan_root),
        )

    def _draw_worker_ram_estimate(self, per_worker_x: float) -> None:
        """Estimated-RAM line under the Workers control. Per-worker footprint is
        ~``per_worker_x`` x one raw camera volume; total = that x worker count.
        Colored amber when it nears, red when it exceeds, *available* RAM —
        compared against free memory (not total) so baseline usage counts, same
        basis as ``_ram_capped_fuse_workers``."""
        vol_gb = getattr(self, "_iso_vol_gb", 0.0)
        if vol_gb <= 0:
            return
        per_worker = vol_gb * per_worker_x
        total = per_worker * max(1, self._workers)
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            avail_gb = None
        label = f"~{total:.0f} GB RAM  ({per_worker:.1f} GB/worker x {self._workers})"
        if avail_gb is not None and total > avail_gb:
            imgui.text_colored(imgui.ImVec4(0.92, 0.32, 0.32, 1.0), label)
        elif avail_gb is not None and total > 0.8 * avail_gb:
            imgui.text_colored(imgui.ImVec4(0.95, 0.75, 0.32, 1.0), label)
        else:
            imgui.text_disabled(label)
        set_tooltip(
            f"Rough estimate: each worker holds ~{per_worker_x:.0f}x one raw "
            f"camera volume ({vol_gb:.1f} GB)."
            + (f" Free RAM: {avail_gb:.0f} GB." if avail_gb is not None else "")
        )

    def _ram_capped_fuse_workers(self, arr: Any, requested: int) -> int:
        """Clamp the fuse worker count to what available RAM can hold.

        Each tiled-fuse worker loads two camera volumes + two masks + the
        fused output, so budget ~6× one Z×Y×X volume per worker. Returns
        ``requested`` unchanged when psutil/shape are unavailable or memory
        is ample; only ever lowers it.
        """
        requested = max(1, int(requested))
        try:
            import psutil
            vol_gb = _iso_volume_gb(arr)
            per_worker_gb = max(1.0, vol_gb * _FUSE_RAM_PER_WORKER_X)
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            ram_cap = max(1, int((avail_gb * 0.8) // per_worker_gb))
        except Exception:
            return requested
        if ram_cap < requested:
            if hasattr(self.parent, "logger"):
                self.parent.logger.info(
                    f"multi_fuse: capping workers {requested} -> {ram_cap} "
                    f"(~{per_worker_gb:.0f} GiB/worker, {avail_gb:.0f} GiB free)"
                )
            return ram_cap
        return requested

    def _submit_fuse(self, arr: Any) -> None:
        # multi_fuse accepts a ``.corrected/`` root directly — isoview's
        # ProcessingConfig.__post_init__ walks SPM##/TM###### from there
        # and strips the ".corrected*" suffix to name the sibling fused
        # output dir. No raw stacks required on disk.
        #
        # IsoviewArray(kind="corrected").scan_root is the SPM## subdir
        # under .corrected/, so walk ancestors for the .corrected* dir
        # (same idiom as _submit_stitcher).
        scan_root = Path(arr.scan_root)
        input_dir: Path | None = None
        for ancestor in (scan_root, *scan_root.parents):
            n = ancestor.name
            idx = n.find(".corrected")
            if idx < 0:
                continue
            rest = n[idx + len(".corrected"):]
            if rest == "" or rest.startswith("_"):
                input_dir = ancestor
                break
        if input_dir is None or not input_dir.is_dir():
            self.parent.logger.error(
                "multi_fuse: could not locate a .corrected tree above "
                f"{scan_root}. Open a dataset whose scan root sits "
                "under a *.corrected[_suffix]/ directory."
            )
            return
        # Resolve the active view's params as the scalar defaults; the
        # full per-view dicts go into the *_by_view kwargs so each pair
        # picks up its own settings during multi_fuse.
        active_view = self._fuse_active_view
        if active_view is None or active_view not in self._fuse_view_params:
            active_view = (self._fuse_view_ids or [0])[0]
        active = self._fuse_view_params.get(
            active_view, self._default_fuse_view_params(),
        )

        def _tp(val: int) -> int | None:
            return None if int(val) < 0 else int(val)

        by_view: dict[str, dict[int, Any]] = {
            "blending_method_by_view": {},
            "blending_range_by_view": {},
            "transition_plane_by_view": {},
            "front_flag_by_view": {},
            "flip_z_by_view": {},
            "flip_horizontal_by_view": {},
            "flip_vertical_by_view": {},
            "rotation_by_view": {},
            "search_offsets_x_by_view": {},
            "search_offsets_y_by_view": {},
        }
        for vid, p in self._fuse_view_params.items():
            by_view["blending_method_by_view"][vid] = p["blending_method"]
            by_view["blending_range_by_view"][vid] = int(p["blending_range"])
            by_view["transition_plane_by_view"][vid] = _tp(p["transition_plane"])
            by_view["front_flag_by_view"][vid] = int(p["front_flag"])
            by_view["flip_z_by_view"][vid] = bool(p["flip_z"])
            by_view["flip_horizontal_by_view"][vid] = bool(p["flip_horizontal"])
            by_view["flip_vertical_by_view"][vid] = bool(p["flip_vertical"])
            by_view["rotation_by_view"][vid] = int(p["rotation"])
            by_view["search_offsets_x_by_view"][vid] = [
                int(p["search_x_start"]),
                int(p["search_x_stop"]),
                int(p["search_x_step"]),
            ]
            by_view["search_offsets_y_by_view"][vid] = [
                int(p["search_y_start"]),
                int(p["search_y_stop"]),
                int(p["search_y_step"]),
            ]

        # RAM-cap the worker count. Each tiled-fuse worker is a separate
        # process that holds two full camera volumes + two masks + the
        # fused output (isoview reads whole volumes via read_volume →
        # arr[:]), so peak footprint is several× one Z×Y×X volume. With
        # large IsoView volumes (e.g. 494×2048×2048 = ~3.9 GiB each), the
        # default min(4, cpu//2) workers can exceed RAM and OOM mid-fusion.
        # Only ever reduces the count; never raises it.
        workers = self._ram_capped_fuse_workers(arr, self._workers)

        args = {
            "input_path": str(input_dir),
            "output_dir": self._output_dir or None,
            "output_format": self._output_format,
            "compression": (
                None if self._compression == "none" else self._compression
            ),
            "compression_level": self._compression_level,
            "overwrite": self._overwrite,
            "workers": workers,
            "pyramid": self._pyramid,
            "pyramid_max_layers": self._pyramid_max_layers,
            "output_suffix": (self._fuse_output_suffix or "").strip() or None,
            # Background subtraction (dataset-wide, scalar config fields).
            "subtract_background": bool(self._fuse_subtract_background),
            "background_mode": self._fuse_background_mode,
            "background_percentile": float(self._fuse_background_percentile),
            # Scalar defaults seeded from the active view — these name the
            # output sub-folder and serve as fall-throughs for any view
            # not present in the per-view dicts.
            "blending_method": active["blending_method"],
            "blending_range": int(active["blending_range"]),
            "transition_plane": _tp(active["transition_plane"]),
            "front_flag": int(active["front_flag"]),
            "flip_z": bool(active["flip_z"]),
            "flip_horizontal": bool(active["flip_horizontal"]),
            "flip_vertical": bool(active["flip_vertical"]),
            "rotation": int(active["rotation"]),
            "search_offsets_x": [
                int(active["search_x_start"]),
                int(active["search_x_stop"]),
                int(active["search_x_step"]),
            ],
            "search_offsets_y": [
                int(active["search_y_start"]),
                int(active["search_y_stop"]),
                int(active["search_y_step"]),
            ],
            **by_view,
        }
        # Always pass an explicit timepoints list for fuse: isoview's
        # ProcessingConfig.__post_init__ auto-detects timepoints from
        # input_dir (the RAW root, all 61 TMs even when only the first
        # 10 were corrected). Scan the .corrected SPM## tree to find
        # which TMs actually have corrected output and intersect that
        # with the user's slicing selection.
        from mbo_utilities.arrays.isoview.array import (
            _find_tm_folders, _extract_timepoint, _SPM_PATTERN,
        )
        corrected_root = scan_root if _SPM_PATTERN.match(scan_root.name) else None
        if corrected_root is None:
            for d in scan_root.iterdir() if scan_root.is_dir() else []:
                if d.is_dir() and _SPM_PATTERN.match(d.name):
                    corrected_root = d
                    break
        available_tms: list[int] = []
        if corrected_root is not None:
            available_tms = [
                _extract_timepoint(d.name)
                for d in _find_tm_folders(corrected_root)
            ]
        tps = self._selected_timepoints()
        if available_tms:
            if tps is None:
                args["timepoints"] = available_tms
            else:
                # Map 0-based selection indices onto the sorted list of
                # available TMs. Keeps the slicing UI consistent with
                # what the user sees as "1..N corrected timepoints".
                args["timepoints"] = [
                    available_tms[i] for i in tps if 0 <= i < len(available_tms)
                ]
        elif tps is not None:
            args["timepoints"] = tps
        args.update(self._microscope_kwargs())
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        args.update(crop_state.to_config_args(arr))
        self._spawn(
            "isoview_fuse", args,
            description=f"multi_fuse ({active['blending_method']}): "
                        f"{input_dir.name}",
            output_path=self._output_dir or str(input_dir),
        )

    def _init_iso_selection_state(self, arr: Any) -> int:
        """Initialize / reset timepoint-slicing state for the current array.

        Returns max_frames (1-based). Resets the selection string back to
        ``1:N`` when the user loads a different dataset or the timepoint
        count changes (e.g. switching kinds).
        """
        try:
            max_frames = int(arr.num_timepoints)
        except AttributeError:
            max_frames = int(arr.shape[0]) if arr.shape else 1
        if max_frames < 1:
            max_frames = 1

        current_fpath = str(arr.scan_root) if getattr(arr, "scan_root", None) else ""
        file_changed = self._iso_last_fpath != current_fpath
        if file_changed:
            self._iso_last_fpath = current_fpath

        if file_changed or not hasattr(self, "_iso_tp_selection"):
            self._iso_tp_selection = f"1:{max_frames}"
            self._iso_tp_error = ""
            self._iso_tp_parsed = None
            self._iso_last_max_tp = max_frames

        if self._iso_last_max_tp != max_frames:
            self._iso_last_max_tp = max_frames
            self._iso_tp_selection = f"1:{max_frames}"
            self._iso_tp_parsed = None
            self._iso_tp_error = ""

        if self._iso_tp_parsed is None and not self._iso_tp_error:
            try:
                from mbo_utilities.arrays.features._slicing import (
                    parse_timepoint_selection,
                )
                self._iso_tp_parsed = parse_timepoint_selection(
                    self._iso_tp_selection, max_frames,
                )
            except ValueError as e:
                self._iso_tp_error = str(e)

        return max_frames

    def _emp_label(self, field: str, text: str) -> None:
        """Render an external label for an inline widget.

        Important params (members of :data:`_ISOVIEW_IMPORTANT_FIELDS`)
        get a boxed bold label; everything else is plain text. Pair with
        ``imgui.<widget>("##field", ...)`` + ``imgui.same_line(0,
        item_inner_spacing.x)`` to replicate imgui's default inline-label
        placement, then call this.
        """
        if field in _ISOVIEW_IMPORTANT_FIELDS:
            bold_font = getattr(self.parent, "_bold_font", None)
            draw_boxed_label(text, font=bold_font)
        else:
            imgui.text(text)

    def _draw_isoview_missing_banner(self) -> None:
        warn = imgui.ImVec4(0.95, 0.55, 0.25, 1.0)
        cmd = imgui.ImVec4(0.6, 0.8, 1.0, 1.0)
        imgui.spacing()
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_colored(warn, "isoview not installed")
            imgui.text_colored(cmd, "pip install isoview")
        finally:
            imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_current_dataset(self, arr: Any) -> None:
        """Current dataset block — path + Files (N) popup + shape + size +
        Data state / dx / dy / dz / fs.
        """
        from mbo_utilities.metadata import get_param

        imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
        set_tooltip(
            "The IsoviewArray currently loaded in the viewer.\n"
            "Data state selects which pipelines apply (raw → Correct, "
            "corrected → Fuse/Consolidate, fused → Consolidate)."
        )
        imgui.spacing()
        imgui.spacing()

        # filenames (concatenation order) — collected up-front so the
        # readout block below can render in one wrapped style scope.
        filenames = list(getattr(arr, "filenames", []) or [])
        n_files = len(filenames)
        if self._iso_files_sizer is None:
            self._iso_files_sizer = PopupAutoSize(
                "Dataset files##current_dataset_files_popup",
                auto_resize=False,
            )

        # 2-col table with a slight indent under the section header. The
        # name cell shows the last folder/file component only; full path
        # appears in a hover tooltip so long paths never blow past the
        # side-panel edge.
        path_str = str(arr.scan_root) if getattr(arr, "scan_root", None) else ""
        if path_str:
            short_name = Path(path_str).name or path_str
        else:
            short_name = "(in-memory)"
        size_bytes = _dataset_size_bytes(self, filenames)

        shape = tuple(arr.shape)
        dims = "TCZYX" if len(shape) == 5 else None
        shape_text = " × ".join(str(s) for s in shape)
        if dims and len(dims) == len(shape):
            # dimension labels on their own line under the sizes
            shape_text = f"{shape_text}\n[{','.join(dims)}]"

        md = dict(getattr(arr, "metadata", {}) or {})
        _warn_color = imgui.ImVec4(0.95, 0.45, 0.35, 1.0)
        rows: list[tuple[str, str, str | None, Any]] = [
            ("Name", short_name, path_str or None, None),
            ("Size on disk", _format_size(size_bytes), None, None),
            ("Shape", shape_text, None, None),
            ("Data state", str(arr.kind), None, None),
        ]
        # Frame rate: isoview metadata does not encode fs. Read the exact
        # ``fs`` key only (NOT get_param, whose aliases resolve the
        # reference camera rate ``fps`` as fs). Prefer a user-set value
        # from the metadata editor (Shift+M); when unset, show the row in
        # red with a hint so the missing value is obvious.
        if not getattr(arr, "is_tiled", False):
            custom = getattr(self.parent, "_custom_metadata", None) or {}
            fs_val = custom.get("fs")
            if fs_val is None:
                fs_val = md.get("fs")
            if fs_val is None:
                rows.append((
                    "Frame rate",
                    "not encoded (Shift+M to set)",
                    "isoview metadata does not currently encode the frame "
                    "rate.\nPress Shift+M to open the metadata editor and set it.",
                    _warn_color,
                ))
            else:
                rows.append(("Frame rate", f"{fs_val} Hz", None, None))
        for label, key, unit in (
            ("dx", "dx", "µm"),
            ("dy", "dy", "µm"),
            ("dz", "dz", "µm"),
        ):
            v = get_param(md, key)
            if v is None:
                continue
            rows.append((label, f"{v} {unit}".rstrip(), None, None))

        imgui.indent(8)
        try:
            tflags = (
                imgui.TableFlags_.sizing_stretch_prop
                | imgui.TableFlags_.no_borders_in_body
                | imgui.TableFlags_.pad_outer_x
            )
            if imgui.begin_table("iso_current_dataset_table", 2, tflags):
                imgui.table_setup_column(
                    "field", imgui.TableColumnFlags_.width_fixed, 90.0
                )
                imgui.table_setup_column(
                    "value", imgui.TableColumnFlags_.width_stretch
                )
                for label, value, hover, color in rows:
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    if color is not None:
                        text_wrapped_cell(label, color)
                    else:
                        imgui.text_unformatted(label)
                    imgui.table_set_column_index(1)
                    text_wrapped_cell(value, color)
                    if hover and imgui.is_item_hovered():
                        imgui.set_tooltip(hover)
                imgui.end_table()

            imgui.spacing()
            if imgui.button(f"Files ({n_files})##iso_dataset_files"):
                self._iso_files_sizer.before_open()
                imgui.open_popup("Dataset files##current_dataset_files_popup")
            _draw_dataset_files_popup(
                filenames, None, sizer=self._iso_files_sizer
            )
        finally:
            imgui.unindent(8)

        imgui.spacing()
        imgui.spacing()

    def _suffix_attr_for_mode(self) -> str | None:
        """Which instance attr holds the active mode's output suffix.

        ``None`` when the mode has no suffix concept (shouldn't happen
        for any mode the widget currently supports).
        """
        return {
            _MODE_CORRECT: "_correct_output_suffix",
            _MODE_FUSE: "_fuse_output_suffix",
            _MODE_STITCHER: "_stitcher_suffix",
            _MODE_CONSOLIDATE: "_consolidate_suffix",
        }.get(self._selected_mode)

    def _output_basename_for_mode(self, arr: Any) -> str:
        """Folder/file basename the suffix attaches to.

        - CORRECT: arr is raw, basename = ``arr.scan_root.name``.
        - FUSE / STITCHER: basename = name of the raw root sibling
          (strips ``.corrected`` / ``.fused``).
        - CONSOLIDATE: basename = ``arr.scan_root.name`` (the suffix
          becomes the trailing portion of the .zarr filename).
        """
        sr = Path(arr.scan_root)
        if self._selected_mode in (_MODE_CORRECT, _MODE_CONSOLIDATE):
            return sr.name
        tree = self._iso_input_tree(arr)
        if tree is not None:
            return self._iso_raw_stem(tree)
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        raw = _sibling_raw_root(arr)
        return raw.name if raw is not None else sr.name

    def _overwrite_attr_for_mode(self) -> str | None:
        """Instance attr holding the active mode's overwrite flag."""
        return {
            _MODE_CORRECT: "_overwrite",
            _MODE_FUSE: "_overwrite",
            _MODE_STITCHER: "_overwrite",
            _MODE_CONSOLIDATE: "_consolidate_overwrite",
        }.get(self._selected_mode)

    def _resolved_output_dir(self, arr: Any) -> "Path | None":
        """Absolute path the current mode will write to (for the exists check).

        Mirrors the per-mode output naming so the Output-options block can
        warn + offer Overwrite only when something is actually there.
        """
        mode = self._selected_mode
        if mode == _MODE_CONSOLIDATE:
            return (
                Path(self._consolidate_output_path)
                if self._consolidate_output_path else None
            )
        if mode == _MODE_STITCHER:
            return self._iso_stitcher_dest(arr)
        prefix = _MODE_PREFIX.get(mode)
        if prefix is None:
            return None
        attr = self._suffix_attr_for_mode()
        variant = _norm_variant(getattr(self, attr, "") if attr else "")
        basename = self._output_basename_for_mode(arr)
        if mode == _MODE_CORRECT:
            parent = Path(arr.scan_root).parent
        else:  # FUSE — output sits next to the loaded .corrected tree
            tree = self._iso_input_tree(arr)
            parent = tree.parent if tree is not None else Path(arr.scan_root).parent
        return parent / f"{basename}{prefix}{variant}"

    def _draw_iso_output_options(self, arr: Any) -> None:
        """Universal Output options block: one variant input + a
        filename preview. Every other I/O knob (codec / workers /
        pyramid / overwrite / format) lives in the Parameters popup.

        For the tree-producing steps the prefix (.corrected/.fused/
        .stitcher) is forced; the field is just the optional variant,
        with a leading underscore added automatically.
        """
        imgui.text_colored(_SUBSECTION_COLOR, "Output options")
        prefix = _MODE_PREFIX.get(self._selected_mode)
        set_tooltip(
            f"Output folder = <dataset>{prefix or ''}[_<suffix>]. The "
            f"{prefix} prefix is automatic; type only a suffix (blank = "
            f"bare {prefix}). A leading underscore is added for you."
            if prefix else
            "Filename suffix appended to the source name.",
            align="right",
        )
        imgui.spacing()

        attr = self._suffix_attr_for_mode()
        if attr is None:
            return

        imgui.set_next_item_width(_input_w())
        cur = getattr(self, attr) or ""
        _, new_val = imgui.input_text("Suffix##iso_suffix", cur)
        setattr(self, attr, new_val)

        basename = self._output_basename_for_mode(arr)
        if prefix:
            # forced prefix + optional variant (auto underscore); dir.
            preview = f"{basename}{prefix}{_norm_variant(new_val)}/"
        else:
            # consolidate: field is the full trailing filename component.
            preview = f"{basename}{new_val}"
        # wrap the result line — basename can be long and would otherwise
        # blow past the right edge of the side panel.
        imgui.push_style_color(
            imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0)
        )
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(f"Result: {preview}")
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()

        # Overwrite — shown only when the output already exists (yellow
        # warning), bound to the per-mode flag. When nothing's there the
        # flag is forced off so a fresh path never carries a stale True.
        ow_attr = self._overwrite_attr_for_mode()
        if ow_attr is not None:
            out_dir = self._resolved_output_dir(arr)
            if out_dir is not None and out_dir.exists():
                imgui.spacing()
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.8, 0.2, 1.0), "Directory already exists"
                )
                _, ow = imgui.checkbox(
                    "Overwrite##iso_overwrite", bool(getattr(self, ow_attr))
                )
                setattr(self, ow_attr, ow)
            else:
                setattr(self, ow_attr, False)
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_crop_readout(self, arr: Any) -> None:
        """Crop-bounds Run-tab block — corrected (fuse) data only.

        Mirrors the segmentation / dead-pixel readouts: Edit opens the
        floating crop window, Clear resets, and the summary shows the
        current per-camera bounds. Trailing separator emitted inline.
        """
        if getattr(arr, "kind", None) != "corrected":
            return
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        from mbo_utilities.gui.widgets import isoview_crop as crop_window

        imgui.text_colored(_SUBSECTION_COLOR, "Crop bounds")
        set_tooltip(
            "Per-camera crop applied at fuse time. Edit opens the "
            "side-by-side crop window with draggable bounds.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_crop_edit", imgui.ImVec2(_BTN_W, 0)):
            crop_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Clear##iso_crop_clear", imgui.ImVec2(_BTN_W, 0)):
            crop_state.clear(arr)
        imgui.push_style_color(
            imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0)
        )
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(crop_state.summary(arr))
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_segment_readout(self, arr: Any) -> None:
        """Segmentation threshold Run-tab block — only meaningful on
        raw data (where ``correct_stack`` will run). Hidden otherwise.
        Trailing separator emitted inline so a hidden readout leaves
        no orphan separator behind.
        """
        if getattr(arr, "kind", None) != "raw":
            return
        # Segmentation drives correct_stack only; BigStitcher XML ignores it.
        if self._selected_mode == _MODE_STITCHER:
            return
        from mbo_utilities.gui.widgets import isoview_segment as seg_window

        imgui.text_colored(_SUBSECTION_COLOR, "Segmentation threshold")
        set_tooltip(
            "Threshold + gaussian smoothing + mask percentile that drive "
            "foreground/background separation in correct_stack. Edit "
            "opens a live preview with sliders and per-view masks.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_seg_edit", imgui.ImVec2(_BTN_W, 0)):
            seg_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Reset##iso_seg_reset", imgui.ImVec2(_BTN_W, 0)):
            self._correct_segment_threshold = 0.4
            self._correct_mask_percentile = 1.0
            self._correct_gauss_sigma = 2.0
            self._correct_gauss_kernel = 5
        imgui.push_style_color(
            imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0)
        )
        imgui.push_text_wrap_pos(0.0)
        try:
            imgui.text_unformatted(
                f"threshold={self._correct_segment_threshold:.3f}  "
                f"mask_pct={self._correct_mask_percentile:.2f}  "
                f"sigma={self._correct_gauss_sigma:.2f}  "
                f"kernel={self._correct_gauss_kernel}"
            )
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_deadpixel_readout(self, arr: Any) -> None:
        """Dead-pixel Run-tab block — raw only."""
        if getattr(arr, "kind", None) != "raw":
            return
        # Dead-pixel correction drives correct_stack only; XML ignores it.
        if self._selected_mode == _MODE_STITCHER:
            return
        from mbo_utilities.gui.widgets import isoview_deadpixel as dp_window

        imgui.text_colored(_SUBSECTION_COLOR, "Dead-pixel correction")
        set_tooltip(
            "Median-filter window + background percentile for "
            "dead-pixel detection in correct_stack. Edit opens a live "
            "preview with the flagged pixels highlighted.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Edit...##iso_dp_edit", imgui.ImVec2(_BTN_W, 0)):
            dp_window.open_window(self.parent)
        imgui.same_line()
        if imgui.button("Reset##iso_dp_reset", imgui.ImVec2(_BTN_W, 0)):
            self._correct_background_percentile = 5.0
            self._correct_median_kernel_size = 3
            self._correct_median_kernel_enabled = True
        imgui.push_style_color(
            imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.65, 1.0)
        )
        imgui.push_text_wrap_pos(0.0)
        try:
            on = "on" if self._correct_median_kernel_enabled else "off"
            imgui.text_unformatted(
                f"median={on}  kernel={self._correct_median_kernel_size}  "
                f"bg_pct={self._correct_background_percentile:.2f}"
            )
        finally:
            imgui.pop_text_wrap_pos()
            imgui.pop_style_color()
        imgui.spacing()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_iso_data_slicing(self, max_frames: int) -> None:
        """Data-slicing section — Set slice button + count preview.

        The row is labeled with the loaded array's T-axis slider name
        ("Tile" for tiled trees, "Timepoint" otherwise); the selected
        indices forwarded to the pipeline are unchanged.
        """
        tp_label, _, _ = resolve_dim_labels(self.parent)

        imgui.text_colored(_SUBSECTION_COLOR, "Data slicing")
        set_tooltip(
            f"Restrict which {tp_label.lower()} feed the run. Click Set "
            "slice to define a range using start:stop[:step] syntax "
            "(1-based, inclusive). Leaving the full range selects all.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Set slice##iso_slice", imgui.ImVec2(_BTN_W, 0)):
            self._iso_slicing_open = True

        # preview
        n_tp = (
            self._iso_tp_parsed.count
            if self._iso_tp_parsed is not None
            else max_frames
        )
        imgui.text(f"{tp_label}: {n_tp}/{max_frames}")

        if self._selected_mode == _MODE_CORRECT:
            self._draw_iso_camera_subset()

        self._draw_iso_slicing_popup(max_frames, tp_label)
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_slicing_popup(self, max_frames: int, tp_label: str) -> None:
        """Slicing popup — pick which timepoints to process."""
        popup_id = f"{tp_label}##iso_slice"
        if self._iso_slicing_open:
            imgui.open_popup(popup_id)
            self._iso_slicing_open = False

        imgui.set_next_window_size(
            imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever,
        )
        if imgui.begin_popup(popup_id):
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.8, 0.2, 1.0), tp_label,
            )
            imgui.same_line()
            imgui.text_disabled("(?)")
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.push_text_wrap_pos(imgui.get_font_size() * 30.0)
                imgui.text_unformatted(
                    "Format: start:stop or start:stop:step\n"
                    "Exclude: 1:100,50:60 = 1-100 excluding 50-60\n"
                    "1-based, inclusive."
                )
                imgui.pop_text_wrap_pos()
                imgui.end_tooltip()
            imgui.dummy(imgui.ImVec2(0, 4))

            # num_planes=1 → draw_selection_table will skip the Z-Planes row
            draw_selection_table(
                self,
                max_frames,
                1,
                tp_attr="_iso_tp",
                z_attr="_iso_z",
                id_suffix="_iso",
                num_channels=1,
                c_attr="_iso_c",
                tp_label=tp_label,
            )

            imgui.spacing()
            if imgui.button("Close##iso_slice_close", imgui.ImVec2(80, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

    def _available_cameras(self, arr: Any) -> list[int]:
        """Camera indices present in the loaded array. raw/corrected view_keys
        are plain camera ints; fused view_keys are tuples (take the first)."""
        vk = getattr(arr, "_view_keys", None)
        if vk is None:
            vk = getattr(arr, "view_keys", None) or []
        out: list[int] = []
        for v in vk:
            if isinstance(v, (tuple, list)) and v:
                out.append(int(v[0]))
            else:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    continue
        return sorted(set(out))

    def _selected_cameras(self, arr: Any) -> list[int] | None:
        """Selected camera indices, or None when the full set is selected (in
        which case the pipeline auto-detects every camera)."""
        avail = self._available_cameras(arr)
        if len(avail) <= 1:
            return None
        sel = self._iso_selected_cameras
        if sel is None:
            return None
        chosen = [c for c in avail if c in sel]
        if not chosen or len(chosen) == len(avail):
            return None
        return chosen

    def _draw_iso_camera_subset(self) -> None:
        """Camera checkboxes — process only the checked cameras (correct mode)."""
        arr = self._get_array()
        cams = self._available_cameras(arr) if arr is not None else []
        if len(cams) <= 1:
            return
        # (re)seed on first draw or when the dataset's cameras changed.
        if self._iso_selected_cameras is None or not (self._iso_selected_cameras <= set(cams)):
            self._iso_selected_cameras = set(cams)
        imgui.spacing()
        imgui.text_colored(_SUBSECTION_COLOR, "Cameras")
        set_tooltip("Process only the checked views.", align="right")
        from mbo_utilities.arrays.isoview.array import camera_view_label
        for i, c in enumerate(cams):
            checked = c in self._iso_selected_cameras
            changed, new = imgui.checkbox(
                f"{camera_view_label(c)}##iso_cam_{c}", checked
            )
            if changed:
                if new:
                    self._iso_selected_cameras.add(c)
                else:
                    self._iso_selected_cameras.discard(c)
            if i < len(cams) - 1:
                imgui.same_line()
        n_sel = len([c for c in cams if c in self._iso_selected_cameras])
        imgui.text_disabled(f"{n_sel}/{len(cams)} cameras")

    def _selected_timepoints(self) -> list[int] | None:
        """Return a 0-based list of selected timepoint indices, or None
        when the full range is selected (in which case the pipeline
        auto-detects all available timepoints).
        """
        parsed = self._iso_tp_parsed
        if parsed is None:
            return None
        try:
            count = int(parsed.count)
            max_tp = int(self._iso_last_max_tp)
        except (TypeError, ValueError):
            return None
        # full range → leave it None so the pipeline picks up everything
        if count >= max_tp and not getattr(parsed, "exclude_str", ""):
            return None
        # TimeSelection.final_indices is already 0-based and post-exclude.
        return [int(i) for i in parsed.final_indices]

    def draw_config(self) -> None:
        """Render the Suite2p-style side panel.

        Layout (top → bottom):
          1. Current dataset (path + Files popup + shape + size + dx/dy/dz/fs)
          2. Action (mode combo — shown only when multiple apply)
          3. Output options (format / compressor / workers / pyramid / overwrite)
          4. Data slicing (timepoint subset selector)
          5. Parameters and settings (small "Open" button → popup with
             algorithm-only knobs)
          6. Run button (big, green, centered)

        The output path is derived from the raw-data folder name in
        ``_ensure_defaults`` — no user-facing path picker.
        """
        arr = self._get_array()
        if arr is None:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                "No isoview array loaded.",
            )
            return

        self._ensure_defaults(arr)

        # Top-of-tab banner: an isoview dataset is loaded but the python
        # package isn't importable. The Correct/Fuse/BigStitcher XML
        # actions all need it.
        if not _isoview_pkg_available():
            self._draw_isoview_missing_banner()

        modes = _available_modes(arr)
        if not modes:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                f"No isoview action available for kind={arr.kind!r}.",
            )
            return
        if self._selected_mode not in modes:
            self._selected_mode = modes[0]

        # Crop window just closed and we owe the user a reopened
        # Parameters popup. open_popup() must be called BEFORE
        # _draw_settings_popup so begin_popup_modal sees the open state
        # this frame.
        crop_open_now = getattr(self.parent, "_iso_crop_window_open", False)
        if (
            self._reopen_settings_after_crop
            and self._crop_was_open
            and not crop_open_now
        ):
            self._show_settings_popup = True
            self._popup_just_opened = True
            imgui.open_popup(
                "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
            )
            self._reopen_settings_after_crop = False
        self._crop_was_open = crop_open_now

        # popup lifecycle: open_popup() must be called from the same
        # imgui frame as begin_popup_modal(); we keep the visibility flag
        # for state but always re-open here when it's True and not
        # already open (handles re-renders).
        self._draw_settings_popup(arr)

        # initialize slicing state up-front so the preview reflects edits
        max_frames = self._init_iso_selection_state(arr)

        imgui.spacing()
        self._draw_iso_current_dataset(arr)

        imgui.separator()
        imgui.spacing()

        # PROCESSING STEP — centered, no trailing separator so the
        # combo reads as a header for everything below.
        label = "Processing Step"
        avail = imgui.get_content_region_avail().x
        label_w = imgui.calc_text_size(label).x
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max(0.0, (avail - label_w) * 0.5)
        )
        imgui.text_colored(_SUBSECTION_COLOR, label)
        set_tooltip(
            "Which IsoView pipeline to run on this dataset.\n"
            "Correct/Fuse require `pip install isoview`.",
            align="right",
        )
        imgui.spacing()
        try:
            idx = modes.index(self._selected_mode)
        except (ValueError, TypeError):
            idx = 0
        combo_w = 180.0
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max(0.0, (avail - combo_w) * 0.5)
        )
        imgui.set_next_item_width(combo_w)
        if len(modes) > 1:
            changed, new_idx = imgui.combo("##iso_mode", idx, modes)
            if changed:
                self._selected_mode = modes[new_idx]
        else:
            imgui.begin_disabled()
            imgui.combo("##iso_mode", 0, modes)
            imgui.end_disabled()
        imgui.spacing()

        # OUTPUT OPTIONS — format / compressor / workers / pyramid /
        # overwrite. Moved out of the Parameters popup so the Run tab
        # owns all file-I/O knobs.
        self._draw_iso_output_options(arr)
        imgui.separator()
        imgui.spacing()

        # DATA SLICING — restrict the run to a subset of timepoints (tiles
        # for tiled trees). BigStitcher XML can only filter tiles, so hide
        # the section there for non-tiled trees rather than show a no-op.
        if not (
            self._selected_mode == _MODE_STITCHER
            and not bool(getattr(arr, "is_tiled", False))
        ):
            self._draw_iso_data_slicing(max_frames)
            imgui.separator()
            imgui.spacing()

        # CROP BOUNDS — corrected (fuse) only. Edit opens the floating
        # crop window. Mirrors the seg/dead-pixel readouts below.
        self._draw_iso_crop_readout(arr)

        # SEGMENTATION — adaptive threshold + gauss params (non-fused
        # only). Emits its own trailing separator when content is drawn.
        self._draw_iso_segment_readout(arr)

        # DEAD-PIXEL — median filter + background percentile (non-fused).
        self._draw_iso_deadpixel_readout(arr)

        # PARAMETERS AND SETTINGS — small Open button → popup.
        imgui.text_colored(_SUBSECTION_COLOR, "Parameters and settings")
        set_tooltip(
            "Open the algorithm-parameter popup: segmentation, blending, "
            "registration search, microscope overrides. File-I/O knobs "
            "live in the Output options section above.",
            align="right",
        )
        imgui.spacing()
        if imgui.button("Open##iso_pipe_settings", imgui.ImVec2(_BTN_W, 0)):
            self._show_settings_popup = True
            self._popup_just_opened = True
            # NOTE: before_open() is called inside _draw_settings_popup
            # (right before begin_popup_modal) because the popup body is
            # rendered earlier in draw_config than this button — buffering
            # a set_next_window_pos here would be discarded at end of frame.
            imgui.open_popup(
                "Isoview Pipeline Settings##isoview_pipeline_settings_popup"
            )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # RUN — big green, centered.
        has_path = bool(self._current_output_path())
        run_label = _RUN_LABELS.get(self._selected_mode or "", "Run")
        avail = imgui.get_content_region_avail().x
        btn_w = min(_RUN_W, max(80.0, avail))
        if avail > btn_w:
            imgui.set_cursor_pos_x(
                imgui.get_cursor_pos_x() + (avail - btn_w) * 0.5
            )
        with _green_button():
            if not has_path:
                imgui.begin_disabled()
            clicked = imgui.button(run_label, imgui.ImVec2(btn_w, 0))
            if not has_path:
                imgui.end_disabled()
        if (
            not has_path
            and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        ):
            imgui.set_tooltip(
                "No output path resolved for this dataset."
            )
        if clicked and has_path:
            self._submit_active(arr)

        if self._last_status:
            imgui.spacing()
            color = (
                imgui.ImVec4(0.8, 0.4, 0.4, 1.0)
                if self._last_status.startswith(("Failed", "Output", "Dialog"))
                else imgui.ImVec4(0.5, 0.85, 0.5, 1.0)
            )
            imgui.text_colored(color, self._last_status)
