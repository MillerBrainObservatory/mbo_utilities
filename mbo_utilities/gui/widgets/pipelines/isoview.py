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
    set_tooltip,
    tooltip_marks_right,
)
from mbo_utilities.gui._selection_ui import draw_selection_table
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


def _default_workers() -> int:
    """Conservative default: half the logical CPUs, capped at 4."""
    cpu = os.cpu_count() or 1
    return max(1, min(4, cpu // 2))


def _auto_detect_spark_jar() -> str:
    """Find a bigstitcher-spark fat JAR built locally, else return empty."""
    home = Path.home()
    candidates = list((home / "repos" / "BigStitcher-Spark" / "target").glob(
        "BigStitcher-Spark-*-SNAPSHOT.jar"
    ))
    # prefer non-sources, non-tests jars
    real = [
        p for p in candidates
        if "sources" not in p.name and "tests" not in p.name
    ]
    return str(real[0]) if real else ""


def _auto_detect_spark_jdk() -> str:
    """Find a portable JDK 11 in the conventional side-dir, else empty."""
    home = Path.home()
    candidates = sorted(
        (home / "repos" / "bigstitcher-spark-build").glob("jdk-11*")
    )
    for p in candidates:
        if (p / "bin" / "java.exe").is_file() or (p / "bin" / "java").is_file():
            return str(p)
    return ""


def _input_w() -> float:
    """Default width for numeric inputs in the popup."""
    return hello_imgui.em_size(8)


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
        self._stitcher_suffix: str = ".stitcher"
        # Opt-in feature-based coarse alignment (3D blob detection +
        # pairwise-difference histogram). When False, only the
        # calibration affines are emitted; the user runs BigStitcher's
        # interest-point registration after opening the XML.
        self._stitcher_coarse_align: bool = True

        # bigstitcher-spark registration step (opt-in via separate "Run
        # registration" button — mutates the exported dataset.xml).
        # ``spark_root`` is either the executables directory produced
        # by bigstitcher-spark's install script, or a fat JAR file.
        # ``java_home`` overrides the system Java (Spark 3.3.x requires
        # Java 8-17; >=21 fails at runtime). Defaults auto-detect the
        # built JAR + JDK 11 in ~/repos/ if present.
        self._stitcher_spark_root: str = _auto_detect_spark_jar()
        self._stitcher_spark_java_home: str = _auto_detect_spark_jdk()
        # Detection / match defaults are tuned for cellular fluorescence
        # data (the common isoview case — GCaMP-labelled neurons,
        # spheroids, etc.), not sub-resolution beads. sigma=4.0 matches
        # ~6-pixel cell bodies, threshold=0.0004 keeps ~80-150 detections
        # per view (we need >=12 correspondences after descriptor
        # matching), and FAST_ROTATION tolerates the orthogonal
        # VW00<->VW90 geometry where PRECISE_TRANSLATION's local
        # descriptor fails. For bead datasets, dial sigma=1.8 /
        # threshold=0.008 / PRECISE_TRANSLATION back in via the popup.
        self._stitcher_spark_label: str = "beads"
        self._stitcher_spark_sigma: float = 4.0
        self._stitcher_spark_threshold: float = 0.0004
        self._stitcher_spark_min_intensity: float = 0.0
        self._stitcher_spark_max_intensity: float = 65535.0
        self._stitcher_spark_downsample_xy: int = 2
        self._stitcher_spark_downsample_z: int = 1
        self._stitcher_spark_match_algorithm: str = "FAST_ROTATION"
        self._stitcher_spark_transformation_model: str = "AFFINE"
        self._stitcher_spark_regularization_model: str = "RIGID"
        self._stitcher_spark_dialog: Any = None

        # Fused-tree top-level suffix. Distinct from
        # `_fuse_output_suffix` which is a per-method tag inside the
        # `<raw>.fused/<method>[_<output_suffix>]/` layout.
        self._fuse_fused_suffix: str = ".fused"

        # Consolidate-mode filename suffix (relative to scan_root.name).
        # Re-derived per dataset in :meth:`_ensure_defaults` so it
        # reflects the kind that was loaded.
        self._consolidate_suffix: str = "_isoview-corrected.zarr"

        # Correct-mode state
        self._correct_corrected_suffix: str = ".corrected"
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

        # Fuse-mode state
        self._fuse_blending_method: str = "geometric"
        self._fuse_output_suffix: str = ""  # appended to <method> folder name
        self._fuse_blending_range: int = 20
        self._fuse_flip_z: bool = False
        self._fuse_flip_horizontal: bool = True
        self._fuse_flip_vertical: bool = False
        self._fuse_rotation: int = 0
        self._fuse_transition_plane: int = -1  # -1 = None (center)
        self._fuse_front_flag: int = 2
        self._fuse_search_x_start: int = -50
        self._fuse_search_x_stop: int = 50
        self._fuse_search_x_step: int = 10
        self._fuse_search_y_start: int = -50
        self._fuse_search_y_stop: int = 50
        self._fuse_search_y_step: int = 10

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

        # Files popup lazy state
        self._iso_files_sizer: PopupAutoSize | None = None
        self._iso_slicing_sizer: PopupAutoSize | None = None

    def _get_array(self) -> Any | None:
        iw = getattr(self.parent, "image_widget", None)
        if iw is None or not iw.data:
            return None
        return _unwrap_array(iw.data[0])

    def _ensure_defaults(self, arr: Any) -> None:
        """Compute per-dataset defaults once. Re-runs when the loaded
        IsoviewArray's scan_root changes (different dataset opened).
        """
        sr = Path(arr.scan_root)
        if self._initialized_for_scan_root == sr:
            return
        self._initialized_for_scan_root = sr

        # Default consolidate output: drop a .zarr sibling next to the
        # input scan_root. Suffix is kind-aware so the result for a
        # corrected SPM00 lands at .corrected/SPM00_isoview-corrected.zarr
        # and a fused method dir lands at <method>_isoview-fused.zarr.
        self._consolidate_suffix = f"_isoview-{arr.kind}.zarr"
        self._consolidate_output_path = str(
            sr.parent / f"{sr.name}{self._consolidate_suffix}"
        )

        # Default BigStitcher destination: <raw_root><stitcher_suffix>/
        # (sibling of the corrected and fused trees, matching how isoview
        # computes config.stitcher_dir in ProcessingConfig.__post_init__).
        # Resolves via _sibling_raw_root which handles either
        # ``.corrected`` or ``.fused`` ancestors. Falls back to
        # sr.parent when no raw sibling is found so the readout still
        # shows something sensible.
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        raw = _sibling_raw_root(arr) if arr.kind in ("corrected", "fused") else None
        if raw is None:
            anchor = sr.parent
            for suf in (".corrected", ".fused"):
                if anchor.name.endswith(suf):
                    raw = anchor.parent / anchor.name[: -len(suf)]
                    break
        if raw is not None:
            self._stitcher_output_path = str(
                raw.parent / f"{raw.name}{self._stitcher_suffix}"
            )
        else:
            self._stitcher_output_path = ""

        # Correct / Fuse output_dir defaults:
        #   raw       → leave EMPTY so isoview's ProcessingConfig derives
        #               output_dir from input_dir + corrected_suffix.
        #               Pre-filling here freezes a stale ".corrected"
        #               path that overrides the user's suffix at submit
        #               time (isoview honors output_dir when set and
        #               ignores corrected_suffix).
        #   corrected → .corrected/ root (parent of scan_root). The
        #               fuse task reads from here; it doesn't write a
        #               new corrected tree, so no suffix conflict.
        if arr.kind == "corrected":
            self._output_dir = str(sr.parent)
        else:
            self._output_dir = ""

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

    def _current_output_path(self) -> str:
        explicit = getattr(self, self._output_path_attr(), "") or ""
        if explicit:
            return explicit
        # CORRECT mode against raw data intentionally leaves _output_dir
        # empty so isoview's ProcessingConfig derives the output from
        # input_dir + corrected_suffix. Mirror that derivation here so
        # the Run-tab gate doesn't block on the empty string.
        if self._selected_mode == _MODE_CORRECT:
            arr = self._get_array()
            sr = getattr(arr, "scan_root", None) if arr is not None else None
            if sr is not None:
                suffix = self._correct_corrected_suffix or ""
                return f"{Path(sr).resolve()}{suffix}"
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
            ("Microscope (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_correct_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("I/O options", self._draw_correct_io_box),
            ("Segmentation", self._draw_correct_segmentation_box),
            ("Pixel correction", self._draw_correct_advanced_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Shared", self._draw_correct_shared_box),
            ("Microscope (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_fuse_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("I/O options", self._draw_fuse_io_box),
            ("Fusion", self._draw_fuse_blending_box),
            ("View transforms", self._draw_fuse_transforms_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Crop bounds", self._draw_fuse_crop_box),
            ("Registration search", self._draw_fuse_search_box, True),
            ("Microscope (override XML)",
             self._draw_microscope_overrides_box, True),
        ])

    def _draw_fuse_crop_box(self) -> None:
        """Crop-bounds editor inside the Fuse Parameters popup.

        Edit closes this popup and hands off to the floating crop window
        (preview + drags); when that window closes the popup is
        reopened. Manual numeric entries below the buttons write
        straight into ``_isoview_crop_state`` so they're picked up by
        the floating window the next time Edit is pressed.
        """
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        from mbo_utilities.gui.widgets import isoview_crop as crop_window

        arr = self._get_array()
        shape = getattr(arr, "shape", None) if arr is not None else None
        if not shape or len(shape) != 5:
            imgui.text_disabled("(no compatible dataset)")
            return
        _, _, nz, ny, nx = shape

        meta = getattr(arr, "metadata", None) or {}
        cv = meta.get("camera_view_map") or {0: 0, 1: 0, 2: 90, 3: 90}
        views = sorted({int(v) for v in cv.values()})
        if not views:
            views = [0, 90]

        with tooltip_marks_right():
            if imgui.button("Edit...##fuse_crop_edit", imgui.ImVec2(_BTN_W, 0)):
                self._show_settings_popup = False
                imgui.close_current_popup()
                self._reopen_settings_after_crop = True
                crop_window.open_window(self.parent)
            imgui.same_line()
            if imgui.button("Clear##fuse_crop_clear", imgui.ImVec2(_BTN_W, 0)):
                crop_state.clear(arr)
            set_tooltip(
                "Edit opens the side-by-side crop window. Manual values "
                "below are reactive — typing them then clicking Edit "
                "pre-loads them into the preview."
            )

        imgui.spacing()
        crops = crop_state.get_crops(arr)
        for view in views:
            existing = crops.get(view) or {}
            bounds = {
                "Z": existing.get("z", (0, nz)),
                "Y": existing.get("y", (0, ny)),
                "X": existing.get("x", (0, nx)),
            }
            limits = {"Z": (0, nz), "Y": (0, ny), "X": (0, nx)}
            imgui.text_colored(_SUBSECTION_COLOR, f"VW{view:02d}")
            imgui.indent(12)
            changed = False
            new_bounds: dict[str, tuple[int, int]] = {}
            for axis in ("Z", "Y", "X"):
                lo, hi = bounds[axis]
                vmin, vmax = limits[axis]
                new_lo, new_hi = self._draw_axis_inputs(
                    view, axis, lo, hi, vmin, vmax,
                )
                new_bounds[axis] = (new_lo, new_hi)
                if (new_lo, new_hi) != (lo, hi):
                    changed = True
            imgui.unindent(12)
            if changed:
                crop_state.set_view_bounds(
                    arr, view,
                    z=new_bounds["Z"], y=new_bounds["Y"], x=new_bounds["X"],
                    shape=(nz, ny, nx),
                )
            imgui.spacing()

    def _draw_axis_inputs(
        self, view: int, axis: str, lo: int, hi: int, vmin: int, vmax: int,
    ) -> tuple[int, int]:
        """One row: ``{axis}: [lo] [hi]  /max``. Clamps lo<hi inline."""
        imgui.text(f"{axis}:")
        imgui.same_line()
        imgui.set_next_item_width(72)
        _, lo_new = imgui.input_int(
            f"##fuse_crop_vw{view}_{axis}_lo", int(lo), 0, 0,
        )
        imgui.same_line()
        imgui.set_next_item_width(72)
        _, hi_new = imgui.input_int(
            f"##fuse_crop_vw{view}_{axis}_hi", int(hi), 0, 0,
        )
        imgui.same_line()
        imgui.text_disabled(f"/ {vmax}")
        lo_new = max(int(vmin), min(int(lo_new), int(vmax) - 1))
        hi_new = max(lo_new + 1, min(int(hi_new), int(vmax)))
        return int(lo_new), int(hi_new)

    def _draw_stitcher_popup_rows(self) -> None:
        """BigStitcher Parameters popup.

        Single row, two columns:
        - Column 1: a "BigStitcher / Spark registration" box that
          stacks every registration-related subgroup as a collapsing
          header (View transforms + Interest point detection open by
          default, Spark runtime + Matching & solver collapsed).
        - Column 2: Microscope (override XML), unrelated to BigStitcher
          but reused across isoview pipelines.
        """
        self._draw_popup_columns([
            ("BigStitcher / Spark registration",
             self._draw_stitcher_grouped_box),
            ("Microscope (override XML)",
             self._draw_microscope_overrides_box),
        ])

    def _draw_stitcher_grouped_box(self) -> None:
        """Render all BigStitcher / Spark subgroups in a single column
        as colored subsection headers (always visible — no collapsing).
        """
        subgroups = [
            ("View transforms", self._draw_stitcher_transforms_box),
            ("Interest point detection",
             self._draw_stitcher_spark_detect_box),
            ("Spark runtime", self._draw_stitcher_spark_runtime_box),
            ("Matching & solver", self._draw_stitcher_spark_match_box),
        ]
        for i, (title, draw_fn) in enumerate(subgroups):
            if i > 0:
                imgui.spacing()
                imgui.spacing()
            imgui.text_colored(_SUBSECTION_COLOR, title)
            imgui.separator()
            imgui.spacing()
            draw_fn()

    def _draw_stitcher_transforms_box(self) -> None:
        with tooltip_marks_right():
            _, self._stitcher_coarse_align = imgui.checkbox(
                "Coarse align (blob features)", self._stitcher_coarse_align,
            )
            set_tooltip(
                "Opt-in. Runs a feature-based coarse registration "
                "(3D blob detection in both views, pairwise-difference "
                "histogram in shared world space) and bakes the "
                "resulting translation into VW00's calibration as a "
                "second ViewTransform. Off by default — when off, "
                "only the per-view calibration affines are written "
                "and you'd run BigStitcher's interest-point "
                "registration manually to align the views."
            )

    def _draw_stitcher_spark_runtime_box(self) -> None:
        """Spark JAR / JDK paths — set once per machine, rarely re-tuned."""
        with tooltip_marks_right():
            _text_disabled_wrapped("Spark JAR or executables dir:")
            # Reserve room for the folder-browse button so the input
            # field doesn't push the icon off the right edge of the
            # column. Inner_spacing + button width on the right.
            style = imgui.get_style()
            btn_w = imgui.calc_text_size(fa.ICON_FA_FOLDER_OPEN).x \
                + 2 * style.frame_padding.x
            avail_x = imgui.get_content_region_avail().x
            imgui.set_next_item_width(
                max(40.0, avail_x - style.item_inner_spacing.x - btn_w)
            )
            _, self._stitcher_spark_root = imgui.input_text(
                "##spark_root", self._stitcher_spark_root or "",
            )
            imgui.same_line(0, style.item_inner_spacing.x)
            if imgui.button(fa.ICON_FA_FOLDER_OPEN + "##spark_root_browse"):
                default = self._stitcher_spark_root or str(Path.home())
                try:
                    self._stitcher_spark_dialog = pfd.select_folder(
                        "Select bigstitcher-spark executables dir", default,
                    )
                except Exception:
                    self._stitcher_spark_dialog = None
            set_tooltip(
                "Path to bigstitcher-spark — either the directory "
                "produced by its ./install script (containing "
                "detect-interestpoints, match-interestpoints, solver "
                "scripts) or a fat JAR (mvn -P fatjar)."
            )

            _text_disabled_wrapped(
                "Java home (JDK 8-17 — Spark 3.3 incompat with 21+):"
            )
            imgui.set_next_item_width(-1)
            _, self._stitcher_spark_java_home = imgui.input_text(
                "##spark_java_home", self._stitcher_spark_java_home or "",
            )
            set_tooltip(
                "Path to a JDK 8-17 install (Spark 3.3.x requires it; "
                "Java >=21 fails at runtime). Leave blank to use the "
                "system default `java` on PATH."
            )

    def _draw_stitcher_spark_detect_box(self) -> None:
        """Difference-of-Gaussian detection parameters. The two knobs
        users actually retune between runs are ``sigma`` (feature size
        in voxels at the downsampled scale) and ``threshold`` (peak
        cutoff after normalization). Defaults are bead-tuned — for
        calcium-imaging / cellular data sigma ~3-4 and threshold
        ~0.0005-0.001 usually find enough features.
        """
        with tooltip_marks_right():
            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_label = imgui.input_text(
                "Label", self._stitcher_spark_label or "beads",
            )
            set_tooltip(
                "Interest-point label written into the XML (`-l`). "
                "Distinct labels let multiple detection runs coexist "
                "in the same dataset."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_sigma = imgui.input_float(
                "DoG sigma", self._stitcher_spark_sigma, 0.1, 1.0, "%.2f",
            )
            set_tooltip(
                "Difference-of-Gaussian sigma (`-s`), in voxels at "
                "the downsampled scale. Match your typical feature "
                "diameter / ~2.355 (Gaussian FWHM relation). 1.8 for "
                "beads, 3-4 for cell bodies, larger for blobs."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_threshold = imgui.input_float(
                "DoG threshold", self._stitcher_spark_threshold,
                0.0005, 0.005, "%.4f",
            )
            set_tooltip(
                "DoG response threshold (`-t`) after intensity "
                "normalization. Lower = more (and weaker) detections. "
                "0.008 = bead default, 0.001-0.002 for cellular data. "
                "Aim for 100-500 points/view in the worker log."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_min_intensity = imgui.input_float(
                "Min intensity", self._stitcher_spark_min_intensity,
                0.0, 100.0, "%.1f",
            )
            set_tooltip(
                "Block normalization minimum (`--minIntensity`). "
                "Mandatory — Spark blocks can't auto-detect this. "
                "Use 0 unless you have a hot background."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_max_intensity = imgui.input_float(
                "Max intensity", self._stitcher_spark_max_intensity,
                0.0, 1000.0, "%.1f",
            )
            set_tooltip(
                "Block normalization maximum (`--maxIntensity`). "
                "65535 for uint16, 255 for uint8. The DoG threshold "
                "applies to the [0, 1]-normalized signal."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_downsample_xy = imgui.input_int(
                "Downsample XY", self._stitcher_spark_downsample_xy, 1, 1,
            )
            set_tooltip(
                "`-dsxy`. Detection is performed on this downsampled "
                "resolution; results are then re-projected. 2 is "
                "usually fine and 4x faster than 1."
            )

            imgui.set_next_item_width(_input_w())
            _, self._stitcher_spark_downsample_z = imgui.input_int(
                "Downsample Z", self._stitcher_spark_downsample_z, 1, 1,
            )
            set_tooltip(
                "`-dsz`. Z downsampling. Usually 1 for already-"
                "anisotropic light-sheet data."
            )

    def _draw_stitcher_spark_match_box(self) -> None:
        """Pairwise matching algorithm + solver model. These change
        much less often than detection params — set once per dataset
        type, leave alone.

        Labels live above each combo (not to the right) so the long
        enum previews like ``PRECISE_TRANSLATION`` aren't clipped by
        a narrow column.
        """
        with tooltip_marks_right():
            algos = [
                "PRECISE_TRANSLATION", "FAST_TRANSLATION",
                "FAST_ROTATION", "ICP",
            ]
            try:
                idx = algos.index(self._stitcher_spark_match_algorithm)
            except ValueError:
                idx = 0
            _text_disabled_wrapped("Match algorithm")
            imgui.set_next_item_width(-1)
            changed, new_idx = imgui.combo(
                "##spark_match_algo", idx, algos,
            )
            if changed:
                self._stitcher_spark_match_algorithm = algos[new_idx]
            set_tooltip(
                "`-m`. PRECISE_TRANSLATION = high-accuracy descriptor "
                "matcher, needs >=12 points/view. FAST_TRANSLATION = "
                "geometric hashing, tolerates very different content. "
                "FAST_ROTATION = handles small rotations between "
                "views. ICP = refinement after coarse alignment."
            )

            tms = ["AFFINE", "RIGID", "TRANSLATION"]
            try:
                idx = tms.index(self._stitcher_spark_transformation_model)
            except ValueError:
                idx = 0
            _text_disabled_wrapped("Transformation")
            imgui.set_next_item_width(-1)
            changed, new_idx = imgui.combo(
                "##spark_match_tm", idx, tms,
            )
            if changed:
                self._stitcher_spark_transformation_model = tms[new_idx]
            set_tooltip(
                "`-tm`. Solver's per-view transformation model. "
                "AFFINE allows shear/anisotropic scaling; RIGID is "
                "rotation+translation only; TRANSLATION is pure shift."
            )

            rms = ["RIGID", "AFFINE", "TRANSLATION", "IDENTITY", "NONE"]
            try:
                idx = rms.index(self._stitcher_spark_regularization_model)
            except ValueError:
                idx = 0
            _text_disabled_wrapped("Regularize with")
            imgui.set_next_item_width(-1)
            changed, new_idx = imgui.combo(
                "##spark_match_rm", idx, rms,
            )
            if changed:
                self._stitcher_spark_regularization_model = rms[new_idx]
            set_tooltip(
                "`-rm`. Regularization model. Blends the main "
                "transformation toward a simpler one (lambda=0.1 by "
                "default). RIGID keeps the affine close to a rigid "
                "transform — good for two-view orthogonal stitching."
            )

    def _draw_consolidate_io_box(self) -> None:
        """Consolidate I/O options for the Parameters popup."""
        with tooltip_marks_right():
            _, self._consolidate_overwrite = imgui.checkbox(
                "Overwrite", self._consolidate_overwrite,
            )
            set_tooltip("Replace any existing zarr at the output path.")

            _, self._consolidate_pyramid = imgui.checkbox(
                "Pyramid", self._consolidate_pyramid,
            )
            set_tooltip(
                "Generate the OME-NGFF resolution pyramid (recommended)."
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
            set_tooltip("Inner-chunk compression codec.")

            imgui.set_next_item_width(_input_w())
            _, self._consolidate_compression_level = imgui.input_int(
                "Level", self._consolidate_compression_level, 1, 1,
            )
            set_tooltip("Compression level (0–9).")

    def _draw_correct_io_box(self) -> None:
        """Correct-mode I/O options for the Parameters popup. The
        ``corrected_suffix`` lives in the Run-tab Output section now;
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
            set_tooltip(
                "Volume container format written per (timepoint, camera)."
            )

            if self._output_format == "zarr":
                compressors = ["zstd", "gzip", "blosc-zstd", "blosc-lz4", "none"]
                try:
                    cidx = compressors.index(self._compression)
                except ValueError:
                    cidx = 0
                imgui.set_next_item_width(_input_w())
                c_changed, c_new = imgui.combo("Compressor", cidx, compressors)
                if c_changed:
                    self._compression = compressors[c_new]
                set_tooltip("Codec applied inside the volume containers.")

                imgui.set_next_item_width(_input_w())
                _, self._compression_level = imgui.input_int(
                    "Level", self._compression_level, 1, 1,
                )
                set_tooltip("Compression level (0–9; higher = smaller / slower).")

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers", self._workers, 1, 2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel workers for timepoint/tile processing.\n"
                "Sequential = 1. Bottleneck is RAM, not cores: each "
                "worker holds a full 3D camera volume in memory. Start "
                f"at 2–4 and watch RAM before raising (soft cap {_MAX_WORKERS})."
            )

            _, self._overwrite = imgui.checkbox("Overwrite", self._overwrite)
            set_tooltip(
                "Re-process even when an output already exists at the target."
            )

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Generate the resolution pyramid (zarr / OME-TIFF only)."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels", self._pyramid_max_layers, 1, 1,
                )
                set_tooltip("Max pyramid levels beyond the full-res /0.")

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
                "Zero out background pixels in the saved volume using "
                "the segmentation mask. Matches MATLAB segmentFlag == 1."
            )

            _, self._correct_do_tenengrad = imgui.checkbox(
                "Tenengrad diagnostic", self._correct_do_tenengrad,
            )
            set_tooltip(
                "Per-Z sharpness plot per camera pair. Useful for "
                "picking a transition_plane or blending_range later."
            )

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_kernel = imgui.input_int(
                "Gaussian kernel", self._correct_gauss_kernel, 1, 1,
            )
            set_tooltip("Square kernel size for the Gaussian pre-filter.")

            imgui.set_next_item_width(_input_w())
            _, self._correct_gauss_sigma = imgui.input_float(
                "Gaussian sigma", self._correct_gauss_sigma, 0.1, 1.0, "%.2f",
            )
            set_tooltip("Sigma for the Gaussian pre-filter (in pixels).")

            imgui.set_next_item_width(_input_w())
            _, self._correct_segment_threshold = imgui.input_float(
                "##segment_threshold", self._correct_segment_threshold,
                0.01, 0.1, "%.3f",
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("segment_threshold", "Threshold")
            set_tooltip("Foreground threshold after Gaussian smoothing (0–1).")

            imgui.set_next_item_width(_input_w())
            _, self._correct_mask_percentile = imgui.input_float(
                "Mask percentile", self._correct_mask_percentile,
                0.1, 1.0, "%.2f",
            )
            set_tooltip("Percentile (0–100) used to threshold the mask.")

            imgui.set_next_item_width(_input_w())
            _, self._correct_splitting = imgui.input_int(
                "Splitting", self._correct_splitting, 1, 5,
            )
            set_tooltip(
                "Block size (along Z) for memory-bounded Gaussian "
                "filtering.\nSmaller = less RAM, more block edges."
            )

    def _draw_correct_advanced_box(self) -> None:
        with tooltip_marks_right():
            _, self._correct_median_kernel_enabled = imgui.checkbox(
                "Median filter", self._correct_median_kernel_enabled,
            )
            set_tooltip("Dead-pixel correction via per-plane median filter.")
            if self._correct_median_kernel_enabled:
                imgui.set_next_item_width(_input_w())
                _, self._correct_median_kernel_size = imgui.input_int(
                    "Kernel (N×N)", self._correct_median_kernel_size, 1, 1,
                )
                set_tooltip("Square kernel size for the median filter.")

            imgui.set_next_item_width(_input_w())
            _, self._correct_background_percentile = imgui.input_float(
                "Background pct",
                self._correct_background_percentile,
                0.5, 5.0, "%.2f",
            )
            set_tooltip(
                "Percentile (0–100) used to estimate the camera "
                "background intensity."
            )

    def _draw_correct_shared_box(self) -> None:
        with tooltip_marks_right():
            imgui.set_next_item_width(_input_w())
            _, self._correct_subsample_factor = imgui.input_int(
                "Subsample factor", self._correct_subsample_factor, 1, 10,
            )
            set_tooltip(
                "Stride for percentile estimation.\n"
                "Used by both segmentation and pixel correction.\n"
                "Higher = faster but noisier."
            )

    def _draw_fuse_io_box(self) -> None:
        """Fuse-mode I/O options for the Parameters popup. The
        top-level ``fused_suffix`` lives in the Run-tab Output section
        now (per-method ``output_suffix`` stays under Fusion).
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
            set_tooltip(
                "Volume container format written per (timepoint, camera)."
            )

            if self._output_format == "zarr":
                compressors = ["zstd", "gzip", "blosc-zstd", "blosc-lz4", "none"]
                try:
                    cidx = compressors.index(self._compression)
                except ValueError:
                    cidx = 0
                imgui.set_next_item_width(_input_w())
                c_changed, c_new = imgui.combo("Compressor", cidx, compressors)
                if c_changed:
                    self._compression = compressors[c_new]
                set_tooltip("Codec applied inside the volume containers.")

                imgui.set_next_item_width(_input_w())
                _, self._compression_level = imgui.input_int(
                    "Level", self._compression_level, 1, 1,
                )
                set_tooltip("Compression level (0–9).")

            imgui.set_next_item_width(_input_w())
            _, new_workers = imgui.input_int(
                "Workers", self._workers, 1, 2,
            )
            self._workers = max(1, min(_MAX_WORKERS, new_workers))
            set_tooltip(
                "Parallel workers for timepoint/tile processing.\n"
                "Bottleneck is RAM, not cores: each worker holds both "
                "camera volumes plus a same-size output. Adaptive "
                "blending roughly doubles per-worker footprint vs "
                f"geometric — start at 2–4 with adaptive (soft cap {_MAX_WORKERS})."
            )

            _, self._overwrite = imgui.checkbox("Overwrite", self._overwrite)
            set_tooltip(
                "Re-process even when an output already exists at the target."
            )

            _, self._pyramid = imgui.checkbox("Pyramid", self._pyramid)
            set_tooltip(
                "Generate the resolution pyramid (zarr / OME-TIFF only)."
            )

            if self._pyramid:
                imgui.set_next_item_width(_input_w())
                _, self._pyramid_max_layers = imgui.input_int(
                    "Pyramid levels", self._pyramid_max_layers, 1, 1,
                )
                set_tooltip("Max pyramid levels beyond the full-res /0.")

    def _draw_fuse_blending_box(self) -> None:
        with tooltip_marks_right():
            methods = ["geometric", "adaptive", "auto", "average"]
            try:
                idx = methods.index(self._fuse_blending_method)
            except ValueError:
                idx = 0
            imgui.set_next_item_width(_input_w())
            changed, new_idx = imgui.combo("##blending_method", idx, methods)
            if changed:
                self._fuse_blending_method = methods[new_idx]
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_method", "Blending method")
            set_tooltip(
                "  geometric  fixed transition plane\n"
                "  adaptive   per-XY tenengrad-driven blend\n"
                "  auto       picks geometric / adaptive based on data\n"
                "  average    plain mean (no blending)."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_output_suffix = imgui.input_text(
                "Output suffix", self._fuse_output_suffix or "",
            )
            set_tooltip(
                "Appended to the fused output folder name so multiple "
                "parameter sweeps coexist:\n"
                "  <raw>.fused/<method>[_<suffix>]/\n"
                "Leave blank to use the bare <method>/ folder."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_blending_range = imgui.input_int(
                "##blending_range", self._fuse_blending_range, 1, 5,
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("blending_range", "Blending range")
            set_tooltip(
                "Transition zone width in Z-planes for geometric blending."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_transition_plane = imgui.input_int(
                "##transition_plane", self._fuse_transition_plane, 1, 5,
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("transition_plane", "Transition plane")
            set_tooltip(
                "Z-index for geometric blending.\n−1 = use volume center."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_front_flag = imgui.input_int(
                "##front_flag", self._fuse_front_flag, 1, 1,
            )
            imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
            self._emp_label("front_flag", "Front camera")
            set_tooltip(
                "Which camera carries the sharper signal at low Z.\n"
                "  1 = reference camera (cam0)\n"
                "  2 = transformed camera (cam1)"
            )

    def _draw_fuse_transforms_box(self) -> None:
        with tooltip_marks_right():
            _, self._fuse_flip_z = imgui.checkbox("Flip Z", self._fuse_flip_z)
            set_tooltip(
                "Flip the second camera along Z before fusion.\n"
                "Use for opposing camera pairs."
            )

            _, self._fuse_flip_horizontal = imgui.checkbox(
                "Flip horizontal", self._fuse_flip_horizontal,
            )
            set_tooltip("Flip the second camera horizontally before fusion.")

            _, self._fuse_flip_vertical = imgui.checkbox(
                "Flip vertical", self._fuse_flip_vertical,
            )
            set_tooltip("Flip the second camera vertically before fusion.")

            rotations = ["0", "+90 (cw)", "−90 (ccw)"]
            idx = {0: 0, 1: 1, -1: 2}.get(self._fuse_rotation, 0)
            imgui.set_next_item_width(_input_w())
            r_changed, r_new = imgui.combo("Rotation", idx, rotations)
            if r_changed:
                self._fuse_rotation = {0: 0, 1: 1, 2: -1}[r_new]
            set_tooltip(
                "Rotation applied to the second camera before fusion."
            )

    def _draw_fuse_search_box(self) -> None:
        with tooltip_marks_right():
            imgui.text_colored(_SUBSECTION_COLOR, "X offsets")
            for label, attr in (
                ("Start##sx", "_fuse_search_x_start"),
                ("Stop##sx", "_fuse_search_x_stop"),
                ("Step##sx", "_fuse_search_x_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label, getattr(self, attr), 1, 10,
                )
                setattr(self, attr, new_val)
                set_tooltip(
                    "Pixel offsets searched along X during camera-pair "
                    "registration\n(start, stop, step)."
                )

            imgui.spacing()
            imgui.text_colored(_SUBSECTION_COLOR, "Y offsets")
            for label, attr in (
                ("Start##sy", "_fuse_search_y_start"),
                ("Stop##sy", "_fuse_search_y_stop"),
                ("Step##sy", "_fuse_search_y_step"),
            ):
                imgui.set_next_item_width(_input_w())
                _, new_val = imgui.input_int(
                    label, getattr(self, attr), 1, 10,
                )
                setattr(self, attr, new_val)
                set_tooltip(
                    "Pixel offsets searched along Y during camera-pair "
                    "registration\n(start, stop, step)."
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

    def _submit_stitcher(self, arr: Any) -> None:
        """Spawn ``generate_bigstitcher_xml`` against the raw root that
        sits beside the loaded corrected / fused tree.

        The function scans ``config.fused_dir`` itself, so we only need
        to forward the raw root + corrected/fused/stitcher suffixes;
        no fusion-stage args (workers, blending, …) apply.
        """
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        scan_root = Path(arr.scan_root)
        raw_root = _sibling_raw_root(arr)
        if raw_root is None:
            anchor = scan_root.parent
            for suf in (".corrected", ".fused"):
                if anchor.name.endswith(suf):
                    raw_root = anchor.parent / anchor.name[: -len(suf)]
                    break
        if raw_root is None or not raw_root.is_dir():
            self._last_status = (
                "Cannot locate raw acquisition root next to the loaded "
                "tree. Move or symlink the raw folder beside the "
                ".corrected / .fused output before exporting."
            )
            return
        args = {
            "input_path": str(raw_root),
            "corrected_suffix": self._correct_corrected_suffix,
            "fused_suffix": self._fuse_fused_suffix,
            "stitcher_suffix": self._stitcher_suffix,
            "coarse_align": self._stitcher_coarse_align,
        }
        args.update(self._microscope_kwargs())
        self._spawn(
            "isoview_bigstitcher", args,
            description=f"BigStitcher XML: {raw_root.name}",
            output_path=self._stitcher_output_path or str(raw_root),
        )

    def _submit_spark_register(self, arr: Any) -> None:
        """Run bigstitcher-spark's detect → match → solve pipeline against
        the dataset.xml previously emitted by :meth:`_submit_stitcher`.
        """
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root

        if not self._stitcher_spark_root:
            self._last_status = (
                "Set the bigstitcher-spark executables dir (or fat JAR) "
                "first."
            )
            return

        scan_root = Path(arr.scan_root)
        raw_root = _sibling_raw_root(arr)
        if raw_root is None:
            anchor = scan_root.parent
            for suf in (".corrected", ".fused"):
                if anchor.name.endswith(suf):
                    raw_root = anchor.parent / anchor.name[: -len(suf)]
                    break
        if raw_root is None or not raw_root.is_dir():
            self._last_status = (
                "Cannot locate raw acquisition root. Run Export first."
            )
            return

        stitcher_root = raw_root.parent / f"{raw_root.name}{self._stitcher_suffix}"
        xml_candidates = sorted(stitcher_root.glob("*/dataset.xml"))
        if not xml_candidates:
            self._last_status = (
                f"No dataset.xml under {stitcher_root}. Run Export first."
            )
            return
        xml_path = xml_candidates[0]

        args = {
            "xml_path": str(xml_path),
            "spark_root": self._stitcher_spark_root,
            "java_home": self._stitcher_spark_java_home or None,
            "label": self._stitcher_spark_label,
            "sigma": float(self._stitcher_spark_sigma),
            "threshold": float(self._stitcher_spark_threshold),
            "min_intensity": float(self._stitcher_spark_min_intensity),
            "max_intensity": float(self._stitcher_spark_max_intensity),
            "downsample_xy": int(self._stitcher_spark_downsample_xy),
            "downsample_z": int(self._stitcher_spark_downsample_z),
            "match_algorithm": self._stitcher_spark_match_algorithm,
            "transformation_model": self._stitcher_spark_transformation_model,
            "regularization_model": self._stitcher_spark_regularization_model,
        }
        self._spawn(
            "isoview_bigstitcher_register", args,
            description=f"BigStitcher-Spark register: {xml_path.parent.name}",
            output_path=str(xml_path.parent),
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
            "corrected_suffix": self._correct_corrected_suffix,
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
        tps = self._selected_timepoints()
        if tps is not None:
            args["timepoints"] = tps
        args.update(self._microscope_kwargs())
        from mbo_utilities.gui import _isoview_crop_state as crop_state
        args.update(crop_state.to_config_args(arr))
        self._spawn(
            "isoview_correct", args,
            description=f"correct_stack: {Path(arr.scan_root).name}",
            output_path=resolved_out or str(arr.scan_root),
        )

    def _submit_fuse(self, arr: Any) -> None:
        # multi_fuse's `input_dir` is the RAW acquisition root containing
        # the SPC##_TM##_*.stack files — NOT the .corrected/ tree.
        # IsoviewArray(kind="corrected").scan_root is the SPM## subdir
        # under .corrected/, so walk to the .corrected/ root and strip
        # the corrected-suffix to land on the raw root sibling.
        # ProcessingConfig defaults output_dir to <raw_root><suffix>/,
        # so leaving output_dir=None re-uses the existing .corrected/
        # as the corrected tree; fused output goes to the sibling
        # <raw_root>.fused/<method>/.
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        scan_root = Path(arr.scan_root)
        raw_root = _sibling_raw_root(arr)
        if raw_root is None:
            # Fallback: assume scan_root.parent is the .corrected/ root
            # and strip a literal ".corrected" — covers the case where
            # _sibling_raw_root returns None because the sibling was
            # moved/renamed.
            anchor = scan_root.parent
            if anchor.name.endswith(".corrected"):
                raw_root = anchor.parent / anchor.name[: -len(".corrected")]
        if raw_root is None or not raw_root.is_dir():
            self._last_status = (
                "Cannot locate raw acquisition root (sibling of "
                ".corrected/). Move or symlink the raw folder next to "
                "the .corrected/ tree before running multi_fuse."
            )
            return
        transition_plane: int | None = (
            None if self._fuse_transition_plane < 0
            else int(self._fuse_transition_plane)
        )
        args = {
            "input_path": str(raw_root),
            "output_dir": self._output_dir or None,
            "fused_suffix": self._fuse_fused_suffix,
            "output_format": self._output_format,
            "compression": (
                None if self._compression == "none" else self._compression
            ),
            "compression_level": self._compression_level,
            "overwrite": self._overwrite,
            "workers": self._workers,
            "pyramid": self._pyramid,
            "pyramid_max_layers": self._pyramid_max_layers,
            "blending_method": self._fuse_blending_method,
            "output_suffix": (self._fuse_output_suffix or "").strip() or None,
            "blending_range": self._fuse_blending_range,
            "transition_plane": transition_plane,
            "front_flag": self._fuse_front_flag,
            "flip_z": self._fuse_flip_z,
            "flip_horizontal": self._fuse_flip_horizontal,
            "flip_vertical": self._fuse_flip_vertical,
            "rotation": self._fuse_rotation,
            "search_offsets_x": [
                self._fuse_search_x_start,
                self._fuse_search_x_stop,
                self._fuse_search_x_step,
            ],
            "search_offsets_y": [
                self._fuse_search_y_start,
                self._fuse_search_y_stop,
                self._fuse_search_y_step,
            ],
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
            description=f"multi_fuse ({self._fuse_blending_method}): "
                        f"{raw_root.name}",
            output_path=self._output_dir or str(raw_root),
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
            shape_text = f"{shape_text} [{','.join(dims)}]"

        md = dict(getattr(arr, "metadata", {}) or {})
        rows: list[tuple[str, str, str | None]] = [
            ("Name", short_name, path_str or None),
            ("Size on disk", _format_size(size_bytes), None),
            ("Shape", shape_text, None),
            ("Data state", str(arr.kind), None),
        ]
        for label, key, unit in (
            ("Frame rate", "fs", "Hz"),
            ("dx", "dx", "µm"),
            ("dy", "dy", "µm"),
            ("dz", "dz", "µm"),
        ):
            v = get_param(md, key)
            if v is None:
                continue
            rows.append((label, f"{v} {unit}".rstrip(), None))

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
                for label, value, hover in rows:
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    imgui.text_unformatted(label)
                    imgui.table_set_column_index(1)
                    imgui.push_text_wrap_pos(0.0)
                    try:
                        imgui.text_unformatted(value)
                    finally:
                        imgui.pop_text_wrap_pos()
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
            _MODE_CORRECT: "_correct_corrected_suffix",
            _MODE_FUSE: "_fuse_fused_suffix",
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
        from mbo_utilities.arrays.isoview.array import _sibling_raw_root
        raw = _sibling_raw_root(arr)
        if raw is not None:
            return raw.name
        # Fallback: strip a known suffix off the scan_root's parent.
        anchor = sr.parent
        for suf in (".corrected", ".fused"):
            if anchor.name.endswith(suf):
                return anchor.name[: -len(suf)]
        return sr.name

    def _draw_iso_output_options(self, arr: Any) -> None:
        """Universal Output options block: one suffix input + a
        filename preview. Every other I/O knob (codec / workers /
        pyramid / overwrite / format) lives in the Parameters popup.
        """
        imgui.text_colored(_SUBSECTION_COLOR, "Output options")
        set_tooltip(
            "Folder / filename suffix appended to the source name. "
            "All other I/O knobs (codec, workers, pyramid, overwrite, "
            "format) live in the Parameters popup.",
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
        # Consolidate writes a single file; the suffix already carries
        # the extension. Folder modes get a trailing slash so it's
        # obvious the suffix names a directory.
        is_file = self._selected_mode == _MODE_CONSOLIDATE
        preview = f"{basename}{new_val}" + ("" if is_file else "/")
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
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_segment_readout(self, arr: Any) -> None:
        """Segmentation threshold Run-tab block — only meaningful on
        raw data (where ``correct_stack`` will run). Hidden otherwise.
        Trailing separator emitted inline so a hidden readout leaves
        no orphan separator behind.
        """
        if getattr(arr, "kind", None) != "raw":
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
        """Data-slicing section — Set slice button + count preview."""
        imgui.text_colored(_SUBSECTION_COLOR, "Data slicing")
        set_tooltip(
            "Restrict which timepoints feed the run. Click Set slice to "
            "define a range using start:stop[:step] syntax (1-based, "
            "inclusive). Leaving the full range selects all timepoints.",
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
        imgui.text(f"Timepoints: {n_tp}/{max_frames}")

        self._draw_iso_slicing_popup(max_frames)
        imgui.spacing()
        imgui.spacing()

    def _draw_iso_slicing_popup(self, max_frames: int) -> None:
        """Slicing popup — pick which timepoints to process."""
        if self._iso_slicing_open:
            imgui.open_popup("Timepoints##iso_slice")
            self._iso_slicing_open = False

        imgui.set_next_window_size(
            imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever,
        )
        if imgui.begin_popup("Timepoints##iso_slice"):
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Timepoints",
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
            )

            imgui.spacing()
            if imgui.button("Close##iso_slice_close", imgui.ImVec2(80, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

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

        # DATA SLICING — restrict the run to a subset of timepoints.
        self._draw_iso_data_slicing(max_frames)
        imgui.separator()
        imgui.spacing()

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

        # RUN — big green, centered. In BigStitcher mode, the export
        # button sits side-by-side with the Spark "Run registration"
        # button, both centered as a pair. Per-button width shrinks
        # below _RUN_W when the side panel is too narrow to fit both
        # at full size — so the right button never gets clipped.
        has_path = bool(self._current_output_path())
        run_label = _RUN_LABELS.get(self._selected_mode or "", "Run")
        is_stitcher = self._selected_mode == _MODE_STITCHER
        spark_ready = bool(self._stitcher_spark_root) if is_stitcher else False
        spacing_x = imgui.get_style().item_spacing.x
        avail = imgui.get_content_region_avail().x
        if is_stitcher:
            btn_w = max(80.0, min(_RUN_W, (avail - spacing_x) / 2.0))
            total_w = btn_w * 2 + spacing_x
        else:
            btn_w = min(_RUN_W, max(80.0, avail))
            total_w = btn_w
        if avail > total_w:
            imgui.set_cursor_pos_x(
                imgui.get_cursor_pos_x() + (avail - total_w) * 0.5
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

        register_clicked = False
        if is_stitcher:
            imgui.same_line()
            if not spark_ready:
                imgui.begin_disabled()
            register_clicked = imgui.button(
                "Run registration##spark_run",
                imgui.ImVec2(btn_w, 0),
            )
            if not spark_ready:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        "Set the bigstitcher-spark path in the "
                        "Parameters popup first."
                    )
            if register_clicked and spark_ready:
                self._submit_spark_register(arr)

            # Poll the spark_root folder picker dialog if open.
            dlg = self._stitcher_spark_dialog
            if dlg is not None:
                try:
                    ready = dlg.ready(0)
                except TypeError:
                    ready = dlg.ready()
                if ready:
                    try:
                        result = dlg.result()
                    except Exception:
                        result = ""
                    if result:
                        self._stitcher_spark_root = result
                    self._stitcher_spark_dialog = None

        if self._last_status:
            imgui.spacing()
            color = (
                imgui.ImVec4(0.8, 0.4, 0.4, 1.0)
                if self._last_status.startswith(("Failed", "Output", "Dialog"))
                else imgui.ImVec4(0.5, 0.85, 0.5, 1.0)
            )
            imgui.text_colored(color, self._last_status)
