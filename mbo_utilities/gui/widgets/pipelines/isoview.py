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

from imgui_bundle import (
    hello_imgui,
    icons_fontawesome_6 as fa,
    imgui,
    portable_file_dialogs as pfd,
)

from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    set_tooltip,
    tooltip_marks_right,
)
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget


# Style palette matched to Suite2p's settings panel — colors picked
# from mbo_utilities/gui/widgets/pipelines/settings.py so the two
# pipelines look like they belong to the same family.
_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)         # section title (yellow)
_SUBSECTION_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)   # subtitle (light blue)
_MBO_ONLY_COLOR = imgui.ImVec4(0.55, 0.85, 0.55, 1.0)    # mbo-only group title (green)

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


def _section(label: str) -> None:
    """Yellow section title row, matches Suite2p's pipeline-popup style."""
    imgui.spacing()
    imgui.text_colored(_TITLE_COLOR, label)
    imgui.spacing()


def _hint(text: str) -> None:
    """Wrapped subtitle/help text in the same blue Suite2p uses."""
    imgui.push_text_wrap_pos(0.0)
    imgui.text_colored(_SUBSECTION_COLOR, text)
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

_RUN_LABELS = {
    _MODE_CONSOLIDATE: "Run consolidation",
    _MODE_CORRECT: "Run correct_stack",
    _MODE_FUSE: "Run multi_fuse",
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
        modes.append(_MODE_CONSOLIDATE)
    elif arr.kind == "fused":
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
        self._save_dialog: Any = None
        self._save_dialog_target: str = ""  # which path field the dialog fills

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

        # Correct-mode state
        self._correct_corrected_suffix: str = ".corrected"
        self._correct_segment_mode: int = 1
        self._correct_apply_seg_mask: bool = True
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
        self._fuse_output_suffix: str = ""  # appended to MultiFused_<method>
        self._fuse_blending_range: int = 20
        self._fuse_flip_z: bool = False
        self._fuse_flip_horizontal: bool = True
        self._fuse_flip_vertical: bool = False
        self._fuse_rotation: int = 0
        self._fuse_cameras_rotated: bool = False
        self._fuse_transition_plane: int = -1  # -1 = None (center)
        self._fuse_front_flag: int = 1
        self._fuse_search_x_start: int = -50
        self._fuse_search_x_stop: int = 50
        self._fuse_search_x_step: int = 10
        self._fuse_search_y_start: int = -50
        self._fuse_search_y_stop: int = 50
        self._fuse_search_y_step: int = 10

        # Microscope override (all modes). -1 = auto from XML.
        self._mic_pixel_spacing_z: float = -1.0
        self._mic_objective_mag: float = -1.0
        self._mic_pixel_spacing_camera: float = -1.0

        # Popup visibility — set True to open on next frame.
        self._show_settings_popup: bool = False
        self._popup_just_opened: bool = False

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
        # input scan_root. So opening .corrected/SPM00 lands the export
        # at .corrected/SPM00_isoview-corrected.zarr — predictable,
        # close to the source, no separate exports/ tree.
        self._consolidate_output_path = str(
            sr.parent / f"{sr.name}_isoview-{arr.kind}.zarr"
        )

        # Correct / Fuse output_dir defaults: also stay close to the
        # input. correct_stack writes <raw_root>.corrected/ when given
        # an input_dir; multi_fuse writes under the corrected tree's
        # Results/MultiFused_<method>/. For fuse the canonical
        # output_dir is the .corrected/ root; for correct, leave blank
        # so the pipeline falls back to <raw_root><suffix>/.
        if arr.kind == "corrected":
            self._output_dir = str(sr.parent)

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
        return "_output_dir"

    def _current_output_path(self) -> str:
        return getattr(self, self._output_path_attr(), "") or ""

    def _open_save_dialog(self, target: str, default: str, kind: str) -> None:
        """``target`` names which path field the dialog should populate.

        ``kind`` is ``"file"`` for the consolidate .zarr or ``"folder"``
        for the correct/fuse output dir.
        """
        if self._save_dialog is not None:
            return
        self._save_dialog_target = target
        try:
            if kind == "file":
                self._save_dialog = pfd.save_file(
                    "Save as", default,
                    ["Zarr store", "*.zarr"],
                )
            else:
                self._save_dialog = pfd.select_folder(
                    "Select output folder", default,
                )
        except Exception as e:
            self._last_status = f"Dialog failed: {e}"
            self._save_dialog = None

    def _poll_save_dialog(self) -> None:
        if self._save_dialog is None:
            return
        try:
            ready = self._save_dialog.ready(0)
        except TypeError:
            ready = self._save_dialog.ready()
        if not ready:
            return
        try:
            result = self._save_dialog.result()
        except Exception:
            result = ""
        self._save_dialog = None
        if not result:
            return
        if self._save_dialog_target == "consolidate":
            if not result.endswith(".zarr"):
                result = result + ".zarr"
            self._consolidate_output_path = result
        elif self._save_dialog_target == "output_dir":
            self._output_dir = result

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

    def _browse_output(self) -> None:
        """Open the path dialog appropriate for the active mode.

        Consolidate's output is a single .zarr — file dialog. Correct
        and Fuse write under a directory — folder dialog. Targets are
        named to match _poll_save_dialog.
        """
        if self._selected_mode == _MODE_CONSOLIDATE:
            self._open_save_dialog(
                "consolidate",
                self._consolidate_output_path or "consolidated.zarr",
                "file",
            )
        else:
            self._open_save_dialog(
                "output_dir", self._output_dir or "", "folder",
            )

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

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            if imgui.button("Close##iso_settings_close",
                            imgui.ImVec2(_BTN_W, 0)):
                self._show_settings_popup = False
                imgui.close_current_popup()
        finally:
            imgui.end_popup()

    def _draw_popup_columns(self, columns: list[tuple[str, Any]]) -> None:
        """Render a row of equal-width bordered child boxes.

        ``columns`` is a list of ``(title, draw_fn)``. Each box gets a
        yellow ``_TITLE_COLOR`` header, matching Suite2p's pipeline
        popup. Boxes auto-resize their height; widths split the row
        evenly.
        """
        if not columns:
            return
        n = len(columns)
        avail_x = imgui.get_content_region_avail().x
        style = imgui.get_style()
        spacing_x = style.item_spacing.x
        col_w = max(_natural_col_w(), (avail_x - spacing_x * (n - 1)) / n)
        box_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        for i, (title, draw_fn) in enumerate(columns):
            if i > 0:
                imgui.same_line()
            imgui.begin_child(
                f"##iso_col_{title}",
                imgui.ImVec2(col_w, 0),
                box_flags,
            )
            imgui.text_colored(_TITLE_COLOR, title)
            imgui.spacing()
            try:
                draw_fn()
            finally:
                imgui.end_child()

    def _draw_consolidate_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("Output", self._draw_consolidate_output_box),
            ("Codec", self._draw_consolidate_codec_box),
            ("Microscope (override XML)", self._draw_microscope_overrides_box),
        ])

    def _draw_correct_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("Output", self._draw_correct_output_box),
            ("Segmentation", self._draw_correct_segmentation_box),
            ("Pixel correction", self._draw_correct_advanced_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Microscope (override XML)", self._draw_microscope_overrides_box),
        ])

    def _draw_fuse_popup_rows(self) -> None:
        self._draw_popup_columns([
            ("Output", self._draw_fuse_output_box),
            ("Fusion", self._draw_fuse_blending_box),
            ("View transforms", self._draw_fuse_transforms_box),
        ])
        imgui.spacing()
        self._draw_popup_columns([
            ("Registration search", self._draw_fuse_search_box),
            ("Microscope (override XML)", self._draw_microscope_overrides_box),
        ])

    def _draw_consolidate_output_box(self) -> None:
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

    def _draw_correct_output_box(self) -> None:
        with tooltip_marks_right():
            imgui.set_next_item_width(_input_w())
            _, self._correct_corrected_suffix = imgui.input_text(
                "Corrected suffix", self._correct_corrected_suffix or "",
            )
            set_tooltip(
                "Folder suffix the pipeline appends to the raw root when "
                "no Output path is set on the side panel.\n"
                "Effective output dir then becomes <raw_root><suffix>/."
            )

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
            _, self._correct_background_percentile = imgui.input_float(
                "Background pct",
                self._correct_background_percentile,
                0.5, 5.0, "%.2f",
            )
            set_tooltip(
                "Percentile (0–100) used to estimate the background "
                "intensity for dead-pixel correction."
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
            _, self._correct_subsample_factor = imgui.input_int(
                "Subsample factor", self._correct_subsample_factor, 1, 10,
            )
            set_tooltip(
                "Stride for percentile estimation.\n"
                "Higher = faster but noisier."
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
                "Threshold", self._correct_segment_threshold,
                0.01, 0.1, "%.3f",
            )
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

    def _draw_fuse_output_box(self) -> None:
        with tooltip_marks_right():
            _hint(
                "multi_fuse writes its results under "
                "Results/MultiFused_<method>/ inside the output dir."
            )

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
            changed, new_idx = imgui.combo("Blending method", idx, methods)
            if changed:
                self._fuse_blending_method = methods[new_idx]
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
                "  Results/MultiFused_<method>[_<suffix>]/\n"
                "Leave blank to use the bare MultiFused_<method>/ folder."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_blending_range = imgui.input_int(
                "Blending range", self._fuse_blending_range, 1, 5,
            )
            set_tooltip(
                "Transition zone width in Z-planes for geometric blending."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_transition_plane = imgui.input_int(
                "Transition plane", self._fuse_transition_plane, 1, 5,
            )
            set_tooltip(
                "Z-index for geometric blending.\n−1 = use volume center."
            )

            imgui.set_next_item_width(_input_w())
            _, self._fuse_front_flag = imgui.input_int(
                "Front camera", self._fuse_front_flag, 1, 1,
            )
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

            _, self._fuse_cameras_rotated = imgui.checkbox(
                "Cameras rotated", self._fuse_cameras_rotated,
            )
            set_tooltip(
                "Skip the final reslice for VW90 when cameras are rotated."
            )

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

    def _microscope_kwargs(self) -> dict:
        """Pack microscope overrides as a kwargs dict for the task args.

        Sentinels of ``-1`` map to ``None`` (auto from XML); only set
        keys the user explicitly overrode.
        """
        out: dict = {}
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

    def _submit_consolidate(self, arr: Any) -> None:
        if not self._consolidate_output_path:
            self._last_status = "Output path required."
            return
        args = {
            "input_path": str(arr.scan_root),
            "output_path": self._consolidate_output_path,
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
                        f"{Path(self._consolidate_output_path).name}",
            output_path=self._consolidate_output_path,
        )

    def _submit_correct(self, arr: Any) -> None:
        median_kernel = None
        if self._correct_median_kernel_enabled:
            k = max(1, int(self._correct_median_kernel_size))
            median_kernel = [k, k]
        args = {
            "input_path": str(arr.scan_root),
            "output_dir": self._output_dir or None,
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
        args.update(self._microscope_kwargs())
        self._spawn(
            "isoview_correct", args,
            description=f"correct_stack: {Path(arr.scan_root).name}",
            output_path=self._output_dir or str(arr.scan_root),
        )

    def _submit_fuse(self, arr: Any) -> None:
        # multi_fuse's `input_dir` is the RAW acquisition root containing
        # the SPC##_TM##_*.stack files — NOT the .corrected/ tree.
        # IsoviewArray(kind="corrected").scan_root is the SPM## subdir
        # under .corrected/, so walk to the .corrected/ root and strip
        # the corrected-suffix to land on the raw root sibling.
        # ProcessingConfig defaults output_dir to <raw_root><suffix>/,
        # so leaving output_dir=None re-uses the existing .corrected/ as
        # the parent of the MultiFused output.
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
            "cameras_rotated": self._fuse_cameras_rotated,
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
        args.update(self._microscope_kwargs())
        self._spawn(
            "isoview_fuse", args,
            description=f"multi_fuse ({self._fuse_blending_method}): "
                        f"{raw_root.name}",
            output_path=self._output_dir or str(raw_root),
        )

    def draw_config(self) -> None:
        """Render the Suite2p-style side panel.

        Layout (top → bottom):
          1. Current dataset (kind / shape / path)
          2. Select output (folder-icon browse + editable path)
          3. Action (mode combo — shown only when multiple apply)
          4. Parameters and settings (small "Open" button → popup)
          5. Run button (big, green, centered)

        All knobs other than the path + mode live in the popup.
        """
        self._poll_save_dialog()

        arr = self._get_array()
        if arr is None:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                "No isoview array loaded.",
            )
            return

        self._ensure_defaults(arr)

        modes = _available_modes(arr)
        if not modes:
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.5, 0.3, 1.0),
                f"No isoview action available for kind={arr.kind!r}.",
            )
            if arr.kind == "raw" and not _isoview_pkg_available():
                imgui.spacing()
                imgui.text("Install the isoview package to run correct_stack:")
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
                    "uv pip install isoview",
                )
            return
        if self._selected_mode not in modes:
            self._selected_mode = modes[0]

        # popup lifecycle: open_popup() must be called from the same
        # imgui frame as begin_popup_modal(); we keep the visibility flag
        # for state but always re-open here when it's True and not
        # already open (handles re-renders).
        self._draw_settings_popup(arr)

        imgui.spacing()
        imgui.text_colored(_SUBSECTION_COLOR, "Current dataset")
        set_tooltip(
            "The IsoviewArray currently loaded in the viewer.\n"
            "`kind` selects which pipelines apply (raw → Correct, "
            "corrected → Fuse/Consolidate, fused → Consolidate)."
        )
        imgui.indent(10)
        imgui.text(f"kind   {arr.kind}")
        imgui.text(f"shape  {arr.shape}")
        imgui.text("path")
        imgui.same_line()
        imgui.text_wrapped(str(arr.scan_root))
        imgui.unindent(10)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # OUTPUT PATH — text input + Browse, matching Suite2p's row.
        # Label and tooltip are mode-specific. For Fuse, the path is the
        # existing .corrected/ tree (multi_fuse reads from it AND writes
        # its fused output under <path>/Results/MultiFused_<method>/);
        # picking an empty folder will silently produce no output.
        is_consolidate = self._selected_mode == _MODE_CONSOLIDATE
        is_fuse = self._selected_mode == _MODE_FUSE
        if is_consolidate:
            out_label = "Select output .zarr"
            out_tip = (
                "Destination .zarr group for the consolidated output."
            )
        elif is_fuse:
            out_label = "Corrected tree (input/output)"
            out_tip = (
                "Existing .corrected/ root. multi_fuse reads corrected "
                "stacks from <path>/SPM##/TM######/ and writes fused "
                "results to <path>/Results/MultiFused_<method>[_<suffix>]/. "
                "Pick the .corrected/ folder, not an empty target."
            )
        else:
            out_label = "Select output folder"
            out_tip = (
                "Destination directory tree for correct_stack output."
            )
        imgui.text_colored(_SUBSECTION_COLOR, out_label)
        set_tooltip(out_tip, align="right")
        imgui.spacing()

        if imgui.button(f"{fa.ICON_FA_FOLDER_OPEN}##iso_browse"):
            self._browse_output()
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        imgui.set_next_item_width(-1)
        path_attr = self._output_path_attr()
        _, new_path = imgui.input_text(
            "##iso_outpath", getattr(self, path_attr) or "",
        )
        setattr(self, path_attr, new_path)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # MODE PICKER — only when >1 mode applies.
        if len(modes) > 1:
            imgui.text_colored(_SUBSECTION_COLOR, "Action")
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
            imgui.set_next_item_width(180)
            changed, new_idx = imgui.combo("##iso_mode", idx, modes)
            if changed:
                self._selected_mode = modes[new_idx]
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        # PARAMETERS AND SETTINGS — small Open button → popup.
        imgui.text_colored(_SUBSECTION_COLOR, "Parameters and settings")
        set_tooltip(
            "Open the full settings popup: output format/codec, pyramid, "
            "segmentation, blending, registration search, microscope "
            "overrides.",
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
        imgui.spacing()

        # RUN — big green, centered, disabled until output path is set.
        has_path = bool(self._current_output_path())
        run_label = _RUN_LABELS.get(self._selected_mode or "", "Run")
        with _green_button():
            if not has_path:
                imgui.begin_disabled()
            avail = imgui.get_content_region_avail().x
            if avail > _RUN_W:
                imgui.set_cursor_pos_x(
                    imgui.get_cursor_pos_x() + (avail - _RUN_W) * 0.5
                )
            clicked = imgui.button(run_label, imgui.ImVec2(_RUN_W, 0))
            if not has_path:
                imgui.end_disabled()
        if (
            not has_path
            and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
        ):
            imgui.set_tooltip(
                "Set the output path above (folder button or paste a path)."
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
