import logging
import webbrowser
from pathlib import Path
from typing import Literal
import threading
from functools import partial
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

from imgui_bundle import (
    imgui,
    hello_imgui,
    imgui_ctx,
    implot,
    portable_file_dialogs as pfd,
    ImVec2,
    ImVec4,
)

from mbo_utilities.assembly import save_as
from mbo_utilities.file_io import (
    Scan_MBO,
    SAVE_AS_TYPES,
    _get_mbo_dirs,
    read_scan,
)
from mbo_utilities.graphics._imgui import begin_popup_size, ndim_to_frame
from mbo_utilities.graphics._widgets import set_tooltip, checkbox_with_tooltip, draw_scope
from mbo_utilities.graphics.gui_logger import GuiLogger, GuiLogHandler
from mbo_utilities.graphics.progress_bar import (
    draw_zstats_progress,
    draw_saveas_progress,
)
from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings, draw_tab_process
from mbo_utilities.phasecorr import compute_scan_phase_offsets
from mbo_utilities import log

try:
    import cupy as cp  # noqa
    from cusignal import (
        register_translation,
    )  # GPU version of phase_cross_correlation # noqa

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    register_translation = phase_cross_correlation  # noqa

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

REGION_TYPES = ["Full FOV", "Sub-FOV"]
USER_PIPELINES = ["suite2p", "masknmf"]


def style_seaborn():
    style = implot.get_style()
    style.set_color_(implot.Col_.line.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill.value, implot.AUTO_COL)

    style.set_color_(implot.Col_.error_bar.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.frame_bg.value, ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.plot_bg.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.plot_border.value, ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(implot.Col_.legend_bg.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.legend_border.value, ImVec4(0.80, 0.81, 0.85, 1.00))
    style.set_color_(implot.Col_.legend_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.title_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.inlay_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_text.value, ImVec4(0.00, 0.00, 0.00, 1.00))
    style.set_color_(implot.Col_.axis_grid.value, ImVec4(1.00, 1.00, 1.00, 1.00))
    style.set_color_(implot.Col_.axis_bg_hovered.value, ImVec4(0.92, 0.92, 0.95, 1.00))
    style.set_color_(implot.Col_.axis_bg_active.value, ImVec4(0.92, 0.92, 0.95, 0.75))
    style.set_color_(implot.Col_.selection.value, ImVec4(1.00, 0.65, 0.00, 1.00))
    style.set_color_(implot.Col_.crosshairs.value, ImVec4(0.23, 0.10, 0.64, 0.50))

    style.line_weight = 1.5
    style.marker = implot.Marker_.none.value
    style.marker_size = 4
    style.marker_weight = 1
    style.fill_alpha = 1.0
    style.error_bar_size = 5
    style.error_bar_weight = 1.5
    style.digital_bit_height = 8
    style.digital_bit_gap = 4
    style.plot_border_size = 0
    style.minor_alpha = 1.0
    style.major_tick_len = ImVec2(0, 0)
    style.minor_tick_len = ImVec2(0, 0)
    style.major_tick_size = ImVec2(0, 0)
    style.minor_tick_size = ImVec2(0, 0)
    style.major_grid_size = ImVec2(1.2, 1.2)
    style.minor_grid_size = ImVec2(1.2, 1.2)
    style.plot_padding = ImVec2(12, 12)
    style.label_padding = ImVec2(5, 5)
    style.legend_padding = ImVec2(5, 5)
    style.mouse_pos_padding = ImVec2(5, 5)
    style.plot_min_size = ImVec2(300, 225)


def style_seaborn_dark():
    style = implot.get_style()

    # Auto colors for lines and markers
    style.set_color_(implot.Col_.line.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.fill.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_outline.value, implot.AUTO_COL)
    style.set_color_(implot.Col_.marker_fill.value, implot.AUTO_COL)

    # Backgrounds and axes
    style.set_color_(
        implot.Col_.frame_bg.value, ImVec4(0.15, 0.17, 0.2, 1.00)
    )  # dark gray
    style.set_color_(
        implot.Col_.plot_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00)
    )  # darker gray
    style.set_color_(implot.Col_.plot_border.value, ImVec4(0.00, 0.00, 0.00, 0.00))
    style.set_color_(
        implot.Col_.axis_grid.value, ImVec4(0.35, 0.40, 0.45, 0.5)
    )  # light grid
    style.set_color_(
        implot.Col_.axis_text.value, ImVec4(0.9, 0.9, 0.9, 1.0)
    )  # light text
    style.set_color_(implot.Col_.axis_bg_hovered.value, ImVec4(0.25, 0.27, 0.3, 1.00))
    style.set_color_(implot.Col_.axis_bg_active.value, ImVec4(0.25, 0.27, 0.3, 0.75))

    # Legends and labels
    style.set_color_(implot.Col_.legend_bg.value, ImVec4(0.13, 0.15, 0.18, 1.00))
    style.set_color_(implot.Col_.legend_border.value, ImVec4(0.4, 0.4, 0.4, 1.00))
    style.set_color_(implot.Col_.legend_text.value, ImVec4(0.9, 0.9, 0.9, 1.00))
    style.set_color_(implot.Col_.title_text.value, ImVec4(1.0, 1.0, 1.0, 1.00))
    style.set_color_(implot.Col_.inlay_text.value, ImVec4(0.9, 0.9, 0.9, 1.00))

    # Misc
    style.set_color_(implot.Col_.error_bar.value, ImVec4(0.9, 0.9, 0.9, 1.00))
    style.set_color_(implot.Col_.selection.value, ImVec4(1.00, 0.65, 0.00, 1.00))
    style.set_color_(implot.Col_.crosshairs.value, ImVec4(0.8, 0.8, 0.8, 0.5))

    # Sizes
    style.line_weight = 1.5
    style.marker = implot.Marker_.none.value
    style.marker_size = 4
    style.marker_weight = 1
    style.fill_alpha = 1.0
    style.error_bar_size = 5
    style.error_bar_weight = 1.5
    style.digital_bit_height = 8
    style.digital_bit_gap = 4
    style.plot_border_size = 0
    style.minor_alpha = 0.3
    style.major_tick_len = ImVec2(0, 0)
    style.minor_tick_len = ImVec2(0, 0)
    style.major_tick_size = ImVec2(0, 0)
    style.minor_tick_size = ImVec2(0, 0)
    style.major_grid_size = ImVec2(1.2, 1.2)
    style.minor_grid_size = ImVec2(1.2, 1.2)
    style.plot_padding = ImVec2(12, 12)
    style.label_padding = ImVec2(5, 5)
    style.legend_padding = ImVec2(5, 5)
    style.mouse_pos_padding = ImVec2(5, 5)
    style.plot_min_size = ImVec2(300, 225)


def _save_as(
    path: str | Path,
    savedir: str | Path,
    planes: list | tuple = None,
    metadata: dict = None,
    overwrite: bool = True,
    ext: str = ".tiff",
    order: list | tuple = None,
    trim_edge: list | tuple = (0, 0, 0, 0),
    fix_phase: bool = False,
    save_phase_png: bool = False,
    target_chunk_mb: int = 20,
    debug: bool = False,
    progress_callback=None,
    **kwargs,
):
    """
    read scan from path for threading
    there must be a better way to do this
    """
    scan = read_scan(path, roi=kwargs.get("roi", None))
    if fix_phase:
        scan.fix_phase = True
    save_as(
        scan,
        savedir=savedir,
        planes=planes,
        metadata=metadata,
        overwrite=overwrite,
        ext=ext,
        order=order,
        trim_edge=trim_edge,
        save_phase_png=save_phase_png,
        target_chunk_mb=target_chunk_mb,
        progress_callback=progress_callback,
        debug=debug,
    )


@dataclass
class SaveStatus:
    # progressbar
    progress: float = 0.0
    plane: int | None = None
    message: str = ""
    logs: dict = field(default_factory=dict)


class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None = None,
        threading_enabled: bool = True,
        size: int = 350,
        location: Literal["bottom", "right"] = "right",
        title: str = "Data Preview",
        show_title: bool = False,
        movable: bool = False,
        resizable: bool = False,
        scrollable: bool = False,
        auto_resize: bool = True,
        window_flags: int | None = None,
    ):
        """
        Fastplotlib attachment, callable with fastplotlib.ImageWidget.add_gui(PreviewDataWidget)
        """
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        for name in ("mbo", "gui", "scan",):
            lg = log.get(name)
            lg.addHandler(gui_handler)
            lg.setLevel(logging.DEBUG)  # allow debug/info messages through
            lg.disabled = False
        self.logger = log.get("gui")
        self.s2p = Suite2pSettings()
        self.logger.info("Logger initialized.")

        flags = (
                (imgui.WindowFlags_.no_title_bar if not show_title else 0) |
                (imgui.WindowFlags_.no_move if not movable else 0) |
                (imgui.WindowFlags_.no_resize if not resizable else 0) |
                (imgui.WindowFlags_.no_scrollbar if not scrollable else 0) |
                (imgui.WindowFlags_.always_auto_resize if auto_resize else 0) |
                (window_flags or 0)
        )
        super().__init__(
            figure=iw.figure,
            size=size,
            location=location,
            title=title,
            window_flags=flags,
        )
        if implot.get_current_context() is None:
            implot.create_context()

        # backend.create_fonts_texture()
        self.io = imgui.get_io()
        self.io.set_ini_filename("/home/flynn/lbm_data/mbo_settings.ini")

        self.imgui_backend = iw.figure.imgui_renderer.backend
        self.font_size = 12
        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        self._show_debug_panel = False
        self._show_tool_style_editor = False
        self._show_tool_metrics = False
        self._show_debug_panel = False
        self._show_scope = False

        # preview data widget vars
        self._max_offset = 8
        self._gaussian_sigma = 0
        self._current_offset = 0.0
        self._window_size = 1
        self._phase_upsample = 20
        self._border = 0
        self._auto_update = False
        self._proj = "mean"

        self._selected_pipelines = None
        self._selected_roi = 0

        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = SAVE_AS_TYPES.index(".tiff")

        self._selected_planes = set()
        self._planes_str = str(getattr(self, "_planes_str", ""))
        self._overwrite = True
        self._fix_phase = False
        self._debug = False
        self._saveas_save_phase_png = False
        self._saveas_chunk_mb = 20

        # image widget setup
        self.image_widget = iw
        self.shape = self.image_widget.data[0].shape

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1

        if isinstance(self.image_widget.data[0], Scan_MBO):
            self.is_mbo_scan = True
        else:
            self.is_mbo_scan = False
        if self.is_mbo_scan:
            self.image_widget.data[0].fix_phase = False

        for subplot in self.image_widget.figure:
            subplot.toolbar = False

        self.image_widget._image_widget_sliders._loop = True  # noqa

        self._zstats = [
            {"mean": [], "std": [], "snr": []} for _ in range(self.num_arrays)
        ]
        self._zstats_means = [0 for _ in range(self.num_arrays)]
        self._zstats_done = [False] * self.num_arrays
        self._zstats_progress = [0.0] * self.num_arrays
        self._zstats_current_z = [0] * self.num_arrays

        self._saveas_popup_open = False
        self._saveas_done = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0
        self._saveas_outdir = str(getattr(self, "_save_dir", ""))
        self._saveas_total = 0

        self._saveas_selected_roi = set()  # -1 means all ROIs
        self._saveas_rois = False
        self._saveas_selected_roi_mode = "All"

        if threading_enabled:
            threading.Thread(target=self.compute_zstats, daemon=True).start()

    @property
    def fix_phase(self):
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value):
        self._fix_phase = value
        if not value:
            for i, arr in enumerate(self.image_widget.data):
                arr.offset = 0.0
                self.logger.info(f"Resetting phase for array {i}.")
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                if isinstance(arr, Scan_MBO):
                    arr.fix_phase = value
                    self.logger.info(f"Set fix_phase to {value} for MBO Scan object.")
        else:
            self.logger.warning(
                "Fix phase is only applicable to MBO Scan objects. "
                "No action taken."
            )
        # force update
        self.image_widget.current_index = self.image_widget.current_index

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self, value):
        self._border = value
        for arr in self.image_widget.data:
            if isinstance(arr, Scan_MBO):
                arr.max_offset = value
                self.logger.info(f"Border set to {value}.")
            else:
                self.logger.warning(
                    "Max offset is only applicable to MBO Scan objects. "
                    "No action taken."
                )

    @property
    def max_offset(self):
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value):
        self._max_offset = value
        for arr in self.image_widget.data:
            if isinstance(arr, Scan_MBO):
                arr.max_offset = value
                self.logger.info(f"Max offset set to {value}.")
            else:
                self.logger.warning(
                    "Max offset is only applicable to MBO Scan objects. "
                    "No action taken."
                )

    @property
    def num_arrays(self):
        return len(self.image_widget.managed_graphics)

    @property
    def selected_roi(self):
        return self._selected_roi

    @selected_roi.setter
    def selected_roi(self, value):
        if value < 0 or value >= len(self.image_widget.data):
            raise ValueError(
                f"Invalid ROI index: {value}. "
                f"Must be between 0 and {len(self.image_widget.managed_graphics) - 1}."
            )
        self._selected_roi = value
        self.logger.info(f"Selected ROI index set to {value}.")
        # self.image_widget.current_index = {"roi": value}
        self.update_frame_apply()

    @property
    def gaussian_sigma(self):
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value):
        if value > 0:
            self._gaussian_sigma = value
            self.logger.info(f"Gaussian sigma set to {value}.")
            self.update_frame_apply()
        else:
            self.logger.warning(f"Invalid gaussian sigma value: {value}. ")

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, value):
        if value != self._proj:
            if value == "mean-sub":
                self.logger.info("Setting projection to mean-subtracted.")
                self.update_frame_apply()
            else:

                self.logger.info(f"Setting projection to np.{value}.")
                self.image_widget.window_funcs["t"].func = getattr(np, value)
            self._proj = value

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self.image_widget.window_funcs["t"].window_size = value
        self._window_size = value
        self.logger.info(f"Window size set to {value}.")

    @property
    def current_offset(self):
        if all(hasattr(array, "offset") for array in self.image_widget.data):
            if isinstance(self.image_widget.data[0].offset, float):
                self.logger.info(f"All arrays have offset attribute. Setting from array.offset")
                return [array.offset for array in self.image_widget.data]
            else:
                self.logger.info(f"Arrays don't have offset attribute. ")
                return [
                    compute_scan_phase_offsets(
                        arr,
                        "subpix",
                        self.phase_upsample,
                        self.max_offset,
                        self.border
                    ) for i, arr in enumerate(self.image_widget.data)
                ]
        else:
            frame = self.get_raw_frame()
            return compute_scan_phase_offsets(
                frame,
                upsample=self.phase_upsample,
                border=self.border,
                max_offset=self.max_offset,
            )

    @property
    def phase_upsample(self):
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value):
        self._phase_upsample = value
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                if isinstance(arr, Scan_MBO):
                    arr.upsample = value
        else:
            self.logger.warning(
                "Phase upsample is only applicable to MBO Scan objects. "
                "No action taken."
            )

    def update(self):
        # (accessible from the "Tools" menu)
        if self._show_tool_style_editor:
            _, self._show_tool_style_editor = imgui.begin(
                "Style Editor", self._show_tool_style_editor
            )
            imgui.show_style_editor()
            imgui.end()
        if self._show_scope:
            size = begin_popup_size()
            imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
            _, self._show_scope = imgui.begin(
                "Scope Inspector",
                self._show_scope,
            )
            draw_scope()
            imgui.end()
        if self._show_debug_panel:
            size = begin_popup_size()
            imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
            _, self._show_debug_panel = imgui.begin(
                "MBO Debug Panel",
                self._show_debug_panel,
            )
            self.debug_panel.draw()
            imgui.end()

        wflags: imgui.WindowFlags = imgui.WindowFlags_.menu_bar  # noqa
        with imgui_ctx.begin_child(
                "menu",
                window_flags=wflags,
                child_flags=imgui.ChildFlags_.auto_resize_y
                            | imgui.ChildFlags_.always_auto_resize,
        ):
            if imgui.begin_menu_bar():
                if imgui.begin_menu("File", True):
                    if imgui.menu_item(
                        "Save as", "Ctrl+S", p_selected=False, enabled=self.is_mbo_scan
                    )[0]:
                        self._saveas_popup_open = True
                    imgui.end_menu()
                if imgui.begin_menu("Docs", True):
                    if imgui.menu_item(
                        "Open Docs", "Ctrl+I", p_selected=False, enabled=True
                    )[0]:
                        webbrowser.open(
                            "https://millerbrainobservatory.github.io/mbo_utilities/"
                        )
                    imgui.end_menu()
                if imgui.begin_menu("Settings", True):

                    imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Tools")
                    imgui.separator()
                    imgui.spacing()
                    _, self._show_tool_style_editor = imgui.menu_item(
                        "Style Editor", "", self._show_tool_style_editor, True
                    )
                    _, self._show_debug_panel = imgui.menu_item(
                        "Debug Panel",
                        "",
                        p_selected=self._show_debug_panel,
                        enabled=True,
                    )
                    _, self._show_scope = imgui.menu_item(
                        "Scope Inspector", "", self._show_scope, True
                    )
                    imgui.end_menu()
            imgui.end_menu_bar()

        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
                self.draw_preview_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Summary Stats")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
                self.draw_stats_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Process")[0]:
                draw_tab_process(self)
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def draw_stats_section(self):
        if not self._zstats_done:
            return

        stats_list = self._zstats if isinstance(self._zstats, list) else [self._zstats]

        imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats")
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # type: ignore # noqa
        imgui.spacing()

        # ROI selector
        roi_labels = [
            f"ROI {i + 1}"
            for i in range(len(stats_list))
            if stats_list[i] and "mean" in stats_list[i]
        ]
        roi_labels.append("Combined")
        for i, label in enumerate(roi_labels):
            if imgui.radio_button(label, self._selected_roi == i):
                self._selected_roi = i
            if i < len(roi_labels) - 1:
                imgui.same_line()

        imgui.separator()

        if self._selected_roi == len(roi_labels) - 1:  # Combined
            imgui.text("Stats for Combined ROIs")
            mean_vals = np.mean(
                [np.array(s["mean"]) for s in stats_list if s and "mean" in s], axis=0
            )

            if len(mean_vals) == 0:
                return

            std_vals = np.mean(
                [np.array(s["std"]) for s in stats_list if s and "std" in s], axis=0
            )
            snr_vals = np.mean(
                [np.array(s["snr"]) for s in stats_list if s and "snr" in s], axis=0
            )

            z_vals = np.arange(1, len(mean_vals) + 1, dtype=np.float32)

            # Table
            with imgui_ctx.begin_child(
                "##SummaryCombined", size=imgui.ImVec2(0, 0), child_flags=cflags
            ):
                if imgui.begin_table(
                    "Stats, averaged over ROI's",
                    4,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,  # type: ignore # noqa
                ):  # type: ignore # noqa
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )  # type: ignore # noqa
                    imgui.table_headers_row()
                    for i in range(len(z_vals)):
                        imgui.table_next_row()
                        for val in (z_vals[i], mean_vals[i], std_vals[i], snr_vals[i]):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

            with imgui_ctx.begin_child(
                "##PlotsCombined", size=imgui.ImVec2(0, 0), child_flags=cflags
            ):
                imgui.text("Z-plane Signal: Combined")
                if implot.begin_plot("Z-Plane Plot", ImVec2(-1, 0)):
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    implot.setup_axis_limits(implot.ImAxis_.x1.value, 1, self.nz)
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    style_seaborn_dark()

                    for i, stats in enumerate(stats_list):
                        if not stats or "mean" not in stats:
                            continue

                        z_vals_ind = np.arange(1, len(mean_vals) + 1, dtype=np.float32)
                        mean_vals_ind = np.array(stats["mean"])
                        implot.plot_line(f"ROI {i + 1}", z_vals_ind, mean_vals_ind)

                    implot.end_plot()

        else:
            roi_idx = self._selected_roi
            stats = stats_list[roi_idx]

            if not stats or "mean" not in stats:
                return

            imgui.text(f"Stats for ROI {roi_idx + 1}")
            mean_vals = np.array(stats["mean"])
            std_vals = np.array(stats["std"])
            snr_vals = np.array(stats["snr"])
            z_vals = np.arange(1, len(mean_vals) + 1, dtype=np.float32)

            with imgui_ctx.begin_child(
                f"##Summary{roi_idx}", size=imgui.ImVec2(0, 0), child_flags=cflags
            ):
                if imgui.begin_table(
                    f"zstats{roi_idx}",
                    4,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):  # type: ignore # noqa
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )  # type: ignore # noqa
                    imgui.table_headers_row()
                    for i in range(len(z_vals)):
                        imgui.table_next_row()
                        for val in (z_vals[i], mean_vals[i], std_vals[i], snr_vals[i]):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

            with imgui_ctx.begin_child(
                f"##Plots", size=imgui.ImVec2(0, 0), child_flags=cflags
            ):
                imgui.text("Z-plane Signal: Mean ± Std")
                if implot.begin_plot(
                    f"Z-Plane Signal {roi_idx}", size=imgui.ImVec2(-1, 300)
                ):
                    z_vals = np.arange(1, len(mean_vals) + 1, dtype=np.float32)
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    # implot.setup_axis_limits(implot.ImAxis_.x1.value, z_vals[0], z_vals[-1])
                    implot.setup_axis_limits(implot.ImAxis_.x1.value, 1, self.nz)
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    style_seaborn_dark()
                    implot.plot_error_bars("Mean ± Std", z_vals, mean_vals, std_vals)
                    implot.plot_line("Mean", z_vals, mean_vals)
                    implot.end_plot()

    def draw_preview_section(self):
        imgui.dummy(ImVec2(0, 5))
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # noqa
        with imgui_ctx.begin_child(
            "##PreviewChild",
            imgui.ImVec2(0, 0),
            cflags,
        ):
            if getattr(self, "_saveas_popup_open", False):
                imgui.open_popup("Save As")
                self._saveas_popup_open = False

            if imgui.begin_popup_modal("Save As")[0]:
                imgui.dummy(ImVec2(0, 5))

                imgui.set_next_item_width(hello_imgui.em_size(25))

                # Directory + Ext
                _, self._saveas_outdir = imgui.input_text(
                    "Save Dir",
                    str(Path(self._saveas_outdir).expanduser().resolve()),
                    256,
                )
                imgui.same_line()
                if imgui.button("Browse"):
                    home = Path().home()
                    res = pfd.select_folder(str(home))
                    if res:
                        self._saveas_outdir = res.result()

                imgui.set_next_item_width(hello_imgui.em_size(25))
                _, self._ext_idx = imgui.combo("Ext", self._ext_idx, SAVE_AS_TYPES)
                self._ext = SAVE_AS_TYPES[self._ext_idx]

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                # Options Section
                self._saveas_rois = checkbox_with_tooltip(
                    "Save ROI's",
                    self._saveas_rois,
                    "Enable to save each ROI individually. Saved to subfolders like roi1/, roi2/, etc.",
                )
                if self._saveas_rois:
                    try:
                        num_rois = self.image_widget.data[0].num_rois
                    except Exception as e:
                        num_rois = 1

                    imgui.spacing()
                    imgui.separator()
                    imgui.text_colored(
                        imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose ROI(s):"
                    )
                    imgui.dummy(ImVec2(0, 5))

                    if imgui.button("All##roi"):
                        self._saveas_selected_roi = set(range(num_rois))
                    imgui.same_line()
                    if imgui.button("None##roi"):
                        self._saveas_selected_roi = set()

                    imgui.columns(2, borders=False)
                    for i in range(num_rois):
                        imgui.push_id(f"roi_{i}")
                        selected = i in self._saveas_selected_roi
                        _, selected = imgui.checkbox(f"ROI {i + 1}", selected)
                        if selected:
                            self._saveas_selected_roi.add(i)
                        else:
                            self._saveas_selected_roi.discard(i)
                        imgui.pop_id()
                        imgui.next_column()
                    imgui.columns(1)

                imgui.spacing()
                imgui.separator()

                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Options")
                set_tooltip(
                    "Note: Current values for upsample and max-offset are applied during scan-phase correction.",
                    True,
                )

                imgui.dummy(ImVec2(0, 5))

                self._overwrite = checkbox_with_tooltip(
                    "Overwrite", self._overwrite, "Replace any existing output files."
                )
                self.fix_phase = checkbox_with_tooltip(
                    "Fix Phase",
                    self._fix_phase,
                    "Apply scan-phase correction to interleaved lines.",
                )
                self._debug = checkbox_with_tooltip(
                    "Debug",
                    self._debug,
                    "Run with debugging, settings -> debug to view the outputs.",
                )
                self._saveas_save_phase_png = checkbox_with_tooltip(
                    "Save Phase Images",
                    self._saveas_save_phase_png,
                    "Saves pre-post scan-phase images as PNGs to the save-directory.",
                )

                imgui.spacing()
                imgui.text("Chunk Size (MB)")
                set_tooltip(
                    "Target chunk size when saving TIFF or binary. Affects I/O and memory usage."
                )

                imgui.set_next_item_width(hello_imgui.em_size(20))
                _, self._saveas_chunk_mb = imgui.drag_int(
                    "##target_chunk_mb",
                    self._saveas_chunk_mb,
                    v_speed=1,
                    v_min=1,
                    v_max=1024,
                )

                imgui.spacing()
                imgui.separator()

                # Z-plane selection
                imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose z-planes:")
                imgui.dummy(ImVec2(0, 5))

                try:
                    num_planes = self.image_widget.data[0].num_channels  # noqa
                except Exception as e:
                    num_planes = 1
                    hello_imgui.log(
                        hello_imgui.LogLevel.error,
                        f"Could not read number of planes: {e}",
                    )

                if imgui.button("All"):
                    self._selected_planes = set(range(num_planes))
                imgui.same_line()
                if imgui.button("None"):
                    self._selected_planes = set()

                imgui.columns(2, borders=False)
                for i in range(num_planes):
                    imgui.push_id(i)
                    selected = i in self._selected_planes
                    _, selected = imgui.checkbox(f"Plane {i + 1}", selected)
                    if selected:
                        self._selected_planes.add(i)
                    else:
                        self._selected_planes.discard(i)
                    imgui.pop_id()
                    imgui.next_column()
                imgui.columns(1)

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                if imgui.button("Save", imgui.ImVec2(100, 0)):
                    if not self._saveas_outdir:
                        self._saveas_outdir = _get_mbo_dirs()["base"].joinpath("data")
                    try:
                        save_planes = [p + 1 for p in self._selected_planes]
                        self._saveas_total = len(save_planes)
                        if self._saveas_rois:
                            if (
                                not self._saveas_selected_roi
                                or len(self._saveas_selected_roi) == set()
                            ):
                                self._saveas_selected_roi = set(
                                    range(1, self.num_arrays + 1)
                                )
                            rois = sorted(self._saveas_selected_roi)
                        else:
                            rois = None

                        save_kwargs = {
                            "path": self.fpath,
                            "savedir": self._saveas_outdir,
                            "planes": save_planes,
                            "roi": rois,
                            "overwrite": self._overwrite,
                            "fix_phase": self._fix_phase,
                            "debug": self._debug,
                            "ext": self._ext,
                            "save_phase_png": self._saveas_save_phase_png,
                            "target_chunk_mb": self._saveas_chunk_mb,
                            "progress_callback": lambda frac,
                            current_plane: self.gui_progress_callback(
                                frac, current_plane
                            ),
                        }
                        self.logger.info(f"Saving planes {save_planes}")
                        self.logger.info(
                        f"Saving to {self._saveas_outdir} as {self._ext}"
                        )
                        threading.Thread(
                            target=_save_as, kwargs=save_kwargs, daemon=True
                        ).start()
                        imgui.close_current_popup()
                    except Exception as e:
                        self.logger.info(f"Error saving data: {e}")
                        imgui.close_current_popup()

                imgui.same_line()
                if imgui.button("Cancel"):
                    imgui.close_current_popup()

                imgui.end_popup()
            # Section: Window Functions
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
            imgui.spacing()

            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(2, 2))  # noqa

            imgui.begin_group()
            options = ["mean", "max", "std"]
            disabled_label = (
                "mean-sub (pending)" if not self._zstats_done else "mean-sub"
            )
            options.append(disabled_label)

            current_display_idx = options.index(
                self.proj if self._proj != "mean-sub" else disabled_label
            )

            imgui.set_next_item_width(hello_imgui.em_size(15))
            proj_changed, selected_display_idx = imgui.combo(
                "Projection", current_display_idx, options
            )
            set_tooltip(
                "Choose projection method over the sliding window: “mean” (average), “max” (peak), “std” (variance), or “mean-sub” (background-subtracted mean, recommended for motion preview)."
            )

            if proj_changed:
                selected_label = options[selected_display_idx]
                if selected_label == "mean-sub (pending)":
                    pass  # ignore user click while disabled
                else:
                    self.proj = selected_label
                    if self.proj == "mean-sub":
                        self.update_frame_apply()
                    else:
                        self.image_widget.window_funcs["t"].func = getattr(
                            np, self.proj
                        )

            # Window size for projections
            imgui.set_next_item_width(hello_imgui.em_size(15))
            winsize_changed, new_winsize = imgui.input_int(
                "Window Size", self.window_size, step=1, step_fast=2
            )
            set_tooltip(
                "Size of the temporal window (in frames) used for projection."
                " E.g. a value of 3 averages over 3 consecutive frames."
            )
            if winsize_changed and new_winsize > 0:
                self.window_size = new_winsize
                self.logger.info(f"New Window Size: {new_winsize}")

            # Gaussian Filter
            imgui.set_next_item_width(hello_imgui.em_size(15))
            gaussian_changed, new_gaussian_sigma = imgui.slider_float(
                label="sigma",
                v=self.gaussian_sigma,
                v_min=0.0,
                v_max=20.0,
            )
            set_tooltip(
                "Apply a Gaussian blur to the preview image. Sigma is in pixels; larger values yield stronger smoothing."
            )
            if gaussian_changed:
                self.gaussian_sigma = new_gaussian_sigma

            imgui.end_group()

            imgui.pop_style_var()

            # Section: Scan-phase Correction
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction"
            )
            imgui.separator()
            imgui.begin_group()

            imgui.set_next_item_width(hello_imgui.em_size(10))
            phase_changed, phase_value = imgui.checkbox(
                "Fix Phase",
                self._fix_phase,
            )
            set_tooltip("Enable to apply scan-phase correction to interleaved lines.")
            if phase_changed:
                self.fix_phase = phase_value
                self.logger.info(f"Fix Phase: {phase_value}")

            imgui.columns(2, "offsets", False)
            for i, iw in enumerate(self.image_widget.data):
                if not hasattr(iw, "offset"):
                    ofs = self.current_offset[i]
                else:
                    ofs = iw.offset if isinstance(iw.offset, float) else iw.offset[1]
                imgui.text(f"Array {i}:")
                imgui.next_column()
                imgui.text(f"{ofs:.3f}")
                imgui.next_column()
            imgui.columns(1)

            imgui.set_next_item_width(hello_imgui.em_size(10))
            upsample_changed, upsample_val = imgui.input_int(
                "Upsample", self._phase_upsample, step=1, step_fast=2
            )
            set_tooltip(
                "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment."
            )
            if upsample_changed:
                self.phase_upsample = max(1, upsample_val)
                self.logger.info(f"New upsample: {upsample_val}")

            imgui.set_next_item_width(hello_imgui.em_size(10))
            border_changed, border_val = imgui.input_int(
                "Exclude border-px", self._border, step=1, step_fast=2
            )
            set_tooltip(
                "Number of pixels to exclude from the edges of the image when computing the scan-phase offset."
            )
            if border_changed:
                self.border = max(0, border_val)
                self.logger.info(f"New border: {border_val}")

            imgui.set_next_item_width(hello_imgui.em_size(10))
            max_offset_changed, max_offset = imgui.input_int(
                "max-offset", self._max_offset, step=1, step_fast=2
            )
            set_tooltip(
                "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
            )
            if max_offset_changed:
                self.max_offset = max(1, max_offset)
                self.logger.info(f"New max-offset: {max_offset}")

            imgui.end_group()
            imgui.separator()

        imgui.separator()
        draw_zstats_progress(self)
        draw_saveas_progress(self)

    def get_raw_frame(self):
        idx = self.image_widget.current_index
        t = idx.get("t", 0)
        z = idx.get("z", 0)
        return tuple(ndim_to_frame(arr, t, z) for arr in self.image_widget.data)

    def gui_progress_callback(self, fraction, current_plane):
        """Callback for save_as progress updates."""
        self._saveas_current_index = current_plane
        self._saveas_progress = fraction
        self._saveas_done = fraction >= 1.0

    def update_frame_apply(self):
        pass
        """Update the frame_apply function of the image widget."""
        self.image_widget.frame_apply = {
            i: partial(self._combined_frame_apply, roi=i)
            for i in range(len(self.image_widget.managed_graphics))
        }

    def _combined_frame_apply(self, frame: np.ndarray, roi=None) -> np.ndarray:
        """alter final frame only once, in ImageWidget.frame_apply"""
        if self._gaussian_sigma > 0:
            frame = gaussian_filter(frame, sigma=self.gaussian_sigma)
        if self.proj == "mean-sub" and self._zstats_means:
            z = self.image_widget.current_index.get("z", 0)
            frame = frame - self._zstats_means[roi][z]
        return frame

    def calculate_offset(self, ev=None):  # type: ignore # noqa
        """Get the current frame, calculate the offset"""
        raise NotImplementedError()

    def _compute_zstats_single_roi(self, data_ix, arr):
        self.logger.info(f"Computing z-statistics for ROI {data_ix + 1}")

        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]  # TZYX

        stats = {"mean": [], "std": [], "snr": []}
        means = []

        for z in range(self.nz):
            self.logger.info(
                f"--- Processing Z-plane {z + 1}/{self.nz} for ROI {data_ix + 1} --",
            )
            stack = arr[:, z].astype(np.float32)
            mean_img = np.mean(stack, axis=0)
            std_img = np.std(stack, axis=0)
            snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

            stats["mean"].append(np.mean(mean_img))
            stats["std"].append(np.mean(std_img))
            stats["snr"].append(np.mean(snr_img))
            means.append(mean_img)

            self.logger.info(
                f"ROI {data_ix + 1} - Z-plane {z + 1}: "
                f"Mean: {stats['mean'][-1]:.2f}, "
                f"Std: {stats['std'][-1]:.2f}, "
                f"SNR: {stats['snr'][-1]:.2f}",
            )

            self._zstats_progress[data_ix] = (z + 1) / self.nz
            self._zstats_current_z[data_ix] = z

        self._zstats[data_ix] = stats
        self._zstats_means[data_ix] = np.stack(means)
        self._zstats_done[data_ix] = True

    def compute_zstats(self):
        if not self.image_widget or not self.image_widget.data:
            return

        if all(hasattr(arr, "__array__") for arr in self.image_widget.data):
            arrs = [np.array(arr) for arr in self.image_widget.data]
        else:
            arrs = self.image_widget.data
        for data_ix, arr in enumerate(arrs):
            self.logger.debug(f"Sending array index {data_ix} for z-stat computation..")
            threading.Thread(
                target=self._compute_zstats_single_roi, args=(data_ix, arr), daemon=True
            ).start()
