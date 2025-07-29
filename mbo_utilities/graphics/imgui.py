import logging
import webbrowser
from pathlib import Path
from typing import Literal
import threading
from functools import partial
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

from imgui_bundle import (
    imgui,
    hello_imgui,
    imgui_ctx,
    implot,
    portable_file_dialogs as pfd,
    ImVec2,
)

from mbo_utilities.file_io import (
    MBO_SUPPORTED_FTYPES,
    get_mbo_dirs,
)
from mbo_utilities.array_types import MboRawArray
from mbo_utilities.graphics._imgui import (
    begin_popup_size,
    ndim_to_frame,
    style_seaborn_dark,
)
from mbo_utilities.graphics._widgets import (
    set_tooltip,
    checkbox_with_tooltip,
    draw_scope,
)
from mbo_utilities.graphics.progress_bar import (
    draw_zstats_progress,
    draw_saveas_progress,
)
from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings, draw_tab_process
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.phasecorr import apply_scan_phase_offsets
from mbo_utilities.graphics.gui_logger import GuiLogger, GuiLogHandler, GUI_LOGGERS
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

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow

REGION_TYPES = ["Full FOV", "Sub-FOV"]
USER_PIPELINES = ["suite2p", "masknmf"]


def _save_as_worker(path, **imwrite_kwargs):
    data = imread(path, roi=imwrite_kwargs.pop("roi", None))
    imwrite(data, **imwrite_kwargs)


@dataclass
class SaveStatus:
    # progressbar
    progress: float = 0.0
    plane: int | None = None
    message: str = ""
    logs: dict = field(default_factory=dict)


def draw_menu(parent):
    # (accessible from the "Tools" menu)
    if parent.show_style_window:
        _, parent.show_style_window = imgui.begin(
            "Style Editor", parent.show_style_window
        )
        imgui.show_style_editor()
        imgui.end()
    if parent.show_scope_window:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
        _, parent.show_scope_window = imgui.begin(
            "Scope Inspector",
            parent.show_scope_window,
        )
        draw_scope()
        imgui.end()
    if parent.show_debug_panel:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)  # type: ignore # noqa
        _, parent.show_debug_panel = imgui.begin(
            "MBO Debug Panel",
            parent.show_debug_panel,
        )
        parent.debug_panel.draw()
        imgui.end()
    with imgui_ctx.begin_child(
        "menu",
        window_flags=imgui.WindowFlags_.menu_bar,  # noqa,
        child_flags=imgui.ChildFlags_.auto_resize_y
        | imgui.ChildFlags_.always_auto_resize,
    ):
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item(
                    "Save as", "Ctrl+S", p_selected=False, enabled=parent.is_mbo_scan
                )[0]:
                    parent._saveas_popup_open = True
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
                _, parent.show_style_window = imgui.menu_item(
                    "Style Editor", "", parent.show_style_window, True
                )
                _, parent.show_debug_panel = imgui.menu_item(
                    "Debug Panel",
                    "",
                    p_selected=parent.show_debug_panel,
                    enabled=True,
                )
                _, parent.show_scope_window = imgui.menu_item(
                    "Scope Inspector", "", parent.show_scope_window, True
                )
                imgui.end_menu()
        imgui.end_menu_bar()
    pass


def draw_tabs(parent):
    with imgui_ctx.begin_child(
        "tabs",
    ):
        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
                parent.draw_preview_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Summary Stats")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
                parent.draw_stats_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Process")[0]:
                draw_tab_process(parent)
                imgui.end_tab_item()
            imgui.end_tab_bar()


def draw_saveas_popup(parent):
    if getattr(parent, "_saveas_popup_open"):
        imgui.open_popup("Save As")
        parent._saveas_popup_open = False

    if imgui.begin_popup_modal("Save As")[0]:
        imgui.dummy(ImVec2(0, 5))

        imgui.set_next_item_width(hello_imgui.em_size(25))

        # Directory + Ext
        _, parent._saveas_outdir = imgui.input_text(
            "Save Dir",
            str(Path(parent._saveas_outdir).expanduser().resolve()),
            256,
        )
        imgui.same_line()
        if imgui.button("Browse"):
            home = Path().home()
            res = pfd.select_folder(str(home))
            if res:
                parent._saveas_outdir = res.result()

        imgui.set_next_item_width(hello_imgui.em_size(25))
        _, parent._ext_idx = imgui.combo("Ext", parent._ext_idx, MBO_SUPPORTED_FTYPES)
        parent._ext = MBO_SUPPORTED_FTYPES[parent._ext_idx]

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Options Section
        parent._saveas_rois = checkbox_with_tooltip(
            "Save ROI's",
            parent._saveas_rois,
            "Enable to save each ROI individually. Saved to subfolders like roi1/, roi2/, etc.",
        )
        if parent._saveas_rois:
            try:
                num_rois = parent.image_widget.data[0].num_rois
            except Exception as e:
                num_rois = 1

            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose ROI(s):")
            imgui.dummy(ImVec2(0, 5))

            if imgui.button("All##roi"):
                parent._saveas_selected_roi = set(range(num_rois))
            imgui.same_line()
            if imgui.button("None##roi"):
                parent._saveas_selected_roi = set()

            imgui.columns(2, borders=False)
            for i in range(num_rois):
                imgui.push_id(f"roi_{i}")
                selected = i in parent._saveas_selected_roi
                _, selected = imgui.checkbox(f"ROI {i + 1}", selected)
                if selected:
                    parent._saveas_selected_roi.add(i)
                else:
                    parent._saveas_selected_roi.discard(i)
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

        parent._overwrite = checkbox_with_tooltip(
            "Overwrite", parent._overwrite, "Replace any existing output files."
        )

        fix_phase_changed, fix_phase_value = imgui.checkbox(
            "Fix Scan Phase", parent._fix_phase
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted("Apply scan-phase correction to interleaved lines.")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if fix_phase_changed:
            parent.fix_phase = fix_phase_value

        parent._debug = checkbox_with_tooltip(
            "Debug",
            parent._debug,
            "Run with debugging, settings -> debug to view the outputs.",
        )
        parent._saveas_save_phase_png = checkbox_with_tooltip(
            "Save Phase Images",
            parent._saveas_save_phase_png,
            "Saves pre-post scan-phase images as PNGs to the save-directory.",
        )

        imgui.spacing()
        imgui.text("Chunk Size (MB)")
        set_tooltip(
            "Target chunk size when saving TIFF or binary. Affects I/O and memory usage."
        )

        imgui.set_next_item_width(hello_imgui.em_size(20))
        _, parent._saveas_chunk_mb = imgui.drag_int(
            "##target_chunk_mb",
            parent._saveas_chunk_mb,
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
            num_planes = parent.image_widget.data[0].num_channels  # noqa
        except Exception as e:
            num_planes = 1
            hello_imgui.log(
                hello_imgui.LogLevel.error,
                f"Could not read number of planes: {e}",
            )

        if imgui.button("All"):
            parent._selected_planes = set(range(num_planes))
        imgui.same_line()
        if imgui.button("None"):
            parent._selected_planes = set()

        imgui.columns(2, borders=False)
        for i in range(num_planes):
            imgui.push_id(i)
            selected = i in parent._selected_planes
            _, selected = imgui.checkbox(f"Plane {i + 1}", selected)
            if selected:
                parent._selected_planes.add(i)
            else:
                parent._selected_planes.discard(i)
            imgui.pop_id()
            imgui.next_column()
        imgui.columns(1)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if imgui.button("Save", imgui.ImVec2(100, 0)):
            if not parent._saveas_outdir:
                parent._saveas_outdir = get_mbo_dirs()["data"].joinpath("data")
            try:
                save_planes = [p + 1 for p in parent._selected_planes]
                parent._saveas_total = len(save_planes)
                if parent._saveas_rois:
                    if (
                        not parent._saveas_selected_roi
                        or len(parent._saveas_selected_roi) == set()
                    ):
                        parent._saveas_selected_roi = set(
                            range(1, parent.num_arrays + 1)
                        )
                    rois = sorted(parent._saveas_selected_roi)
                else:
                    rois = None

                save_kwargs = {
                    "path": parent.fpath,
                    "outpath": parent._saveas_outdir,
                    "planes": save_planes,
                    "roi": rois,
                    "overwrite": parent._overwrite,
                    "debug": parent._debug,
                    "ext": parent._ext,
                    "target_chunk_mb": parent._saveas_chunk_mb,
                    "progress_callback": lambda frac,
                    current_plane: parent.gui_progress_callback(frac, current_plane),
                }
                parent.logger.info(f"Saving planes {save_planes}")
                parent.logger.info(
                    f"Saving to {parent._saveas_outdir} as {parent._ext}"
                )
                threading.Thread(
                    target=_save_as_worker, kwargs=save_kwargs, daemon=True
                ).start()
                imgui.close_current_popup()
            except Exception as e:
                parent.logger.info(f"Error saving data: {e}")
                imgui.close_current_popup()

        imgui.same_line()
        if imgui.button("Cancel"):
            imgui.close_current_popup()

        imgui.end_popup()


class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None | list = None,
        threading_enabled: bool = True,
        size: int = None,
        location: Literal["bottom", "right"] = "right",
        title: str = "Data Preview",
        show_title: bool = False,
        movable: bool = False,
        resizable: bool = False,
        scrollable: bool = False,
        auto_resize: bool = True,
        window_flags: int | None = None,
        rois: int = 1,
    ):
        """
        Fastplotlib attachment, callable with fastplotlib.ImageWidget.add_gui(PreviewDataWidget)
        """
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        for name in GUI_LOGGERS:
            lg = log.get(name)
            lg.addHandler(gui_handler)
            lg.setLevel(logging.WARNING)
            lg.disabled = False
            lg.propagate = True
        self.logger = log.get("gui")
        self.s2p = Suite2pSettings()
        self.logger.info("Logger initialized.")

        flags = (
            (imgui.WindowFlags_.no_title_bar if not show_title else 0)
            | (imgui.WindowFlags_.no_move if not movable else 0)
            | (imgui.WindowFlags_.no_resize if not resizable else 0)
            | (imgui.WindowFlags_.no_scrollbar if not scrollable else 0)
            | (imgui.WindowFlags_.always_auto_resize if auto_resize else 0)
            | (window_flags or 0)
        )
        super().__init__(
            figure=iw.figure,
            size=350 if size is None else size,
            location=location,
            title=title,
            window_flags=flags,
        )
        if implot.get_current_context() is None:
            implot.create_context()

        # backend.create_fonts_texture()
        io = imgui.get_io()
        fd_settings_dir = (
            Path(get_mbo_dirs()["imgui"])
            .joinpath("assets", "app_settings", "preview_settings.ini")
            .expanduser()
            .resolve()
        )
        io.set_ini_filename(str(fd_settings_dir))

        fonts_path = Path(get_mbo_dirs()["assets"]).joinpath("fonts")
        self._fonts = []
        for font_file in fonts_path.rglob("*.ttf"):
            self._fonts.append(io.fonts.add_font_from_file_ttf(str(font_file), 16))
        io.fonts.build()

        self.font_size = 12
        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # image widget setup
        self.image_widget = iw
        self.rois = rois
        self.shape = self.image_widget.data[0].shape
        if self.image_widget.window_funcs is None:
            self.image_widget.window_funcs = {"t": (np.mean, 0)}

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        self.is_mbo_scan = (
            True if isinstance(self.image_widget.data[0], MboRawArray) else False
        )
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                arr.fix_phase = False

        for subplot in self.image_widget.figure:
            subplot.toolbar = False

        self.image_widget._image_widget_sliders._loop = True  # noqa
        if hasattr(self.image_widget, "rois") or hasattr(
            self.image_widget.data[0], "rois"
        ):
            self._array_type = "roi"
        else:
            self._array_type = "array"

        # boolean flags: imgui.begin() calls that are drawn
        # when these are set to true.
        self.show_style_window = False
        self.show_metrics_window = False
        self.show_debug_panel = False
        self.show_scope_window = False

        # properties: each have a @property getter and setter
        # that controls what to do with the value.
        # different filetypes sometimes require different handling
        # though it may be easier to group these instead into a function or a class.
        self._max_offset = 3
        self._gaussian_sigma = 0
        self._current_offset = [0.0] * self.num_arrays
        self._window_size = 1
        self._phase_upsample = 20
        self._border = 3
        self._auto_update = False
        self._proj = "mean"

        self._selected_pipelines = None
        self._selected_array = 0

        # properties for saving to another filetype
        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = MBO_SUPPORTED_FTYPES.index(".tiff")
        self._selected_planes = set()
        self._planes_str = str(getattr(self, "_planes_str", ""))
        self._overwrite = True
        self._fix_phase = False
        self._debug = False
        self._saveas_save_phase_png = False
        self._saveas_chunk_mb = 20

        # zstats: an entry for each given array
        # these are sent to a thread to compute
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
        self.set_context_info()

        if threading_enabled:
            self.logger.info("Starting zstats computation in a separate thread.")
            threading.Thread(target=self.compute_zstats, daemon=True).start()

    def set_context_info(self):
        if isinstance(self.fpath, list):
            title = f"{[Path(f).stem for f in self.fpath]}"
        else:
            title = f"Filepath: {Path(self.fpath).stem}"
        self.image_widget.figure.canvas.set_title(str(title))

    @property
    def current_offset(self) -> list[float]:
        if not self.fix_phase:
            return [0.0 for _ in self.image_widget.data]
        if not self.is_mbo_scan:
            self.logger.critical(
                "Scan-phase correction is only implemented for MBO scans. "
                "Returning zero offsets."
            )
            return [0.0 for _ in self.image_widget.data]
        if all(hasattr(array, "offset") for array in self.image_widget.data):
            self.logger.debug(
                f"All arrays have offset attribute. Setting from array.offset"
            )
            return [array.offset for array in self.image_widget.data]
        else:
            raise NotImplementedError(
                "Scan-phase correction is only implemented for MBO scans."
            )

    @property
    def fix_phase(self):
        return self._fix_phase

    @fix_phase.setter
    def fix_phase(self, value):
        self._fix_phase = value
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                if isinstance(arr, MboRawArray):
                    arr.fix_phase = value
        else:
            self.update_frame_apply()

        # force update
        self.image_widget.current_index = self.image_widget.current_index

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self, value):
        self._border = value
        for arr in self.image_widget.data:
            if isinstance(arr, MboRawArray):
                arr.border = value
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
            if isinstance(arr, MboRawArray):
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
    def selected_array(self):
        return self._selected_array

    @selected_array.setter
    def selected_array(self, value):
        if value < 0 or value >= len(self.image_widget.data):
            raise ValueError(
                f"Invalid array index: {value}. "
                f"Must be between 0 and {len(self.image_widget.managed_graphics) - 1}."
            )
        self._selected_array = value
        self.logger.info(f"Selected array index set to {value}.")
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
    def phase_upsample(self):
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value):
        self._phase_upsample = value
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                if isinstance(arr, MboRawArray):
                    arr.upsample = value
        else:
            self.logger.warning(
                "Phase upsample is only applicable to MBO Scan objects. "
                "No action taken."
            )

    def update(self):
        draw_saveas_popup(self)
        draw_menu(self)
        draw_tabs(self)

    def draw_stats_section(self):
        if not self._zstats_done:
            return

        stats_list = self._zstats if isinstance(self._zstats, list) else [self._zstats]

        imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats")
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # type: ignore # noqa
        imgui.spacing()

        # ROI selector
        array_labels = [
            f"{self._array_type} {i + 1}"
            for i in range(len(stats_list))
            if stats_list[i] and "mean" in stats_list[i]
        ]
        array_labels.append("Combined")
        avail = imgui.get_content_region_avail().x
        xpos = 0

        for i, label in enumerate(array_labels):
            if imgui.radio_button(label, self._selected_array == i):
                self._selected_array = i
            button_width = (
                imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 4
            )
            xpos += button_width + imgui.get_style().item_spacing.x

            if xpos >= avail:
                xpos = button_width
                imgui.new_line()
            else:
                imgui.same_line()

        imgui.separator()

        if self._selected_array == len(array_labels) - 1:  # Combined
            imgui.text(f"Stats for Combined {self._array_type}s")
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
                    f"Stats, averaged over {self._array_type}s",
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
                if implot.begin_plot("Z-Plane Plot", ImVec2(-1, 300)):
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    implot.setup_axis_limits(implot.ImAxis_.x1.value, 1, self.nz)
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    style_seaborn_dark()
                    if len(z_vals) == 1:
                        implot.plot_bars("Mean", z_vals, mean_vals, 0.5)
                        implot.plot_error_bars("Std", z_vals, mean_vals, std_vals)
                    else:
                        implot.plot_error_bars(
                            "Mean ± Std", z_vals, mean_vals, std_vals
                        )
                        implot.plot_line("Mean", z_vals, mean_vals)
                    implot.end_plot()

        else:
            array_idx = self._selected_array
            stats = stats_list[array_idx]

            if not stats or "mean" not in stats:
                return

            imgui.text(f"Stats for {self._array_type} {array_idx + 1}")
            mean_vals = np.array(stats["mean"])
            std_vals = np.array(stats["std"])
            snr_vals = np.array(stats["snr"])
            z_vals = np.arange(1, len(mean_vals) + 1, dtype=np.float32)

            with imgui_ctx.begin_child(
                f"##Summary{array_idx}", size=imgui.ImVec2(0, 0), child_flags=cflags
            ):
                if imgui.begin_table(
                    f"zstats{array_idx}",
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
                    f"Z-Plane Signal {array_idx}", size=imgui.ImVec2(-1, 300)
                ):
                    z_vals = np.arange(1, len(mean_vals) + 1, dtype=np.float32)
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    implot.setup_axis_limits(implot.ImAxis_.x1.value, 1, self.nz)
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    style_seaborn_dark()
                    if len(z_vals) == 1:
                        implot.plot_bars("Mean", z_vals, mean_vals, 0.5)
                        implot.plot_error_bars("Std", z_vals, mean_vals, std_vals)
                    else:
                        implot.plot_error_bars(
                            "Mean ± Std", z_vals, mean_vals, std_vals
                        )
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
            if not self.is_mbo_scan:
                return

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
                ofs = self.current_offset[i]
                is_sequence = isinstance(ofs, (list, np.ndarray, tuple))

                # Compute the “maximum absolute offset” we'll use to decide if text should be red
                if is_sequence:
                    # turn it into a flat Python list of floats
                    ofs_list = [float(x) for x in ofs]
                    max_abs_offset = max(abs(x) for x in ofs_list) if ofs_list else 0.0
                else:
                    ofs_list = None
                    max_abs_offset = abs(ofs)

                imgui.text(f"{self._array_type} {i + 1}:")
                imgui.next_column()

                if is_sequence:
                    display_text = "avg."
                    if len(ofs_list) > 1:
                        display_text += f" {np.round(np.mean(ofs_list), 2)}"
                    else:
                        display_text += f" {np.round(ofs_list[0], 2):.3f}"
                else:
                    display_text = f"{np.round(ofs, 2):.3f}"

                # ffset ≥ self.max_offset, color red
                if max_abs_offset >= self.max_offset:
                    imgui.push_style_color(
                        imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                    )

                imgui.text(display_text)

                if max_abs_offset >= self.max_offset:
                    imgui.pop_style_color()

                # show all frame offsets in a tooltip
                if is_sequence and imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text_colored(
                        imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Per‐frame offsets:"
                    )
                    for frame_idx, val in enumerate(ofs_list):
                        imgui.text(f"  frame {frame_idx}: {val:.3f}")
                    imgui.end_tooltip()
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

    def get_raw_frame(self) -> tuple[ndarray, ...]:
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
        """Update the frame_apply function of the image widget."""
        self.image_widget.frame_apply = {
            i: partial(self._combined_frame_apply, arr_idx=i)
            for i in range(len(self.image_widget.managed_graphics))
        }

    def _combined_frame_apply(self, frame: np.ndarray, arr_idx: int = 0) -> np.ndarray:
        """alter final frame only once, in ImageWidget.frame_apply"""
        if self._gaussian_sigma > 0:
            frame = gaussian_filter(frame, sigma=self.gaussian_sigma)
        if (not self.is_mbo_scan) and self._fix_phase:
            frame = apply_scan_phase_offsets(frame, self.current_offset[arr_idx])
        if self.proj == "mean-sub" and self._zstats_means:
            z = self.image_widget.current_index.get("z", 0)
            frame = frame - self._zstats_means[arr_idx][z]
        return frame

    def _compute_zstats_single_roi(self, roi, fpath):
        arr = imread(fpath)
        arr.roi = roi

        if HAS_TORCH and isinstance(arr, torch.Tensor):
            arr = arr[:]  # make sure it's dense

        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]

        stats = {"mean": [], "std": [], "snr": []}
        means = []

        self._tiff_lock = threading.Lock()
        for z in range(self.nz):
            self.logger.info(
                f"--- Processing Z-plane {z + 1}/{self.nz} for ROI {roi + 1} --"
            )
            with self._tiff_lock:
                stack = arr[:, z]
                if hasattr(stack, "astype"):  # works for Dask and NumPy
                    stack = stack.astype(np.float32)
                if hasattr(stack, "compute"):
                    stack = stack.compute()

                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

                stats["mean"].append(float(np.mean(mean_img)))
                stats["std"].append(float(np.mean(std_img)))
                stats["snr"].append(float(np.mean(snr_img)))
                means.append(mean_img)

                self.logger.info(
                    f"ROI {roi + 1} - Z-plane {z + 1}: "
                    f"Mean: {stats['mean'][-1]:.2f}, "
                    f"Std: {stats['std'][-1]:.2f}, "
                    f"SNR: {stats['snr'][-1]:.2f}",
                )

                self._zstats_progress[roi] = (z + 1) / self.nz
                self._zstats_current_z[roi] = z

        self._zstats[roi] = stats
        self._zstats_means[roi] = np.stack(means)
        self._zstats_done[roi] = True

    def compute_zstats(self):
        if not self.image_widget or not self.image_widget.data:
            return
        if self.rois is not None:
            for roi in range(self.rois):
                threading.Thread(
                    target=self._compute_zstats_single_roi, args=(roi, self.fpath), daemon=True
                ).start()
        else:
            threading.Thread(
                target=self._compute_zstats_single_roi, args=(self.rois, self.fpath), daemon=True
            ).start()

        # for data_ix, arr in enumerate(self.image_widget.data):
        #     self.logger.debug(f"Sending array index {data_ix} for z-stat computation..")
        #     threading.Thread(
        #         target=self._compute_zstats_single_roi, args=(data_ix, arr), daemon=True
        #     ).start()
