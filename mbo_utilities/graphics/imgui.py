import logging
import webbrowser
from pathlib import Path
from typing import Literal
import threading
from functools import partial

import imgui_bundle
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
)

from mbo_utilities.file_io import (
    MBO_SUPPORTED_FTYPES,
    get_mbo_dirs,
    save_last_savedir,
    get_last_savedir_path,
    load_last_savedir
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
    draw_register_z_progress,
)
from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings, draw_tab_process
from mbo_utilities.lazy_array import imread, imwrite
from mbo_utilities.phasecorr import apply_scan_phase_offsets
from mbo_utilities.graphics.gui_logger import GuiLogger, GuiLogHandler
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
USER_PIPELINES = ["suite2p"]


def _save_as_worker(path, **imwrite_kwargs):
    data = imread(path, roi=imwrite_kwargs.pop("roi", None))
    imwrite(data, **imwrite_kwargs)

def draw_menu(parent):
    # (accessible from the "Tools" menu)
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
        opened, _ = imgui.begin(
            "MBO Debug Panel",
            parent.show_debug_panel,
        )
        if opened:
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
            imgui.begin_disabled(not all(parent._zstats_done))
            if imgui.begin_tab_item("Summary Stats")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
                parent.draw_stats_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()
            imgui.end_disabled()
            if imgui.begin_tab_item("Process")[0]:
                draw_tab_process(parent)
                imgui.end_tab_item()
            imgui.end_tab_bar()


def draw_saveas_popup(parent):
    if getattr(parent, "_saveas_popup_open"):
        imgui.open_popup("Save As")
        parent._saveas_popup_open = False

    if imgui.begin_popup_modal("Save As")[0]:
        imgui.dummy(imgui.ImVec2(0, 5))

        imgui.set_next_item_width(hello_imgui.em_size(25))

        # Directory + Ext
        current_dir_str = str(Path(parent._saveas_outdir).expanduser().resolve()) if parent._saveas_outdir else ""
        changed, new_str = imgui.input_text("Save Dir", current_dir_str)
        if changed:
            parent._saveas_outdir = new_str

        imgui.same_line()
        if imgui.button("Browse"):
            res = pfd.select_folder(parent._saveas_outdir or str(Path.home()))
            if res:
                selected_str = str(res.result())
                parent._saveas_outdir = selected_str
                save_last_savedir(Path(selected_str))

        imgui.set_next_item_width(hello_imgui.em_size(25))
        _, parent._ext_idx = imgui.combo("Ext", parent._ext_idx, MBO_SUPPORTED_FTYPES)
        parent._ext = MBO_SUPPORTED_FTYPES[parent._ext_idx]

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Options Section
        parent._saveas_rois = checkbox_with_tooltip(
            "Save ScanImage multi-ROI Separately",
            parent._saveas_rois,
            "Enable to save each mROI individually."
            " mROI's are saved to subfolders: plane1_roi1, plane1_roi2, etc."
            " These subfolders can be merged later using mbo_utilities.merge_rois()."
            " This can be helpful as often mROI's are non-contiguous and can drift in orthogonal directions over time."
        )
        if parent._saveas_rois:
            try:
                num_rois = parent.image_widget.data[0].num_rois
            except Exception as e:
                num_rois = 1

            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose mROI(s):")
            imgui.dummy(imgui.ImVec2(0, 5))

            if imgui.button("All##roi"):
                parent._saveas_selected_roi = set(range(num_rois))
            imgui.same_line()
            if imgui.button("None##roi"):
                parent._saveas_selected_roi = set()

            imgui.columns(2, borders=False)
            for i in range(num_rois):
                imgui.push_id(f"roi_{i}")
                selected = i in parent._saveas_selected_roi
                _, selected = imgui.checkbox(f"mROI {i + 1}", selected)
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

        imgui.dummy(imgui.ImVec2(0, 5))

        parent._overwrite = checkbox_with_tooltip(
            "Overwrite", parent._overwrite, "Replace any existing output files."
        )
        parent._register_z = checkbox_with_tooltip(
            "Register Z-Planes Axially", parent._register_z, "Register adjacent z-planes to each other using Suite3D."
        )
        fix_phase_changed, fix_phase_value = imgui.checkbox(
            "Fix Scan Phase", parent._fix_phase
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted("Correct for bi-directional scan phase offsets.")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if fix_phase_changed:
            parent.fix_phase = fix_phase_value

        use_fft, use_fft_value = imgui.checkbox(
            "Subpixel Phase Correction", parent._use_fft
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted("Use FFT-based subpixel registration (slower, more precise).")
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()
        if use_fft:
            parent.use_fft = use_fft_value

        parent._debug = checkbox_with_tooltip(
            "Debug",
            parent._debug,
            "Print additional information to the terminal during process.",
        )

        imgui.spacing()
        imgui.text("Chunk Size (MB)")
        set_tooltip(
            "The size of the chunk, in MB, to read and write at a time. Larger chunks may be faster but use more memory.",
        )

        imgui.set_next_item_width(hello_imgui.em_size(20))
        _, parent._saveas_chunk_mb = imgui.drag_int(
            "##chunk_size_mb_mb",
            parent._saveas_chunk_mb,
            v_speed=1,
            v_min=1,
            v_max=1024,
        )

        imgui.spacing()
        imgui.separator()

        # Z-plane selection
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Choose z-planes:")
        imgui.dummy(imgui.ImVec2(0, 5))

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
                last_dir = load_last_savedir(default=Path().home())
                parent._saveas_outdir = last_dir
            try:
                save_planes = [p + 1 for p in parent._selected_planes]
                parent._saveas_total = len(save_planes)
                if parent._saveas_rois:
                    if (
                        not parent._saveas_selected_roi
                        or len(parent._saveas_selected_roi) == set()
                    ):
                        parent._saveas_selected_roi = set(
                            range(1, parent.num_rois + 1)
                        )
                    rois = sorted(parent._saveas_selected_roi)
                else:
                    rois = None

                outdir = Path(parent._saveas_outdir).expanduser()
                if not outdir.exists():
                    outdir.mkdir(parents=True, exist_ok=True)

                save_kwargs = {
                    "path": parent.fpath,
                    "outpath": parent._saveas_outdir,
                    "planes": save_planes,
                    "roi": rois,
                    "overwrite": parent._overwrite,
                    "debug": parent._debug,
                    "ext": parent._ext,
                    "target_chunk_mb": parent._saveas_chunk_mb,
                    "use_fft": parent._use_fft,
                    "register_z": parent._register_z,
                    "progress_callback": lambda frac, current_plane: parent.gui_progress_callback(frac, current_plane),
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
        **kwargs
    ):
        """
        Fastplotlib attachment, callable with fastplotlib.ImageWidget.add_gui(PreviewDataWidget)
        """

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
            size=250 if size is None else size,
            location=location,
            title=title,
            window_flags=flags,
        )

        # logger / debugger
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        gui_handler.setFormatter(logging.Formatter("%(message)s"))
        gui_handler.setLevel(logging.DEBUG)
        log.attach(gui_handler)
        log.set_global_level(logging.DEBUG)
        self.logger = log.get("gui")

        self.logger.info("Logger initialized.")

        self.s2p = Suite2pSettings()
        self._s2p_dir = ""
        self.kwargs = kwargs

        if implot.get_current_context() is None:
            implot.create_context()

        io = imgui.get_io()
        font_config = imgui.ImFontConfig()
        font_config.merge_mode = True

        fd_settings_dir = (
            Path(get_mbo_dirs()["imgui"])
            .joinpath("assets", "app_settings", "preview_settings.ini")
            .expanduser()
            .resolve()
        )
        io.set_ini_filename(str(fd_settings_dir))

        sans_serif_font = str(
            Path(imgui_bundle.__file__).parent.joinpath(
                "assets", "fonts", "Roboto", "Roboto-Regular.ttf"
            )
        )

        self._default_imgui_font = io.fonts.add_font_from_file_ttf(
            sans_serif_font,
            14,
            imgui.ImFontConfig()
        )

        imgui.push_font(self._default_imgui_font, self._default_imgui_font.legacy_size)

        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # image widget setup
        self.image_widget = iw

        self.num_arrays = len(self.image_widget.managed_graphics)
        self.shape = self.image_widget.data[0].shape
        self.is_mbo_scan = (True if isinstance(self.image_widget.data[0], MboRawArray) else False)

        if hasattr(self.image_widget.data[0], "rois"):
            self.num_rois = len(self.image_widget.data[0].rois)
            if self.num_arrays > 1:
                self._array_type = "roi"
            else:
                self._array_type = "array"
        else:
            self.num_rois = 1
            self._array_type = "array"

        print(f"Num rois: {self.num_rois}")
        if self.is_mbo_scan:
            for arr in self.image_widget.data:
                arr.fix_phase = False
                arr.use_fft = False

        self._fix_phase = False
        self._use_fft = False

        if self.image_widget.window_funcs is None:
            self.image_widget.window_funcs = {"t": (np.mean, 0)}

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        for subplot in self.image_widget.figure:
            subplot.toolbar = False
        self.image_widget._image_widget_sliders._loop = True  # noqa

        self._zstats = [{"mean": [], "std": [], "snr": []} for _ in range(self.num_rois)]
        self._zstats_means = [None] * self.num_rois
        self._zstats_mean_scalar = [0.0] * self.num_rois
        self._zstats_done = [False] * self.num_rois
        self._zstats_progress = [0.0] * self.num_rois
        self._zstats_current_z = [0] * self.num_rois
        print(f"zstats: {self._zstats}")

        # Settings menu flags
        self.show_debug_panel = False
        self.show_scope_window = False

        # ------------------------properties
        for arr in self.image_widget.data:
            if hasattr(arr, "border"):
                arr.border = 3
            if hasattr(arr, "max_offset"):
                arr.max_offset = 3
            if hasattr(arr, "upsample"):
                arr.upsample = 20
            if hasattr(arr, "fix_phase"):
                arr.fix_phase = False
            if hasattr(arr, "use_fft"):
                arr.use_fft = False

        self._max_offset = 3
        self._gaussian_sigma = 0
        self._current_offset = [0.0] * self.num_arrays
        self._window_size = 1
        self._phase_upsample = 20
        self._border = 3
        self._auto_update = False
        self._proj = "mean"

        self._register_z = False
        self._register_z_progress = 0.0
        self._register_z_done = False
        self._register_z_current_msg = ""

        self._selected_pipelines = None
        self._selected_array = 0
        self._selected_planes = set()
        self._planes_str = str(getattr(self, "_planes_str", ""))

        # properties for saving to another filetype
        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = MBO_SUPPORTED_FTYPES.index(".tiff")

        self._overwrite = True
        self._debug = False

        self._saveas_chunk_mb = 100

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

    def gui_progress_callback(self, frac, meta=None):
        """
        Handles both saving progress (z-plane) and Suite3D registration progress.
        The `meta` parameter may be a plane index (int) or message (str).
        """
        if isinstance(meta, (int, np.integer)):
            # This is standard save progress
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0

        elif isinstance(meta, str):
            # Suite3D progress message
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0


    @property
    def s2p_dir(self):
        return self._s2p_dir

    @s2p_dir.setter
    def s2p_dir(self, value):
        self.logger.info(f"Setting Suite2p directory to {value}")
        self._s2p_dir = value

    @property
    def register_z(self):
        return self._register_z

    @register_z.setter
    def register_z(self, value):
        self._register_z = value

    @property
    def current_offset(self) -> list[float]:
        if not self.fix_phase:
            return [0.0 for _ in self.image_widget.data]
        if all(hasattr(array, "offset") for array in self.image_widget.data):
            return [array.offset for array in self.image_widget.data]
        return [0.0]

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
        self.image_widget.current_index = self.image_widget.current_index

    @property
    def use_fft(self):
        return self._use_fft

    @use_fft.setter
    def use_fft(self, value):
        self._use_fft = value
        for arr in self.image_widget.data:
            arr.use_fft = value
        self.update_frame_apply()
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
        self.image_widget.current_index = self.image_widget.current_index

    @property
    def max_offset(self):
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value):
        self._max_offset = value
        for arr in self.image_widget.data:
            arr.max_offset = value
            self.logger.info(f"Max offset set to {value}.")
        self.image_widget.current_index = self.image_widget.current_index

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
        self.image_widget.current_index = self.image_widget.current_index

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
        self.image_widget.current_index = self.image_widget.current_index

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self.logger.info(f"Window size set to {value}.")
        self.image_widget.window_funcs["t"].window_size = value
        self._window_size = value

    @property
    def phase_upsample(self):
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value):
        self._phase_upsample = value
        for arr in self.image_widget.data:
            arr.upsample = value
        self.image_widget.current_index = self.image_widget.current_index

    def update(self):
        draw_saveas_popup(self)
        draw_menu(self)
        draw_tabs(self)

    def draw_stats_section(self):
        if not any(self._zstats_done):
            return

        stats_list = self._zstats

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

            z_vals = np.ascontiguousarray(np.arange(1, len(mean_vals) + 1, dtype=np.float64))
            mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
            std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

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

            with imgui_ctx.begin_child("##PlotsCombined", size=imgui.ImVec2(0, 0), child_flags=cflags):
                imgui.text("Z-plane Signal: Combined")
                set_tooltip(
                    f"Gray = per-ROI z-profiles (mean over frames)."
                    f" Blue shade = across-ROI mean ± std; blue line = mean."
                    f" Hover gray lines for values.",
                    True
                )

                # build per-ROI series
                roi_series = [
                    np.asarray(self._zstats[r]["mean"], float)
                    for r in range(self.num_rois)
                ]

                L = min(len(s) for s in roi_series)
                z = np.asarray(z_vals[:L], float)
                roi_series = [s[:L] for s in roi_series]
                stack = np.vstack(roi_series)
                mean_vals = stack.mean(axis=0)
                std_vals = stack.std(axis=0)
                lower = mean_vals - std_vals
                upper = mean_vals + std_vals

                if implot.begin_plot("Z-Plane Plot (Combined)", imgui.ImVec2(-1, 300)):
                    style_seaborn_dark()
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value
                    )

                    implot.setup_axis_limits(
                        implot.ImAxis_.x1.value,
                        float(z[0]),
                        float(z[-1])
                    )
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

                    for i, ys in enumerate(roi_series):
                        label = f"ROI {i + 1}##roi{i}"
                        implot.push_style_var(implot.StyleVar_.line_weight.value, 1)
                        implot.push_style_color(implot.Col_.line.value, (0.6, 0.6, 0.6, 0.35))
                        implot.plot_line(label, z, ys)
                        implot.pop_style_color()
                        implot.pop_style_var()

                    implot.push_style_color(implot.Col_.fill.value, (0.2, 0.4, 0.8, 0.25))
                    implot.plot_shaded("Mean ± Std##band", z, lower, upper)
                    implot.pop_style_color()

                    implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
                    implot.plot_line("Mean##line", z, mean_vals)
                    implot.pop_style_var()

                    implot.end_plot()

        else:
            array_idx = self._selected_array
            stats = stats_list[array_idx]
            if not stats or "mean" not in stats:
                return

            mean_vals = np.array(stats["mean"])
            std_vals = np.array(stats["std"])
            snr_vals = np.array(stats["snr"])
            n = min(len(mean_vals), len(std_vals), len(snr_vals))

            mean_vals, std_vals, snr_vals = mean_vals[:n], std_vals[:n], snr_vals[:n]

            z_vals = np.ascontiguousarray(np.arange(1, n + 1, dtype=np.float64))
            mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
            std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

            imgui.text(f"Stats for {self._array_type} {array_idx + 1}")

            with imgui_ctx.begin_child(f"##Summary{array_idx}", size=imgui.ImVec2(0, 0), child_flags=cflags):
                if imgui.begin_table(f"zstats{array_idx}", 4, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg):
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)
                    imgui.table_headers_row()
                    for j in range(n):
                        imgui.table_next_row()
                        for val in (int(z_vals[j]), mean_vals[j], std_vals[j], snr_vals[j]):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

            style_seaborn_dark()
            with imgui_ctx.begin_child(f"##Plots1{array_idx}", size=imgui.ImVec2(0, 0), child_flags=cflags):
                imgui.text("Z-plane Signal: Mean ± Std")
                if implot.begin_plot(f"Z-Plane Signal {array_idx}", imgui.ImVec2(-1, 300)):
                    implot.setup_axes("Z-Plane", "Mean Fluorescence",
                                      implot.AxisFlags_.auto_fit.value,
                                      implot.AxisFlags_.auto_fit.value)
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    implot.plot_error_bars(f"Mean ± Std {array_idx}",
                                           z_vals, mean_vals, std_vals)
                    implot.plot_line(f"Mean {array_idx}", z_vals, mean_vals)
                    implot.end_plot()

    def draw_preview_section(self):
        imgui.dummy(imgui.ImVec2(0, 5))
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize
        with imgui_ctx.begin_child("##PreviewChild", imgui.ImVec2(0, 0), cflags):

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
            imgui.spacing()

            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(2, 2))
            imgui.begin_group()

            options = ["mean", "max", "std"]
            disabled_label = ("mean-sub (pending)" if not all(self._zstats_done) else "mean-sub")
            options.append(disabled_label)

            current_display_idx = options.index(self.proj if self._proj != "mean-sub" else disabled_label)

            imgui.set_next_item_width(hello_imgui.em_size(6))
            proj_changed, selected_display_idx = imgui.combo("Projection", current_display_idx, options)
            set_tooltip(
                "Choose projection method over the sliding window:\n\n"
                " “mean” (average)\n"
                " “max” (peak)\n"
                " “std” (variance)\n"
                " “mean-sub” (mean-subtracted)."
            )

            if proj_changed:
                selected_label = options[selected_display_idx]
                if selected_label == "mean-sub (pending)":
                    pass
                else:
                    self.proj = selected_label
                    if self.proj == "mean-sub":
                        self.update_frame_apply()
                    else:
                        self.image_widget.window_funcs["t"].func = getattr(
                            np, self.proj
                        )

            # Window size for projections
            imgui.set_next_item_width(hello_imgui.em_size(6))
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
            imgui.set_next_item_width(hello_imgui.em_size(6))
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

            imgui.spacing()
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction"
            )

            imgui.separator()
            imgui.begin_group()

            imgui.set_next_item_width(hello_imgui.em_size(10))
            phase_changed, phase_value = imgui.checkbox("Fix Phase", self._fix_phase)
            set_tooltip("Enable to apply scan-phase correction which shifts every other line/row of pixels "
                        "to maximize correlation between these rows.")
            if phase_changed:
                self.fix_phase = phase_value
                self.logger.info(f"Fix Phase: {phase_value}")

            imgui.set_next_item_width(hello_imgui.em_size(10))
            fft_changed, fft_value = imgui.checkbox("Sub-Pixel (slower)", self._use_fft)
            set_tooltip("Use FFT-based sub-pixel registration (slower but more accurate).")
            if fft_changed:
                self.use_fft = fft_value
                self.logger.info(f"Use-FFT: {fft_value}")

            imgui.columns(2, "offsets", False)
            for i, iw in enumerate(self.image_widget.data):
                ofs = self.current_offset[i]
                is_sequence = isinstance(ofs, (list, np.ndarray, tuple))

                if is_sequence:
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

                if max_abs_offset > self.max_offset:
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
                    imgui.text(display_text)
                    imgui.pop_style_color()
                else:
                    imgui.text(display_text)

                if is_sequence and imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Per‐frame offsets:")
                    for frame_idx, val in enumerate(ofs_list):
                        imgui.text(f"  frame {frame_idx}: {val:.3f}")
                    imgui.end_tooltip()

                imgui.next_column()
            imgui.columns(1)

            imgui.set_next_item_width(hello_imgui.em_size(5))
            upsample_changed, upsample_val = imgui.input_int("Upsample", self._phase_upsample, step=1, step_fast=2)
            set_tooltip(
                "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment.")
            if upsample_changed:
                self.phase_upsample = max(1, upsample_val)
                self.logger.info(f"New upsample: {upsample_val}")
            imgui.set_next_item_width(hello_imgui.em_size(5))
            border_changed, border_val = imgui.input_int(
                "Exclude border-px", self._border, step=1, step_fast=2
            )
            set_tooltip(
                "Number of pixels to exclude from the edges of the image when computing the scan-phase offset."
            )
            if border_changed:
                self.border = max(0, border_val)
                self.logger.info(f"New border: {border_val}")

            imgui.set_next_item_width(hello_imgui.em_size(5))
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
        draw_register_z_progress(self)
        draw_saveas_progress(self)

    def get_raw_frame(self) -> tuple[ndarray, ...]:
        idx = self.image_widget.current_index
        t = idx.get("t", 0)
        z = idx.get("z", 0)
        return tuple(ndim_to_frame(arr, t, z) for arr in self.image_widget.data)

    def update_frame_apply(self):
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
        if self.proj == "mean-sub" and self._zstats_done[arr_idx]:
            z_idx = self.image_widget.current_index.get("z", 0)
            frame = frame - self._zstats_mean_scalar[arr_idx][z_idx]
        return frame

    def _compute_zstats_single_roi(self, roi, fpath):
        arr = imread(fpath)
        if hasattr(arr, "roi"):
            arr.roi = roi

        stats, means = {"mean": [], "std": [], "snr": []}, []
        self._tiff_lock = threading.Lock()
        for z in range(self.nz):
            with self._tiff_lock:
                stack = arr[::10, z].astype(np.float32) # Z, Y, X
                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))
                stats["mean"].append(float(np.mean(mean_img)))
                stats["std"].append(float(np.mean(std_img)))
                stats["snr"].append(float(np.mean(snr_img)))
                means.append(mean_img)
                self._zstats_progress[roi - 1] = (z + 1) / self.nz
                self._zstats_current_z[roi - 1] = z

        self._zstats[roi - 1] = stats
        means_stack = np.stack(means)

        self._zstats_means[roi - 1] = means_stack
        self._zstats_mean_scalar[roi - 1] = means_stack.mean(axis=(1, 2))
        self._zstats_done[roi - 1] = True

    def _compute_zstats_single_array(self, idx, arr):
        stats, means = {"mean": [], "std": [], "snr": []}, []
        self._tiff_lock = threading.Lock()

        for z in ([0] if arr.ndim == 3 else range(self.nz)):
            with (self._tiff_lock):

                stack = arr[::10].astype(np.float32) if arr.ndim == 3 else arr[::10, z].astype(np.float32)

                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

                stats["mean"].append(float(np.mean(mean_img)))
                stats["std"].append(float(np.mean(std_img)))
                stats["snr"].append(float(np.mean(snr_img)))

                means.append(mean_img)
                self._zstats_progress[idx - 1] = (z + 1) / self.nz
                self._zstats_current_z[idx - 1] = z

        self._zstats[idx - 1] = stats
        means_stack = np.stack(means)
        self._zstats_means[idx - 1] = means_stack
        self._zstats_mean_scalar[idx - 1] = means_stack.mean(axis=(1, 2))
        self._zstats_done[idx - 1] = True


    def compute_zstats(self):
        if not self.image_widget or not self.image_widget.data:
            return

        # if arrays have .roi attribute (multi-ROI mode)
        if hasattr(self.image_widget.data[0], "roi") or self.num_rois > 1:
            for roi in range(1, self.num_rois + 1):
                threading.Thread(
                    target=self._compute_zstats_single_roi,
                    args=(roi, self.fpath),
                    daemon=True,
                ).start()
        else:
            # treat each array as a virtual ROI
            for idx, arr in enumerate(self.image_widget.data, start=1):
                threading.Thread(
                    target=self._compute_zstats_single_array,
                    args=(idx, arr),
                    daemon=True,
                ).start()