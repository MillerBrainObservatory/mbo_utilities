import logging
import webbrowser
from pathlib import Path
from typing import Literal
import threading
from functools import partial
import os
import importlib.util

# Force rendercanvas to use Qt backend if PySide6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
if importlib.util.find_spec("PySide6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")

import imgui_bundle
import numpy as np
from numpy import ndarray
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter

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
    load_last_savedir,
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
# Lazy import to avoid loading suite2p/torch/cupy until needed
# from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings, draw_tab_process
from mbo_utilities.lazy_array import imread, imwrite
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
    # Don't pass roi to imread - let it load all ROIs
    # Then imwrite will handle splitting/filtering based on roi parameter
    data = imread(path)
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
    if parent.show_metadata_viewer:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)
        _, parent.show_metadata_viewer = imgui.begin(
            "Metadata Viewer",
            parent.show_metadata_viewer,
        )
        if parent.image_widget and parent.image_widget.data:
            metadata = parent.image_widget.data[0].metadata
            from mbo_utilities.graphics._widgets import draw_metadata_inspector
            draw_metadata_inspector(metadata)
        else:
            imgui.text("No data loaded")
        imgui.end()
    with imgui_ctx.begin_child(
        "menu",
        window_flags=imgui.WindowFlags_.menu_bar,  # noqa,
        child_flags=imgui.ChildFlags_.auto_resize_y
        | imgui.ChildFlags_.always_auto_resize,
    ):
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File", True):
                # Open File - iw-array API
                if imgui.menu_item("Open File", "Ctrl+O", p_selected=False, enabled=True)[0]:
                    # Handle fpath being a list or a string
                    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
                    start_dir = str(Path(fpath).parent) if fpath and Path(fpath).exists() else str(Path.home())
                    parent._file_dialog = pfd.open_file(
                        "Select Data File",
                        start_dir,
                        ["TIFF Files", "*.tif *.tiff", "Raw Files", "*.raw", "All Files", "*.*"]
                    )
                # Open Folder - iw-array API
                if imgui.menu_item("Open Folder", "", p_selected=False, enabled=True)[0]:
                    # Handle fpath being a list or a string
                    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
                    start_dir = str(Path(fpath).parent) if fpath and Path(fpath).exists() else str(Path.home())
                    parent._folder_dialog = pfd.select_folder("Select Data Folder", start_dir)
                imgui.separator()
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
                if imgui.menu_item(
                    "Download User Guide Notebook", "", p_selected=False, enabled=True
                )[0]:
                    webbrowser.open(
                        "https://raw.githubusercontent.com/MillerBrainObservatory/mbo_utilities/master/demos/user_guide.ipynb"
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
    # Don't create an outer child window - let each tab manage its own scrolling
    # For single z-plane data, show all tabs
    # For multi-zplane data, show all tabs (user wants all tabs visible)
    if imgui.begin_tab_bar("MainPreviewTabs"):
        if imgui.begin_tab_item("Preview")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

            # Add metadata button at top of Preview tab
            imgui.spacing()
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.0, 0.0, 0.0, 1.0))  # Black button
            imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))  # White border
            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.0)
            if imgui.button("Show Metadata"):
                parent.show_metadata_viewer = not parent.show_metadata_viewer
            imgui.pop_style_var()
            imgui.pop_style_color(2)
            imgui.spacing()

            parent.draw_preview_section()
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.begin_disabled(not all(parent._zstats_done))
        if imgui.begin_tab_item("Summary Stats")[0]:
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
            # Create scrollable child for stats content
            with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                parent.draw_stats_section()
            imgui.pop_style_var()
            imgui.pop_style_var()
            imgui.end_tab_item()
        imgui.end_disabled()
        if imgui.begin_tab_item("Process")[0]:
            from mbo_utilities.graphics.pipeline_widgets import draw_tab_process
            draw_tab_process(parent)
            imgui.end_tab_item()
        if imgui.begin_tab_item("Suite2p Results")[0]:
            from mbo_utilities.graphics.suite2p_results import draw_tab_suite2p_results
            draw_tab_suite2p_results(parent)
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
        current_dir_str = (
            str(Path(parent._saveas_outdir).expanduser().resolve())
            if parent._saveas_outdir
            else ""
        )
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
            " This can be helpful as often mROI's are non-contiguous and can drift in orthogonal directions over time.",
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
            "Register Z-Planes Axially",
            parent._register_z,
            "Register adjacent z-planes to each other using Suite3D.",
        )
        fix_phase_changed, fix_phase_value = imgui.checkbox(
            "Fix Scan Phase", parent.fix_phase
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
            "Subpixel Phase Correction", parent.use_fft
        )
        imgui.same_line()
        imgui.text_disabled("(?)")
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
            imgui.text_unformatted(
                "Use FFT-based subpixel registration (slower, more precise)."
            )
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

                # Validate that at least one plane is selected
                if not save_planes:
                    parent.logger.error("No z-planes selected! Please select at least one plane.")
                    imgui.close_current_popup()
                    return

                parent._saveas_total = len(save_planes)
                if parent._saveas_rois:
                    if (
                        not parent._saveas_selected_roi
                        or len(parent._saveas_selected_roi) == set()
                    ):
                        # Get mROI count from data array (ScanImage-specific)
                        try:
                            mroi_count = parent.image_widget.data[0].num_rois
                        except Exception:
                            mroi_count = 1
                        parent._saveas_selected_roi = set(range(mroi_count))
                    # Convert 0-indexed UI values to 1-indexed ROI values for MboRawArray
                    rois = sorted([r + 1 for r in parent._saveas_selected_roi])
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
                    "use_fft": parent.use_fft,
                    "register_z": parent._register_z,
                    "progress_callback": lambda frac,
                    current_plane: parent.gui_progress_callback(frac, current_plane),
                }
                parent.logger.info(f"Saving planes {save_planes} with ROIs {rois if rois else 'stitched'}")
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
        **kwargs,
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

        # Also add console handler so logs appear in terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))

        # Only show DEBUG logs if MBO_DEBUG is set
        import os
        if bool(int(os.getenv("MBO_DEBUG", "0"))):
            console_handler.setLevel(logging.DEBUG)
            log.set_global_level(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            log.set_global_level(logging.INFO)

        log.attach(console_handler)
        self.logger = log.get("gui")

        self.logger.info("Logger initialized.")

        from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings
        self.s2p = Suite2pSettings()
        self._s2p_dir = ""
        self._s2p_savepath_flash_start = None  # Track when flash animation starts
        self._s2p_savepath_flash_count = 0  # Number of flashes
        self._s2p_show_savepath_popup = False  # Show popup when save path is missing
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
            sans_serif_font, 14, imgui.ImFontConfig()
        )

        imgui.push_font(self._default_imgui_font, self._default_imgui_font.legacy_size)

        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # image widget setup
        self.image_widget = iw

        # Unified naming: num_graphics matches len(iw.graphics)
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape
        self.is_mbo_scan = (
            True if isinstance(self.image_widget.data[0], MboRawArray) else False
        )

        # Only set if not already configured - the ImageWidget/processor handles defaults
        # We just need to track the window size for our UI
        self._window_size = 1
        self._gaussian_sigma = 0.0  # Track gaussian sigma locally, applied via spatial_func

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        for subplot in self.image_widget.figure:
            subplot.toolbar = False
        self.image_widget._sliders_ui._loop = True  # noqa

        self._zstats = [
            {"mean": [], "std": [], "snr": []} for _ in range(self.num_graphics)
        ]
        self._zstats_means = [None] * self.num_graphics
        self._zstats_mean_scalar = [0.0] * self.num_graphics
        self._zstats_done = [False] * self.num_graphics
        self._zstats_progress = [0.0] * self.num_graphics
        self._zstats_current_z = [0] * self.num_graphics

        # Settings menu flags
        self.show_debug_panel = False
        self.show_scope_window = False
        self.show_metadata_viewer = False

        # Processing properties are now on the processor, not the widget
        # We just track UI state here
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
        # Pre-fill with last saved directory if available
        last_dir = load_last_savedir(default=None)
        self._saveas_outdir = (
            str(last_dir) if last_dir else str(getattr(self, "_save_dir", ""))
        )
        self._saveas_total = 0

        self._saveas_selected_roi = set()  # -1 means all ROIs
        self._saveas_rois = False
        self._saveas_selected_roi_mode = "All"

        # File/folder dialog state for loading new data (iw-array API)
        self._file_dialog = None
        self._folder_dialog = None
        self._load_status_msg = ""
        self._load_status_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

        self.set_context_info()

        if threading_enabled:
            self.logger.info("Starting zstats computation in a separate thread.")
            threading.Thread(target=self.compute_zstats, daemon=True).start()

    def set_context_info(self):
        if self.fpath is None:
            title = "Test Data"
        elif isinstance(self.fpath, list):
            title = f"{[Path(f).stem for f in self.fpath]}"
        else:
            title = f"Filepath: {Path(self.fpath).stem}"
        self.image_widget.figure.canvas.set_title(str(title))

    def _refresh_image_widget(self):
        """
        Trigger a frame refresh on the ImageWidget.

        Uses the internal _set_slider_index method to force a display update
        without changing the actual index value.
        """
        # iw-array API: trigger refresh by re-setting the current t index
        # This forces the widget to re-render the current frame
        names = self.image_widget._slider_dim_names or ()
        if "t" in names:
            current_t = self.image_widget.indices["t"]
            # Use internal method to trigger update without full re-index
            if hasattr(self.image_widget, '_set_slider_index'):
                self.image_widget._set_slider_index(0, current_t)
            else:
                # Fallback: reassign the index to trigger update
                self.image_widget.indices["t"] = current_t

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
    def processors(self) -> list:
        """Access to underlying MboImageProcessor instances."""
        return self.image_widget._image_processors

    @property
    def current_offset(self) -> list[float]:
        """
        Get current phase offset from each processor or underlying array.

        For MboRawArray data, the offset is computed by the array itself.
        For other array types, the processor computes and caches it.
        """
        offsets = []
        for i, proc in enumerate(self.processors):
            # First check if processor has a cached offset
            if proc.current_offset != 0.0:
                offsets.append(proc.current_offset)
            # For MboRawArray, check the array's offset property
            elif hasattr(self.image_widget.data[i], 'offset'):
                arr_offset = self.image_widget.data[i].offset
                # offset can be a scalar or array
                if isinstance(arr_offset, np.ndarray):
                    offsets.append(float(arr_offset.mean()) if arr_offset.size > 0 else 0.0)
                else:
                    offsets.append(float(arr_offset) if arr_offset else 0.0)
            else:
                offsets.append(0.0)
        return offsets

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        return self.processors[0].fix_phase if self.processors else False

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.logger.info(f"Setting fix_phase to {value}.")
        for proc in self.processors:
            proc.fix_phase = value
        self._refresh_image_widget()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        return self.processors[0].use_fft if self.processors else False

    @use_fft.setter
    def use_fft(self, value: bool):
        self.logger.info(f"Setting use_fft to {value}.")
        for proc in self.processors:
            proc.use_fft = value
        self._refresh_image_widget()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        return self.processors[0].border if self.processors else 3

    @border.setter
    def border(self, value: int):
        self.logger.info(f"Setting border to {value}.")
        for proc in self.processors:
            proc.border = value
        self._refresh_image_widget()

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        return self.processors[0].max_offset if self.processors else 3

    @max_offset.setter
    def max_offset(self, value: int):
        self.logger.info(f"Setting max_offset to {value}.")
        for proc in self.processors:
            proc.max_offset = value
        self._refresh_image_widget()

    @property
    def selected_array(self) -> int:
        return self._selected_array

    @selected_array.setter
    def selected_array(self, value: int):
        if value < 0 or value >= self.num_graphics:
            raise ValueError(
                f"Invalid array index: {value}. "
                f"Must be between 0 and {self.num_graphics - 1}."
            )
        self._selected_array = value
        self.logger.info(f"Selected array index set to {value}.")

    @property
    def gaussian_sigma(self) -> float:
        """Sigma for Gaussian blur (0 = disabled). Uses fastplotlib spatial_func API."""
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        """Set gaussian blur using fastplotlib's spatial_func API."""
        self._gaussian_sigma = max(0.0, value)
        self.logger.info(f"Setting gaussian_sigma to {self._gaussian_sigma}.")

        if self._gaussian_sigma > 0:
            # Use partial to create a spatial_func with the current sigma
            spatial_func = partial(gaussian_filter, sigma=self._gaussian_sigma)
        else:
            # Use identity function instead of None to work around fastplotlib bug
            # (processor.spatial_func setter has incorrect None check)
            def _identity(x):
                return x
            spatial_func = _identity

        # Apply to all graphics via ImageWidget's spatial_func API
        self.image_widget.spatial_func = spatial_func
        self._refresh_image_widget()

    @property
    def proj(self) -> str:
        """Current projection mode (mean, max, std, mean-sub)."""
        return self._proj

    @proj.setter
    def proj(self, value: str):
        if value != self._proj:
            self.logger.info(f"Setting projection to {value}.")
            self._proj = value
            # For mean-sub, update processor mean_image when z-stats are ready
            self._update_mean_subtraction()
        self._refresh_image_widget()

    def _update_mean_subtraction(self):
        """Update processor mean_image based on current projection mode."""
        if self._proj == "mean-sub":
            # Set mean image on each processor from z-stats
            names = self.image_widget._slider_dim_names or ()
            z_idx = self.image_widget.indices["z"] if "z" in names else 0
            for i, proc in enumerate(self.processors):
                if self._zstats_done[i] and self._zstats_mean_scalar[i] is not None:
                    # Use the scalar mean for the current z-plane
                    proc.mean_image = self._zstats_mean_scalar[i][z_idx]
                else:
                    proc.mean_image = None
        else:
            # Clear mean image
            for proc in self.processors:
                proc.mean_image = None

    @property
    def window_size(self) -> int:
        """
        Window size for temporal projection.

        This sets the window size for the first slider dimension (typically 't').
        Uses fastplotlib's window_sizes API which expects a tuple per slider dim.
        """
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        if value < 1:
            self.logger.warning(f"Window size must be >= 1, got {value}. Setting to 1.")
            value = 1
        elif value < 4 and self.fix_phase:
            self.logger.warning(
                f"Window size ({value}) < 4 with phase correction enabled. "
                f"Phase correction requires >= 4 frames for reliable results. "
                f"Consider increasing window size or disabling phase correction."
            )

        self._window_size = value
        self.logger.info(f"Window size set to {value}.")

        # Use fastplotlib's window_sizes API
        # ImageWidget.window_sizes expects a list with one entry per processor
        # Each entry is a tuple with one value per slider dim
        # For 4D data (t, z, y, x) we have 2 slider dims: (t_window, z_window)
        # For 3D data (t, y, x) we have 1 slider dim: (t_window,)
        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1

        if n_slider_dims == 1:
            # Only temporal dimension
            per_processor_sizes = (value,)
        elif n_slider_dims == 2:
            # Temporal and z dimensions - only apply window to temporal (first dim)
            per_processor_sizes = (value, None)
        else:
            # More dimensions - apply to first, None for rest
            per_processor_sizes = (value,) + (None,) * (n_slider_dims - 1)

        # Create list with same window_sizes for each processor
        window_sizes = [per_processor_sizes] * len(self.processors)

        # Set via ImageWidget API (applies to all graphics)
        self.image_widget.window_sizes = window_sizes
        self._refresh_image_widget()

    @property
    def phase_upsample(self) -> int:
        """Upsampling factor for subpixel phase correlation."""
        return self.processors[0].phase_upsample if self.processors else 5

    @phase_upsample.setter
    def phase_upsample(self, value: int):
        self.logger.info(f"Setting phase_upsample to {value}.")
        for proc in self.processors:
            proc.phase_upsample = value
        self._refresh_image_widget()

    def update(self):
        # Check for file/folder dialog results (iw-array API)
        self._check_file_dialogs()
        draw_saveas_popup(self)
        draw_menu(self)
        draw_tabs(self)

    def _check_file_dialogs(self):
        """Check if file/folder dialogs have results and load data if so."""
        # Check file dialog
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result and len(result) > 0:
                self._load_new_data(result[0])
            self._file_dialog = None

        # Check folder dialog
        if self._folder_dialog is not None and self._folder_dialog.ready():
            result = self._folder_dialog.result()
            if result:
                self._load_new_data(result)
            self._folder_dialog = None

    def _load_new_data(self, path: str):
        """
        Load new data from the specified path using iw-array API.

        Uses iw.set_data() to swap data arrays, which handles shape changes.
        """
        from mbo_utilities.lazy_array import imread

        path_obj = Path(path)
        if not path_obj.exists():
            self.logger.error(f"Path does not exist: {path}")
            self._load_status_msg = f"Error: Path does not exist"
            self._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
            return

        try:
            self.logger.info(f"Loading data from: {path}")
            self._load_status_msg = "Loading..."
            self._load_status_color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)

            new_data = imread(path)

            # iw-array API: use data indexer for replacing data
            # iw.data[0] = new_array handles shape changes automatically
            self.image_widget.data[0] = new_data

            # Reset indices
            self.image_widget.indices["t"] = 0
            if new_data.ndim >= 4:
                self.image_widget.indices["z"] = 0

            # Update internal state
            self.fpath = path
            self.shape = new_data.shape
            self.is_mbo_scan = isinstance(new_data, MboRawArray)

            # Update nz for z-plane count
            if len(self.shape) == 4:
                self.nz = self.shape[1]
            elif len(self.shape) == 3:
                self.nz = 1
            else:
                self.nz = 1

            self._load_status_msg = f"Loaded: {path_obj.name}"
            self._load_status_color = imgui.ImVec4(0.3, 1.0, 0.3, 1.0)
            self.logger.info(f"Loaded successfully, shape: {new_data.shape}")
            self.set_context_info()

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self._load_status_msg = f"Error: {str(e)}"
            self._load_status_color = imgui.ImVec4(1.0, 0.3, 0.3, 1.0)

    def draw_stats_section(self):
        if not any(self._zstats_done):
            return

        stats_list = self._zstats
        is_single_zplane = self.nz == 1

        # Different title for single vs multi z-plane
        if is_single_zplane:
            imgui.text_colored(
                imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Signal Quality Summary"
            )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats"
            )

        imgui.spacing()

        # ROI selector
        array_labels = [
            f"{"graphic"} {i + 1}"
            for i in range(len(stats_list))
            if stats_list[i] and "mean" in stats_list[i]
        ]
        # Only show "Combined" if there are multiple arrays
        if len(array_labels) > 1:
            array_labels.append("Combined")

        # Ensure selected array is within bounds
        if self._selected_array >= len(array_labels):
            self._selected_array = 0

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

        # Check if "Combined" view is selected (only valid if there are multiple arrays)
        has_combined = len(array_labels) > 1 and array_labels[-1] == "Combined"
        is_combined_selected = has_combined and self._selected_array == len(array_labels) - 1

        if is_combined_selected:  # Combined
            imgui.text(f"Stats for Combined {"graphic"}s")
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

            z_vals = np.ascontiguousarray(
                np.arange(1, len(mean_vals) + 1, dtype=np.float64)
            )
            mean_vals = np.ascontiguousarray(mean_vals, dtype=np.float64)
            std_vals = np.ascontiguousarray(std_vals, dtype=np.float64)

            # For single z-plane, show simplified combined view
            if is_single_zplane:
                # Show just the single plane combined stats
                if imgui.begin_table(
                    f"Stats (averaged over {"graphic"}s)",
                    3,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    for col in ["Metric", "Value", "Unit"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )
                    imgui.table_headers_row()

                    metrics = [
                        ("Mean Fluorescence", mean_vals[0], "a.u."),
                        ("Std. Deviation", std_vals[0], "a.u."),
                        ("Signal-to-Noise", snr_vals[0], "ratio"),
                    ]

                    for metric_name, value, unit in metrics:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(metric_name)
                        imgui.table_next_column()
                        imgui.text(f"{value:.2f}")
                        imgui.table_next_column()
                        imgui.text(unit)
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text("Signal Quality Comparison")
                set_tooltip(
                    f"Comparison of mean fluorescence across all {"graphic"}s",
                    True,
                )

                # Get per-graphic mean values
                graphic_means = [
                    np.asarray(self._zstats[r]["mean"][0], float)
                    for r in range(self.num_graphics)
                    if self._zstats[r] and "mean" in self._zstats[r]
                ]

                plot_width = imgui.get_content_region_avail().x
                if graphic_means and implot.begin_plot(
                    "Signal Comparison", imgui.ImVec2(plot_width, 350)
                ):
                    style_seaborn_dark()
                    implot.setup_axes(
                        "Graphic",
                        "Mean Fluorescence (a.u.)",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )

                    x_pos = np.arange(len(graphic_means), dtype=np.float64)
                    heights = np.array(graphic_means, dtype=np.float64)

                    labels = [f"{i + 1}" for i in range(len(graphic_means))]
                    implot.setup_axis_limits(
                        implot.ImAxis_.x1.value, -0.5, len(graphic_means) - 0.5
                    )
                    implot.setup_axis_ticks_custom(
                        implot.ImAxis_.x1.value, x_pos, labels
                    )

                    implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                    implot.push_style_color(
                        implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                    )
                    implot.plot_bars(
                        "Graphic Signal",
                        x_pos,
                        heights,
                        0.6,
                    )
                    implot.pop_style_color()
                    implot.pop_style_var()

                    # Add mean line
                    mean_line = np.full_like(heights, mean_vals[0])
                    implot.push_style_var(implot.StyleVar_.line_weight.value, 2)
                    implot.push_style_color(
                        implot.Col_.line.value, (1.0, 0.4, 0.2, 0.8)
                    )
                    implot.plot_line("Average", x_pos, mean_line)
                    implot.pop_style_color()
                    implot.pop_style_var()

                    implot.end_plot()

            else:
                # Multi-z-plane: show original table and combined plot
                # Table
                if imgui.begin_table(
                    f"Stats, averaged over {"graphic"}s",
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
                        for val in (
                            z_vals[i],
                            mean_vals[i],
                            std_vals[i],
                            snr_vals[i],
                        ):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                imgui.text("Z-plane Signal: Combined")
                set_tooltip(
                    f"Gray = per-ROI z-profiles (mean over frames)."
                    f" Blue shade = across-ROI mean ± std; blue line = mean."
                    f" Hover gray lines for values.",
                    True,
                )

                # build per-graphic series
                graphic_series = [
                    np.asarray(self._zstats[r]["mean"], float)
                    for r in range(self.num_graphics)
                ]

                L = min(len(s) for s in graphic_series)
                z = np.asarray(z_vals[:L], float)
                graphic_series = [s[:L] for s in graphic_series]
                stack = np.vstack(graphic_series)
                mean_vals = stack.mean(axis=0)
                std_vals = stack.std(axis=0)
                lower = mean_vals - std_vals
                upper = mean_vals + std_vals

                # Use available width to prevent cutoff
                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    "Z-Plane Plot (Combined)", imgui.ImVec2(plot_width, 300)
                ):
                    style_seaborn_dark()
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )

                    implot.setup_axis_limits(
                        implot.ImAxis_.x1.value, float(z[0]), float(z[-1])
                    )
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")

                    for i, ys in enumerate(graphic_series):
                        label = f"ROI {i + 1}##roi{i}"
                        implot.push_style_var(implot.StyleVar_.line_weight.value, 1)
                        implot.push_style_color(
                            implot.Col_.line.value, (0.6, 0.6, 0.6, 0.35)
                        )
                        implot.plot_line(label, z, ys)
                        implot.pop_style_color()
                        implot.pop_style_var()

                    implot.push_style_color(
                        implot.Col_.fill.value, (0.2, 0.4, 0.8, 0.25)
                    )
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

            imgui.text(f"Stats for {"graphic"} {array_idx + 1}")

            # For single z-plane, show simplified table and visualization
            if is_single_zplane:
                # Show just the single plane stats in a nice format
                if imgui.begin_table(
                    f"stats{array_idx}",
                    3,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    for col in ["Metric", "Value", "Unit"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )
                    imgui.table_headers_row()

                    metrics = [
                        ("Mean Fluorescence", mean_vals[0], "a.u."),
                        ("Std. Deviation", std_vals[0], "a.u."),
                        ("Signal-to-Noise", snr_vals[0], "ratio"),
                    ]

                    for metric_name, value, unit in metrics:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(metric_name)
                        imgui.table_next_column()
                        imgui.text(f"{value:.2f}")
                        imgui.table_next_column()
                        imgui.text(unit)
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                style_seaborn_dark()
                imgui.text("Signal Quality Metrics")
                set_tooltip(
                    "Bar chart showing mean fluorescence, standard deviation, and SNR",
                    True,
                )

                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    f"Signal Metrics {array_idx}", imgui.ImVec2(plot_width, 350)
                ):
                    implot.setup_axes(
                        "Metric",
                        "Value (normalized)",
                        implot.AxisFlags_.none.value,
                        implot.AxisFlags_.auto_fit.value,
                    )

                    # Normalize values for better visualization
                    norm_mean = mean_vals[0]
                    norm_std = std_vals[0]
                    norm_snr = snr_vals[0] * (
                        norm_mean / max(snr_vals[0], 1.0)
                    )  # Scale SNR to be comparable

                    x_pos = np.array([0.0, 1.0, 2.0], dtype=np.float64)
                    heights = np.array(
                        [norm_mean, norm_std, norm_snr], dtype=np.float64
                    )

                    implot.setup_axis_limits(implot.ImAxis_.x1.value, -0.5, 2.5)
                    implot.setup_axis_ticks(
                        implot.ImAxis_.x1.value, x_pos, ["Mean", "Std Dev", "SNR"], False
                    )

                    implot.push_style_var(implot.StyleVar_.fill_alpha.value, 0.8)
                    implot.push_style_color(
                        implot.Col_.fill.value, (0.2, 0.6, 0.9, 0.8)
                    )
                    implot.plot_bars("Signal Metrics", x_pos, heights, 0.6)
                    implot.pop_style_color()
                    implot.pop_style_var()

                    implot.end_plot()

            else:
                # Multi-z-plane: show original table and line plot
                if imgui.begin_table(
                    f"zstats{array_idx}",
                    4,
                    imgui.TableFlags_.borders | imgui.TableFlags_.row_bg,
                ):
                    for col in ["Z", "Mean", "Std", "SNR"]:
                        imgui.table_setup_column(
                            col, imgui.TableColumnFlags_.width_stretch
                        )
                    imgui.table_headers_row()
                    for j in range(n):
                        imgui.table_next_row()
                        for val in (
                            int(z_vals[j]),
                            mean_vals[j],
                            std_vals[j],
                            snr_vals[j],
                        ):
                            imgui.table_next_column()
                            imgui.text(f"{val:.2f}")
                    imgui.end_table()

                imgui.spacing()
                imgui.separator()
                imgui.spacing()

                style_seaborn_dark()
                imgui.text("Z-plane Signal: Mean ± Std")
                plot_width = imgui.get_content_region_avail().x
                if implot.begin_plot(
                    f"Z-Plane Signal {array_idx}", imgui.ImVec2(plot_width, 300)
                ):
                    implot.setup_axes(
                        "Z-Plane",
                        "Mean Fluorescence",
                        implot.AxisFlags_.auto_fit.value,
                        implot.AxisFlags_.auto_fit.value,
                    )
                    implot.setup_axis_format(implot.ImAxis_.x1.value, "%g")
                    implot.plot_error_bars(
                        f"Mean ± Std {array_idx}", z_vals, mean_vals, std_vals
                    )
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
            disabled_label = (
                "mean-sub (pending)" if not all(self._zstats_done) else "mean-sub"
            )
            options.append(disabled_label)

            current_display_idx = options.index(
                self.proj if self._proj != "mean-sub" else disabled_label
            )

            imgui.set_next_item_width(hello_imgui.em_size(6))
            proj_changed, selected_display_idx = imgui.combo(
                "Projection", current_display_idx, options
            )
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
                    # The proj setter handles updating window_funcs and calling update_frame_apply
                    self.proj = selected_label

            # Window size for projections (temporal dimension)
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

            # Gaussian Filter - slider for fine control, +/- buttons for integer steps
            imgui.text("Gaussian Blur")

            # Slider for fine decimal control (0.0 to 5.0)
            imgui.set_next_item_width(hello_imgui.em_size(8))
            slider_changed, slider_val = imgui.slider_float(
                "##sigma_slider",
                self.gaussian_sigma,
                v_min=0.0,
                v_max=5.0,
                format="%.2f",
            )
            if slider_changed:
                self.gaussian_sigma = slider_val

            imgui.same_line()

            # +/- buttons for integer steps
            if imgui.button("-##sigma"):
                self.gaussian_sigma = max(0.0, self.gaussian_sigma - 1.0)
            imgui.same_line()
            if imgui.button("+##sigma"):
                self.gaussian_sigma = self.gaussian_sigma + 1.0

            imgui.same_line()

            # Input field for direct entry
            imgui.set_next_item_width(hello_imgui.em_size(4))
            input_changed, input_val = imgui.input_float(
                "##sigma_input",
                self.gaussian_sigma,
                step=0.1,
                step_fast=1.0,
                format="%.2f",
            )
            if input_changed:
                self.gaussian_sigma = max(0.0, input_val)

            set_tooltip(
                "Apply a Gaussian blur to the preview image. Sigma is in pixels; larger values yield stronger smoothing."
            )

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
            phase_changed, phase_value = imgui.checkbox("Fix Phase", self.fix_phase)
            set_tooltip(
                "Enable to apply scan-phase correction which shifts every other line/row of pixels "
                "to maximize correlation between these rows."
            )
            if phase_changed:
                self.fix_phase = phase_value

            imgui.set_next_item_width(hello_imgui.em_size(10))
            fft_changed, fft_value = imgui.checkbox("Sub-Pixel (slower)", self.use_fft)
            set_tooltip(
                "Use FFT-based sub-pixel registration (slower but more accurate)."
            )
            if fft_changed:
                self.use_fft = fft_value

            # Display current offsets - use self.current_offset which checks both
            # processor cache and MboRawArray.offset
            current_offsets = self.current_offset
            imgui.columns(2, "offsets", False)
            for i, ofs in enumerate(current_offsets):
                max_abs_offset = abs(ofs)

                imgui.text(f"graphic {i + 1}:")
                imgui.next_column()

                display_text = f"{ofs:.3f}"

                if max_abs_offset > self.max_offset:
                    imgui.push_style_color(
                        imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                    )
                    imgui.text(display_text)
                    imgui.pop_style_color()
                else:
                    imgui.text(display_text)

                imgui.next_column()
            imgui.columns(1)

            imgui.set_next_item_width(hello_imgui.em_size(5))
            upsample_changed, upsample_val = imgui.input_int(
                "Upsample", self.phase_upsample, step=1, step_fast=2
            )
            set_tooltip(
                "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment."
            )
            if upsample_changed:
                self.phase_upsample = max(1, upsample_val)

            imgui.set_next_item_width(hello_imgui.em_size(5))
            border_changed, border_val = imgui.input_int(
                "Exclude border-px", self.border, step=1, step_fast=2
            )
            set_tooltip(
                "Number of pixels to exclude from the edges of the image when computing the scan-phase offset."
            )
            if border_changed:
                self.border = max(0, border_val)

            imgui.set_next_item_width(hello_imgui.em_size(5))
            max_offset_changed, max_offset_val = imgui.input_int(
                "max-offset", self.max_offset, step=1, step_fast=2
            )
            set_tooltip(
                "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
            )
            if max_offset_changed:
                self.max_offset = max(1, max_offset_val)

            imgui.end_group()
            imgui.separator()

        imgui.separator()

        draw_zstats_progress(self)
        draw_register_z_progress(self)
        draw_saveas_progress(self)

    def get_raw_frame(self) -> tuple[ndarray, ...]:
        # iw-array API: use indices property for named dimension access
        idx = self.image_widget.indices
        names = self.image_widget._slider_dim_names or ()
        t = idx["t"] if "t" in names else 0
        z = idx["z"] if "z" in names else 0
        return tuple(ndim_to_frame(arr, t, z) for arr in self.image_widget.data)

    # NOTE: _compute_phase_offsets, update_frame_apply, and _combined_frame_apply
    # have been removed. Processing logic is now on MboImageProcessor.get()

    def _compute_zstats_single_roi(self, roi, fpath):
        arr = imread(fpath)
        if hasattr(arr, "fix_phase"):
            arr.fix_phase = False
        if hasattr(arr, "roi"):
            arr.roi = roi

        stats, means = {"mean": [], "std": [], "snr": []}, []
        self._tiff_lock = threading.Lock()
        for z in range(self.nz):
            with self._tiff_lock:
                stack = arr[::10, z].astype(np.float32)  # Z, Y, X
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

        for z in [0] if arr.ndim == 3 else range(self.nz):
            with self._tiff_lock:
                stack = (
                    arr[::10].astype(np.float32)
                    if arr.ndim == 3
                    else arr[::10, z].astype(np.float32)
                )

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

        # Compute z-stats for each graphic (array)
        for idx, arr in enumerate(self.image_widget.data, start=1):
            threading.Thread(
                target=self._compute_zstats_single_array,
                args=(idx, arr),
                daemon=True,
            ).start()
