"""
PreviewDataWidget - Main GUI widget for data preview and processing.

This module contains the PreviewDataWidget class which provides:
- Time series data visualization
- Z-stats signal quality analysis
- Suite2p pipeline integration
- File format conversion

The widget uses modular components:
- _menu_bar.py: Menu bar and status indicator
- _popups.py: Tool popups and process console
- _save_as.py: Save As dialog
- _keyboard.py: Keyboard shortcuts
- _dialogs.py: File dialog handling
- _stats.py: Z-stats computation and display
"""

import logging
import threading
from pathlib import Path
from typing import Literal
import os
import importlib.util
import time

# Force rendercanvas to use Qt backend if PyQt6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
if importlib.util.find_spec("PyQt6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PyQt6  # noqa: F401 - Must be imported before rendercanvas.qt can load

    # Fix suite2p PyQt6 compatibility - must happen before any suite2p GUI imports
    from PyQt6.QtWidgets import QSlider
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks

import imgui_bundle
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter

from imgui_bundle import imgui, hello_imgui, imgui_ctx, implot

from mbo_utilities.file_io import get_mbo_dirs
from mbo_utilities.reader import MBO_SUPPORTED_FTYPES
from mbo_utilities.arrays.features import PhaseCorrectionFeature
from mbo_utilities.preferences import get_last_dir
from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.gui.widgets.gui_logger import GuiLogger, GuiLogHandler
from mbo_utilities.gui.widgets.progress_bar import start_output_capture
from mbo_utilities.gui.widgets import get_supported_widgets, draw_all_widgets
from mbo_utilities import log

# Import modular components
from mbo_utilities.gui.widgets.menu_bar import draw_menu_bar, draw_keybinds_popup
from mbo_utilities.gui._popups import draw_tools_popups, draw_process_console_popup
from mbo_utilities.gui._save_as import draw_saveas_popup
from mbo_utilities.gui._keyboard import handle_keyboard_shortcuts
from mbo_utilities.gui._dialogs import check_file_dialogs
from mbo_utilities.gui._stats import compute_zstats, refresh_zstats, draw_stats_section
from mbo_utilities.gui._help_viewer import draw_help_popup

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
import contextlib

__all__ = ["PreviewDataWidget"]


class PreviewDataWidget(EdgeWindow):
    """
    Main GUI widget for data preview and processing.

    This widget provides:
    - Time series data visualization with temporal/spatial projections
    - Z-stats signal quality analysis
    - Suite2p pipeline integration
    - File format conversion (tiff, zarr, etc.)

    Parameters
    ----------
    iw : fastplotlib.ImageWidget
        The ImageWidget to attach to.
    fpath : str | list | None
        Path(s) to the data file(s).
    threading_enabled : bool
        Whether to compute z-stats in background threads.
    size : int | None
        Width of the widget panel.
    location : str
        Panel location ("right" or "bottom").
    """

    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None | list = None,
        threading_enabled: bool = True,
        size: int | None = None,
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

        # Initialize logging
        self._init_logging()

        # Initialize Suite2p settings
        self._init_suite2p()

        # Initialize ImPlot context
        if implot.get_current_context() is None:
            implot.create_context()

        # Setup ImGui fonts
        self._init_fonts()

        # Store kwargs and paths
        self.kwargs = kwargs
        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # Image widget setup
        self.image_widget = iw
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape

        # Determine data type (ScanImage or volumetric TIFF)
        from mbo_utilities.arrays import TiffArray
        self.is_mbo_scan = (
            isinstance(self.image_widget.data[0], ScanImageArray) or
            isinstance(self.image_widget.data[0], TiffArray)
        )
        self.logger.info(f"Data type: {type(self.image_widget.data[0]).__name__}, is_mbo_scan: {self.is_mbo_scan}")

        # Initialize state
        self._init_state()

        # Initialize z-stats tracking
        self._init_zstats()

        # Initialize save dialog state
        self._init_saveas_state()

        # Initialize viewer
        self._init_viewer()

        # Start z-stats computation
        if threading_enabled:
            self.logger.info("Starting zstats computation...")
            for i in range(self.num_graphics):
                self._zstats_running[i] = True
            threading.Thread(target=lambda: compute_zstats(self), daemon=True).start()

    def _init_logging(self):
        """Initialize logging system."""
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        gui_handler.setFormatter(logging.Formatter("%(message)s"))
        gui_handler.setLevel(logging.DEBUG)
        log.attach(gui_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))

        if bool(int(os.getenv("MBO_DEBUG", "0"))):
            console_handler.setLevel(logging.DEBUG)
            log.set_global_level(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
            log.set_global_level(logging.INFO)

        log.attach(console_handler)
        self.logger = log.get("gui")
        self.logger.info("Logger initialized.")
        start_output_capture()

    def _init_suite2p(self):
        """Initialize Suite2p settings and start background preload."""
        # start background preload of pipeline widgets (non-blocking)
        from mbo_utilities.gui.widgets.pipelines import start_preload
        start_preload()

        # defer Suite2pSettings creation until actually needed
        self._s2p = None  # lazy init
        self._s2p_dir = ""
        self._s2p_savepath_flash_start = None
        self._s2p_savepath_flash_count = 0
        self._s2p_show_savepath_popup = False
        self._s2p_folder_dialog = None

    @property
    def s2p(self):
        """Get Suite2pSettings (lazy init)."""
        if self._s2p is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pSettings
            self._s2p = Suite2pSettings()
        return self._s2p

    @s2p.setter
    def s2p(self, value):
        """Set Suite2pSettings."""
        self._s2p = value

    def _init_fonts(self):
        """Initialize ImGui fonts."""
        io = imgui.get_io()

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

    def _init_state(self):
        """Initialize widget state."""
        for subplot in self.image_widget.figure:
            subplot.toolbar = False
        self.image_widget._sliders_ui._loop = True

        # Determine nz
        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1
        else:
            self.nz = 1

        # Window/projection state
        self._window_size = 1
        self._gaussian_sigma = 0.0
        self._auto_update = False
        self._proj = "mean"
        self._mean_subtraction = False
        self._last_z_idx = 0

        # Registration state
        self._register_z = False
        self._register_z_progress = 0.0
        self._register_z_done = False
        self._register_z_running = False
        self._register_z_current_msg = ""

        # Selection state
        self._selected_pipelines = None
        self._selected_array = 0
        self._selected_planes = None
        self._planes_str = ""

        # Settings menu flags
        self.show_debug_panel = False
        self.show_scope_window = False
        self.show_metadata_viewer = False
        self.show_diagnostics_window = False
        self._diagnostics_widget = None
        self._show_progress_overlay = True

        # Process monitoring
        self._viewing_process_pid = None

        # File dialogs
        self._file_dialog = None
        self._folder_dialog = None
        self._load_status_msg = ""
        self._load_status_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

        # Initialize widgets
        self._widgets = get_supported_widgets(self)

    def _init_zstats(self):
        """Initialize z-stats tracking state."""
        self._zstats = [
            {"mean": [], "std": [], "snr": []} for _ in range(self.num_graphics)
        ]
        self._zstats_means = [None] * self.num_graphics
        self._zstats_mean_scalar = [0.0] * self.num_graphics
        self._zstats_done = [False] * self.num_graphics
        self._zstats_running = [False] * self.num_graphics
        self._zstats_progress = [0.0] * self.num_graphics
        self._zstats_current_z = [0] * self.num_graphics

    def _init_saveas_state(self):
        """Initialize save-as dialog state."""
        self._ext = ".tiff"
        self._ext_idx = MBO_SUPPORTED_FTYPES.index(".tiff")
        self._overwrite = True
        self._debug = False
        self._saveas_chunk_mb = 100

        # Zarr options
        self._zarr_sharded = True
        self._zarr_ome = True
        self._zarr_compression_level = 1
        self._zarr_pyramid = False
        self._zarr_pyramid_max_layers = 4
        self._zarr_pyramid_method = "mean"

        # Save dialog state
        self._saveas_popup_open = False
        self._saveas_done = False
        self._saveas_running = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0

        # Directories
        save_as_dir = get_last_dir("save_as")
        self._saveas_outdir = str(save_as_dir) if save_as_dir else ""
        s2p_output_dir = get_last_dir("suite2p_output")
        self._s2p_outdir = str(s2p_output_dir) if s2p_output_dir else ""
        self._saveas_folder_dialog = None
        self._saveas_total = 0

        # ROI selection
        self._saveas_selected_roi = set()
        self._saveas_rois = False
        self._saveas_selected_roi_mode = "All"

        # Metadata
        self._saveas_custom_metadata = {}
        self._saveas_custom_key = ""
        self._saveas_custom_value = ""

        # Output suffix
        self._saveas_output_suffix = ""

        # Options
        self._saveas_background = True

        # Scan-phase correction for save/export (separate from display settings)
        # defaults to True for save operations
        self._saveas_fix_phase = True
        self._saveas_use_fft = True

    def _init_viewer(self):
        """Initialize the viewer based on data type."""
        from mbo_utilities.gui.viewers import get_viewer_class
        viewer_cls = get_viewer_class(self.image_widget.data[0])
        self._viewer = viewer_cls(self.image_widget, self.fpath, parent=self)
        self.logger.info(f"Viewer: {self._viewer.name}")
        self._main_widget = self._viewer._main_widget
        self.set_context_info()

    def set_context_info(self):
        """Update app title with dataset name."""
        try:
            if self.fpath is None:
                return
            name = Path(self.fpath[0]).parent.name if isinstance(self.fpath, list) else Path(self.fpath).name
            hello_imgui.get_runner_params().app_shallow_settings.window_title = f"MBO Utilities - {name}"
        except (RuntimeError, TypeError):
            pass

    # === Properties ===

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
        """Access to underlying NDImageProcessor instances."""
        return self.image_widget._image_processors

    def _get_data_arrays(self) -> list:
        """Get underlying data arrays from image processors."""
        return [proc.data for proc in self.processors]

    @property
    def current_offset(self) -> list[float]:
        """Get current phase offset from each data array."""
        offsets = []
        for arr in self._get_data_arrays():
            if hasattr(arr, "offset"):
                arr_offset = arr.offset
                if isinstance(arr_offset, np.ndarray):
                    offsets.append(float(arr_offset.mean()) if arr_offset.size > 0 else 0.0)
                else:
                    offsets.append(float(arr_offset) if arr_offset else 0.0)
            else:
                offsets.append(0.0)
        return offsets

    @property
    def has_raster_scan_support(self) -> bool:
        """Check if any data array supports raster scan phase correction."""
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                return True
            if hasattr(arr, "fix_phase") and hasattr(arr, "use_fft"):
                return True
        return False

    @property
    def has_frame_averaging_support(self) -> bool:
        """Check if any data array supports frame averaging."""
        for arr in self._get_data_arrays():
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                return True
        return False

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.enabled
        return getattr(arr, "fix_phase", False)

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.logger.info(f"Setting fix_phase to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.enabled = value
            elif hasattr(arr, "fix_phase"):
                arr.fix_phase = value
        self._refresh_image_widget()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.use_fft
        return getattr(arr, "use_fft", False)

    @use_fft.setter
    def use_fft(self, value: bool):
        self.logger.info(f"Setting use_fft to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.use_fft = value
            elif hasattr(arr, "use_fft"):
                arr.use_fft = value
        self._refresh_image_widget()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        arrays = self._get_data_arrays()
        if not arrays:
            return 3
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.border
        return getattr(arr, "border", 3)

    @border.setter
    def border(self, value: int):
        self.logger.info(f"Setting border to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.border = value
            elif hasattr(arr, "border"):
                arr.border = value
        self._refresh_image_widget()

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], "max_offset", 3) if arrays else 3

    @max_offset.setter
    def max_offset(self, value: int):
        self.logger.info(f"Setting max_offset to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "max_offset"):
                arr.max_offset = value
        self._refresh_image_widget()

    @property
    def selected_array(self) -> int:
        return self._selected_array

    @selected_array.setter
    def selected_array(self, value: int):
        if value < 0 or value >= self.num_graphics:
            raise ValueError(f"Invalid array index: {value}")
        self._selected_array = value

    @property
    def gaussian_sigma(self) -> float:
        """Sigma for Gaussian blur (0 = disabled)."""
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        self._gaussian_sigma = max(0.0, value)
        self._rebuild_spatial_func()
        self._refresh_image_widget()
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

    @property
    def proj(self) -> str:
        """Current projection mode (mean, max, std)."""
        return self._proj

    @proj.setter
    def proj(self, value: str):
        if value != self._proj:
            self._proj = value
            self._update_window_funcs()

    @property
    def mean_subtraction(self) -> bool:
        """Whether mean subtraction is enabled."""
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if value != self._mean_subtraction:
            self._mean_subtraction = value
            self._update_mean_subtraction()

    @property
    def window_size(self) -> int:
        """Window size for temporal projection."""
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = value
        self.logger.info(f"Window size set to {value}.")
        if not self.processors:
            return
        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1
        per_processor_sizes = (self._window_size,) + (None,) * (n_slider_dims - 1)
        self._set_processor_attr("window_sizes", per_processor_sizes)
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

    @property
    def phase_upsample(self) -> int:
        """Upsampling factor for subpixel phase correlation."""
        if not self.has_raster_scan_support:
            return 5
        arrays = self._get_data_arrays()
        return getattr(arrays[0], "upsample", 5) if arrays else 5

    @phase_upsample.setter
    def phase_upsample(self, value: int):
        if not self.has_raster_scan_support:
            return
        self.logger.info(f"Setting phase_upsample to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "upsample"):
                arr.upsample = value
        self._refresh_image_widget()

    # === Internal methods ===

    def _refresh_image_widget(self):
        """Trigger a frame refresh on the ImageWidget."""
        current_indices = list(self.image_widget.indices)
        self.image_widget.indices = current_indices

    def _set_processor_attr(self, attr: str, value):
        """Set processor attribute without expensive histogram recomputation."""
        if not self.processors:
            return

        original_states = []
        for proc in self.processors:
            original_states.append(proc._compute_histogram)
            proc._compute_histogram = False

        try:
            if attr == "window_funcs":
                if isinstance(value, tuple):
                    value = [value] * len(self.processors)
                self.image_widget.window_funcs = value
            elif attr == "window_sizes":
                if isinstance(value, (tuple, list)) and not isinstance(value[0], (tuple, list, type(None))):
                    value = [value] * len(self.processors)
                self.image_widget.window_sizes = value
            elif attr == "spatial_func":
                self.image_widget.spatial_func = value
            else:
                for proc in self.processors:
                    setattr(proc, attr, value)
                self._refresh_image_widget()
        except Exception as e:
            self.logger.exception(f"Error setting {attr}: {e}")
        finally:
            for proc, orig in zip(self.processors, original_states, strict=False):
                proc._compute_histogram = orig

        with contextlib.suppress(Exception):
            self.image_widget.reset_vmin_vmax_frame()

    def _refresh_widgets(self):
        """Refresh widgets based on current data capabilities."""
        self._widgets = get_supported_widgets(self)

    def _update_mean_subtraction(self):
        """Update spatial_func to apply mean subtraction."""
        self._rebuild_spatial_func()
        self._refresh_image_widget()
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

    def _rebuild_spatial_func(self):
        """Rebuild and apply the combined spatial function."""
        names = self.image_widget._slider_dim_names or ()
        try:
            z_idx = self.image_widget.indices["z"] if "z" in names else 0
        except (IndexError, KeyError):
            z_idx = 0

        sigma = self.gaussian_sigma if self.gaussian_sigma > 0 else None

        any_mean_sub = self._mean_subtraction and any(
            self._zstats_done[i] and self._zstats_means[i] is not None
            for i in range(self.num_graphics)
        )

        if not any_mean_sub and sigma is None:
            def identity(frame):
                return frame
            self.image_widget.spatial_func = identity
            return

        spatial_funcs = []
        for i in range(self.num_graphics):
            mean_img = None
            if self._mean_subtraction and self._zstats_done[i] and self._zstats_means[i] is not None:
                mean_img = self._zstats_means[i][z_idx].astype(np.float32)

            spatial_funcs.append(self._make_spatial_func(mean_img, sigma))

        self.image_widget.spatial_func = spatial_funcs

    def _make_spatial_func(self, mean_img: np.ndarray | None, sigma: float | None):
        """Create a spatial function that applies mean subtraction and/or gaussian blur."""
        def spatial_func(frame):
            result = frame
            if mean_img is not None:
                result = result.astype(np.float32) - mean_img
            if sigma is not None and sigma > 0:
                result = gaussian_filter(result, sigma=sigma)
            return result
        return spatial_func

    def _update_window_funcs(self):
        """Update window_funcs on image widget based on current projection mode."""
        if not self.processors:
            return

        def mean_wrapper(data, axis, keepdims):
            return np.mean(data, axis=axis, keepdims=keepdims)

        def max_wrapper(data, axis, keepdims):
            return np.max(data, axis=axis, keepdims=keepdims)

        def std_wrapper(data, axis, keepdims):
            return np.std(data, axis=axis, keepdims=keepdims)

        proj_funcs = {"mean": mean_wrapper, "max": max_wrapper, "std": std_wrapper}
        proj_func = proj_funcs.get(self._proj, mean_wrapper)

        n_slider_dims = self.processors[0].n_slider_dims if self.processors else 1

        if n_slider_dims == 1:
            window_funcs = (proj_func,)
        elif n_slider_dims == 2:
            window_funcs = (proj_func, None)
        else:
            window_funcs = (proj_func,) + (None,) * (n_slider_dims - 1)

        self._set_processor_attr("window_funcs", window_funcs)
        if self.image_widget:
            self.image_widget.reset_vmin_vmax_frame()

    def gui_progress_callback(self, frac, meta=None):
        """Handle progress callbacks from save operations."""
        if isinstance(meta, (int, np.integer)):
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0
            if frac >= 1.0:
                self._saveas_running = False
                self._saveas_complete_time = time.time()
                self.logger.info("Save complete")
        elif isinstance(meta, str):
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0
            if frac >= 1.0:
                self._register_z_running = False
                self._register_z_complete_time = time.time()

    def _clear_stale_progress(self):
        """Clear completed progress indicators after a delay."""
        now = time.time()
        clear_delay = 5.0

        if getattr(self, "_saveas_done", False):
            complete_time = getattr(self, "_saveas_complete_time", 0)
            if now - complete_time > clear_delay:
                self._saveas_done = False
                self._saveas_progress = 0.0

        if getattr(self, "_register_z_done", False):
            complete_time = getattr(self, "_register_z_complete_time", 0)
            if now - complete_time > clear_delay:
                self._register_z_done = False
                self._register_z_progress = 0.0
                self._register_z_current_msg = None

    # === Rendering ===

    def draw_window(self):
        """Override parent to handle keyboard shortcuts and global popups."""
        handle_keyboard_shortcuts(self)
        check_file_dialogs(self)

        # Draw independent floating windows
        draw_tools_popups(self)
        draw_saveas_popup(self)
        draw_process_console_popup(self)
        draw_keybinds_popup(self)
        draw_help_popup(self)

        super().draw_window()

    def update(self):
        """Main render callback."""
        import time
        t0 = time.perf_counter()
        draw_menu_bar(self)
        t1 = time.perf_counter()
        self._viewer.draw()
        t2 = time.perf_counter()
        menu_ms = (t1 - t0) * 1000
        draw_ms = (t2 - t1) * 1000
        if menu_ms > 50 or draw_ms > 50:
            print(f"SLOW FRAME: menu={menu_ms:.1f}ms, draw={draw_ms:.1f}ms")

        # Update mean subtraction when z-plane changes
        names = self.image_widget._slider_dim_names or ()
        try:
            z_idx = self.image_widget.indices["z"] if "z" in names else 0
        except (IndexError, KeyError):
            z_idx = 0

        if z_idx != self._last_z_idx:
            self._last_z_idx = z_idx
            if self._mean_subtraction:
                self._update_mean_subtraction()
            elif self.image_widget:
                self.image_widget.reset_vmin_vmax_frame()

    def draw_stats_section(self):
        """Draw z-stats visualization."""
        draw_stats_section(self)

    def draw_preview_section(self):
        """Draw preview section using modular UI widgets."""
        imgui.dummy(imgui.ImVec2(0, 5))
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize
        with imgui_ctx.begin_child("##PreviewChild", imgui.ImVec2(0, 0), cflags):
            draw_all_widgets(self, self._widgets)

    def get_raw_frame(self) -> tuple[ndarray, ...]:
        """Get raw frame data at current indices."""
        idx = self.image_widget.indices
        names = self.image_widget._slider_dim_names or ()
        t = idx["t"] if "t" in names else 0
        z = idx["z"] if "z" in names else 0

        def _ndim_to_frame(arr, t=0, z=0):
            if arr.ndim == 4:
                return arr[t, z]
            if arr.ndim == 3:
                return arr[t]
            if arr.ndim == 2:
                return arr
            raise ValueError(f"Unsupported data shape: {arr.shape}")

        return tuple(_ndim_to_frame(arr, t, z) for arr in self.image_widget.data)

    def compute_zstats(self):
        """Compute z-stats for all graphics."""
        compute_zstats(self)

    def refresh_zstats(self):
        """Reset and recompute z-stats for all arrays."""
        refresh_zstats(self)

    def cleanup(self):
        """Clean up resources when the GUI is closing."""
        from mbo_utilities.gui.widgets.pipelines import cleanup_pipelines
        from mbo_utilities.gui.widgets import cleanup_all_widgets

        cleanup_pipelines(self)
        cleanup_all_widgets(self._widgets)

        self._file_dialog = None
        self._folder_dialog = None
        if hasattr(self, "_s2p_folder_dialog"):
            self._s2p_folder_dialog = None

        self.logger.info("GUI cleanup complete")
