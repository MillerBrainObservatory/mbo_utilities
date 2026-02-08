import pathlib
import threading
from pathlib import Path
import time
from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd, hello_imgui

from mbo_utilities.gui._imgui_helpers import set_tooltip, settings_row_with_popup, _popup_states
from mbo_utilities.gui._selection_ui import draw_selection_table
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities._parsing import _convert_paths_to_strings

# lazy availability check - avoid heavy import at module load
_HAS_LSP: bool | None = None


def _check_lsp_available() -> bool:
    """check if lbm_suite2p_python is available (lazy, cached)."""
    global _HAS_LSP
    if _HAS_LSP is None:
        import importlib.util
        _HAS_LSP = importlib.util.find_spec("lbm_suite2p_python") is not None
    return _HAS_LSP


USER_PIPELINES = ["suite2p"]


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

    if readonly:
        imgui.text(f"  [{'x' if settings.do_registration else ' '}] do_registration")
    else:
        _, settings.do_registration = imgui.checkbox(
            "do_registration##panel", settings.do_registration
        )
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

    if readonly:
        imgui.text(f"  [{'x' if settings.roidetect else ' '}] roidetect")
    else:
        _, settings.roidetect = imgui.checkbox("roidetect##panel", settings.roidetect)
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Run cell detection")

    if readonly:
        imgui.text(f"  [{'x' if settings.sparse_mode else ' '}] sparse_mode")
    else:
        _, settings.sparse_mode = imgui.checkbox(
            "sparse_mode##panel", settings.sparse_mode
        )
    imgui.same_line(hello_imgui.em_size(20))
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Sparse detection (faster)")

    imgui.text("  diameter")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.diameter}")
    else:
        _, settings.diameter = imgui.input_int("##diam_panel", settings.diameter)
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Cell diameter (px)")

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

    imgui.text("  neucoeff")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.neucoeff:.2f}")
    else:
        _, settings.neucoeff = imgui.input_float(
            "##neuc_panel", settings.neucoeff, 0.05, 0.1, "%.2f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Neuropil coefficient")

    if readonly:
        imgui.text(f"  [{'x' if settings.spikedetect else ' '}] spikedetect")
    else:
        _, settings.spikedetect = imgui.checkbox(
            "spikedetect##panel", settings.spikedetect
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
    Suite2p pipeline configuration settings.
    Organized by functional sections matching Suite2p documentation.
    Defaults are optimized for LBM datasets based on LBM-Suite2p-Python.
    """

    # main settings
    tau: float = 1.3  # Timescale of sensor (LBM default for GCaMP6m-like)
    frames_include: int = -1
    target_timepoints: int = -1

    # processing control
    keep_raw: bool = False  # keep raw binary (data_raw.bin) after processing
    keep_reg: bool = True  # Keep registered binary (data.bin) after processing
    force_reg: bool = False  # Force re-registration even if already done
    force_detect: bool = False  # Force ROI detection even if stat.npy exists
    dff_window_size: int = 300  # Frames for rolling percentile baseline in ΔF/F
    dff_percentile: int = 20  # Percentile for baseline F₀ estimation
    dff_smooth_window: int = 0  # Smooth ΔF/F trace (0 = disabled)

    # output settings
    preclassify: float = 0.0  # apply classifier before extraction (0.0 = keep all)
    save_nwb: bool = False
    save_mat: bool = False  # Save results in Fall.mat
    combined: bool = True  # Combine results across planes
    aspect: float = 1.0  # Ratio of um/pixels X to Y (for GUI only)
    report_time: bool = True  # Return timing dictionary

    # registration settings
    do_registration: bool = True  # whether to run registration
    align_by_chan: int = 1  # Channel to use for alignment (1-based)
    nimg_init: int = 300  # Frames to compute reference image
    batch_size: int = 500  # Frames to register simultaneously
    maxregshift: float = 0.1  # Max shift as fraction of frame size
    smooth_sigma: float = 1.15  # Gaussian stddev for phase correlation (>4 for 1P)
    smooth_sigma_time: float = 0.0  # Gaussian stddev in time frames
    keep_movie_raw: bool = False  # Keep non-registered binary
    two_step_registration: bool = False  # Run registration twice (low SNR)
    reg_tif: bool = False  # Write registered binary to tiff
    reg_tif_chan2: bool = False  # Write registered chan2 to tiff
    subpixel: int = 10  # Precision of subpixel registration (1/subpixel steps)
    th_badframes: float = 1.0  # Threshold for excluding frames
    norm_frames: bool = True  # Normalize frames when detecting shifts
    force_refImg: bool = False  # Use refImg stored in ops
    pad_fft: bool = False  # Pad image during FFT registration

    # 1P registration
    do_1Preg: bool = False  # perform 1P-specific registration
    spatial_hp_reg: int = 42  # Window for spatial high-pass filtering (1P)
    pre_smooth: float = 0.0  # Gaussian smoothing before high-pass (1P)
    spatial_taper: float = 40.0  # Pixels to ignore on edges (1P)

    # non-rigid registration
    nonrigid: bool = True  # perform non-rigid registration
    block_size: list = (
        None  # Block size for non-rigid (default [128, 128], power of 2/3)
    )
    snr_thresh: float = 1.2  # Phase correlation peak threshold (1.5 for 1P)
    maxregshiftNR: float = 5.0  # Max block shift relative to rigid shift

    # roi detection settings (functional)
    functional_chan: int = 1  # channel for functional ROI extraction (1-based)
    roidetect: bool = True  # run ROI detection and extraction
    sparse_mode: bool = True  # Use sparse_mode cell detection
    spatial_scale: int = 1  # Optimal recording scale (1=6-pixel cells, LBM default)
    connected: bool = True  # Require ROIs to be fully connected
    threshold_scaling: float = 1.0  # Detection threshold (higher=fewer ROIs)
    spatial_hp_detect: int = 25  # High-pass window for neuropil subtraction
    max_overlap: float = 0.75  # Max overlap fraction before discarding ROI
    high_pass: int = 100  # Running mean subtraction window (<10 for 1P)
    smooth_masks: bool = True  # Smooth masks in final detection pass
    max_iterations: int = 20  # Max iterations for cell extraction
    nbinned: int = 5000  # Max binned frames for ROI detection
    denoise: bool = False  # Denoise binned movie (requires sparse_mode)

    # cellpose detection settings (lbm-optimized defaults)
    anatomical_only: int = 3  # use enhanced mean image (lbm default)
    diameter: int = 4  # Expected cell diameter in pixels (LBM datasets)
    cellprob_threshold: float = -6.0  # More permissive detection threshold
    flow_threshold: float = 0.0  # Standard Cellpose flow threshold
    spatial_hp_cp: float = 0  # High-pass filtering strength for Cellpose

    # signal extraction settings
    neuropil_extract: bool = True  # extract neuropil signal
    allow_overlap: bool = False  # Extract from overlapping pixels
    min_neuropil_pixels: int = 350  # Min pixels for neuropil computation
    inner_neuropil_radius: int = 2  # Pixels between ROI and neuropil
    lam_percentile: int = 50  # Lambda percentile for neuropil exclusion

    # spike deconvolution settings
    spikedetect: bool = True  # run spike deconvolution
    neucoeff: float = 0.7  # Neuropil coefficient for all ROIs
    baseline: str = "maximin"  # Baseline method (maximin/constant/constant_percentile)
    win_baseline: float = 60.0  # Window for maximin filter (seconds)
    sig_baseline: float = 10.0  # Gaussian filter width (seconds)
    prctile_baseline: float = 8.0  # Percentile for constant_percentile baseline

    # classification settings
    soma_crop: bool = True  # crop dendrites for classification stats
    use_builtin_classifier: bool = False  # Use built-in classifier
    classifier_path: str = ""  # Path to custom classifier

    # channel 2 settings
    chan2_file: str = ""  # path to channel 2 data file
    chan2_thres: float = 0.65  # Threshold for ROI detection on channel 2

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.block_size is None:
            self.block_size = [128, 128]

    def to_dict(self):
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()  # type: ignore # noqa
        }

    def to_file(self, filepath):
        """Save settings to a .npy file."""
        # Convert Path objects to strings for cross-platform compatibility
        np.save(filepath, _convert_paths_to_strings(self.to_dict()), allow_pickle=True)


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
            if hasattr(data, "num_color_channels"):
                num_channels = data.num_color_channels
            elif hasattr(data, "_num_color_channels"):
                num_channels = data._num_color_channels
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

    # ensure _selected_planes is initialized (used by run_process)
    if not hasattr(self, "_selected_planes"):
        self._selected_planes = set(range(1, num_planes + 1))

    # parse timepoint selection if needed
    if self._s2p_tp_parsed is None and not self._s2p_tp_error:
        try:
            self._s2p_tp_parsed = parse_timepoint_selection(self._s2p_tp_selection, max_frames)
        except ValueError as e:
            self._s2p_tp_error = str(e)

    return max_frames, num_planes, num_channels


def _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels=1):
    """Draw compact selection preview with frame/plane/channel counts."""
    # calculate counts
    if self._s2p_tp_parsed:
        n_frames = self._s2p_tp_parsed.count
    else:
        n_frames = max_frames

    n_planes = len(range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step))
    n_channels = len(range(self._s2p_c_start, self._s2p_c_stop + 1, self._s2p_c_step))

    # compact preview line
    parts = [f"{n_frames} frames"]
    if num_channels > 1:
        parts.append(f"{n_channels} channels")
    if num_planes > 1:
        parts.append(f"{n_planes} planes")
    imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), ", ".join(parts))


def _draw_s2p_selection_popup(self):
    """Draw selection popup with output path and timepoint/z-plane selection."""
    from mbo_utilities.arrays.features._slicing import parse_timepoint_selection

    if getattr(self, "_s2p_selection_open", False):
        imgui.open_popup("Selection##s2p")
        self._s2p_selection_open = False

    imgui.set_next_window_size(imgui.ImVec2(480, 0), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Selection##s2p"):
        max_frames = getattr(self, "_s2p_last_max_tp", 1000)
        num_planes = getattr(self, "_s2p_last_num_planes", 1)
        num_channels = getattr(self, "_s2p_last_num_channels", 1)

        # === OUTPUT PATH ===
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Output")
        imgui.dummy(imgui.ImVec2(0, 2))

        s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")

        imgui.text("Path:")
        imgui.same_line()
        display_path = s2p_path if s2p_path else "(not set)"
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x - 80)
        if s2p_path:
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), display_path)
        else:
            imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.4, 1.0), display_path)
        imgui.pop_text_wrap_pos()

        imgui.same_line()
        if imgui.button("Browse##s2p_sel"):
            default_dir = s2p_path or str(get_last_dir("suite2p_output") or pathlib.Path().home())
            self._s2p_folder_dialog = pfd.select_folder("Select Suite2p output folder", default_dir)

        # check async folder dialog
        if self._s2p_folder_dialog is not None and self._s2p_folder_dialog.ready():
            result = self._s2p_folder_dialog.result()
            if result:
                self._s2p_outdir = str(result)
                set_last_dir("suite2p_output", result)
            self._s2p_folder_dialog = None

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # === SLICING SECTION ===
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Slicing")
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
        imgui.dummy(imgui.ImVec2(0, 3))

        # draw selection table using shared component
        tp_parsed, z_start, z_stop, z_step, c_start, c_stop, c_step = draw_selection_table(
            self,
            max_frames,
            num_planes,
            tp_attr="_s2p_tp",
            z_attr="_s2p_z",
            id_suffix="_s2p",
            num_channels=num_channels,
            c_attr="_s2p_c",
        )

        # update _selected_planes for run_process
        if num_planes > 1:
            self._selected_planes = set(range(z_start, z_stop + 1, z_step))
        else:
            self._selected_planes = {1}

        # update s2p.target_timepoints from selection
        if tp_parsed:
            self.s2p.target_timepoints = tp_parsed.count

        imgui.spacing()
        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _draw_s2p_settings_popup(self):
    """Draw settings popup with main Suite2p parameters."""
    INPUT_WIDTH = 120

    if getattr(self, "_s2p_settings_open", False):
        imgui.open_popup("Settings##s2p_main")
        self._s2p_settings_open = False

    imgui.set_next_window_size(imgui.ImVec2(350, 0), imgui.Cond_.first_use_ever)
    if imgui.begin_popup("Settings##s2p_main"):
        # tau
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
        set_tooltip(
            "Calcium indicator decay timescale in seconds.\n"
            "GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
        )

        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip("Denoise binned movie before cell detection.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # processing control options
        _, self.s2p.keep_raw = imgui.checkbox("Keep Raw Binary", self.s2p.keep_raw)
        set_tooltip("Keep data_raw.bin after processing (uses disk space)")

        _, self.s2p.keep_reg = imgui.checkbox("Keep Registered Binary", self.s2p.keep_reg)
        set_tooltip("Keep data.bin after processing (useful for QC)")

        _, self.s2p.force_reg = imgui.checkbox("Force Re-registration", self.s2p.force_reg)
        set_tooltip("Force re-registration even if already processed")

        _, self.s2p.force_detect = imgui.checkbox("Force Re-detection", self.s2p.force_detect)
        set_tooltip("Force ROI detection even if stat.npy exists")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # dF/F settings
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "ΔF/F")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.dff_window_size = imgui.input_int("Window", self.s2p.dff_window_size)
        set_tooltip("Frames for rolling percentile baseline (default: 300)")

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.dff_percentile = imgui.input_int("Percentile", self.s2p.dff_percentile)
        set_tooltip("Percentile for baseline F₀ estimation (default: 20)")

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.dff_smooth_window = imgui.input_int("Smooth", self.s2p.dff_smooth_window)
        set_tooltip("Smooth ΔF/F trace with rolling window (0 = disabled)")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # processing options
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "Processing")

        if not hasattr(self, "_s2p_background"):
            self._s2p_background = True
        _, self._s2p_background = imgui.checkbox("Run in background", self._s2p_background)
        set_tooltip("Run as separate process that continues after closing GUI")

        if not hasattr(self, "_parallel_processing"):
            self._parallel_processing = False
        if not hasattr(self, "_max_parallel_jobs"):
            self._max_parallel_jobs = 2

        # get num_planes from data
        num_planes = 1
        n_selected = 1
        try:
            if hasattr(self, "image_widget") and self.image_widget.data:
                data = self.image_widget.data[0]
                if data.ndim == 4:
                    num_planes = data.shape[1]
            n_selected = len(getattr(self, "_selected_planes", {1}))
        except Exception:
            pass

        if num_planes > 1 and n_selected > 1:
            _, self._parallel_processing = imgui.checkbox("Parallel plane processing", self._parallel_processing)
            set_tooltip("Process multiple planes simultaneously (uses more memory)")

            if self._parallel_processing:
                imgui.set_next_item_width(INPUT_WIDTH)
                _, self._max_parallel_jobs = imgui.input_int("Max parallel jobs", self._max_parallel_jobs, step=1)
                self._max_parallel_jobs = max(1, min(self._max_parallel_jobs, n_selected))

        imgui.spacing()
        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


# def _load_suite2p_results(self):
#     """Load Suite2p results (stat.npy) from a folder."""
#     s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")
#     default_dir = s2p_path or str(get_last_dir("suite2p_output") or pathlib.Path().home())
#
#     # open folder dialog
#     self._s2p_load_dialog = pfd.select_folder("Select Suite2p results folder", default_dir)
#     self._s2p_load_type = "suite2p"
#
#
# def _load_cellpose_results(self):
#     """Load Cellpose masks from a file."""
#     s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")
#     default_dir = s2p_path or str(get_last_dir("cellpose_masks") or pathlib.Path().home())
#
#     # open file dialog
#     self._s2p_load_dialog = pfd.open_file(
#         "Select Cellpose masks file",
#         default_dir,
#         ["*.npy", "*.png", "*_seg.npy"],
#     )
#     self._s2p_load_type = "cellpose"
#
#
# def _check_load_results_dialog(self):
#     """Check if load results dialog has finished and process results."""
#     if not hasattr(self, "_s2p_load_dialog") or self._s2p_load_dialog is None:
#         return
#
#     if not self._s2p_load_dialog.ready():
#         return
#
#     result = self._s2p_load_dialog.result()
#     self._s2p_load_dialog = None
#
#     if not result:
#         return
#
#     load_type = getattr(self, "_s2p_load_type", "suite2p")
#
#     if load_type == "suite2p":
#         _process_suite2p_load(self, result)
#     elif load_type == "cellpose":
#         # result is a list of files
#         if result:
#             _process_cellpose_load(self, result[0])
#
#
# def _process_suite2p_load(self, folder_path: str):
#     """Process loaded Suite2p results folder."""
#     import numpy as np
#     from pathlib import Path
#
#     folder = Path(folder_path)
#     try:
#         # find stat.npy
#         stat_files = list(folder.rglob("stat.npy"))
#         if not stat_files:
#             self.logger.error(f"No stat.npy found in {folder}")
#             return
#
#         stat_path = stat_files[0]
#         ops_path = stat_path.parent / "ops.npy"
#         iscell_path = stat_path.parent / "iscell.npy"
#
#         stat = np.load(stat_path, allow_pickle=True)
#         ops = np.load(ops_path, allow_pickle=True).item() if ops_path.exists() else {}
#         iscell = np.load(iscell_path, allow_pickle=True)[:, 0].astype(bool) if iscell_path.exists() else np.ones(len(stat), dtype=bool)
#
#         n_cells = iscell.sum()
#         self.logger.info(f"Loaded {n_cells} cells from {stat_path}")
#
#         # store for visualization
#         self._s2p_stat = stat
#         self._s2p_ops = ops
#         self._s2p_iscell = iscell
#
#         set_last_dir("suite2p_output", folder_path)
#
#     except Exception as e:
#         self.logger.error(f"Failed to load Suite2p results: {e}")
#
#
# def _process_cellpose_load(self, file_path: str):
#     """Process loaded Cellpose masks file."""
#     import numpy as np
#     from pathlib import Path
#
#     try:
#         masks = np.load(file_path, allow_pickle=True)
#         if isinstance(masks, np.ndarray) and masks.dtype == object:
#             # may be a dict stored in npy
#             masks = masks.item() if masks.ndim == 0 else masks
#
#         self.logger.info(f"Loaded Cellpose masks from {file_path}")
#         self._cellpose_masks = masks
#
#         set_last_dir("cellpose_masks", str(Path(file_path).parent))
#
#     except Exception as e:
#         self.logger.error(f"Failed to load Cellpose masks: {e}")


def _draw_data_options_content(self):
    """Draw data options content showing settings that affect Suite2p processing."""
    from mbo_utilities.gui._availability import HAS_SUITE3D

    INPUT_WIDTH = 100
    has_phase_support = getattr(self, "has_raster_scan_support", False)
    nz = getattr(self, "nz", 1)

    # check if data already has z-registration applied
    already_registered = False
    arrays = self._get_data_arrays() if hasattr(self, "_get_data_arrays") else []
    if arrays and hasattr(arrays[0], "metadata"):
        already_registered = arrays[0].metadata.get("apply_shift", False)

    has_z_reg = HAS_SUITE3D and nz > 1 and not already_registered

    has_any_options = has_phase_support or has_z_reg

    # ensure s2p phase attributes exist with defaults
    if not hasattr(self, "_s2p_fix_phase"):
        self._s2p_fix_phase = True
    if not hasattr(self, "_s2p_use_fft"):
        self._s2p_use_fft = True

    # scan-phase correction section
    if has_phase_support:
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction")
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

        # upsample factor
        imgui.set_next_item_width(INPUT_WIDTH)
        upsample_changed, upsample_val = imgui.input_int("Upsample", self.phase_upsample, step=1)
        set_tooltip("Upsampling factor for sub-pixel alignment")
        if upsample_changed:
            self.phase_upsample = max(1, upsample_val)

        # show current offset values
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Current offsets:")
        for i, ofs in enumerate(self.current_offset):
            imgui.text(f"  Array {i + 1}: {ofs:.3f} px")

    # z-registration section (suite3d) - only show if not already registered
    if nz > 1 and not already_registered:
        if has_phase_support:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Z-Registration (Suite3D)")
        imgui.spacing()

        if has_z_reg:
            reg_z_val = getattr(self, "_register_z", False)
            reg_changed, reg_value = imgui.checkbox("Register Z-Planes Axially", reg_z_val)
            set_tooltip(
                "Use Suite3D to register z-planes axially before processing.\n"
                "This corrects for z-drift between planes."
            )
            if reg_changed:
                self._register_z = reg_value
        else:
            # show disabled option with install hint
            imgui.begin_disabled()
            imgui.checkbox("Register Z-Planes Axially", False)
            imgui.end_disabled()
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Install suite3d to enable")

    if not has_any_options:
        imgui.text_disabled("No available options for this data type.")


def draw_section_suite2p(self):
    """Draw Suite2p configuration UI with button-based popups."""
    imgui.spacing()

    # consistent input width
    INPUT_WIDTH = 120

    # set proper padding and spacing using context manager for safety
    with imgui_ctx.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4)):
        with imgui_ctx.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2)):
            _draw_section_suite2p_content(self)


def _draw_section_suite2p_content(self):
    """inner content for suite2p section (called within style context)."""
    INPUT_WIDTH = 120

    # initialize selection state (timepoints, z-planes, channels)
    max_frames, num_planes, num_channels = _init_s2p_selection_state(self)

    # get output path
    s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")
    has_save_path = bool(s2p_path)

    # === SELECTION BUTTON + PREVIEW ===
    imgui.spacing()
    if imgui.button("Selection"):
        self._s2p_selection_open = True
    set_tooltip("Output path and frame/plane selection")

    # draw the selection popup
    _draw_s2p_selection_popup(self)

    # selection preview info underneath
    imgui.indent(8)
    _draw_s2p_selection_preview(self, max_frames, num_planes, num_channels)
    # show output path
    if s2p_path:
        # truncate long paths
        display_path = s2p_path
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        imgui.text_colored(imgui.ImVec4(0.5, 0.7, 0.9, 1.0), display_path)
    else:
        imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.4, 1.0), "(output path not set)")
    imgui.unindent(8)

    imgui.spacing()

    # === RUN SUITE2P BUTTON (centered) ===
    imgui.spacing()
    avail_width = imgui.get_content_region_avail().x
    button_width = 120
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_width - button_width) / 2)

    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0))

    if not has_save_path:
        imgui.begin_disabled()

    button_clicked = imgui.button("Run Suite2p", imgui.ImVec2(button_width, 0))

    if not has_save_path:
        imgui.end_disabled()

    imgui.pop_style_color(3)

    if not has_save_path and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip("Set output path via Selection button")

    # handle run button click
    if button_clicked and has_save_path:
        self.logger.info(f"Running Suite2p pipeline on {len(self._selected_planes)} planes...")
        run_process(self)
        if getattr(self, "_s2p_background", True):
            self.logger.info("Suite2p processing started in background.")
        else:
            self.logger.info("Suite2p processing submitted.")

    if self._install_error:
        if self._show_red_text:
            imgui.text_colored(imgui.ImVec4(1.0, 0.0, 0.0, 1.0), "Error: lbm_suite2p_python is not installed.")
        if self._show_green_text:
            imgui.text_colored(imgui.ImVec4(0.0, 1.0, 0.0, 1.0), "lbm_suite2p_python install success.")
        if self._show_install_button and imgui.button("Install"):
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

    imgui.spacing()

    # === PIPELINE SETTINGS ===
    imgui.separator_text("Pipeline Settings")

    # draw the main settings popup (triggered from table below)
    _draw_s2p_settings_popup(self)

    # --- Registration ---
    def draw_registration_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.align_by_chan = imgui.input_int(
            "Align by Channel", self.s2p.align_by_chan
        )
        set_tooltip("Channel index used for alignment (1-based).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nimg_init = imgui.input_int("Initial Frames", self.s2p.nimg_init)
        set_tooltip("Number of frames used to build the reference image.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.batch_size = imgui.input_int("Batch Size", self.s2p.batch_size)
        set_tooltip("Number of frames processed per registration batch.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.maxregshift = imgui.input_float(
            "Max Shift Fraction", self.s2p.maxregshift
        )
        set_tooltip("Maximum allowed shift as a fraction of the image size.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.smooth_sigma = imgui.input_float(
            "Smooth Sigma", self.s2p.smooth_sigma
        )
        set_tooltip("Gaussian smoothing sigma (pixels) before registration.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.smooth_sigma_time = imgui.input_float(
            "Smooth Sigma Time", self.s2p.smooth_sigma_time
        )
        set_tooltip("Temporal smoothing sigma (frames) before registration.")
        _, self.s2p.keep_movie_raw = imgui.checkbox(
            "Keep Raw Movie", self.s2p.keep_movie_raw
        )
        set_tooltip("Keep unregistered binary movie after processing.")
        _, self.s2p.two_step_registration = imgui.checkbox(
            "Two-Step Registration", self.s2p.two_step_registration
        )
        set_tooltip("Perform registration twice for low-SNR data.")
        _, self.s2p.reg_tif = imgui.checkbox("Export Registered TIFF", self.s2p.reg_tif)
        set_tooltip("Export registered movie as TIFF files.")
        _, self.s2p.reg_tif_chan2 = imgui.checkbox(
            "Export Chan2 TIFF", self.s2p.reg_tif_chan2
        )
        set_tooltip("Export registered TIFFs for channel 2.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.subpixel = imgui.input_int("Subpixel Precision", self.s2p.subpixel)
        set_tooltip("Subpixel precision level (1/subpixel step).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.th_badframes = imgui.input_float(
            "Bad Frame Threshold", self.s2p.th_badframes
        )
        set_tooltip("Threshold for excluding low-quality frames.")
        _, self.s2p.norm_frames = imgui.checkbox(
            "Normalize Frames", self.s2p.norm_frames
        )
        set_tooltip("Normalize frames during registration.")
        _, self.s2p.force_refImg = imgui.checkbox("Force refImg", self.s2p.force_refImg)
        set_tooltip("Use stored reference image instead of recomputing.")
        _, self.s2p.pad_fft = imgui.checkbox("Pad FFT", self.s2p.pad_fft)
        set_tooltip("Pad image for FFT registration to reduce edge artifacts.")

        imgui.spacing()
        imgui.text("Channel 2 File:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
        imgui.text(self.s2p.chan2_file if self.s2p.chan2_file else "(none)")
        imgui.pop_text_wrap_pos()
        if imgui.button("Browse##chan2"):
            default_dir = str(get_last_dir("suite2p_chan2") or pathlib.Path().home())
            res = pfd.open_file("Select channel 2 file", default_dir)
            if res and res.result():
                self.s2p.chan2_file = res.result()[0]
                set_last_dir("suite2p_chan2", res.result()[0])
        set_tooltip("Path to channel 2 binary file for cross-channel registration.")

        if imgui.tree_node("1-Photon Registration"):
            _, self.s2p.do_1Preg = imgui.checkbox(
                "Enable 1P Registration", self.s2p.do_1Preg
            )
            set_tooltip("Apply high-pass filtering and tapering for 1-photon data.")

            imgui.begin_disabled(not self.s2p.do_1Preg)
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.spatial_hp_reg = imgui.input_int(
                "Spatial HP Window", self.s2p.spatial_hp_reg
            )
            set_tooltip(
                "Window size for spatial high-pass filtering before registration."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.pre_smooth = imgui.input_float(
                "Pre-smooth Sigma", self.s2p.pre_smooth
            )
            set_tooltip(
                "Gaussian smoothing stddev before high-pass filtering (0=disabled)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.spatial_taper = imgui.input_float(
                "Spatial Taper", self.s2p.spatial_taper
            )
            set_tooltip(
                "Pixels to set to zero on edges (important for vignetted windows)."
            )
            imgui.end_disabled()
            imgui.tree_pop()

        if imgui.tree_node("Non-rigid Registration"):
            _, self.s2p.nonrigid = imgui.checkbox("Enable Non-rigid", self.s2p.nonrigid)
            set_tooltip(
                "Split FOV into blocks and compute registration offsets per block."
            )

            imgui.begin_disabled(not self.s2p.nonrigid)

            if self.s2p.block_size is None:
                self.s2p.block_size = [128, 128]
            imgui.set_next_item_width(INPUT_WIDTH)
            block_y_changed, block_y = imgui.input_int(
                "Block Height", self.s2p.block_size[0]
            )
            set_tooltip(
                "Block height for non-rigid registration (power of 2/3 recommended)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            block_x_changed, block_x = imgui.input_int(
                "Block Width", self.s2p.block_size[1]
            )
            set_tooltip(
                "Block width for non-rigid registration (power of 2/3 recommended)."
            )
            if block_y_changed or block_x_changed:
                self.s2p.block_size = [block_y, block_x]

            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.snr_thresh = imgui.input_float(
                "SNR Threshold", self.s2p.snr_thresh
            )
            set_tooltip("Phase correlation peak threshold (1.5 recommended for 1P).")
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.maxregshiftNR = imgui.input_float(
                "Max NR Shift", self.s2p.maxregshiftNR
            )
            set_tooltip("Max pixel shift of block relative to rigid shift.")

            imgui.end_disabled()
            imgui.tree_pop()

    # --- ROI Detection ---
    def draw_roi_detection_settings():
        # determine detection mode for greying out (re-check in popup context)
        use_anatomical_local = self.s2p.anatomical_only > 0

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.anatomical_only = imgui.input_int(
            "Anatomical Only", self.s2p.anatomical_only
        )
        set_tooltip(
            "0=disabled (use functional detection)\n"
            "1=max_proj / mean_img combined\n"
            "2=mean_img only\n"
            "3=enhanced mean_img (LBM default, recommended)\n"
            "4=max_proj only"
        )

        # Grey out Cellpose settings when anatomical_only = 0
        imgui.begin_disabled(not use_anatomical_local)

        if not use_anatomical_local:
            imgui.text_colored(
                imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                "(Enable anatomical_only to use Cellpose)",
            )

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.diameter = imgui.input_int("Cell Diameter", self.s2p.diameter)
        set_tooltip(
            "Expected cell diameter in pixels (6 = LBM default for ~6μm cells). Passed to Cellpose."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.cellprob_threshold = imgui.input_float(
            "CellProb Threshold", self.s2p.cellprob_threshold
        )
        set_tooltip(
            "Cell probability threshold for Cellpose. Default: 0.0\n\n"
            "DECREASE this threshold if:\n"
            "  - Cellpose is not returning as many masks as expected\n"
            "  - Masks are too small\n\n"
            "INCREASE this threshold if:\n"
            "  - Cellpose is returning too many masks\n"
            "  - Getting false positives from dull/dim areas\n\n"
            "LBM default: -6 (very permissive)"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.flow_threshold = imgui.input_float(
            "Flow Threshold", self.s2p.flow_threshold
        )
        set_tooltip(
            "Maximum allowed error of flows for each mask. Default: 0.4\n\n"
            "INCREASE this threshold if:\n"
            "  - Cellpose is not returning as many masks as expected\n"
            "  - Set to 0.0 to turn off flow checking completely\n\n"
            "DECREASE this threshold if:\n"
            "  - Cellpose is returning too many ill-shaped masks\n\n"
            "LBM default: 0 (flow checking disabled)"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.spatial_hp_cp = imgui.input_float(
            "Spatial HP (Cellpose)", self.s2p.spatial_hp_cp
        )
        set_tooltip(
            "Spatial high-pass filtering before Cellpose, as a multiple of diameter.\n"
            "0.5 = LBM default"
        )

        imgui.end_disabled()

        # functional detection settings (greyed if using anatomical)
        if imgui.tree_node("Functional Detection"):
            imgui.begin_disabled(use_anatomical_local)
            if use_anatomical_local:
                imgui.text_colored(
                    imgui.ImVec4(0.7, 0.7, 0.7, 1.0),
                    "(Skipped when anatomical_only > 0)",
                )

            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.functional_chan = imgui.input_int(
                "Functional Channel", self.s2p.functional_chan
            )
            set_tooltip("Channel used for functional ROI extraction (1-based).")
            _, self.s2p.sparse_mode = imgui.checkbox(
                "Sparse Mode", self.s2p.sparse_mode
            )
            set_tooltip("Use sparse detection (recommended for soma).")
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.spatial_scale = imgui.input_int(
                "Spatial Scale", self.s2p.spatial_scale
            )
            set_tooltip(
                "ROI size scale: 0=auto, 1=6-pixel cells (LBM default), 2=medium, 3=large, 4=very large."
            )
            _, self.s2p.connected = imgui.checkbox("Connected ROIs", self.s2p.connected)
            set_tooltip("Require ROIs to be connected regions.")
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.threshold_scaling = imgui.input_float(
                "Threshold Scaling", self.s2p.threshold_scaling
            )
            set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.spatial_hp_detect = imgui.input_int(
                "Spatial HP Detect", self.s2p.spatial_hp_detect
            )
            set_tooltip("Spatial high-pass filter size for neuropil subtraction.")
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.max_iterations = imgui.input_int(
                "Max Iterations", self.s2p.max_iterations
            )
            set_tooltip("Maximum number of cell-detection iterations.")
            imgui.end_disabled()
            imgui.tree_pop()

        # shared settings for both detection methods
        imgui.spacing()
        imgui.text("Shared Settings:")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.max_overlap = imgui.input_float("Max Overlap", self.s2p.max_overlap)
        set_tooltip("Maximum allowed fraction of overlapping ROI pixels.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.high_pass = imgui.input_int("High-Pass Window", self.s2p.high_pass)
        set_tooltip("Running mean subtraction window for temporal high-pass filtering.")
        _, self.s2p.smooth_masks = imgui.checkbox("Smooth Masks", self.s2p.smooth_masks)
        set_tooltip("Smooth masks in the final ROI detection pass.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
        set_tooltip("Maximum number of binned frames for ROI detection.")

    # --- Signal Extraction ---
    def draw_signal_extraction_settings():
        _, self.s2p.allow_overlap = imgui.checkbox(
            "Allow Overlap", self.s2p.allow_overlap
        )
        set_tooltip("Allow overlapping ROI pixels during extraction.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.min_neuropil_pixels = imgui.input_int(
            "Min Neuropil Pixels", self.s2p.min_neuropil_pixels
        )
        set_tooltip("Minimum neuropil pixels per ROI.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.inner_neuropil_radius = imgui.input_int(
            "Inner Neuropil Radius", self.s2p.inner_neuropil_radius
        )
        set_tooltip("Pixels to exclude between ROI and neuropil region.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.lam_percentile = imgui.input_int(
            "Lambda Percentile", self.s2p.lam_percentile
        )
        set_tooltip("Percentile of Lambda used for neuropil exclusion.")

    # --- Classification ---
    def draw_classification_settings():
        _, self.s2p.soma_crop = imgui.checkbox("Soma Crop", self.s2p.soma_crop)
        set_tooltip("Crop dendrites for soma classification.")

        imgui.spacing()
        imgui.text("Classifier Path:")
        path_display = self.s2p.classifier_path if self.s2p.classifier_path else "(none)"
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x - 80)
        imgui.text(path_display)
        imgui.pop_text_wrap_pos()
        imgui.same_line()
        if imgui.button("Browse##classifier"):
            default_dir = str(Path(self.s2p.classifier_path).parent) if self.s2p.classifier_path else str(Path.home())
            self._classifier_dialog = pfd.open_file(
                "Select classifier file",
                default_dir,
                ["Classifier files", "*.npy *.pkl *.pickle", "All files", "*"],
            )
        set_tooltip("Select custom classifier file (e.g., .npy or .pkl)")

        # handle file dialog result
        if hasattr(self, "_classifier_dialog") and self._classifier_dialog is not None:
            if self._classifier_dialog.ready():
                result = self._classifier_dialog.result()
                if result:
                    self.s2p.classifier_path = result[0] if isinstance(result, list) else result
                self._classifier_dialog = None

    # --- Spike Deconvolution ---
    def draw_spike_deconv_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.neucoeff = imgui.input_float(
            "Neuropil Coefficient", self.s2p.neucoeff
        )
        set_tooltip(
            "Neuropil coefficient for all ROIs (F_corrected = F - coeff * F_neu)."
        )

        # Baseline method as combo box
        baseline_options = ["maximin", "constant", "constant_percentile"]
        current_baseline_idx = (
            baseline_options.index(self.s2p.baseline)
            if self.s2p.baseline in baseline_options
            else 0
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        baseline_changed, selected_baseline_idx = imgui.combo(
            "Baseline Method", current_baseline_idx, baseline_options
        )
        if baseline_changed:
            self.s2p.baseline = baseline_options[selected_baseline_idx]
        set_tooltip(
            "maximin: moving baseline with min/max filters. "
            "constant: minimum of Gaussian-filtered trace. "
            "constant_percentile: percentile of trace."
        )

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.win_baseline = imgui.input_float(
            "Baseline Window (s)", self.s2p.win_baseline
        )
        set_tooltip("Window for maximin filter in seconds.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.sig_baseline = imgui.input_float(
            "Baseline Sigma (s)", self.s2p.sig_baseline
        )
        set_tooltip("Gaussian filter width in seconds for baseline computation.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.prctile_baseline = imgui.input_float(
            "Baseline Percentile", self.s2p.prctile_baseline
        )
        set_tooltip("Percentile of trace for constant_percentile baseline method.")

    # --- Output Settings ---
    def draw_output_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.preclassify = imgui.input_float(
            "Preclassify Threshold", self.s2p.preclassify
        )
        set_tooltip("Probability threshold to apply classifier before extraction.")
        _, self.s2p.save_nwb = imgui.checkbox("Save NWB", self.s2p.save_nwb)
        _, self.s2p.save_mat = imgui.checkbox("Save MATLAB File", self.s2p.save_mat)
        set_tooltip("Export results to Fall.mat for MATLAB analysis.")
        _, self.s2p.combined = imgui.checkbox(
            "Combine Across Planes", self.s2p.combined
        )
        set_tooltip("Combine per-plane results into one GUI-loadable folder.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.aspect = imgui.input_float("Aspect Ratio", self.s2p.aspect)
        set_tooltip("um/pixel ratio X/Y for correct GUI aspect display.")
        _, self.s2p.report_time = imgui.checkbox("Report Timing", self.s2p.report_time)
        set_tooltip("Return timing dictionary for each processing stage.")

        # Channel 2 settings
        imgui.spacing()
        imgui.separator()
        imgui.text("Channel 2:")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.chan2_thres = imgui.input_float(
            "Chan2 Detection Threshold", self.s2p.chan2_thres
        )
        set_tooltip("Threshold for calling ROI detected on channel 2.")

    # === PIPELINE SETTINGS TABLE ===
    # tabular layout: [checkbox/empty] | [label] | [settings button]
    table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    if imgui.begin_table("pipeline_settings_table", 3, table_flags):
        imgui.table_setup_column("checkbox", imgui.TableColumnFlags_.width_fixed, 24)
        imgui.table_setup_column("label", imgui.TableColumnFlags_.width_fixed, 100)
        imgui.table_setup_column("button", imgui.TableColumnFlags_.width_stretch)

        # --- Main settings row (no checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()  # empty checkbox column
        imgui.table_next_column()
        imgui.text("Main")
        imgui.table_next_column()
        if imgui.button("Settings##main"):
            self._s2p_settings_open = True
        set_tooltip("Main Suite2p parameters (Tau, denoise, dF/F, etc.)")

        # --- Data Options row (no checkbox) - shows available data-specific widgets ---
        imgui.table_next_row()
        imgui.table_next_column()  # empty checkbox column
        imgui.table_next_column()
        imgui.text("Data Options")
        imgui.table_next_column()
        if imgui.button("Settings##data_opts"):
            _popup_states["data_options"] = True
            imgui.open_popup("Data Options##data_options")
        set_tooltip("Data-specific options (scan-phase correction, frame averaging, etc.)")

        # --- Registration row (with checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()
        _, self.s2p.do_registration = imgui.checkbox("##reg_cb", self.s2p.do_registration)
        set_tooltip("Enable/disable motion registration")
        imgui.table_next_column()
        imgui.text("Registration")
        imgui.table_next_column()
        if imgui.button("Settings##reg"):
            _popup_states["reg_settings"] = True
            imgui.open_popup("Registration##reg_settings")
        set_tooltip("Configure motion correction and registration parameters")

        # --- ROI Detection row (with checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()
        _, self.s2p.roidetect = imgui.checkbox("##roi_cb", self.s2p.roidetect)
        set_tooltip("Enable/disable ROI detection")
        imgui.table_next_column()
        imgui.text("ROI Detection")
        imgui.table_next_column()
        if imgui.button("Settings##roi"):
            _popup_states["roi_settings"] = True
            imgui.open_popup("ROI Detection##roi_settings")
        set_tooltip("Configure cell detection parameters")

        # --- Signal Extraction row (with checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()
        _, self.s2p.neuropil_extract = imgui.checkbox("##extract_cb", self.s2p.neuropil_extract)
        set_tooltip("Enable/disable signal extraction")
        imgui.table_next_column()
        imgui.text("Signal Extraction")
        imgui.table_next_column()
        if imgui.button("Settings##extract"):
            _popup_states["extract_settings"] = True
            imgui.open_popup("Signal Extraction##extract_settings")
        set_tooltip("Configure signal extraction parameters")

        # --- Classification row ---
        imgui.table_next_row()
        imgui.table_next_column()
        _, self.s2p.use_builtin_classifier = imgui.checkbox("##classify_cb", self.s2p.use_builtin_classifier)
        set_tooltip("Enable/disable ROI classification")
        imgui.table_next_column()
        imgui.text("Classification")
        imgui.table_next_column()
        if imgui.button("Settings##classify"):
            _popup_states["classify_settings"] = True
            imgui.open_popup("Classification##classify_settings")
        set_tooltip("Configure ROI classification settings")

        # --- Spike Deconvolution row (with checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()
        _, self.s2p.spikedetect = imgui.checkbox("##spike_cb", self.s2p.spikedetect)
        set_tooltip("Enable/disable spike deconvolution")
        imgui.table_next_column()
        imgui.text("Spike Deconv")
        imgui.table_next_column()
        if imgui.button("Settings##spike"):
            _popup_states["spike_settings"] = True
            imgui.open_popup("Spike Deconv##spike_settings")
        set_tooltip("Configure spike deconvolution parameters")

        # --- Output row (no checkbox) ---
        imgui.table_next_row()
        imgui.table_next_column()  # empty checkbox column
        imgui.table_next_column()
        imgui.text("Output")
        imgui.table_next_column()
        if imgui.button("Settings##output"):
            _popup_states["output_settings"] = True
            imgui.open_popup("Output##output_settings")
        set_tooltip("Configure output options")

        # --- Draw all popups (inside table context) ---

        # Registration popup
        imgui.set_next_window_size(imgui.ImVec2(450, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "Registration##reg_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["reg_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_registration_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##reg", imgui.ImVec2(80, 0)):
                        _popup_states["reg_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # ROI Detection popup
        imgui.set_next_window_size(imgui.ImVec2(450, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "ROI Detection##roi_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["roi_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_roi_detection_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##roi", imgui.ImVec2(80, 0)):
                        _popup_states["roi_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # Signal Extraction popup
        imgui.set_next_window_size(imgui.ImVec2(400, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "Signal Extraction##extract_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["extract_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_signal_extraction_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##extract", imgui.ImVec2(80, 0)):
                        _popup_states["extract_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # Classification popup
        imgui.set_next_window_size(imgui.ImVec2(400, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "Classification##classify_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["classify_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_classification_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##classify", imgui.ImVec2(80, 0)):
                        _popup_states["classify_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # Spike Deconv popup
        imgui.set_next_window_size(imgui.ImVec2(400, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "Spike Deconv##spike_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["spike_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_spike_deconv_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##spike", imgui.ImVec2(80, 0)):
                        _popup_states["spike_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # Output popup
        imgui.set_next_window_size(imgui.ImVec2(400, 0), imgui.Cond_.first_use_ever)
        opened, visible = imgui.begin_popup_modal(
            "Output##output_settings",
            p_open=True,
            flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
        )
        if opened:
            try:
                if not visible:
                    _popup_states["output_settings"] = False
                    imgui.close_current_popup()
                else:
                    draw_output_settings()
                    imgui.spacing()
                    imgui.separator()
                    if imgui.button("Close##output", imgui.ImVec2(80, 0)):
                        _popup_states["output_settings"] = False
                        imgui.close_current_popup()
            finally:
                imgui.end_popup()

        # Data Options popup - shows data settings that affect Suite2p processing
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

        imgui.end_table()
    # if imgui.button("Load Suite2p Masks"):
    #     try:
    #         import numpy as np
    #         from pathlib import Path
    #
    #         res = pfd.select_folder(self._saveas_outdir or str(Path().home()))
    #         if res:
    #             self.s2p_dir = res.result()
    #
    #         s2p_dir = Path(self._saveas_outdir)
    #         ops = np.load(next(s2p_dir.rglob("ops.npy")), allow_pickle=True).item()
    #         stat = np.load(next(s2p_dir.rglob("stat.npy")), allow_pickle=True)
    #         iscell = np.load(next(s2p_dir.rglob("iscell.npy")), allow_pickle=True)[:, 0].astype(bool)
    #
    #         Ly, Lx = ops["Ly"], ops["Lx"]
    #         mask_rgb = np.zeros((Ly, Lx, 3), dtype=np.float32)
    #
    #         # build ROI overlay (green for accepted cells)
    #         for s, ok in zip(stat, iscell):
    #             if not ok:
    #                 continue
    #             ypix, xpix, lam = s["ypix"], s["xpix"], s["lam"]
    #             lam = lam / lam.max()
    #             mask_rgb[ypix, xpix, 1] = np.maximum(mask_rgb[ypix, xpix, 1], lam)  # G channel
    #
    #         self._mask_color_strength = 0.5
    #         self._mask_rgb = mask_rgb
    #         self._mean_img = ops["meanImg"].astype(np.float32)
    #         self._show_mask_slider = True
    #
    #         combined = self._mean_img[..., None].repeat(3, axis=2)
    #         combined = combined / combined.max()
    #         combined = np.clip(combined + self._mask_color_strength * self._mask_rgb, 0, 1)
    #         self.image_widget.graphics[1].data = combined
    #         self.logger.info(f"Loaded and displayed {iscell.sum()} Suite2p masks.")
    #
    #     except Exception as e:
    #         self.logger.error(f"Mask load failed: {e}")

    # if getattr(self, "_show_mask_slider", False):
    #     imgui.separator_text("Mask Overlay")
    #     changed, self._mask_color_strength = imgui.slider_float(
    #         "Color Strength", self._mask_color_strength, 0.0, 2.0
    #     )
    #     if changed:
    #         combined = self._mean_img[..., None].repeat(3, axis=2)
    #         combined = combined / combined.max()
    #         combined = np.clip(combined + self._mask_color_strength * self._mask_rgb, 0, 1)
    #         self.image_widget.graphics[1].data = combined


def _build_channel_dirname(self, channel: int) -> str:
    """build dimension-tagged dirname for a single channel run."""
    from mbo_utilities.arrays.features import DimensionTag, TAG_REGISTRY

    tags = []
    # timepoints
    tp = self._s2p_tp_parsed
    if tp and tp.final_indices:
        tp_start = tp.final_indices[0] + 1
        tp_stop = tp.final_indices[-1] + 1
        tags.append(DimensionTag(TAG_REGISTRY["T"], start=tp_start, stop=tp_stop, step=1))
    # channel (single)
    tags.append(DimensionTag(TAG_REGISTRY["C"], start=channel, stop=None, step=1))
    # z-planes
    z_start = getattr(self, "_s2p_z_start", 1)
    z_stop = getattr(self, "_s2p_z_stop", 1)
    z_step = getattr(self, "_s2p_z_step", 1)
    tags.append(DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=z_stop, step=z_step))
    return "_".join(tag.to_string() for tag in tags)


def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline != "suite2p":
        if self._current_pipeline == "masknmf":
            self.logger.info("Running MaskNMF pipeline (not yet implemented).")
        else:
            self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
        return

    self.logger.info(f"Running Suite2p pipeline with settings: {self.s2p}")
    if not _check_lsp_available():
        self.logger.warning(
            "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
            "`uv pip install lbm_suite2p_python`",
        )
        self._install_error = True
        return

    if not self._install_error:
        # Get selected planes (1-indexed)
        selected_planes = getattr(self, "_selected_planes", None)
        if not selected_planes:
            # Fallback to current plane
            names = self.image_widget._slider_dim_names or ()
            try:
                current_z = self.image_widget.indices["z"] if "z" in names else 0
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

            # get input path (full list or single path)
            if isinstance(self.fpath, (list, tuple)):
                input_path = [str(f) for f in self.fpath]
            else:
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

            for channel in selected_channels:
                # per-channel output subdir when multiple channels selected
                if multi_channel:
                    output_dir = str(Path(s2p_path) / _build_channel_dirname(self, channel))
                else:
                    output_dir = s2p_path

                worker_args = {
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "planes": sorted(selected_planes),
                    "roi": roi,
                    "num_timepoints": self.s2p.target_timepoints,
                    "ops": self.s2p.to_dict(),
                    "fix_phase": self._s2p_fix_phase,
                    "use_fft": self._s2p_use_fft,
                    "channel": channel if multi_channel else None,
                    "s2p_settings": {
                        "keep_raw": self.s2p.keep_raw,
                        "keep_reg": self.s2p.keep_reg,
                        "force_reg": self.s2p.force_reg,
                        "force_detect": self.s2p.force_detect,
                        "dff_window_size": self.s2p.dff_window_size,
                        "dff_percentile": self.s2p.dff_percentile,
                        "dff_smooth_window": self.s2p.dff_smooth_window,
                    },
                }

                description = f"Suite2p: {len(selected_planes)} plane(s)"
                if multi_channel:
                    description += f" ch{channel}"
                if roi:
                    description += f" ROI {roi}"

                pid = pm.spawn(
                    task_type="suite2p",
                    args=worker_args,
                    description=description,
                    output_path=output_dir,
                )

                if pid:
                    self.logger.info(f"Started background process {pid} for {description}")
                else:
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

            s2p_path = getattr(self, "_s2p_outdir", "") or getattr(
                self, "_saveas_outdir", ""
            )

            # build list of (arr_idx, z_plane, channel, base_out) jobs
            jobs = []
            for i, _arr in enumerate(self.image_widget.data):
                for channel in selected_channels:
                    if multi_channel:
                        base_out = str(Path(s2p_path) / _build_channel_dirname(self, channel))
                    else:
                        base_out = None  # use default path logic
                    for plane in sorted(selected_planes):
                        jobs.append((i, plane - 1, channel if multi_channel else None, base_out))

            # Check if parallel processing is enabled
            use_parallel = getattr(self, "_parallel_processing", False)
            max_jobs = getattr(self, "_max_parallel_jobs", 2)

            if use_parallel and len(jobs) > 1:
                # Parallel processing with limited concurrency
                from concurrent.futures import ThreadPoolExecutor

                def run_parallel():
                    self.logger.info(
                        f"Starting parallel processing with max {max_jobs} concurrent jobs..."
                    )
                    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
                        futures = {}
                        for job_idx, (arr_idx, z_plane, channel, base_out) in enumerate(jobs):
                            future = executor.submit(
                                run_plane_from_data, self, arr_idx, z_plane,
                                channel=channel, base_out_override=base_out,
                            )
                            futures[future] = (z_plane, channel, job_idx)

                        for future in futures:
                            z_plane, channel, job_idx = futures[future]
                            try:
                                future.result()
                                desc = f"Plane {z_plane + 1}"
                                if channel is not None:
                                    desc += f" ch{channel}"
                                self.logger.info(
                                    f"{desc} completed ({job_idx + 1}/{len(jobs)})"
                                )
                            except Exception as e:
                                self.logger.exception(
                                    f"Error processing plane {z_plane + 1}: {e}"
                                )
                    self.logger.info("Suite2p parallel processing complete.")

                threading.Thread(target=run_parallel, daemon=True).start()
            else:
                # Sequential processing in a single background thread
                def run_all_planes_sequential():
                    for job_idx, (arr_idx, z_plane, channel, base_out) in enumerate(jobs):
                        desc = f"plane {z_plane + 1}"
                        if channel is not None:
                            desc += f" ch{channel}"
                        self.logger.info(
                            f"Processing {desc} ({job_idx + 1}/{len(jobs)})..."
                        )
                        try:
                            run_plane_from_data(
                                self, arr_idx, z_plane,
                                channel=channel, base_out_override=base_out,
                            )
                        except Exception as e:
                            self.logger.exception(
                                f"Error processing {desc}: {e}"
                            )
                    self.logger.info("Suite2p processing complete.")

                threading.Thread(target=run_all_planes_sequential, daemon=True).start()


def run_plane_from_data(self, arr_idx, z_plane=None, channel=None, base_out_override=None):
    if not _check_lsp_available():
        self.logger.error("lbm_suite2p_python is not installed.")
        self._install_error = True
        return

    arr = self.image_widget.data[arr_idx]
    # Use provided z_plane or fall back to current index
    if z_plane is not None:
        current_z = z_plane
    else:
        names = self.image_widget._slider_dim_names or ()
        try:
            current_z = self.image_widget.indices["z"] if "z" in names else 0
        except (IndexError, KeyError):
            current_z = 0

    # Use base_out_override (per-channel subdir), or fall back to _s2p_outdir
    if base_out_override:
        base_out = Path(base_out_override)
    else:
        s2p_path = getattr(self, "_s2p_outdir", "") or getattr(self, "_saveas_outdir", "")
        base_out = Path(s2p_path) if s2p_path else None
    if not base_out:
        from mbo_utilities.file_io import get_mbo_dirs, get_last_savedir_path

        # find last saved dir
        last_savedir = get_last_savedir_path()
        base_out = Path(last_savedir) if last_savedir else get_mbo_dirs()["data"]
    if not base_out.exists():
        base_out.mkdir(exist_ok=True)

    if len(self.image_widget.graphics) > 1:
        plane_dir = base_out / f"plane{current_z + 1:02d}_roi{arr_idx + 1:02d}"
        roi = arr_idx + 1
        plane = current_z + 1
    else:
        plane_dir = base_out / f"plane{current_z + 1:02d}_stitched"
        roi = None
        plane = current_z + 1

    ops_path = plane_dir / "ops.npy"

    lazy_mdata = getattr(arr, "metadata", {}).copy()

    # Get dimensions without extracting - let imwrite handle extraction lazily
    # For 4D: shape is (T, Z, H, W), imwrite with planes=N will write that z-plane
    # For 3D: shape is (T, H, W), imwrite writes all frames
    Lx = arr.shape[-1]
    Ly = arr.shape[-2]

    # Use standard metadata accessors to ensure we get correct values from any alias
    from mbo_utilities.metadata import get_param, get_voxel_size

    vs = get_voxel_size(lazy_mdata)

    # extract only scalar metadata needed for suite2p - do NOT pass shape arrays
    # that could confuse the pipeline when processing 4D data
    md = {
        "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_timepoints": self.s2p.target_timepoints,
        "num_frames": self.s2p.target_timepoints,  # legacy alias
        "nframes": self.s2p.target_timepoints,  # suite2p alias
        "n_frames": self.s2p.target_timepoints,
        "original_file": str(self.fpath),
        "roi_index": arr_idx,
        "z_index": current_z,
        "plane": plane,
        "Ly": Ly,
        "Lx": Lx,
        "fs": get_param(lazy_mdata, "fs", 15.0),
        "dx": vs.dx,
        "dy": vs.dy,
        "dz": vs.dz,
        "ops_path": str(ops_path),
        "save_path": str(plane_dir),
        "raw_file": str((plane_dir / "data_raw.bin").resolve()),
    }

    from lbm_suite2p_python import default_ops

    ops = self.s2p.to_dict()
    defaults = default_ops()
    defaults.update(ops)
    # Only update with md dict, not the full lazy_mdata which may have 4D shape
    defaults.update(md)

    # set the correct 3D shape for the output binary (T, Ly, Lx)
    # this is needed by write_ops to create ops.npy
    # CRITICAL: override any 4D shape from the source array with the correct 3D shape
    defaults["shape"] = (self.s2p.target_timepoints, Ly, Lx)
    defaults["num_timepoints"] = self.s2p.target_timepoints
    defaults["num_frames"] = self.s2p.target_timepoints  # legacy alias
    defaults["nframes"] = self.s2p.target_timepoints  # suite2p alias

    # single-channel extraction: suite2p only sees 1 channel
    if channel is not None:
        defaults["functional_chan"] = 1
        defaults["align_by_chan"] = 1

    # also clean lazy_mdata to prevent shape contamination from arr.metadata
    # TODO: Do we need this?
    lazy_mdata.pop("shape", None)
    lazy_mdata.pop("num_timepoints", None)
    lazy_mdata.pop("num_frames", None)
    lazy_mdata.pop("nframes", None)
    lazy_mdata.pop("n_frames", None)

    # CRITICAL: Also clean arr.metadata directly to prevent imwrite from getting
    # the full shape when it does file_metadata = dict(lazy_array.metadata)
    # Note: Some arrays have read-only metadata, so wrap in try/except
    if hasattr(arr, "metadata"):
        try:
            arr.metadata.pop("shape", None)
            arr.metadata.pop("num_timepoints", None)
            arr.metadata.pop("num_frames", None)
            arr.metadata.pop("nframes", None)
            arr.metadata.pop("n_frames", None)
        except (TypeError, AttributeError):
            # read-only metadata, skip cleaning
            pass

    from mbo_utilities.writer import imwrite

    # Ensure plane_dir exists
    plane_dir.mkdir(parents=True, exist_ok=True)

    imwrite(
        arr,  # Keep it lazy
        plane_dir,  # Write directly to plane directory
        ext=".bin",
        overwrite=True,
        register_z=False,
        planes=plane,
        channels=[channel] if channel is not None else None,
        output_name="data_raw.bin",
        roi=roi,
        metadata=defaults,
        num_frames=self.s2p.target_timepoints,
        fix_phase=self._s2p_fix_phase,
        use_fft=self._s2p_use_fft,
    )

    # Use run_plane instead of run_plane_bin - it handles initialization properly
    from lbm_suite2p_python import run_plane

    raw_file = plane_dir / "data_raw.bin"

    # Log detection settings for debugging
    self.logger.info(
        f"Suite2p settings - roidetect: {defaults.get('roidetect')}, force_detect: {self.s2p.force_detect}"
    )
    self.logger.info(
        f"Detection params - diameter: {defaults.get('diameter')}, sparse_mode: {defaults.get('sparse_mode')}"
    )

    try:
        run_plane(
            input_path=raw_file,
            save_path=plane_dir,
            ops=defaults,
            keep_raw=self.s2p.keep_raw,
            keep_reg=self.s2p.keep_reg,
            force_reg=self.s2p.force_reg,
            force_detect=self.s2p.force_detect,
            dff_window_size=self.s2p.dff_window_size,
            dff_percentile=self.s2p.dff_percentile,
            dff_smooth_window=self.s2p.dff_smooth_window
            if self.s2p.dff_smooth_window > 0
            else None,
        )
        self.logger.info(
            f"Suite2p processing complete for plane {current_z}, roi {arr_idx}. Results in {plane_dir}"
        )

        # Check if detection outputs were created
        stat_file = plane_dir / "stat.npy"
        if stat_file.exists():
            self.logger.info("Detection succeeded - stat.npy created")
        else:
            self.logger.warning(
                "Detection did not run - stat.npy not found. Check Suite2p output logs."
            )
    except Exception as e:
        self.logger.exception(
            f"Suite2p processing failed for plane {current_z}, roi {arr_idx}: {e}"
        )
        import traceback

        traceback.print_exc()


#
# def _run_plane_worker(
#     source_file,
#     arr_idx,
#     plane_num,
#     base_out,
#     roi,
#     num_frames,
#     user_ops,
#     s2p_settings,
# ):
#     """
#     Worker function for processing a single plane in a separate process.
#     This function is module-level (not a method) so it can be pickled for multiprocessing.
#     """
#     try:
#         # Import here to avoid issues with multiprocessing pickling
#         from pathlib import Path
#         import numpy as np
#         from mbo_utilities.lazy_array import imread, imwrite
#         from lbm_suite2p_python.run_lsp import run_plane
#
#         print(f"Process ROI={arr_idx}, Plane={plane_num} started (PID={os.getpid()})")
#
#         base_out = Path(base_out)
#
#         # Reload array (lazy loading)
#         print(f"Loading from source: {source_file}")
#         print(f"roi parameter: {roi}")
#         try:
#             arr = imread(source_file, roi=roi)
#             print(f"Loaded array shape: {arr.shape}, ndim: {arr.ndim}")
#             print(f"Array type: {type(arr)}")
#             print(f"Has num_rois: {hasattr(arr, 'num_rois')}")
#             if hasattr(arr, 'num_rois'):
#                 print(f"Array num_rois: {arr.num_rois}")
#         except Exception as e:
#             import traceback
#             print(f"ERROR in imread: {e}")
#             print(traceback.format_exc())
#             raise
#
#         # For 4D arrays, extract the specific plane before writing
#         # For 3D arrays, write directly (they're already single-plane)
#         if arr.ndim == 4:
#             # Extract the specific z-plane (0-indexed)
#             z_idx = plane_num - 1
#             if z_idx >= arr.shape[1]:
#                 raise IndexError(
#                     f"Plane {plane_num} requested but array only has {arr.shape[1]} planes. "
#                     f"Array shape: {arr.shape}"
#                 )
#             # Extract plane: arr[:, z_idx, :, :] gives us (T, H, W)
#             plane_data = arr[:, z_idx, :, :]
#             print(f"Extracted plane_data type: {type(plane_data)}, shape: {plane_data.shape}")
#             write_planes = None  # Don't specify planes for extracted 3D data
#         else:
#             # 3D array - already a single plane
#             plane_data = arr
#             print(f"Using 3D array directly, type: {type(plane_data)}, shape: {plane_data.shape}")
#             write_planes = None
#
#         print(f"plane_data has num_rois: {hasattr(plane_data, 'num_rois')}")
#
#         # Write functional channel using imwrite (lazy!)
#         print(f"Writing plane {plane_num} for ROI {arr_idx} to {base_out}")
#
#         # Update metadata with plane-specific info
#         plane_metadata = user_ops.copy()
#         plane_metadata.update({
#             "plane": plane_num,
#             "z_index": plane_num - 1,
#             "num_rois": arr.num_rois if hasattr(arr, 'num_rois') else 1,
#         })
#
#         imwrite(
#             plane_data,
#             base_out,
#             ext=".bin",
#             planes=write_planes,  # None for extracted plane data
#             num_frames=num_frames,
#             metadata=plane_metadata,
#             overwrite=True,
#         )
#
#         # Determine the plane directory that imwrite() created
#         if roi is None:
#             plane_dir = base_out / f"plane{plane_num:02d}_stitched"
#         else:
#             plane_dir = base_out / f"plane{plane_num:02d}_roi{roi}"
#
#         # Handle channel 2 if specified (only if path is valid and exists)
#         chan2_path = user_ops.get("chan2_file")
#         if chan2_path and Path(chan2_path).exists():
#             try:
#                 print(f"Loading channel 2 from: {chan2_path}")
#                 chan2_arr = imread(chan2_path, roi=roi)
#
#                 # Extract plane for 4D arrays
#                 if chan2_arr.ndim == 4:
#                     z_idx = plane_num - 1
#                     if z_idx >= chan2_arr.shape[1]:
#                         raise IndexError(
#                             f"Plane {plane_num} requested but channel 2 array only has {chan2_arr.shape[1]} planes"
#                         )
#                     chan2_plane_data = chan2_arr[:, z_idx, :, :]
#                 else:
#                     chan2_plane_data = chan2_arr
#
#                 chan2_metadata = user_ops.copy()
#                 chan2_metadata["structural"] = True
#                 chan2_metadata.update({
#                     "plane": plane_num,
#                     "z_index": plane_num - 1,
#                     "num_rois": chan2_arr.num_rois if hasattr(chan2_arr, 'num_rois') else 1,
#                 })
#
#                 imwrite(
#                     chan2_plane_data,
#                     base_out,
#                     ext=".bin",
#                     planes=None,  # Already extracted
#                     num_frames=num_frames,
#                     metadata=chan2_metadata,
#                     overwrite=True,
#                     structural=True,
#                 )
#             except Exception as e:
#                 print(f"WARNING: Could not load channel 2 data: {e}")
#
#         # Define file paths
#         raw_file = plane_dir / "data_raw.bin"
#         ops_path = plane_dir / "ops.npy"
#
#         # Load ops
#         ops_dict = np.load(ops_path, allow_pickle=True).item() if ops_path.exists() else {}
#
#         # Run Suite2p processing
#         print(f"Running Suite2p for plane {plane_num}, ROI {arr_idx}")
#         print(f"="*60)
#         print(f"Suite2p run_plane() parameters:")
#         print(f"  input_path: {raw_file}")
#         print(f"  save_path: {plane_dir}")
#         print(f"  input_path exists: {raw_file.exists()}")
#         print(f"  save_path exists: {plane_dir.exists()}")
#
#         # Only pass chan2_file if it's actually set (not empty string)
#         chan2 = user_ops.get("chan2_file")
#         if chan2 and Path(chan2).exists():
#             chan2_file_arg = chan2
#             print(f"  chan2_file: {chan2_file_arg}")
#         else:
#             chan2_file_arg = None
#             print(f"  chan2_file: None")
#
#         print(f"  keep_raw: {s2p_settings.get('keep_raw', False)}")
#         print(f"  keep_reg: {s2p_settings.get('keep_reg', True)}")
#         print(f"  force_reg: {s2p_settings.get('force_reg', False)}")
#         print(f"  force_detect: {s2p_settings.get('force_detect', False)}")
#         print(f"="*60)
#
#         print(f"CALLING run_plane() NOW...")
#         result_ops = run_plane(
#             input_path=raw_file,
#             save_path=plane_dir,
#             ops=ops_dict,
#             chan2_file=chan2_file_arg,
#             keep_raw=s2p_settings.get("keep_raw", False),
#             keep_reg=s2p_settings.get("keep_reg", True),
#             force_reg=s2p_settings.get("force_reg", False),
#             force_detect=s2p_settings.get("force_detect", False),
#             dff_window_size=s2p_settings.get("dff_window_size", 300),
#             dff_percentile=s2p_settings.get("dff_percentile", 20),
#             save_json=s2p_settings.get("save_json", False),
#         )
#         print(f"run_plane() RETURNED!")
#         print(f"  Return type: {type(result_ops)}")
#         print(f"  Return value: {result_ops}")
#         print(f"="*60)
#
#         print(f"Suite2p complete for plane {plane_num}, ROI {arr_idx}")
#         return (arr_idx, plane_num, "success", {"result_ops": str(result_ops)})
#
#     except ValueError as e:
#         print(f"WARNING: No cells found for plane {plane_num}, ROI {arr_idx}: {e}")
#         return (arr_idx, plane_num, "no_cells", {"error": str(e)})
#     except Exception as e:
#         print(f"ERROR: Suite2p failed for plane {plane_num}, ROI {arr_idx}: {e}")
#         import traceback
#         return (arr_idx, plane_num, "error", {"error": str(e), "traceback": traceback.format_exc()})
#
#
# def run_process(self):
#     """Runs the selected processing pipeline using parallel processing."""
#     print(f"DEBUG: run_process called, pipeline={self._current_pipeline}")
#
#     if self._current_pipeline != "suite2p":
#         if self._current_pipeline == "masknmf":
#             self.logger.info("Running MaskNMF pipeline (not yet implemented).")
#         else:
#             self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
#         return
#
#     print(f"DEBUG: About to check HAS_LSP={HAS_LSP}")
#     self.logger.info(f"Running Suite2p pipeline with settings: {self.s2p}")
#     if not HAS_LSP:
#         self.logger.warning(
#             "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline. "
#             "`uv pip install lbm_suite2p_python`",
#         )
#         self._install_error = True
#         return
#
#     if self._install_error:
#         return
#
#     from mbo_utilities.file_io import load_last_savedir, save_last_savedir
#
#     # Prepare tasks for all selected planes and ROIs
#     tasks = []
#
#     # Determine if we have a single 4D array or multiple 3D arrays
#     data_arrays = (
#         self.image_widget.data
#         if isinstance(self.image_widget.data, list)
#         else [self.image_widget.data]
#     )
#     first_array = data_arrays[0] if len(data_arrays) > 0 else None
#
#     # Case 1: Single 4D array (T, Z, H, W) - one file with multiple planes
#     # Case 2: Multiple 3D arrays (T, H, W) - multiple files, one plane each
#     is_single_4d = (
#         first_array is not None
#         and first_array.ndim == 4
#         and len(data_arrays) == 1
#     )
#
#     if is_single_4d:
#         # Single 4D array: loop over selected planes only
#         arr = data_arrays[0]
#         # Pass all files if it's a multi-file volume, or single file if it's a merged volume
#         source_file = self.fpath
#         roi = None  # No multi-ROI for single 4D case
#
#         base_out = Path(self._saveas_outdir or load_last_savedir())
#         base_out.mkdir(exist_ok=True)
#
#         # Build metadata and ops dict
#         user_ops = {}
#         if hasattr(self, "s2p"):
#             try:
#                 user_ops = (
#                     vars(self.s2p).copy()
#                     if hasattr(self.s2p, "__dict__")
#                     else dict(self.s2p)
#                 )
#             except Exception as e:
#                 self.logger.warning(f"Could not merge Suite2p params: {e}")
#
#         # Determine num_frames
#         num_frames = None
#         if user_ops.get("frames_include", -1) > 0:
#             num_frames = user_ops["frames_include"]
#
#         # Create tasks for each selected plane
#         for plane_num in self._selected_planes:
#             # Update metadata for this specific plane
#             task_ops = user_ops.copy()
#             task_ops.update({
#                 "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "original_file": str(source_file),
#                 "roi_index": 0,
#                 "mroi": roi,
#                 "roi": roi,
#                 "z_index": plane_num - 1,  # 0-indexed
#                 "plane": plane_num,  # 1-indexed
#                 "fs": arr.metadata.get("frame_rate", 15.0),
#                 "dx": arr.metadata.get("pixel_size_xy", 1.0),
#                 "dz": arr.metadata.get("z_step", 1.0),
#             })
#
#             # Extract s2p settings as dict for pickling
#             s2p_dict = self.s2p.to_dict() if hasattr(self.s2p, "to_dict") else vars(self.s2p)
#
#             tasks.append({
#                 "source_file": str(source_file),
#                 "arr_idx": 0,
#                 "plane_num": plane_num,
#                 "base_out": str(base_out),
#                 "roi": roi,
#                 "num_frames": num_frames,
#                 "user_ops": task_ops,
#                 "s2p_settings": s2p_dict,
#             })
#     else:
#         # Multiple 3D arrays: each array is already a single plane/ROI
#         for i, arr in enumerate(data_arrays):
#             # Determine source file
#             if isinstance(self.fpath, list):
#                 source_file = self.fpath[i]
#             else:
#                 source_file = self.fpath
#
#             # Determine ROI
#             roi = self.image_widget.data[0].roi
#             # if self.num_rois > 1 and i < self.num_rois:
#             #     roi = i + 1
#             # else:
#             #     roi = None
#
#             # For 3D arrays, plane_num should be derived from array index or metadata
#             # Use i+1 as plane_num (1-indexed) if planes are selected
#             if self._selected_planes and (i + 1) in self._selected_planes:
#                 plane_num = i + 1
#             elif not self._selected_planes:
#                 # If no planes selected, skip
#                 continue
#             else:
#                 # This array's plane is not in selected_planes
#                 continue
#
#             # Output base directory
#             base_out = Path(self._saveas_outdir or load_last_savedir())
#             base_out.mkdir(exist_ok=True)
#
#             # Build metadata and ops dict
#             user_ops = {}
#             if hasattr(self, "s2p"):
#                 try:
#                     user_ops = (
#                         vars(self.s2p).copy()
#                         if hasattr(self.s2p, "__dict__")
#                         else dict(self.s2p)
#                     )
#                 except Exception as e:
#                     self.logger.warning(f"Could not merge Suite2p params: {e}")
#
#             # Determine num_frames
#             num_frames = None
#             if user_ops.get("frames_include", -1) > 0:
#                 num_frames = user_ops["frames_include"]
#
#             # Update metadata for this specific plane
#             task_ops = user_ops.copy()
#             task_ops.update({
#                 "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 "original_file": str(source_file),
#                 "roi_index": i,
#                 "mroi": roi,
#                 "roi": roi,
#                 "z_index": plane_num - 1,  # 0-indexed
#                 "plane": plane_num,  # 1-indexed
#                 "fs": arr.metadata.get("frame_rate", 15.0),
#                 "dx": arr.metadata.get("pixel_size_xy", 1.0),
#                 "dz": arr.metadata.get("z_step", 1.0),
#             })
#
#             # Extract s2p settings as dict for pickling
#             s2p_dict = self.s2p.to_dict() if hasattr(self.s2p, "to_dict") else vars(self.s2p)
#
#             tasks.append({
#                 "source_file": str(source_file),
#                 "arr_idx": i,
#                 "plane_num": plane_num,
#                 "base_out": str(base_out),
#                 "roi": roi,
#                 "num_frames": num_frames,
#                 "user_ops": task_ops,
#                 "s2p_settings": s2p_dict,
#             })
#
#     print(f"DEBUG: Created {len(tasks)} tasks")
#     for i, task in enumerate(tasks):
#         print(f"  Task {i}: plane={task['plane_num']}, arr_idx={task['arr_idx']}, source={task['source_file']}")
#
#     if not tasks:
#         self.logger.warning("No planes selected for processing.")
#         return
#
#     # Determine optimal number of workers
#     # Use min of: number of tasks, CPU count, or 4 (to avoid memory issues)
#     max_workers = min(len(tasks), os.cpu_count() or 4, 4)
#
#     self.logger.info(f"Starting parallel processing of {len(tasks)} tasks with {max_workers} workers...")
#
#     # Run tasks in parallel using ProcessPoolExecutor
#     completed_count = 0
#     error_count = 0
#     no_cells_count = 0
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # Submit all tasks
#         future_to_task = {
#             executor.submit(_run_plane_worker, **task): task
#             for task in tasks
#         }
#
#         # Process results as they complete
#         for future in as_completed(future_to_task):
#             task = future_to_task[future]
#             try:
#                 arr_idx, plane_num, status, info = future.result()
#
#                 if status == "success":
#                     completed_count += 1
#                     self.logger.info(
#                         f"[{completed_count}/{len(tasks)}] ✓ Plane {plane_num}, ROI {arr_idx} complete"
#                     )
#                     # Save the last successful directory
#                     if "result_ops" in info:
#                         save_last_savedir(Path(info["result_ops"]).parent)
#
#                 elif status == "no_cells":
#                     no_cells_count += 1
#                     self.logger.warning(
#                         f"[{completed_count + error_count + no_cells_count}/{len(tasks)}] "
#                         f"WARNING: Plane {plane_num}, ROI {arr_idx}: No cells found"
#                     )
#                 else:  # error
#                     error_count += 1
#                     self.logger.error(
#                         f"[{completed_count + error_count + no_cells_count}/{len(tasks)}] "
#                         f"✗ Plane {plane_num}, ROI {arr_idx} failed: {info.get('error', 'Unknown error')}"
#                     )
#
#             except Exception as e:
#                 error_count += 1
#                 self.logger.error(
#                     f"Task for plane {task['plane_num']}, ROI {task['arr_idx']} "
#                     f"raised exception: {e}"
#                 )
#
#     # Final summary
#     self.logger.info(
#         f"Processing complete: {completed_count} succeeded, "
#         f"{no_cells_count} had no cells, {error_count} failed"
#     )
