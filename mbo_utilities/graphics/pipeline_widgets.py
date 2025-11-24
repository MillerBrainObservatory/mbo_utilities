import pathlib
import threading
from pathlib import Path
import time
from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd, hello_imgui

from mbo_utilities.graphics._widgets import set_tooltip, compact_header

try:
    from lbm_suite2p_python.run_lsp import run_plane, run_plane_bin

    HAS_LSP = True
except ImportError as e:
    print(f"Error importing lbm_suite2p_python: \n {e}")
    HAS_LSP = False
    run_plane = None


USER_PIPELINES = ["suite2p"]


@dataclass
class Suite2pSettings:
    """
    Suite2p pipeline configuration settings.
    Organized by functional sections matching Suite2p documentation.
    Defaults are optimized for LBM datasets based on LBM-Suite2p-Python.
    """

    # main settings
    functional_chan: int = 1  # channel for functional ROI extraction (1-based)
    tau: float = 1.3  # Timescale of sensor (LBM default for GCaMP6m-like)
    frames_include: int = -1
    target_frames: int = -1
    # multiplane_parallel: bool = False  # Run pipeline on server
    # ignore_flyback: list = field(default_factory=list)  # Planes to ignore

    # processing control
    keep_raw: bool = False  # keep raw binary (data_raw.bin) after processing
    keep_reg: bool = True  # Keep registered binary (data.bin) after processing
    force_reg: bool = False  # Force re-registration even if already done
    force_detect: bool = False  # Force ROI detection even if stat.npy exists
    dff_window_size: int = 300  # Frames for rolling percentile baseline in ΔF/F
    dff_percentile: int = 20  # Percentile for baseline F₀ estimation

    # output settings
    preclassify: float = 0.0  # apply classifier before extraction (0.0 = keep all)
    save_nwb: bool = False  # Save output as NWB file
    save_mat: bool = False  # Save results in Fall.mat
    save_json: bool = False  # Save ops as JSON in addition to .npy
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

    # --- 1P Registration ---
    do_1Preg: bool = False  # Perform 1P-specific registration
    spatial_hp_reg: int = 42  # Window for spatial high-pass filtering (1P)
    pre_smooth: float = 0.0  # Gaussian smoothing before high-pass (1P)
    spatial_taper: float = 40.0  # Pixels to ignore on edges (1P)

    # --- Non-rigid Registration ---
    nonrigid: bool = True  # Perform non-rigid registration
    block_size: list = (
        None  # Block size for non-rigid (default [128, 128], power of 2/3)
    )
    snr_thresh: float = 1.2  # Phase correlation peak threshold (1.5 for 1P)
    maxregshiftNR: float = 5.0  # Max block shift relative to rigid shift

    # roi detection settings
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
    spatial_hp_cp: float = 0.5  # High-pass filtering strength for Cellpose
    pretrained_model: str = "cpsam"  # Cellpose model path or type

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
        """Save settings to a JSON file."""
        np.save(filepath, self.to_dict(), allow_pickle=True)


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


def draw_section_suite2p(self):
    """Draw Suite2p configuration UI with collapsible sections and proper styling."""
    imgui.spacing()

    # Consistent input width matching pipeline selector
    INPUT_WIDTH = 120

    # Set proper padding and spacing
    imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 4))
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 2))

    imgui.spacing()
    imgui.separator_text("Processing Controls")
    imgui.spacing()

    # Save path display with proper text wrapping
    imgui.text("Save path:")
    imgui.same_line()

    # Flash animation logic for "(not set)" text
    text_color = imgui.ImVec4(0.6, 0.8, 1.0, 1.0)  # Default cyan color
    if not self._saveas_outdir and self._s2p_savepath_flash_start is not None:
        elapsed = time.time() - self._s2p_savepath_flash_start
        flash_duration = 0.3  # Duration of each flash in seconds
        total_flashes = 4

        if elapsed < total_flashes * flash_duration:
            # Determine if we should show red or cyan
            current_flash = int(elapsed / flash_duration)
            if current_flash % 2 == 0:  # Even flashes = red
                text_color = imgui.ImVec4(1.0, 0.2, 0.2, 1.0)  # Red
        else:
            # Animation finished, reset
            self._s2p_savepath_flash_start = None

    # Display path with wrapping to prevent clipping
    display_path = self._saveas_outdir if self._saveas_outdir else "(not set)"
    imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
    imgui.text_colored(text_color, display_path)
    imgui.pop_text_wrap_pos()

    # Browse button
    if imgui.button("Browse##savepath"):
        home = pathlib.Path().home()
        res = pfd.select_folder(str(home))
        if res:
            self._saveas_outdir = res.result()

    # Get max frames from data
    # iw-array API: data is an ImageWidgetProperty indexer, access with data[0]
    if hasattr(self, "image_widget") and self.image_widget.data:
        try:
            first_array = self.image_widget.data[0]
            max_frames = first_array.shape[0]
        except (IndexError, AttributeError):
            max_frames = 1000
    else:
        max_frames = 1000

    # Frames to process slider
    imgui.spacing()
    # Initialize target_frames only once, or when max_frames changes
    if not hasattr(self, '_frames_initialized'):
        self._frames_initialized = True
        self._last_max_frames = max_frames
        if self.s2p.target_frames == -1:
            self.s2p.target_frames = max_frames
    elif hasattr(self, '_last_max_frames') and self._last_max_frames != max_frames:
        # Max frames changed (different data loaded), reset to new max
        self._last_max_frames = max_frames
        self.s2p.target_frames = max_frames

    # frames input with slider
    imgui.set_next_item_width(INPUT_WIDTH)
    changed, new_value = imgui.input_int("##frames_input", self.s2p.target_frames, step=1, step_fast=100)
    if changed:
        self.s2p.target_frames = max(1, min(new_value, max_frames))
    imgui.same_line()
    imgui.text("Frames")
    set_tooltip(
        f"Number of frames to process (1-{max_frames}). "
        "Use arrows or type exact value. Useful for testing on subsets."
    )

    imgui.set_next_item_width(INPUT_WIDTH)
    slider_changed, slider_value = imgui.slider_int(
        "##frames_slider", self.s2p.target_frames, 1, max_frames
    )
    if slider_changed:
        self.s2p.target_frames = slider_value

    # Get current z index and total planes from image widget
    # iw-array API: use indices property with named dimension access
    names = self.image_widget._slider_dim_names or ()
    current_z = self.image_widget.indices["z"] if "z" in names else 0
    current_plane = current_z + 1  # Convert 0-indexed to 1-indexed

    # Get number of planes from data shape
    try:
        first_array = self.image_widget.data[0]
        if first_array.ndim == 4:
            num_planes = first_array.shape[1]  # (T, Z, H, W)
        else:
            num_planes = 1
    except (IndexError, AttributeError):
        num_planes = 1

    # Initialize multi-z state
    if not hasattr(self, "_selected_planes"):
        self._selected_planes = {current_plane}
    if not hasattr(self, "_show_plane_popup"):
        self._show_plane_popup = False
    if not hasattr(self, "_parallel_processing"):
        self._parallel_processing = False
    if not hasattr(self, "_max_parallel_jobs"):
        self._max_parallel_jobs = 2

    imgui.spacing()
    imgui.separator_text("Plane Selection")

    # Check if multi-z mode (more than just current plane selected)
    is_multi_z = len(self._selected_planes) > 1 or (
        len(self._selected_planes) == 1 and current_plane not in self._selected_planes
    )

    if num_planes > 1:
        # Show selected planes summary
        if is_multi_z:
            selected_str = ", ".join(str(p) for p in sorted(self._selected_planes))
            imgui.text(f"Selected: {selected_str}")
            # Clear button on next line, only when multiple planes selected
            if len(self._selected_planes) > 1:
                # Match Browse button size
                if imgui.button("Clear Selection", imgui.ImVec2(100, 0)):
                    self._selected_planes = {current_plane}
        else:
            imgui.text(f"Current plane: {current_plane}")

        # Smaller rounded button for plane selection
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 8.0)
        if imgui.button("Z-Planes...", imgui.ImVec2(80, 0)):
            self._show_plane_popup = True
        imgui.pop_style_var()
        set_tooltip("Click to select which z-planes to process")
    else:
        self._selected_planes = {1}
        imgui.text("Single plane data")

    # Plane selection popup
    if self._show_plane_popup:
        imgui.open_popup("Select Z-Planes##popup")

    # Calculate popup size to fit all planes
    popup_height = 130 + num_planes * 24
    popup_height = max(150, min(400, popup_height))
    imgui.set_next_window_size(imgui.ImVec2(220, popup_height), imgui.Cond_.appearing)

    if imgui.begin_popup_modal("Select Z-Planes##popup", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Select planes to process:")
        imgui.spacing()

        if imgui.button("All"):
            self._selected_planes = set(range(1, num_planes + 1))
        imgui.same_line()
        if imgui.button("None"):
            self._selected_planes = set()
        imgui.same_line()
        if imgui.button("Current"):
            self._selected_planes = {current_plane}

        imgui.separator()
        imgui.spacing()

        for i in range(num_planes):
            plane_num = i + 1
            checked = plane_num in self._selected_planes
            label = f"Plane {plane_num}"
            if plane_num == current_plane:
                label += " (current)"
            changed, checked = imgui.checkbox(label, checked)
            if changed:
                if checked:
                    self._selected_planes.add(plane_num)
                else:
                    self._selected_planes.discard(plane_num)

        imgui.spacing()
        imgui.separator()

        if imgui.button("Done", imgui.ImVec2(-1, 0)):
            self._show_plane_popup = False
            imgui.close_current_popup()

        imgui.end_popup()
    else:
        self._show_plane_popup = False

    # Parallel processing options (only show when multiple planes selected)
    if len(self._selected_planes) > 1:
        imgui.spacing()
        imgui.separator_text("Multi-Plane Processing")

        _, self._parallel_processing = imgui.checkbox(
            "Parallel Processing", self._parallel_processing
        )
        set_tooltip(
            "Process multiple planes simultaneously. Faster but uses more memory.\n"
            "Sequential processing is safer for large datasets."
        )

        if self._parallel_processing:
            imgui.set_next_item_width(hello_imgui.em_size(5))
            _, self._max_parallel_jobs = imgui.input_int(
                "Max parallel jobs", self._max_parallel_jobs, step=1, step_fast=2
            )
            self._max_parallel_jobs = max(1, min(self._max_parallel_jobs, len(self._selected_planes)))
            set_tooltip("Maximum number of planes to process simultaneously")

            # Memory warning estimate
            try:
                arr = self.image_widget.data[0]
                # Estimate memory per plane: shape * dtype size * 2 (for processing overhead)
                plane_shape = arr.shape
                if len(plane_shape) == 4:  # T, Z, H, W
                    frames_per_plane = plane_shape[0] * plane_shape[2] * plane_shape[3]
                else:  # T, H, W
                    frames_per_plane = plane_shape[0] * plane_shape[1] * plane_shape[2]
                bytes_per_element = 2  # assume uint16
                mem_per_plane_gb = (frames_per_plane * bytes_per_element) / (1024**3)
                total_mem_gb = mem_per_plane_gb * self._max_parallel_jobs * 2  # 2x for processing overhead

                if total_mem_gb > 16:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.3, 0.3, 1.0),
                        f"Warning: ~{total_mem_gb:.1f} GB RAM needed"
                    )
                elif total_mem_gb > 8:
                    imgui.text_colored(
                        imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
                        f"Est. memory: ~{total_mem_gb:.1f} GB"
                    )
                else:
                    imgui.text(f"Est. memory: ~{total_mem_gb:.1f} GB")
            except Exception:
                pass  # Skip memory estimate if we can't calculate

    imgui.spacing()
    imgui.separator_text("Pipeline")

    imgui.push_item_width(100)
    current_display_idx = USER_PIPELINES.index(self._current_pipeline)
    changed, selected_idx = imgui.combo("##Pipeline", current_display_idx, USER_PIPELINES)
    imgui.pop_item_width()

    if changed:
        self._current_pipeline = USER_PIPELINES[selected_idx]
    set_tooltip("Select a processing pipeline to configure.")

    # Processing Options (pipeline-specific)
    imgui.spacing()
    imgui.separator_text("Suite2p Options")

    _, self.s2p.keep_raw = imgui.checkbox("Keep Raw Binary", self.s2p.keep_raw)
    set_tooltip("Keep data_raw.bin after processing (uses disk space)")

    _, self.s2p.keep_reg = imgui.checkbox("Keep Registered Binary", self.s2p.keep_reg)
    set_tooltip("Keep data.bin after processing (useful for QC)")

    _, self.s2p.force_reg = imgui.checkbox("Force Re-registration", self.s2p.force_reg)
    set_tooltip("Force re-registration even if already processed")

    _, self.s2p.force_detect = imgui.checkbox("Force Re-detection", self.s2p.force_detect)
    set_tooltip("Force ROI detection even if stat.npy exists")

    imgui.spacing()
    imgui.set_next_item_width(INPUT_WIDTH)
    _, self.s2p.dff_window_size = imgui.input_int(
        "ΔF/F Window", self.s2p.dff_window_size
    )
    set_tooltip("Frames for rolling percentile baseline in ΔF/F (default: 300)")

    imgui.set_next_item_width(INPUT_WIDTH)
    _, self.s2p.dff_percentile = imgui.input_int(
        "ΔF/F Percentile", self.s2p.dff_percentile
    )
    set_tooltip("Percentile for baseline F₀ estimation (default: 20)")

    _, self.s2p.save_json = imgui.checkbox("Save JSON ops", self.s2p.save_json)
    set_tooltip("Save ops as JSON in addition to .npy")

    imgui.spacing()
    # Green Run button, smaller size
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.2, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.8, 0.3, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.6, 0.1, 1.0))
    button_clicked = imgui.button("Run Suite2p", imgui.ImVec2(100, 0))
    imgui.pop_style_color(3)

    if button_clicked:
        print("Run button clicked")
        print(f"Save path: {self._saveas_outdir}")
        print(f"Selected planes: {self._selected_planes}")
        print(f"Image widget data: {type(self.image_widget.data)}")

        # Validate save path is set
        if not self._saveas_outdir:
            self.logger.warning("Please select a save path before running.")
            # Trigger flash animation
            self._s2p_savepath_flash_start = time.time()
            self._s2p_savepath_flash_count = 0
            self._s2p_show_savepath_popup = True
        else:
            self.logger.info(f"Running Suite2p pipeline on {len(self._selected_planes)} planes...")
            run_process(self)
            self.logger.info("Suite2p processing submitted (running in background).")

    # Popup for missing save path
    if self._s2p_show_savepath_popup:
        imgui.open_popup("Missing Save Path")
        self._s2p_show_savepath_popup = False

    if imgui.begin_popup_modal("Missing Save Path")[0]:
        imgui.text("Please select a save path before running.")
        imgui.spacing()
        if imgui.button("OK", imgui.ImVec2(120, 0)):
            imgui.close_current_popup()
        imgui.end_popup()

    if self._install_error:
        imgui.same_line()
        if self._show_red_text:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.0, 0.0, 1.0),
                "Error: lbm_suite2p_python is not installed.",
            )
        if self._show_green_text:
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.0, 0.0, 1.0),
                "lbm_suite2p_python install success.",
            )
        if self._show_install_button:
            if imgui.button("Install"):
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

    if compact_header("Main Settings", default_open=True):
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.functional_chan = imgui.input_int(
            "Functional Channel", self.s2p.functional_chan
        )
        set_tooltip("Channel used for functional ROI extraction (1-based).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
        set_tooltip(
            "Sensor decay timescale: GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
        )

    if compact_header("Registration Settings", default_open=True):
        # Main registration toggle
        _, self.s2p.do_registration = imgui.checkbox(
            "Enable Registration", self.s2p.do_registration
        )
        set_tooltip("Run motion registration on the movie.")

        # Disable all registration settings if do_registration is False
        imgui.begin_disabled(not self.s2p.do_registration)

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.align_by_chan = imgui.input_int(
            "Align by Channel", self.s2p.align_by_chan
        )
        set_tooltip("Channel index used for alignment (1-based).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nimg_init = imgui.input_int(
            "Initial Frames", self.s2p.nimg_init
        )
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
        _, self.s2p.reg_tif = imgui.checkbox(
            "Export Registered TIFF", self.s2p.reg_tif
        )
        set_tooltip("Export registered movie as TIFF files.")
        _, self.s2p.reg_tif_chan2 = imgui.checkbox(
            "Export Chan2 TIFF", self.s2p.reg_tif_chan2
        )
        set_tooltip("Export registered TIFFs for channel 2.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.subpixel = imgui.input_int(
            "Subpixel Precision", self.s2p.subpixel
        )
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
        _, self.s2p.force_refImg = imgui.checkbox(
            "Force refImg", self.s2p.force_refImg
        )
        set_tooltip("Use stored reference image instead of recomputing.")
        _, self.s2p.pad_fft = imgui.checkbox("Pad FFT", self.s2p.pad_fft)
        set_tooltip("Pad image for FFT registration to reduce edge artifacts.")

        imgui.spacing()
        imgui.text("Channel 2 File:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
        imgui.text(self.s2p.chan2_file if self.s2p.chan2_file else "(none)")
        imgui.pop_text_wrap_pos()
        if imgui.button("Browse##chan2"):
            home = pathlib.Path().home()
            res = pfd.open_file("Select channel 2 file", str(home))
            if res and res.result():
                self.s2p.chan2_file = res.result()[0]
        set_tooltip("Path to channel 2 binary file for cross-channel registration.")

        # --- 1P Registration Collapsible Subsection ---
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

        # --- Non-rigid Registration Collapsible Subsection ---
        if imgui.tree_node("Non-rigid Registration"):
            _, self.s2p.nonrigid = imgui.checkbox(
                "Enable Non-rigid", self.s2p.nonrigid
            )
            set_tooltip(
                "Split FOV into blocks and compute registration offsets per block."
            )

            imgui.begin_disabled(not self.s2p.nonrigid)

            # Block size as two separate inputs
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
            set_tooltip(
                "Phase correlation peak threshold (1.5 recommended for 1P)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.maxregshiftNR = imgui.input_float(
                "Max NR Shift", self.s2p.maxregshiftNR
            )
            set_tooltip("Max pixel shift of block relative to rigid shift.")

            imgui.end_disabled()
            imgui.tree_pop()

        imgui.end_disabled()  # End registration disabled block

    if compact_header("ROI Detection Settings", default_open=True):
        _, self.s2p.roidetect = imgui.checkbox(
            "Enable ROI Detection", self.s2p.roidetect
        )
        set_tooltip("Run ROI detection and extraction.")

        imgui.begin_disabled(not self.s2p.roidetect)

        _, self.s2p.sparse_mode = imgui.checkbox(
            "Sparse Mode", self.s2p.sparse_mode
        )
        set_tooltip("Use sparse detection (recommended for soma).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.spatial_scale = imgui.input_int(
            "Spatial Scale", self.s2p.spatial_scale
        )
        set_tooltip(
            "ROI size scale: 0=auto, 1=6-pixel cells (LBM default), 2=medium, 3=large, 4=very large"
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
        set_tooltip("Spatial high-pass filter size before ROI detection.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.max_overlap = imgui.input_float(
            "Max Overlap", self.s2p.max_overlap
        )
        set_tooltip("Maximum allowed fraction of overlapping ROI pixels.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.high_pass = imgui.input_int(
            "High-Pass Window", self.s2p.high_pass
        )
        set_tooltip("Running mean subtraction window (frames).")
        _, self.s2p.smooth_masks = imgui.checkbox(
            "Smooth Masks", self.s2p.smooth_masks
        )
        set_tooltip("Smooth masks in the final ROI detection pass.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.max_iterations = imgui.input_int(
            "Max Iterations", self.s2p.max_iterations
        )
        set_tooltip("Maximum number of cell-detection iterations.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
        set_tooltip("Number of frames binned for ROI detection.")
        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip("Denoise binned movie before ROI detection.")

        imgui.end_disabled()

    if compact_header("Cellpose / Anatomical Detection"):
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.anatomical_only = imgui.input_int(
            "Anatomical Only", self.s2p.anatomical_only
        )
        set_tooltip(
            "0=disabled; 1=mean, 2=max, 3=enhanced (LBM default), 4=correlation"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.diameter = imgui.input_int("Cell Diameter", self.s2p.diameter)
        set_tooltip("Expected cell diameter in pixels (6 = LBM default for ~6μm cells)")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.cellprob_threshold = imgui.input_float(
            "CellProb Threshold", self.s2p.cellprob_threshold
        )
        set_tooltip("Cellpose detection threshold (-6 = LBM default for permissive detection)")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.flow_threshold = imgui.input_float(
            "Flow Threshold", self.s2p.flow_threshold
        )
        set_tooltip("Cellpose flow field threshold (0 = standard)")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.spatial_hp_cp = imgui.input_float(
            "Spatial HP (Cellpose)", self.s2p.spatial_hp_cp
        )
        set_tooltip("High-pass filtering strength (0.5 = LBM default)")
        imgui.text("Pretrained Model:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
        imgui.text(
            self.s2p.pretrained_model if self.s2p.pretrained_model else "cyto"
        )
        imgui.pop_text_wrap_pos()
        set_tooltip("Cellpose model name or custom path (e.g., 'cyto').")

    if compact_header("Classification Settings"):
        _, self.s2p.soma_crop = imgui.checkbox("Soma Crop", self.s2p.soma_crop)
        set_tooltip("Crop dendrites for soma classification.")
        _, self.s2p.use_builtin_classifier = imgui.checkbox(
            "Use Built-in Classifier", self.s2p.use_builtin_classifier
        )
        set_tooltip("Use Suite2p's built-in ROI classifier.")
        imgui.text("Classifier Path:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
        imgui.text(
            self.s2p.classifier_path if self.s2p.classifier_path else "(none)"
        )
        imgui.pop_text_wrap_pos()
        set_tooltip("Path to external classifier if not using built-in.")

    if compact_header("Output Settings"):
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.preclassify = imgui.input_float(
            "Preclassify Threshold", self.s2p.preclassify
        )
        set_tooltip("Probability threshold to apply classifier before extraction.")
        _, self.s2p.save_nwb = imgui.checkbox("Save NWB", self.s2p.save_nwb)
        set_tooltip("Export processed data to NWB format.")
        _, self.s2p.save_mat = imgui.checkbox("Save MATLAB File", self.s2p.save_mat)
        set_tooltip("Export results to Fall.mat for MATLAB analysis.")
        _, self.s2p.combined = imgui.checkbox(
            "Combine Across Planes", self.s2p.combined
        )
        set_tooltip("Combine per-plane results into one GUI-loadable folder.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.aspect = imgui.input_float("Aspect Ratio", self.s2p.aspect)
        set_tooltip("um/pixel ratio X/Y for correct GUI aspect display.")
        _, self.s2p.report_time = imgui.checkbox(
            "Report Timing", self.s2p.report_time
        )
        set_tooltip("Return timing dictionary for each processing stage.")

    if compact_header("Signal Extraction Settings"):
        _, self.s2p.neuropil_extract = imgui.checkbox(
            "Extract Neuropil", self.s2p.neuropil_extract
        )
        set_tooltip("Extract neuropil signal for background correction.")
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

    if compact_header("Spike Deconvolution Settings"):
        _, self.s2p.spikedetect = imgui.checkbox(
            "Enable Spike Deconvolution", self.s2p.spikedetect
        )
        set_tooltip(
            "Detect spikes from neuropil-corrected and baseline-corrected traces."
        )

        imgui.begin_disabled(not self.s2p.spikedetect)

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

        imgui.end_disabled()

    if compact_header("Channel 2 Settings"):
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.chan2_thres = imgui.input_float(
            "Chan2 Detection Threshold", self.s2p.chan2_thres
        )
        set_tooltip("Threshold for calling ROI detected on channel 2.")

    # Pop style variables
    imgui.pop_style_var(2)  # Pop both style vars
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




def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline != "suite2p":
        if self._current_pipeline == "masknmf":
            self.logger.info("Running MaskNMF pipeline (not yet implemented).")
        else:
            self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
        return

    self.logger.info(f"Running Suite2p pipeline with settings: {self.s2p}")
    if not HAS_LSP:
        self.logger.warning("lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
                            "`uv pip install lbm_suite2p_python`",)
        self._install_error = True
        return

    if not self._install_error:
        # Get selected planes (1-indexed)
        selected_planes = getattr(self, "_selected_planes", None)
        if not selected_planes:
            # Fallback to current plane
            names = self.image_widget._slider_dim_names or ()
            current_z = self.image_widget.indices["z"] if "z" in names else 0
            selected_planes = {current_z + 1}

        self.logger.info(f"Running Suite2p pipeline on {len(selected_planes)} plane(s)...")

        # Build list of all (arr_idx, z_plane) pairs to process
        jobs = []
        for i, arr in enumerate(self.image_widget.data):
            for plane in sorted(selected_planes):
                jobs.append((i, plane - 1))  # Convert to 0-indexed

        # Check if parallel processing is enabled
        use_parallel = getattr(self, "_parallel_processing", False)
        max_jobs = getattr(self, "_max_parallel_jobs", 2)

        if use_parallel and len(jobs) > 1:
            # Parallel processing with limited concurrency
            from concurrent.futures import ThreadPoolExecutor

            def run_parallel():
                self.logger.info(f"Starting parallel processing with max {max_jobs} concurrent jobs...")
                with ThreadPoolExecutor(max_workers=max_jobs) as executor:
                    futures = {}
                    for job_idx, (arr_idx, z_plane) in enumerate(jobs):
                        future = executor.submit(run_plane_from_data, self, arr_idx, z_plane)
                        futures[future] = (z_plane, job_idx)

                    for future in futures:
                        z_plane, job_idx = futures[future]
                        try:
                            future.result()
                            self.logger.info(f"Plane {z_plane + 1} completed ({job_idx + 1}/{len(jobs)})")
                        except Exception as e:
                            self.logger.error(f"Error processing plane {z_plane + 1}: {e}")
                self.logger.info("Suite2p parallel processing complete.")

            threading.Thread(target=run_parallel, daemon=True).start()
        else:
            # Sequential processing in a single background thread
            def run_all_planes_sequential():
                for job_idx, (arr_idx, z_plane) in enumerate(jobs):
                    self.logger.info(f"Processing plane {z_plane + 1} ({job_idx + 1}/{len(jobs)})...")
                    try:
                        run_plane_from_data(self, arr_idx, z_plane)
                    except Exception as e:
                        self.logger.error(f"Error processing plane {z_plane + 1}: {e}")
                self.logger.info("Suite2p processing complete.")

            threading.Thread(target=run_all_planes_sequential, daemon=True).start()

def run_plane_from_data(self, arr_idx, z_plane=None):
    if not HAS_LSP:
        self.logger.error("lbm_suite2p_python is not installed.")
        self._install_error = True
        return

    arr = self.image_widget.data[arr_idx]
    # Use provided z_plane or fall back to current index
    if z_plane is not None:
        current_z = z_plane
    else:
        names = self.image_widget._slider_dim_names or ()
        current_z = self.image_widget.indices["z"] if "z" in names else 0

    base_out = Path(self._saveas_outdir) if getattr(self, "_saveas_outdir", None) else None
    if not base_out:
        from mbo_utilities.file_io import get_mbo_dirs, get_last_savedir_path
        # find last saved dir
        last_savedir = get_last_savedir_path()
        if last_savedir:
            base_out = Path(last_savedir)
        else:
            base_out = get_mbo_dirs()["data"]
    if not base_out.exists():
        base_out.mkdir(exist_ok=True)

    if len(self.image_widget.graphics) > 1:
        plane_dir = base_out / f"plane{current_z+1:02d}_roi{arr_idx+1:02d}"
        roi = arr_idx + 1
        plane = current_z + 1
    else:
        plane_dir = base_out / f"plane{current_z+1:02d}_stitched"
        roi = None
        plane = current_z + 1

    ops_path = plane_dir / "ops.npy"

    lazy_mdata = getattr(arr, "metadata", {}).copy()

    # Get dimensions without extracting - let imwrite handle extraction lazily
    # For 4D: shape is (T, Z, H, W), imwrite with planes=N will write that z-plane
    # For 3D: shape is (T, H, W), imwrite writes all frames
    Lx = arr.shape[-1]
    Ly = arr.shape[-2]

    # Extract only scalar metadata needed for Suite2p - do NOT pass shape arrays
    # that could confuse the pipeline when processing 4D data
    md = {
        "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_frames": self.s2p.target_frames,
        "nframes": self.s2p.target_frames,
        "n_frames": self.s2p.target_frames,
        "original_file": str(self.fpath),
        "roi_index": arr_idx,
        "z_index": current_z,
        "plane": plane,
        "Ly": Ly,
        "Lx": Lx,
        "fs": lazy_mdata.get("frame_rate", 15.0),
        "dx": lazy_mdata.get("pixel_size_xy", 1.0),
        "dz": lazy_mdata.get("z_step", 1.0),
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

    # Remove any shape-related keys that could confuse the pipeline
    # imwrite will set the correct shape based on what it actually writes
    # CRITICAL: Remove these AFTER updating with md, but BEFORE passing to imwrite
    for key in ['shape', 'num_frames', 'nframes', 'n_frames']:
        defaults.pop(key, None)

    # Also clean lazy_mdata to prevent shape contamination from arr.metadata
    lazy_mdata.pop('shape', None)
    lazy_mdata.pop('num_frames', None)
    lazy_mdata.pop('nframes', None)
    lazy_mdata.pop('n_frames', None)

    # CRITICAL: Also clean arr.metadata directly to prevent imwrite from getting
    # the full shape when it does file_metadata = dict(lazy_array.metadata)
    if hasattr(arr, 'metadata'):
        arr.metadata.pop('shape', None)
        arr.metadata.pop('num_frames', None)
        arr.metadata.pop('nframes', None)
        arr.metadata.pop('n_frames', None)

    from mbo_utilities.lazy_array import imwrite
    imwrite(
        arr,  # Keep it lazy
        base_out, #plane_dir,
        ext=".bin",
        overwrite=True,
        register_z=False,
        planes= plane,
        output_name="data_raw.bin",
        roi=roi,
        metadata=defaults,
        num_frames=self.s2p.target_frames,
    )

    # Use run_plane instead of run_plane_bin - it handles initialization properly
    from lbm_suite2p_python import run_plane
    raw_file = plane_dir / "data_raw.bin"

    try:
        result = run_plane(
            input_path=raw_file,
            save_path=plane_dir,
            ops=defaults,
            keep_raw=self.s2p.keep_raw,
            keep_reg=self.s2p.keep_reg,
            force_reg=self.s2p.force_reg,
            force_detect=self.s2p.force_detect,
            dff_window_size=self.s2p.dff_window_size,
            dff_percentile=self.s2p.dff_percentile,
        )
        self.logger.info(f"Suite2p processing complete for plane {current_z}, roi {arr_idx}. Results in {plane_dir}")
    except Exception as e:
        self.logger.error(f"Suite2p processing failed for plane {current_z}, roi {arr_idx}: {e}")
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
