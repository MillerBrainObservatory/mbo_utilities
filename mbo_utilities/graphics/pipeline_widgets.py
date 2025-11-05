import pathlib
from pathlib import Path
import threading
import time
from dataclasses import dataclass

import numpy as np

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd

from mbo_utilities.graphics._widgets import set_tooltip

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

    # ==================== Main Settings ====================
    # nplanes: int = 1  # Number of planes in each tiff
    functional_chan: int = 1  # Channel for functional ROI extraction (1-based)
    tau: float = 1.3  # Timescale of sensor (LBM default for GCaMP6m-like)
    frames_include: int = -1  # Only process this many frames (for testing)
    # multiplane_parallel: bool = False  # Run pipeline on server
    # ignore_flyback: list = field(default_factory=list)  # Planes to ignore

    # ==================== Processing Control ====================
    keep_raw: bool = False  # Keep raw binary (data_raw.bin) after processing
    keep_reg: bool = True  # Keep registered binary (data.bin) after processing
    force_reg: bool = False  # Force re-registration even if already done
    force_detect: bool = False  # Force ROI detection even if stat.npy exists
    dff_window_size: int = 300  # Frames for rolling percentile baseline in ΔF/F
    dff_percentile: int = 20  # Percentile for baseline F₀ estimation

    # ==================== Output Settings ====================
    preclassify: float = 0.0  # Apply classifier before extraction (0.0 = keep all)
    save_nwb: bool = False  # Save output as NWB file
    save_mat: bool = False  # Save results in Fall.mat
    save_json: bool = False  # Save ops as JSON in addition to .npy
    combined: bool = True  # Combine results across planes
    aspect: float = 1.0  # Ratio of um/pixels X to Y (for GUI only)
    report_time: bool = True  # Return timing dictionary

    # ==================== Registration Settings ====================
    do_registration: bool = True  # Whether to run registration
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

    # ==================== ROI Detection Settings ====================
    roidetect: bool = True  # Run ROI detection and extraction
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

    # ==================== Cellpose Detection Settings ====================
    # LBM-optimized defaults for Cellpose-based detection
    anatomical_only: int = 3  # Use enhanced mean image (LBM default)
    diameter: int = 6  # Expected cell diameter in pixels (LBM datasets)
    cellprob_threshold: float = -6.0  # More permissive detection threshold
    flow_threshold: float = 0.0  # Standard Cellpose flow threshold
    spatial_hp_cp: float = 0.5  # High-pass filtering strength for Cellpose
    pretrained_model: str = "cyto"  # Cellpose model path or type

    # ==================== Signal Extraction Settings ====================
    neuropil_extract: bool = True  # Extract neuropil signal
    allow_overlap: bool = False  # Extract from overlapping pixels
    min_neuropil_pixels: int = 350  # Min pixels for neuropil computation
    inner_neuropil_radius: int = 2  # Pixels between ROI and neuropil
    lam_percentile: int = 50  # Lambda percentile for neuropil exclusion

    # ==================== Spike Deconvolution Settings ====================
    spikedetect: bool = True  # Run spike deconvolution
    neucoeff: float = 0.7  # Neuropil coefficient for all ROIs
    baseline: str = "maximin"  # Baseline method (maximin/constant/constant_percentile)
    win_baseline: float = 60.0  # Window for maximin filter (seconds)
    sig_baseline: float = 10.0  # Gaussian filter width (seconds)
    prctile_baseline: float = 8.0  # Percentile for constant_percentile baseline

    # ==================== Classification Settings ====================
    soma_crop: bool = True  # Crop dendrites for classification stats
    use_builtin_classifier: bool = False  # Use built-in classifier
    classifier_path: str = ""  # Path to custom classifier

    # ==================== Channel 2 Settings ====================
    chan2_file: str = ""  # Path to channel 2 data file
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

    imgui.begin_group()
    imgui.dummy(imgui.ImVec2(0, 5))

    imgui.text_colored(
        imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Select a processing pipeline:"
    )

    current_display_idx = USER_PIPELINES.index(self._current_pipeline)
    changed, selected_idx = imgui.combo("Pipeline", current_display_idx, USER_PIPELINES)

    if changed:
        self._current_pipeline = USER_PIPELINES[selected_idx]
    set_tooltip("Select a processing pipeline to configure.")

    if self._current_pipeline == "suite2p":
        draw_section_suite2p(self)
    elif self._current_pipeline == "masknmf":
        imgui.text("MaskNMF pipeline not yet implemented.")
    imgui.spacing()
    imgui.end_group()


def draw_section_suite2p(self):
    """Draw Suite2p configuration UI with collapsible sections and proper styling."""
    imgui.spacing()

    # Use proper child window flags to prevent scrollbar issues
    child_flags = imgui.WindowFlags_.none

    with imgui_ctx.begin_child("##Processing", imgui.ImVec2(0, 0), child_flags):
        # Set proper padding and spacing
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 6))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))

        # Set widget width to be responsive
        avail_w = imgui.get_content_region_avail().x * 0.35
        imgui.push_item_width(avail_w)

        # ==================================================================
        # PROCESSING CONTROLS SECTION
        # ==================================================================
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
        if hasattr(self, "image_widget") and self.image_widget.data:
            data_arrays = (
                self.image_widget.data
                if isinstance(self.image_widget.data, list)
                else [self.image_widget.data]
            )
            if len(data_arrays) > 0:
                max_frames = data_arrays[0].shape[0]
            else:
                max_frames = 1000
        else:
            max_frames = 1000

        # Frames to process slider
        imgui.spacing()
        if self.s2p.frames_include == -1:
            # Initialize to max frames if set to -1
            self.s2p.frames_include = max_frames

        _, self.s2p.frames_include = imgui.slider_int(
            "Frames to process", self.s2p.frames_include, 1, max_frames
        )
        set_tooltip(
            f"Number of frames to process (1-{max_frames}). "
            "Useful for testing on a subset of data."
        )

        # Get number of planes from data
        if hasattr(self, "image_widget") and self.image_widget.data:
            data_arrays = (
                self.image_widget.data
                if isinstance(self.image_widget.data, list)
                else [self.image_widget.data]
            )
            if len(data_arrays) > 0:
                num_planes = data_arrays[0].shape[1] if data_arrays[0].ndim > 1 else 1
            else:
                num_planes = 1
        else:
            num_planes = 1

        # Initialize selected planes if not already set
        if not hasattr(self, "_selected_planes"):
            self._selected_planes = set(range(1, num_planes + 1))  # All selected by default

        # Plane selection UI
        imgui.spacing()
        imgui.separator_text("Plane Selection")
        imgui.text("Select planes to process:")

        # [All] and [None] buttons
        if imgui.button("All##planes"):
            self._selected_planes = set(range(1, num_planes + 1))
        imgui.same_line()
        if imgui.button("None##planes"):
            self._selected_planes = set()

        # Checkboxes for each plane (in rows of 5)
        for i in range(num_planes):
            plane_num = i + 1
            checked = plane_num in self._selected_planes
            changed, checked = imgui.checkbox(f"Plane {plane_num}", checked)
            if changed:
                if checked:
                    self._selected_planes.add(plane_num)
                else:
                    self._selected_planes.discard(plane_num)
            if (i + 1) % 5 != 0 and i < num_planes - 1:
                imgui.same_line()

        # Processing Control Options
        imgui.spacing()
        imgui.separator_text("Processing Options")

        _, self.s2p.keep_raw = imgui.checkbox("Keep Raw Binary", self.s2p.keep_raw)
        set_tooltip("Keep data_raw.bin after processing (uses disk space)")

        _, self.s2p.keep_reg = imgui.checkbox("Keep Registered Binary", self.s2p.keep_reg)
        set_tooltip("Keep data.bin after processing (useful for QC)")

        _, self.s2p.force_reg = imgui.checkbox("Force Re-registration", self.s2p.force_reg)
        set_tooltip("Force re-registration even if already processed")

        _, self.s2p.force_detect = imgui.checkbox("Force Re-detection", self.s2p.force_detect)
        set_tooltip("Force ROI detection even if stat.npy exists")

        imgui.spacing()
        _, self.s2p.dff_window_size = imgui.input_int(
            "ΔF/F Window (frames)", self.s2p.dff_window_size
        )
        set_tooltip("Frames for rolling percentile baseline in ΔF/F (default: 300)")

        _, self.s2p.dff_percentile = imgui.input_int(
            "ΔF/F Percentile", self.s2p.dff_percentile
        )
        set_tooltip("Percentile for baseline F₀ estimation (default: 20)")

        _, self.s2p.save_json = imgui.checkbox("Save JSON ops", self.s2p.save_json)
        set_tooltip("Save ops as JSON in addition to .npy")

        imgui.spacing()
        if imgui.button("Run Suite2p", imgui.ImVec2(150, 30)):
            print("Run button clicked")
            # Validate save path is set
            if not self._saveas_outdir:
                self.logger.warning("Please select a save path before running.")
                # Trigger flash animation
                self._s2p_savepath_flash_start = time.time()
                self._s2p_savepath_flash_count = 0
                self._s2p_show_savepath_popup = True
            else:
                self.logger.info("Running Suite2p pipeline...")
                run_process(self)
            self.logger.info("Suite2p pipeline completed.")

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

        # ==================================================================
        # MAIN SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        imgui.spacing()
        if imgui.collapsing_header("Main Settings", imgui.TreeNodeFlags_.default_open):
            imgui.indent()
            _, self.s2p.functional_chan = imgui.input_int(
                "Functional Channel", self.s2p.functional_chan
            )
            set_tooltip("Channel used for functional ROI extraction (1-based).")
            _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
            set_tooltip(
                "Sensor decay timescale: GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
            )
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # REGISTRATION SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header(
            "Registration Settings", imgui.TreeNodeFlags_.default_open
        ):
            imgui.indent()

            # Main registration toggle
            _, self.s2p.do_registration = imgui.checkbox(
                "Enable Registration", self.s2p.do_registration
            )
            set_tooltip("Run motion registration on the movie.")

            # Disable all registration settings if do_registration is False
            imgui.begin_disabled(not self.s2p.do_registration)

            imgui.spacing()
            _, self.s2p.align_by_chan = imgui.input_int(
                "Align by Channel", self.s2p.align_by_chan
            )
            set_tooltip("Channel index used for alignment (1-based).")
            _, self.s2p.nimg_init = imgui.input_int(
                "Initial Frames", self.s2p.nimg_init
            )
            set_tooltip("Number of frames used to build the reference image.")
            _, self.s2p.batch_size = imgui.input_int("Batch Size", self.s2p.batch_size)
            set_tooltip("Number of frames processed per registration batch.")
            _, self.s2p.maxregshift = imgui.input_float(
                "Max Shift Fraction", self.s2p.maxregshift
            )
            set_tooltip("Maximum allowed shift as a fraction of the image size.")
            _, self.s2p.smooth_sigma = imgui.input_float(
                "Smooth Sigma", self.s2p.smooth_sigma
            )
            set_tooltip("Gaussian smoothing sigma (pixels) before registration.")
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
            _, self.s2p.subpixel = imgui.input_int(
                "Subpixel Precision", self.s2p.subpixel
            )
            set_tooltip("Subpixel precision level (1/subpixel step).")
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
            imgui.spacing()
            if imgui.tree_node("1-Photon Registration"):
                _, self.s2p.do_1Preg = imgui.checkbox(
                    "Enable 1P Registration", self.s2p.do_1Preg
                )
                set_tooltip("Apply high-pass filtering and tapering for 1-photon data.")

                imgui.begin_disabled(not self.s2p.do_1Preg)
                _, self.s2p.spatial_hp_reg = imgui.input_int(
                    "Spatial HP Window", self.s2p.spatial_hp_reg
                )
                set_tooltip(
                    "Window size for spatial high-pass filtering before registration."
                )
                _, self.s2p.pre_smooth = imgui.input_float(
                    "Pre-smooth Sigma", self.s2p.pre_smooth
                )
                set_tooltip(
                    "Gaussian smoothing stddev before high-pass filtering (0=disabled)."
                )
                _, self.s2p.spatial_taper = imgui.input_float(
                    "Spatial Taper", self.s2p.spatial_taper
                )
                set_tooltip(
                    "Pixels to set to zero on edges (important for vignetted windows)."
                )
                imgui.end_disabled()

                imgui.tree_pop()

            # --- Non-rigid Registration Collapsible Subsection ---
            imgui.spacing()
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
                block_y_changed, block_y = imgui.input_int(
                    "Block Height", self.s2p.block_size[0]
                )
                set_tooltip(
                    "Block height for non-rigid registration (power of 2/3 recommended)."
                )
                block_x_changed, block_x = imgui.input_int(
                    "Block Width", self.s2p.block_size[1]
                )
                set_tooltip(
                    "Block width for non-rigid registration (power of 2/3 recommended)."
                )
                if block_y_changed or block_x_changed:
                    self.s2p.block_size = [block_y, block_x]

                _, self.s2p.snr_thresh = imgui.input_float(
                    "SNR Threshold", self.s2p.snr_thresh
                )
                set_tooltip(
                    "Phase correlation peak threshold (1.5 recommended for 1P)."
                )
                _, self.s2p.maxregshiftNR = imgui.input_float(
                    "Max NR Shift", self.s2p.maxregshiftNR
                )
                set_tooltip("Max pixel shift of block relative to rigid shift.")

                imgui.end_disabled()
                imgui.tree_pop()

            imgui.end_disabled()  # End registration disabled block
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # ROI DETECTION SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header(
            "ROI Detection Settings", imgui.TreeNodeFlags_.default_open
        ):
            imgui.indent()

            _, self.s2p.roidetect = imgui.checkbox(
                "Enable ROI Detection", self.s2p.roidetect
            )
            set_tooltip("Run ROI detection and extraction.")

            imgui.begin_disabled(not self.s2p.roidetect)

            imgui.spacing()
            _, self.s2p.sparse_mode = imgui.checkbox(
                "Sparse Mode", self.s2p.sparse_mode
            )
            set_tooltip("Use sparse detection (recommended for soma).")
            _, self.s2p.spatial_scale = imgui.input_int(
                "Spatial Scale", self.s2p.spatial_scale
            )
            set_tooltip(
                "ROI size scale: 0=auto, 1=6-pixel cells (LBM default), 2=medium, 3=large, 4=very large"
            )
            _, self.s2p.connected = imgui.checkbox("Connected ROIs", self.s2p.connected)
            set_tooltip("Require ROIs to be connected regions.")
            _, self.s2p.threshold_scaling = imgui.input_float(
                "Threshold Scaling", self.s2p.threshold_scaling
            )
            set_tooltip("Scale ROI detection threshold; higher = fewer ROIs.")
            _, self.s2p.spatial_hp_detect = imgui.input_int(
                "Spatial HP Detect", self.s2p.spatial_hp_detect
            )
            set_tooltip("Spatial high-pass filter size before ROI detection.")
            _, self.s2p.max_overlap = imgui.input_float(
                "Max Overlap", self.s2p.max_overlap
            )
            set_tooltip("Maximum allowed fraction of overlapping ROI pixels.")
            _, self.s2p.high_pass = imgui.input_int(
                "High-Pass Window", self.s2p.high_pass
            )
            set_tooltip("Running mean subtraction window (frames).")
            _, self.s2p.smooth_masks = imgui.checkbox(
                "Smooth Masks", self.s2p.smooth_masks
            )
            set_tooltip("Smooth masks in the final ROI detection pass.")
            _, self.s2p.max_iterations = imgui.input_int(
                "Max Iterations", self.s2p.max_iterations
            )
            set_tooltip("Maximum number of cell-detection iterations.")
            _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
            set_tooltip("Number of frames binned for ROI detection.")
            _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
            set_tooltip("Denoise binned movie before ROI detection.")

            imgui.end_disabled()
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # CELLPOSE / ANATOMICAL DETECTION SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Cellpose / Anatomical Detection"):
            imgui.indent()
            _, self.s2p.anatomical_only = imgui.input_int(
                "Anatomical Only", self.s2p.anatomical_only
            )
            set_tooltip(
                "0=disabled; 1=mean, 2=max, 3=enhanced (LBM default), 4=correlation"
            )
            _, self.s2p.diameter = imgui.input_int("Cell Diameter", self.s2p.diameter)
            set_tooltip("Expected cell diameter in pixels (6 = LBM default for ~6μm cells)")
            _, self.s2p.cellprob_threshold = imgui.input_float(
                "CellProb Threshold", self.s2p.cellprob_threshold
            )
            set_tooltip("Cellpose detection threshold (-6 = LBM default for permissive detection)")
            _, self.s2p.flow_threshold = imgui.input_float(
                "Flow Threshold", self.s2p.flow_threshold
            )
            set_tooltip("Cellpose flow field threshold (0 = standard)")
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
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # CLASSIFICATION SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Classification Settings"):
            imgui.indent()
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
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # OUTPUT SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Output Settings"):
            imgui.indent()
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
            _, self.s2p.aspect = imgui.input_float("Aspect Ratio", self.s2p.aspect)
            set_tooltip("um/pixel ratio X/Y for correct GUI aspect display.")
            _, self.s2p.report_time = imgui.checkbox(
                "Report Timing", self.s2p.report_time
            )
            set_tooltip("Return timing dictionary for each processing stage.")
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # SIGNAL EXTRACTION SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Signal Extraction Settings"):
            imgui.indent()
            _, self.s2p.neuropil_extract = imgui.checkbox(
                "Extract Neuropil", self.s2p.neuropil_extract
            )
            set_tooltip("Extract neuropil signal for background correction.")
            _, self.s2p.allow_overlap = imgui.checkbox(
                "Allow Overlap", self.s2p.allow_overlap
            )
            set_tooltip("Allow overlapping ROI pixels during extraction.")
            _, self.s2p.min_neuropil_pixels = imgui.input_int(
                "Min Neuropil Pixels", self.s2p.min_neuropil_pixels
            )
            set_tooltip("Minimum neuropil pixels per ROI.")
            _, self.s2p.inner_neuropil_radius = imgui.input_int(
                "Inner Neuropil Radius", self.s2p.inner_neuropil_radius
            )
            set_tooltip("Pixels to exclude between ROI and neuropil region.")
            _, self.s2p.lam_percentile = imgui.input_int(
                "Lambda Percentile", self.s2p.lam_percentile
            )
            set_tooltip("Percentile of Lambda used for neuropil exclusion.")
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # SPIKE DECONVOLUTION SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Spike Deconvolution Settings"):
            imgui.indent()
            _, self.s2p.spikedetect = imgui.checkbox(
                "Enable Spike Deconvolution", self.s2p.spikedetect
            )
            set_tooltip(
                "Detect spikes from neuropil-corrected and baseline-corrected traces."
            )

            imgui.begin_disabled(not self.s2p.spikedetect)

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

            _, self.s2p.win_baseline = imgui.input_float(
                "Baseline Window (s)", self.s2p.win_baseline
            )
            set_tooltip("Window for maximin filter in seconds.")
            _, self.s2p.sig_baseline = imgui.input_float(
                "Baseline Sigma (s)", self.s2p.sig_baseline
            )
            set_tooltip("Gaussian filter width in seconds for baseline computation.")
            _, self.s2p.prctile_baseline = imgui.input_float(
                "Baseline Percentile", self.s2p.prctile_baseline
            )
            set_tooltip("Percentile of trace for constant_percentile baseline method.")

            imgui.end_disabled()
            imgui.unindent()
            imgui.spacing()

        # ==================================================================
        # CHANNEL 2 SETTINGS SECTION
        # ==================================================================
        imgui.spacing()
        if imgui.collapsing_header("Channel 2 Settings"):
            imgui.indent()
            _, self.s2p.chan2_thres = imgui.input_float(
                "Chan2 Detection Threshold", self.s2p.chan2_thres
            )
            set_tooltip("Threshold for calling ROI detected on channel 2.")
            imgui.unindent()
            imgui.spacing()

        # Pop style variables and item width
        imgui.pop_item_width()
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
        #         self.image_widget.managed_graphics[1].data = combined
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
        #         self.image_widget.managed_graphics[1].data = combined


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
        self.logger.warning(
            "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
            "`uv pip install lbm_suite2p_python`",
        )
        self._install_error = True
        return

    if not self._install_error:
        # Iterate over each array (ROI) and each selected plane
        for i, arr in enumerate(self.image_widget.data):
            for plane_num in self._selected_planes:
                kwargs = {"self": self, "arr_idx": i, "plane_num": plane_num}
                threading.Thread(
                    target=run_plane_from_data, kwargs=kwargs, daemon=True
                ).start()


def run_plane_from_data(self, arr_idx, plane_num):
    """Process a single plane for a single ROI/array."""
    print(f"Thread ROI={arr_idx}, Plane={plane_num} started")
    from mbo_utilities.file_io import load_last_savedir, save_last_savedir
    from mbo_utilities.lazy_array import imread, imwrite

    if isinstance(self.fpath, list):
        source_file = self.fpath[arr_idx]
    else:
        source_file = self.fpath

    # Reload full array with correct ROI selection (lazy - not loaded into memory)
    if self.num_rois > 1 and arr_idx < self.num_rois:
        arr = imread(source_file, roi=arr_idx + 1)
        roi = arr_idx + 1
    else:
        arr = imread(source_file, roi=None)
        roi = None

    # output base - let imwrite() create the plane/roi subdirectories
    base_out = Path(self._saveas_outdir or load_last_savedir())
    base_out.mkdir(exist_ok=True)

    # Build metadata
    user_ops = {}
    if hasattr(self, "s2p"):
        try:
            user_ops = (
                vars(self.s2p).copy()
                if hasattr(self.s2p, "__dict__")
                else dict(self.s2p)
            )
        except Exception as e:
            self.logger.warning(f"Could not merge Suite2p params: {e}")

    # Add metadata
    user_ops.update({
        "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_file": str(source_file),
        "roi_index": arr_idx,
        "mroi": roi,
        "roi": roi,
        "z_index": plane_num - 1,  # 0-indexed
        "plane": plane_num,  # 1-indexed
        "fs": arr.metadata.get("frame_rate", 15.0),
        "dx": arr.metadata.get("pixel_size_xy", 1.0),
        "dz": arr.metadata.get("z_step", 1.0),
    })

    # Determine num_frames
    num_frames = None
    if user_ops.get("frames_include", -1) > 0:
        num_frames = user_ops["frames_include"]

    # Write functional channel using imwrite (lazy!)
    # imwrite will create plane_dir automatically based on plane/roi
    print(f"Writing plane {plane_num} for ROI {arr_idx} to {base_out}")
    imwrite(
        arr,
        base_out,
        ext=".bin",
        planes=[plane_num],  # 1-indexed
        num_frames=num_frames,
        metadata=user_ops,
        overwrite=True,
    )

    # Determine the plane directory that imwrite() created
    if roi is None:
        plane_dir = base_out / f"plane{plane_num:02d}_stitched"
    else:
        plane_dir = base_out / f"plane{plane_num:02d}_roi{roi}"

    if user_ops.get("chan2_file"):
        try:
            self.logger.info(f"Loading channel 2 from: {user_ops['chan2_file']}")
            chan2_arr = imread(user_ops["chan2_file"], roi=roi)

            chan2_metadata = user_ops.copy()
            chan2_metadata["structural"] = True

            imwrite(
                chan2_arr,
                base_out,
                ext=".bin",
                planes=[plane_num],
                num_frames=num_frames,
                metadata=chan2_metadata,
                overwrite=True,
                structural=True,
            )
        except Exception as e:
            self.logger.warning(f"Could not load channel 2 data: {e}")

    save_last_savedir(plane_dir)  # cache this location

    # Define file paths (imwrite already created these)
    raw_file = plane_dir / "data_raw.bin"
    ops_path = plane_dir / "ops.npy"

    # Load ops and merge with user settings
    # LBM-Suite2p-Python will merge this with defaults automatically
    ops_dict = np.load(ops_path, allow_pickle=True).item() if ops_path.exists() else {}

    # Run Suite2p processing with full parameter set
    print(f"Running Suite2p for plane {plane_num}, ROI {arr_idx}")
    try:
        result_ops = run_plane(
            input_path=raw_file,
            save_path=plane_dir,
            ops=ops_dict,  # Pass dict instead of path for proper merging
            chan2_file=user_ops.get("chan2_file"),
            keep_raw=self.s2p.keep_raw,
            keep_reg=self.s2p.keep_reg,
            force_reg=self.s2p.force_reg,
            force_detect=self.s2p.force_detect,
            dff_window_size=self.s2p.dff_window_size,
            dff_percentile=self.s2p.dff_percentile,
            save_json=self.s2p.save_json,
        )
        self.logger.info(
            f"Suite2p processing complete for plane {plane_num}, ROI {arr_idx}. "
            f"Results saved to {result_ops}"
        )
    except ValueError as e:
        self.logger.warning(
            f"No cells found for plane {plane_num}, ROI {arr_idx}: \n{e}"
        )
        return
    except Exception as e:
        self.logger.error(
            f"Suite2p processing failed for plane {plane_num}, ROI {arr_idx}: \n{e}"
        )
        return
