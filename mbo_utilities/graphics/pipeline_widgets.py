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
    """

    # ==================== Main Settings ====================
    nplanes: int = 1  # Number of planes in each tiff
    nchannels: int = 1  # Number of channels per plane
    functional_chan: int = 1  # Channel for functional ROI extraction (1-based)
    tau: float = 1.0  # Timescale of sensor (GCaMP6f=0.7, 6m=1.0, 6s=1.25-1.5)
    fs: float = 10.0  # Sampling rate per plane (Hz)
    force_sktiff: bool = False  # Use scikit-image for reading tiffs
    do_bidiphase: bool = False  # Compute bidirectional phase offset
    bidiphase: int = 0  # Bidirectional phase offset (user-set)
    bidi_corrected: bool = False  # Whether bidi correction is done
    frames_include: int = -1  # Only process this many frames (for testing)
    multiplane_parallel: bool = False  # Run pipeline on server
    # ignore_flyback: list = field(default_factory=list)  # Planes to ignore

    # ==================== Output Settings ====================
    preclassify: float = 0.0  # Apply classifier before extraction (0.0 = keep all)
    save_nwb: bool = False  # Save output as NWB file
    save_mat: bool = False  # Save results in Fall.mat
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
    block_size: list = None  # Block size for non-rigid (default [128, 128], power of 2/3)
    snr_thresh: float = 1.2  # Phase correlation peak threshold (1.5 for 1P)
    maxregshiftNR: float = 5.0  # Max block shift relative to rigid shift

    # ==================== ROI Detection Settings ====================
    roidetect: bool = True  # Run ROI detection and extraction
    sparse_mode: bool = True  # Use sparse_mode cell detection
    spatial_scale: int = 0  # Optimal recording scale (0=auto, 1-4 manual)
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
    anatomical_only: int = 0  # Use Cellpose (1-4 for different projections)
    diameter: int = 0  # Cellpose diameter (0=auto-estimate)
    cellprob_threshold: float = 0.0  # Cellpose cell detection threshold
    flow_threshold: float = 1.5  # Cellpose flow threshold
    spatial_hp_cp: int = 0  # High-pass window for Cellpose image
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

    if not hasattr(self, "_rectangle_selectors"):
        self._rectangle_selectors = {}
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
    if not hasattr(self, "_region_idx"):
        self._region_idx = 0
    if not hasattr(self, "_subregions"):
        self._subregions = {}
    if not hasattr(self, "_array_type"):
        self._array_type = "array"

    imgui.begin_group()
    imgui.dummy(imgui.ImVec2(0, 5))

    imgui.text_colored(
        imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Spatial-crop before processing:"
    )

    for i, graphic in enumerate(self.image_widget.managed_graphics):
        selected = self._rectangle_selectors.get(i) is not None
        label = f"{'Remove Crop Selector: ' if selected else 'Add Crop Selector: '}{self._array_type} {i + 1}"
        if imgui.button(label):
            g = self.image_widget.managed_graphics[i]
            sel = self._rectangle_selectors.get(i)
            if sel:  # already exists → remove
                self.image_widget.figure[0, i].delete_graphic(sel)
                self._rectangle_selectors[i] = None
            else:  # doesn’t exist → add
                g.add_rectangle_selector()
                self._rectangle_selectors[i] = g._plot_area.selectors[0]

    imgui.dummy(imgui.ImVec2(0, 5))
    imgui.separator()

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
    imgui.spacing()
    with imgui_ctx.begin_child("##Processing"):
        avail_w = imgui.get_content_region_avail().x * 0.3
        imgui.push_item_width(avail_w)
        imgui.new_line()

        # --------------------------------------------------------------
        imgui.separator_text("Main Settings")
        _, self.s2p.nplanes = imgui.input_int("Number of Planes", self.s2p.nplanes)
        set_tooltip("Each tiff has this many planes in sequence.")
        _, self.s2p.nchannels = imgui.input_int("Number of Channels", self.s2p.nchannels)
        set_tooltip("Each tiff has this many channels per plane.")
        _, self.s2p.functional_chan = imgui.input_int(
            "Functional Channel", self.s2p.functional_chan
        )
        set_tooltip("Channel used for functional ROI extraction (1-based).")
        _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
        set_tooltip(
            "Sensor decay timescale: GCaMP6f=0.7, GCaMP6m=1.0, GCaMP6s=1.25-1.5"
        )
        _, self.s2p.fs = imgui.input_float("Sampling Rate (Hz)", self.s2p.fs)
        set_tooltip("Sampling rate per plane (e.g., 30Hz / 10 planes = 3Hz per plane).")
        _, self.s2p.force_sktiff = imgui.checkbox(
            "Force scikit-image TIFF", self.s2p.force_sktiff
        )
        set_tooltip("Use scikit-image for reading TIFFs instead of ScanImageTiffReader.")
        _, self.s2p.do_bidiphase = imgui.checkbox(
            "Compute Bidi Phase", self.s2p.do_bidiphase
        )
        set_tooltip("Compute bidirectional phase offset from data (2P only).")
        _, self.s2p.bidiphase = imgui.input_int("Bidi Phase Offset", self.s2p.bidiphase)
        set_tooltip("User-specified bidirectional phase offset (if known).")
        _, self.s2p.bidi_corrected = imgui.checkbox(
            "Bidi Corrected", self.s2p.bidi_corrected
        )
        set_tooltip("Whether bidirectional correction has been applied.")
        _, self.s2p.frames_include = imgui.input_int(
            "Frames to Process", self.s2p.frames_include
        )
        set_tooltip("Only process this many frames (-1 = all frames, for testing).")
        _, self.s2p.multiplane_parallel = imgui.checkbox(
            "Multiplane Parallel", self.s2p.multiplane_parallel
        )
        set_tooltip("Run pipeline in parallel across planes on server.")

        # --------------------------------------------------------------
        imgui.spacing()
        imgui.separator_text("Registration Settings")
        _, self.s2p.do_registration = imgui.checkbox(
            "Do Registration", self.s2p.do_registration
        )
        set_tooltip("Run motion registration on the movie.")
        _, self.s2p.align_by_chan = imgui.input_int(
            "Align by Channel", self.s2p.align_by_chan
        )
        set_tooltip("Channel index used for alignment (1-based).")
        _, self.s2p.nimg_init = imgui.input_int("Initial Frames", self.s2p.nimg_init)
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
        _, self.s2p.reg_tif = imgui.checkbox("Export Registered TIFF", self.s2p.reg_tif)
        set_tooltip("Export registered movie as TIFF files.")
        _, self.s2p.reg_tif_chan2 = imgui.checkbox(
            "Export Chan2 TIFF", self.s2p.reg_tif_chan2
        )
        set_tooltip("Export registered TIFFs for channel 2.")
        _, self.s2p.subpixel = imgui.input_int("Subpixel Precision", self.s2p.subpixel)
        set_tooltip("Subpixel precision level (1/subpixel step).")
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
        _, self.s2p.chan2_file = imgui.input_text(
            "Channel 2 File", self.s2p.chan2_file, 256
        )
        set_tooltip("Path to channel 2 binary file for cross-channel registration.")
        imgui.same_line()
        if imgui.button("Browse##chan2"):
            home = pathlib.Path().home()
            res = pfd.open_file("Select channel 2 file", str(home))
            if res and res.result():
                self.s2p.chan2_file = res.result()[0]

        # --- 1P Registration Subsection ---
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "1-Photon Registration:")
        _, self.s2p.do_1Preg = imgui.checkbox("Enable 1P Registration", self.s2p.do_1Preg)
        set_tooltip("Apply high-pass filtering and tapering for 1-photon data.")
        _, self.s2p.spatial_hp_reg = imgui.input_int(
            "Spatial HP Window (1P)", self.s2p.spatial_hp_reg
        )
        set_tooltip("Window size for spatial high-pass filtering before registration.")
        _, self.s2p.pre_smooth = imgui.input_float("Pre-smooth Sigma (1P)", self.s2p.pre_smooth)
        set_tooltip("Gaussian smoothing stddev before high-pass filtering (0=disabled).")
        _, self.s2p.spatial_taper = imgui.input_float(
            "Spatial Taper (1P)", self.s2p.spatial_taper
        )
        set_tooltip("Pixels to set to zero on edges (important for vignetted windows).")

        # --- Non-rigid Registration Subsection ---
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.6, 0.8, 1.0, 1.0), "Non-rigid Registration:")
        _, self.s2p.nonrigid = imgui.checkbox("Enable Non-rigid", self.s2p.nonrigid)
        set_tooltip("Split FOV into blocks and compute registration offsets per block.")

        # Block size as two separate inputs
        if self.s2p.block_size is None:
            self.s2p.block_size = [128, 128]
        block_y_changed, block_y = imgui.input_int("Block Height", self.s2p.block_size[0])
        set_tooltip("Block height for non-rigid registration (power of 2/3 recommended).")
        block_x_changed, block_x = imgui.input_int("Block Width", self.s2p.block_size[1])
        set_tooltip("Block width for non-rigid registration (power of 2/3 recommended).")
        if block_y_changed or block_x_changed:
            self.s2p.block_size = [block_y, block_x]

        _, self.s2p.snr_thresh = imgui.input_float("SNR Threshold", self.s2p.snr_thresh)
        set_tooltip("Phase correlation peak threshold (1.5 recommended for 1P).")
        _, self.s2p.maxregshiftNR = imgui.input_float(
            "Max NR Shift", self.s2p.maxregshiftNR
        )
        set_tooltip("Max pixel shift of block relative to rigid shift.")

        imgui.spacing()
        imgui.separator_text("ROI Detection Settings")
        _, self.s2p.roidetect = imgui.checkbox("Detect ROIs", self.s2p.roidetect)
        set_tooltip("Run ROI detection and extraction.")
        _, self.s2p.sparse_mode = imgui.checkbox("Sparse Mode", self.s2p.sparse_mode)
        set_tooltip("Use sparse detection (recommended for soma).")
        _, self.s2p.spatial_scale = imgui.input_int(
            "Spatial Scale", self.s2p.spatial_scale
        )
        set_tooltip("ROI size scale. 0=auto, 1=small, 2=medium, 3=large, 4=very large.")
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
        _, self.s2p.max_overlap = imgui.input_float("Max Overlap", self.s2p.max_overlap)
        set_tooltip("Maximum allowed fraction of overlapping ROI pixels.")
        _, self.s2p.high_pass = imgui.input_int("High-Pass Window", self.s2p.high_pass)
        set_tooltip("Running mean subtraction window (frames).")
        _, self.s2p.smooth_masks = imgui.checkbox("Smooth Masks", self.s2p.smooth_masks)
        set_tooltip("Smooth masks in the final ROI detection pass.")
        _, self.s2p.max_iterations = imgui.input_int(
            "Max Iterations", self.s2p.max_iterations
        )
        set_tooltip("Maximum number of cell-detection iterations.")
        _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
        set_tooltip("Number of frames binned for ROI detection.")
        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip("Denoise binned movie before ROI detection.")

        imgui.spacing()
        imgui.separator_text("Cellpose / Anatomical Detection")
        _, self.s2p.anatomical_only = imgui.input_int(
            "Anatomical Only", self.s2p.anatomical_only
        )
        set_tooltip("0=disabled; 1-4 select Cellpose image type (mean, max, enhanced).")
        _, self.s2p.diameter = imgui.input_int("Cell Diameter", self.s2p.diameter)
        set_tooltip("Expected cell diameter; 0=auto-estimate.")
        _, self.s2p.cellprob_threshold = imgui.input_float(
            "CellProb Threshold", self.s2p.cellprob_threshold
        )
        set_tooltip("Cellpose detection probability threshold.")
        _, self.s2p.flow_threshold = imgui.input_float(
            "Flow Threshold", self.s2p.flow_threshold
        )
        set_tooltip("Cellpose flow field threshold.")
        _, self.s2p.spatial_hp_cp = imgui.input_int(
            "Spatial HP (Cellpose)", self.s2p.spatial_hp_cp
        )
        set_tooltip("Spatial high-pass window for Cellpose preprocessing.")
        _, self.s2p.pretrained_model = imgui.input_text(
            "Pretrained Model", self.s2p.pretrained_model, 128
        )
        set_tooltip("Cellpose model name or custom path (e.g., 'cyto').")

        imgui.spacing()
        imgui.separator_text("Classification Settings")
        _, self.s2p.soma_crop = imgui.checkbox("Soma Crop", self.s2p.soma_crop)
        set_tooltip("Crop dendrites for soma classification.")
        _, self.s2p.use_builtin_classifier = imgui.checkbox(
            "Use Built-in Classifier", self.s2p.use_builtin_classifier
        )
        set_tooltip("Use Suite2p's built-in ROI classifier.")
        _, self.s2p.classifier_path = imgui.input_text(
            "Classifier Path", self.s2p.classifier_path, 256
        )
        set_tooltip("Path to external classifier if not using built-in.")

        imgui.spacing()
        imgui.separator_text("Output Settings")
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
        _, self.s2p.report_time = imgui.checkbox("Report Timing", self.s2p.report_time)
        set_tooltip("Return timing dictionary for each processing stage.")

        imgui.spacing()
        imgui.separator_text("Signal Extraction Settings")
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

        imgui.spacing()
        imgui.separator_text("Spike Deconvolution Settings")
        _, self.s2p.spikedetect = imgui.checkbox(
            "Run Spike Deconvolution", self.s2p.spikedetect
        )
        set_tooltip("Detect spikes from neuropil-corrected and baseline-corrected traces.")
        _, self.s2p.neucoeff = imgui.input_float("Neuropil Coefficient", self.s2p.neucoeff)
        set_tooltip("Neuropil coefficient for all ROIs (F_corrected = F - coeff * F_neu).")

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

        imgui.spacing()
        imgui.separator_text("Channel 2 Settings")
        _, self.s2p.chan2_thres = imgui.input_float(
            "Chan2 Detection Threshold", self.s2p.chan2_thres
        )
        set_tooltip("Threshold for calling ROI detected on channel 2.")

        imgui.spacing()
        imgui.input_text("Save folder", self._saveas_outdir, 256)
        imgui.same_line()
        if imgui.button("Browse"):
            home = pathlib.Path().home()
            res = pfd.select_folder(str(home))
            if res:
                self._saveas_outdir = res.result()

        imgui.separator()
        if imgui.button("Run"):
            self.logger.info("Running Suite2p pipeline...")
            run_process(self)
            self.logger.info("Suite2p pipeline completed.")
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
                    except Exception as e:
                        try:
                            self.logger.log(
                                "error",
                                f"Installation failed: {e}",
                            )
                            subprocess.check_call(
                                ["uv", "pip", "install", "lbm_suite2p_python"]
                            )
                            self._show_red_text = False
                            self._show_green_text = True
                        except Exception as e:
                            self.logger.log("error", f"Installation failed: {e}")
                            self._show_red_text = True
                            self._show_install_button = False
                            self._show_green_text = True

        imgui.pop_item_width()
        imgui.spacing()
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
        for i, arr in enumerate(self.image_widget.data):
            kwargs = {"self": self, "arr_idx": i}
            threading.Thread(
                target=run_plane_from_data, kwargs=kwargs, daemon=True
            ).start()


def run_plane_from_data(self, arr_idx):
    from mbo_utilities._writers import _write_bin
    from mbo_utilities.file_io import get_last_savedir_path, save_last_savedir
    from mbo_utilities.lazy_array import imread

    arr = self.image_widget.data[arr_idx]
    dims = self.image_widget.current_index
    current_z = dims.get("z", 0)

    # ROI selection
    if arr_idx in self._rectangle_selectors and self._rectangle_selectors[arr_idx]:
        ind_x, ind_y = self._rectangle_selectors[arr_idx].get_selected_indices()
    else:
        ind_x, ind_y = slice(None), slice(None)

    # output base
    base_out = Path(self._saveas_outdir or get_last_savedir_path())
    base_out.mkdir(exist_ok=True)

    # Determine plane directory name based on ROI configuration
    # Match naming convention from array_types.py:1155-1158
    roi = getattr(arr, "roi", None)
    if roi is None:
        # Stitched data
        plane_dirname = f"plane{current_z + 1:02d}_stitched"
    else:
        # Separate ROI data
        plane_dirname = f"plane{current_z + 1:02d}_roi{roi}"

    plane_dir = base_out / plane_dirname
    plane_dir.mkdir(parents=True, exist_ok=True)
    raw_file = plane_dir / "data_raw.bin"
    ops_path = plane_dir / "ops.npy"

    # slice the selected z and optional ROI region
    data = arr[:, current_z, ind_x, ind_y]

    # Force computation for lazy arrays (dask, etc.)
    # np.asarray handles both dask arrays and memmap arrays correctly
    data = np.asarray(data, dtype=np.int16)

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

    # Handle chan2_file if provided - CRITICAL: both bins must have same frame count
    chan2_data = None
    chan2_raw_file = None
    min_frames = data.shape[0]  # default to functional data frames

    if user_ops.get("chan2_file"):
        try:
            self.logger.info(f"Loading channel 2 from: {user_ops['chan2_file']}")
            chan2_arr = imread(user_ops["chan2_file"])
            # Slice same z and ROI from chan2 data
            chan2_data = chan2_arr[:, current_z, ind_x, ind_y]
            # Force computation for lazy arrays (dask, etc.)
            chan2_data = np.asarray(chan2_data, dtype=np.int16)
            # Get minimum frames between functional and structural
            min_frames = min(data.shape[0], chan2_data.shape[0])
            # Trim both to same number of frames
            data = data[:min_frames]
            chan2_data = chan2_data[:min_frames]
            chan2_raw_file = plane_dir / "data_chan2.bin"
            self.logger.info(f"Channel 2 loaded and trimmed to {min_frames} frames")
        except Exception as e:
            self.logger.warning(f"Could not load channel 2 data: {e}")
            chan2_data = None
            min_frames = data.shape[0]

    md = {
        "process_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_file": str(self.fpath),
        "roi_index": arr_idx,  # Array index in ImageWidget
        "mroi": roi,  # Actual ROI number or None for stitched
        "roi": roi,  # Alias for mroi (matches array_types.py convention)
        "z_index": current_z,
        "plane": current_z + 1,
        "num_frames": min_frames,  # Use minimum frame count
        "nframes": min_frames,  # Suite2p also uses nframes
        "shape": (min_frames, data.shape[-2], data.shape[-1]),
        "Ly": data.shape[-2],
        "Lx": data.shape[-1],
        "fs": arr.metadata.get("frame_rate", 15.0),
        "dx": arr.metadata.get("pixel_size_xy", 1.0),
        "dz": arr.metadata.get("z_step", 1.0),
        "ops_path": str(ops_path),
        "save_path": str(plane_dir),
        "raw_file": str((plane_dir / "data_raw.bin").resolve()),
        "reg_file": str((plane_dir / "data.bin").resolve()),
    }

    user_ops.update(md)

    # Write functional channel (channel 1) binary with num_frames constraint
    _write_bin(raw_file, data, overwrite=True, metadata=user_ops)

    # Write structural channel (channel 2) binary if provided
    # Both bins MUST have the same num_frames for suite2p
    if chan2_data is not None and chan2_raw_file is not None:
        chan2_metadata = user_ops.copy()
        # CRITICAL: Keep the same num_frames for both bins
        chan2_metadata["num_frames"] = min_frames
        chan2_metadata["nframes"] = min_frames
        chan2_metadata["shape"] = (
            min_frames,
            chan2_data.shape[-2],
            chan2_data.shape[-1],
        )
        # Update the metadata to point to the chan2 file
        chan2_metadata["chan2_file"] = str(chan2_raw_file.resolve())
        _write_bin(chan2_raw_file, chan2_data, overwrite=True, metadata=chan2_metadata)
        # Update ops to reference the chan2 file path
        user_ops["chan2_file"] = str(chan2_raw_file.resolve())

    save_last_savedir(plane_dir)  # cache this location

    # re-save user ops with correct num_frames
    ops_dict = np.load(ops_path, allow_pickle=True).item()
    ops_dict.update(user_ops)
    np.save(ops_path, ops_dict)

    try:
        _ = run_plane(
            raw_file, save_path=plane_dir, ops=ops_path, keep_raw=True, keep_reg=True
        )
    except ValueError as e:
        self.logger.warning(
            f"No cells found for plane {current_z}, ROI {arr_idx}: \n{e}"
        )
        return
    self.logger.info(
        f"Suite2p processing complete for plane {current_z}, ROI {arr_idx}."
    )
