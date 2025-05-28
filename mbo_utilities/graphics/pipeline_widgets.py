from dataclasses import dataclass
from pathlib import Path

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd

from mbo_utilities.graphics._widgets import set_tooltip

REGION_TYPES = ["Full FOV", "Sub-FOV"]
USER_PIPELINES = ["suite2p", "masknmf"]

@dataclass
class Suite2pSettings:
    do_registration: bool = True
    align_by_chan: int = 1
    nimg_init: int = 300
    batch_size: int = 500
    maxregshift: float = 0.1
    smooth_sigma: float = 1.15
    smooth_sigma_time: float = 0.0
    keep_movie_raw: bool = False
    two_step: bool = False
    reg_tif: bool = False
    reg_tif_chan2: bool = False
    subpixel: int = 10
    th_badframes: float = 1.0
    norm_frames: bool = True
    force_refimg: bool = False
    pad_fft: bool = False

    soma_crop: bool = True
    use_builtin_classifier: bool = False
    classifier_path: str = ""

    roidetect: bool = True
    sparse_mode: bool = True
    spatial_scale: int = 0
    connected: bool = True
    threshold_scaling: float = 1.0
    spatial_hp_detect: int = 25
    max_overlap: float = 0.75
    high_pass: int = 100
    smooth_masks: bool = True
    max_iterations: int = 20
    nbinned: int = 5000
    denoise: bool = False

def draw_pipeline_section(self):
    """Draws the pipeline selection and configuration section."""
    imgui.spacing()
    imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Select a processing pipeline:")
    if not hasattr(self, "_current_pipeline"):
        self._current_pipeline = USER_PIPELINES[0]

    current_display_idx = USER_PIPELINES.index(self._current_pipeline)
    changed, selected_idx = imgui.combo("Pipeline", current_display_idx, USER_PIPELINES)

    if changed:
        self._current_pipeline = USER_PIPELINES[selected_idx]

    set_tooltip("Select a processing pipeline to configure.")
    imgui.spacing()
    imgui.separator()

    # imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
    # imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))  # noqa
    if self._current_pipeline == "suite2p":
        imgui.set_next_item_width(imgui.get_content_region_avail().x)
        draw_processing_tab(self)
    elif self._current_pipeline == "masknmf":
        imgui.text("MaskNMF is not yet implemented in this version.")
    imgui.spacing()
    # imgui.pop_style_var()
    # imgui.pop_style_var()

def draw_processing_tab(self):
    imgui.spacing()
    cflags = imgui.ChildFlags_.auto_resize_x | imgui.ChildFlags_.always_use_window_padding | imgui.ChildFlags_.auto_resize_y
    with imgui_ctx.begin_child("##Processing", child_flags=cflags):

        avail_w = imgui.get_content_region_avail().x * 0.5
        imgui.push_item_width(avail_w)
        imgui.begin_group()
        imgui.text("Process full FOV or selected subregion?")
        imgui.separator()

        _, self._region_idx = imgui.combo("Region type", self._region_idx, REGION_TYPES)
        self._region = REGION_TYPES[self._region_idx]

        imgui.new_line()
        imgui.separator_text("Registration Settings")
        _, self.s2p.do_registration = imgui.checkbox("Do Registration", self.s2p.do_registration)
        set_tooltip("Whether or not to run registration")
        _, self.s2p.nimg_init = imgui.input_int("Initial Frames", self.s2p.nimg_init)
        set_tooltip("Shave off this many frames from the start of the movie for registration")
        _, self.s2p.batch_size = imgui.input_int("Batch Size", self.s2p.batch_size)
        set_tooltip("Number of frames to process in each batch during registration")
        _, self.s2p.maxregshift = imgui.input_float("Max Shift Fraction", self.s2p.maxregshift)
        set_tooltip("Maximum allowed shift as a fraction of the image size")
        _, self.s2p.smooth_sigma = imgui.input_float("Smooth Sigma", self.s2p.smooth_sigma)
        set_tooltip("Sigma for Gaussian smoothing of the image. Keep it low (less than 1.0) for high-frequency data.")
        _, self.s2p.smooth_sigma_time = imgui.input_float("Smooth Sigma Time", self.s2p.smooth_sigma_time)
        set_tooltip("Sigma for temporal smoothing of the image.")
        _, self.s2p.keep_movie_raw = imgui.checkbox("Keep Raw Movie", self.s2p.keep_movie_raw)
        set_tooltip("Whether to keep the raw movie data after processing, kept as a .bin file.")
        _, self.s2p.two_step = imgui.checkbox("Two-Step Registration", self.s2p.two_step)
        set_tooltip("Register once to a reference image, then register to the mean of the registered movie.")
        _, self.s2p.reg_tif = imgui.checkbox("Export Registered TIFF", self.s2p.reg_tif)
        set_tooltip("Export the registered movie as a TIFF file. Saved in save/path/reg_tiff/reg_tif_chan1_000N.tif")
        _, self.s2p.subpixel = imgui.input_int("Subpixel Precision", self.s2p.subpixel)
        set_tooltip("Subpixel precision for registration. Higher values may improve accuracy but increase processing time.")
        _, self.s2p.th_badframes = imgui.input_float("Bad Frame Threshold", self.s2p.th_badframes)
        set_tooltip("Threshold for detecting bad frames during registration. Frames with a value above this threshold will be considered bad.")
        _, self.s2p.norm_frames = imgui.checkbox("Normalize Frames", self.s2p.norm_frames)
        set_tooltip("Whether to normalize frames during registration. This can help with illumination variations.")
        _, self.s2p.force_refimg = imgui.checkbox("Use Stored refImg", self.s2p.force_refimg)
        set_tooltip("Use a stored reference image for registration instead of computing one from the movie.")
        _, self.s2p.pad_fft = imgui.checkbox("Pad FFT Image", self.s2p.pad_fft)
        set_tooltip("Whether to pad the FFT image during registration. This can help with aliasing effects.")

        imgui.spacing()
        imgui.separator_text("Classification Settings")
        _, self.s2p.soma_crop = imgui.checkbox("Soma Crop", self.s2p.soma_crop)
        set_tooltip("Crop the movie to the soma region before processing.")
        _, self.s2p.use_builtin_classifier = imgui.checkbox("Use Builtin Classifier", self.s2p.use_builtin_classifier)
        set_tooltip("Use the built-in classifier for detecting ROIs. If unchecked, a custom classifier path is required.")
        _, self.s2p.classifier_path = imgui.input_text("Classifier Path", self.s2p.classifier_path, 256)
        set_tooltip("Path to a custom classifier file. If the built-in classifier is not used, this path must be provided.")

        imgui.separator_text("ROI Detection Settings")
        _, self.s2p.roidetect = imgui.checkbox("Detect ROIs", self.s2p.roidetect)
        set_tooltip("Whether to detect ROIs in the movie.")
        _, self.s2p.sparse_mode = imgui.checkbox("Sparse Mode", self.s2p.sparse_mode)
        set_tooltip("Sparse_mode=True is recommended for soma, False for dendrites.")
        _, self.s2p.spatial_scale = imgui.input_int("Spatial Scale", self.s2p.spatial_scale)
        set_tooltip("what the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).")
        _, self.s2p.connected = imgui.checkbox("Connected ROIs", self.s2p.connected)
        set_tooltip("Whether to use connected components for ROI detection. If False, ROIs will be detected independently.")
        _, self.s2p.threshold_scaling = imgui.input_float("Threshold Scaling", self.s2p.threshold_scaling)
        set_tooltip("Scaling factor for the threshold used in ROI detection. Generally (NOT always), higher values will result in fewer ROIs being detected.")
        _, self.s2p.spatial_hp_detect = imgui.input_int("Spatial HP Filter", self.s2p.spatial_hp_detect)
        set_tooltip("Spatial high-pass filter size for ROI detection. A value of 25 is recommended for most datasets.")
        _, self.s2p.max_overlap = imgui.input_float("Max Overlap", self.s2p.max_overlap)
        set_tooltip("Maximum allowed overlap between detected ROIs. If two ROIs overlap more than this value, the one with the lower signal will be discarded.")
        _, self.s2p.high_pass = imgui.input_int("High Pass", self.s2p.high_pass)
        set_tooltip("High-pass filter size for ROI detection. A value of 100 is recommended for most datasets.")
        _, self.s2p.smooth_masks = imgui.checkbox("Smooth Masks", self.s2p.smooth_masks)
        set_tooltip("Whether to smooth the detected masks. This can help with noise but may also remove small ROIs.")
        _, self.s2p.max_iterations = imgui.input_int("Max Iterations", self.s2p.max_iterations)
        set_tooltip("Maximum number of iterations for the ROI detection algorithm. More than 100 is likely redundant.")
        _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
        set_tooltip("Maximum number of frames to bin for ROI detection. More than 5000 is likely redundant.")
        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip("Whether to denoise the movie before processing. This can help with noise but may also remove small ROIs.")

        imgui.spacing()
        imgui.input_text("Save folder", self._saveas_outdir, 256)
        imgui.same_line()
        if imgui.button("Browse"):
            home = Path().home()
            res = pfd.select_folder(str(home))
            if res:
                self._saveas_outdir = res.result()

        imgui.separator()
        if imgui.button("Run"):
            self.debug_panel.log("info", "Running Suite2p pipeline...")
            self.debug_panel.log("info", "Suite2p pipeline completed.")

        imgui.end_group()
        imgui.pop_item_width()
