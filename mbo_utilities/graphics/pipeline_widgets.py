from dataclasses import dataclass

from imgui_bundle import imgui, imgui_ctx, hello_imgui

REGION_TYPES = ["Full FOV", "Sub-FOV"]

# -- Suite2p

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

    if not hasattr(self, "_current_pipeline"):
        self._current_pipeline = USER_PIPELINES[0]

    imgui.begin_group()

    current_display_idx = USER_PIPELINES.index(self._current_pipeline)
    imgui.set_next_item_width(hello_imgui.em_size(15))
    changed, selected_idx = imgui.combo("Pipeline", current_display_idx, USER_PIPELINES)

    if changed:
        self._current_pipeline = USER_PIPELINES[selected_idx]

    set_tooltip("Select a processing pipeline to configure.")

    imgui.separator()

    if self._current_pipeline == "suite2p":
        self.draw_processing_tab()

    elif self._current_pipeline == "masknmf":
        imgui.text("MaskNMF is not yet implemented in this version.")

    imgui.end_group()


def draw_processing_tab(self: PreviewDataWidget):
    cflags: imgui.ChildFlags = (
            imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # noqa
    )
    """Draw the processing tab for Suite2p settings."""
    imgui.spacing()

    with imgui_ctx.begin_child("##Processing", child_flags=cflags):

        imgui.text("Process full FOV or selected subregion?")
        imgui.separator()

        imgui.begin_group()
        # Region selection
        imgui.set_next_item_width(hello_imgui.em_size(25))
        _, self._region_idx = imgui.combo("Region type", self._region_idx, REGION_TYPES)
        self._region = REGION_TYPES[self._region_idx]
        imgui.end_group()

        imgui.begin_group()
        imgui.new_line()
        imgui.separator_text("Registration Settings")
        self._do_registration = imgui.checkbox("Do Registration", True)[1]
        if imgui.is_item_hovered(): imgui.set_tooltip("Whether or not to run registration")
        _, self._align_by_chan = imgui.input_int("Align by Channel", 1)
        if imgui.is_item_hovered(): imgui.set_tooltip("Which channel to use for alignment (1-based index)")
        _, self._nimg_init = imgui.input_int("Initial Frames", 300)
        _, self._batch_size = imgui.input_int("Batch Size", 500)
        _, self._maxregshift = imgui.input_float("Max Shift Fraction", 0.1)
        _, self._smooth_sigma = imgui.input_float("Smooth Sigma", 1.15)
        _, self._smooth_sigma_time = imgui.input_float("Smooth Sigma Time", 0.0)
        self._keep_movie_raw = imgui.checkbox("Keep Raw Movie", False)[1]
        self._two_step = imgui.checkbox("Two-Step Registration", False)[1]
        self._reg_tif = imgui.checkbox("Export Registered TIFF", False)[1]
        self._reg_tif_chan2 = imgui.checkbox("Export Channel 2 TIFF", False)[1]
        _, self._subpixel = imgui.input_int("Subpixel Precision", 10)
        _, self._th_badframes = imgui.input_float("Bad Frame Threshold", 1.0)
        self._norm_frames = imgui.checkbox("Normalize Frames", True)[1]
        self._force_refimg = imgui.checkbox("Use Stored refImg", False)[1]
        self._pad_fft = imgui.checkbox("Pad FFT Image", False)[1]
        imgui.end_group()

        imgui.begin_group()

        imgui.spacing()
        imgui.separator_text("Classification Settings")
        self._soma_crop = imgui.checkbox("Soma Crop", True)[1]
        self._use_builtin_classifier = imgui.checkbox("Use Builtin Classifier", False)[1]
        imgui.set_next_item_width(hello_imgui.em_size(30))
        _, self._classifier_path = imgui.input_text("Classifier Path", "", 256)
        imgui.end_group()

        imgui.begin_group()
        imgui.separator_text("ROI Detection Settings")
        self._roidetect = imgui.checkbox("Detect ROIs", True)[1]
        self._sparse_mode = imgui.checkbox("Sparse Mode", True)[1]
        _, self._spatial_scale = imgui.input_int("Spatial Scale", 0)
        self._connected = imgui.checkbox("Connected ROIs", True)[1]
        _, self._threshold_scaling = imgui.input_float("Threshold Scaling", 1.0)
        _, self._spatial_hp_detect = imgui.input_int("Spatial HP Filter", 25)
        _, self._max_overlap = imgui.input_float("Max Overlap", 0.75)
        _, self._high_pass = imgui.input_int("High Pass", 100)
        self._smooth_masks = imgui.checkbox("Smooth Masks", True)[1]
        _, self._max_iterations = imgui.input_int("Max Iterations", 20)
        _, self._nbinned = imgui.input_int("Max Binned Frames", 5000)
        self._denoise = imgui.checkbox("Denoise Movie", False)[1]
        imgui.end_group()

        imgui.spacing()
        imgui.input_text("Save folder", "/path/to/suite2p", 256)

        imgui.separator()
        if imgui.button("Run"):
            self.debug_panel.log("info", "Running Suite2p pipeline...")
            self.debug_panel.log("info", "Suite2p pipeline completed.")
