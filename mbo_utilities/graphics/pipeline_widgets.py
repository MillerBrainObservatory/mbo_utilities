import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd
from mbo_utilities import get_metadata, get_mbo_dirs
from mbo_utilities._parsing import _make_json_serializable
from mbo_utilities.graphics._widgets import set_tooltip
from mbo_utilities.lazy_array import imread

try:
    import lbm_suite2p_python as lsp

    HAS_LSP = True
except ImportError:
    HAS_LSP = False
    lsp = None

try:
    import masknmf

    HAS_MASKNMF = True
except ImportError:
    HAS_MASKNMF = False
    masknmf = None


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
        draw_section_masknmf(self)
    imgui.spacing()
    imgui.end_group()


def draw_section_suite2p(self):
    imgui.spacing()
    with imgui_ctx.begin_child("##Processing"):
        avail_w = imgui.get_content_region_avail().x * 0.3
        imgui.push_item_width(avail_w)
        imgui.new_line()
        imgui.separator_text("Registration Settings")
        _, self.s2p.do_registration = imgui.checkbox(
            "Do Registration", self.s2p.do_registration
        )
        set_tooltip("Whether or not to run registration")
        _, self.s2p.nimg_init = imgui.input_int("Initial Frames", self.s2p.nimg_init)
        set_tooltip(
            "Shave off this many frames from the start of the movie for registration"
        )
        _, self.s2p.batch_size = imgui.input_int("Batch Size", self.s2p.batch_size)
        set_tooltip("Number of frames to process in each batch during registration")
        _, self.s2p.maxregshift = imgui.input_float(
            "Max Shift Fraction", self.s2p.maxregshift
        )
        set_tooltip("Maximum allowed shift as a fraction of the image size")
        _, self.s2p.smooth_sigma = imgui.input_float(
            "Smooth Sigma", self.s2p.smooth_sigma
        )
        set_tooltip(
            "Sigma for Gaussian smoothing of the image. Keep it low (less than 1.0) for high-frequency data."
        )
        _, self.s2p.smooth_sigma_time = imgui.input_float(
            "Smooth Sigma Time", self.s2p.smooth_sigma_time
        )
        set_tooltip("Sigma for temporal smoothing of the image.")
        _, self.s2p.keep_movie_raw = imgui.checkbox(
            "Keep Raw Movie", self.s2p.keep_movie_raw
        )
        set_tooltip(
            "Whether to keep the raw movie data after processing, kept as a .bin file."
        )
        _, self.s2p.two_step = imgui.checkbox(
            "Two-Step Registration", self.s2p.two_step
        )
        set_tooltip(
            "Register once to a reference image, then register to the mean of the registered movie."
        )
        _, self.s2p.reg_tif = imgui.checkbox("Export Registered TIFF", self.s2p.reg_tif)
        set_tooltip(
            "Export the registered movie as a TIFF file. Saved in save/path/reg_tiff/reg_tif_chan1_000N.tif"
        )
        _, self.s2p.subpixel = imgui.input_int("Subpixel Precision", self.s2p.subpixel)
        set_tooltip(
            "Subpixel precision for registration. Higher values may improve accuracy but increase processing time."
        )
        _, self.s2p.th_badframes = imgui.input_float(
            "Bad Frame Threshold", self.s2p.th_badframes
        )
        set_tooltip(
            "Threshold for detecting bad frames during registration. Frames with a value above this threshold will be considered bad."
        )
        _, self.s2p.norm_frames = imgui.checkbox(
            "Normalize Frames", self.s2p.norm_frames
        )
        set_tooltip(
            "Whether to normalize frames during registration. This can help with illumination variations."
        )
        _, self.s2p.force_refimg = imgui.checkbox(
            "Use Stored refImg", self.s2p.force_refimg
        )
        set_tooltip(
            "Use a stored reference image for registration instead of computing one from the movie."
        )
        _, self.s2p.pad_fft = imgui.checkbox("Pad FFT Image", self.s2p.pad_fft)
        set_tooltip(
            "Whether to pad the FFT image during registration. This can help with aliasing effects."
        )

        imgui.spacing()
        imgui.separator_text("Classification Settings")
        _, self.s2p.soma_crop = imgui.checkbox("Soma Crop", self.s2p.soma_crop)
        set_tooltip("Crop the movie to the soma region before processing.")
        _, self.s2p.use_builtin_classifier = imgui.checkbox(
            "Use Builtin Classifier", self.s2p.use_builtin_classifier
        )
        set_tooltip(
            "Use the built-in classifier for detecting ROIs. If unchecked, a custom classifier path is required."
        )
        _, self.s2p.classifier_path = imgui.input_text(
            "Classifier Path", self.s2p.classifier_path, 256
        )
        set_tooltip(
            "Path to a custom classifier file. If the built-in classifier is not used, this path must be provided."
        )

        imgui.separator_text("ROI Detection Settings")
        _, self.s2p.roidetect = imgui.checkbox("Detect ROIs", self.s2p.roidetect)
        set_tooltip("Whether to detect ROIs in the movie.")
        _, self.s2p.sparse_mode = imgui.checkbox("Sparse Mode", self.s2p.sparse_mode)
        set_tooltip("Sparse_mode=True is recommended for soma, False for dendrites.")
        _, self.s2p.spatial_scale = imgui.input_int(
            "Spatial Scale", self.s2p.spatial_scale
        )
        set_tooltip(
            "what the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels)."
        )
        _, self.s2p.connected = imgui.checkbox("Connected ROIs", self.s2p.connected)
        set_tooltip(
            "Whether to use connected components for ROI detection. If False, ROIs will be detected independently."
        )
        _, self.s2p.threshold_scaling = imgui.input_float(
            "Threshold Scaling", self.s2p.threshold_scaling
        )
        set_tooltip(
            "Scaling factor for the threshold used in ROI detection. Generally (NOT always), higher values will result in fewer ROIs being detected."
        )
        _, self.s2p.spatial_hp_detect = imgui.input_int(
            "Spatial HP Filter", self.s2p.spatial_hp_detect
        )
        set_tooltip(
            "Spatial high-pass filter size for ROI detection. A value of 25 is recommended for most datasets."
        )
        _, self.s2p.max_overlap = imgui.input_float("Max Overlap", self.s2p.max_overlap)
        set_tooltip(
            "Maximum allowed overlap between detected ROIs. If two ROIs overlap more than this value, the one with the lower signal will be discarded."
        )
        _, self.s2p.high_pass = imgui.input_int("High Pass", self.s2p.high_pass)
        set_tooltip(
            "High-pass filter size for ROI detection. A value of 100 is recommended for most datasets."
        )
        _, self.s2p.smooth_masks = imgui.checkbox("Smooth Masks", self.s2p.smooth_masks)
        set_tooltip(
            "Whether to smooth the detected masks. This can help with noise but may also remove small ROIs."
        )
        _, self.s2p.max_iterations = imgui.input_int(
            "Max Iterations", self.s2p.max_iterations
        )
        set_tooltip(
            "Maximum number of iterations for the ROI detection algorithm. More than 100 is likely redundant."
        )
        _, self.s2p.nbinned = imgui.input_int("Max Binned Frames", self.s2p.nbinned)
        set_tooltip(
            "Maximum number of frames to bin for ROI detection. More than 5000 is likely redundant."
        )
        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip(
            "Whether to denoise the movie before processing. This can help with noise but may also remove small ROIs."
        )

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


def draw_section_masknmf(self):
    imgui.text("MaskNMF is not yet implemented in this version.")


def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline == "suite2p":
        self.logger.info(f"Running Suite2p pipeline with settings: {self.s2p}")
        try:
            import lbm_suite2p_python as lsp
        except ImportError:
            self.logger.warning(
                "error",
                "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline.",
            )
            self._install_error = True
            return
    if not self._install_error:
        for i, arr in enumerate(self.image_widget.data):
            kwargs = {"self": self, "arr_idx": i}
            threading.Thread(target=run_plane_from_data, kwargs=kwargs).start()
    elif self._current_pipeline == "masknmf":
        self.logger.info("Running MaskNMF pipeline (not yet implemented).")
    else:
        self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")


def run_plane_from_data(self, arr_idx):
    if not HAS_LSP:
        self.logger.error(
            "lbm_suite2p_python is not installed. Please install it to run the Suite2p pipeline."
        )
        self._install_error = True
        return

    data_shape = self.image_widget.data[arr_idx].shape
    metadata = getattr(self, "metadata", None)
    input_file = None

    dims = self.image_widget.current_index
    if "z" in dims:
        current_z = self.image_widget.current_index["z"]
    else:
        current_z = 0

    if arr_idx in self._rectangle_selectors.keys():
        ind_xy = self._rectangle_selectors[arr_idx].get_selected_indices()
        ind_x = list(ind_xy[0])
        ind_y = list(ind_xy[1])
        self.logger.debug(f"Sub-indices selected: {ind_xy}")
    else:
        ind_x, ind_y = slice(None), slice(None)

    if self.is_mbo_scan:
        # TODO
        raise NotImplementedError()
        # save_as(
        #     self.image_widget.data[arr_idx],
        #     self._saveas_outdir,
        #     planes=[current_z + 1],
        #     ext=".bin",
        # )

    # move to property?
    if not self._saveas_outdir:
        current_time_fmt = time.strftime("%Y%m%d_%H%M%S")
        self._saveas_outdir = get_mbo_dirs()["data"].joinpath(
            f"{current_time_fmt}_{self._current_pipeline}_output"
        )

    if len(data_shape) == 4:
        data = self.image_widget.data[arr_idx][:, current_z, ind_x, ind_y]
    elif len(data_shape) == 3:
        data = self.image_widget.data[arr_idx][:, ind_x, ind_y]
    else:
        data = self.image_widget.data[arr_idx][ind_x, ind_y]

    self.logger.info(f"shape of selected data: {data.shape}")
    out_dir = Path(self._saveas_outdir)
    out_dir.mkdir(exist_ok=True)

    self.fpath = self.fpath
    loader = imread(self.fpath)
    if hasattr(loader, "roi") or hasattr(loader, "rois"):
        self.logger.info("Using LazyArrayLoader with ROI support. ")
        arr = loader.rois

    if self.fpath.suffix.lower() in [".tif", ".tiff"]:
        metadata = get_metadata(self.fpath)
        metadata = _make_json_serializable(metadata)
        input_file = out_dir / "plane_7.tif"
        metadata["shape"] = data.shape
        # TODO
        raise NotImplementedError()
        # save_as(
        #     data,
        #     input_file,
        #     metadata
        # )
        # tifffile.imwrite(input_file, data, metadata=metadata)
        # self.logger.info(f"Temporary file created:"
        #             f" {input_file}")

    elif self.fpath.suffix.lower() == ".bin":
        metadata = np.load(self.fpath.parent / "ops.npy", allow_pickle=True).item()
        input_file = self.fpath

    self.logger.info(f"Metadata provided:")
    for k, v in metadata.items():
        self.logger.info(f"{k}: {v}")
    ops = self.s2p.to_dict()
    self.logger.info(f"User ops provided:")
    for k, v in ops.items():
        self.logger.info(f"{k}: {v}")
    ops.update(metadata)
    if input_file:
        lsp.run_plane(input_file, out_dir, ops=ops)
        self.logger.info(f"Plane 7 saved to {out_dir / 'plane_7.tif'}")
    else:
        self.logger.error("No valid input file found for processing.")
        return
