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
from mbo_utilities._writers import _convert_paths_to_strings

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

    # do_registration is int 0/1/2 (skip/run/force) in upstream
    _reg_labels = ["Skip", "Run", "Force"]
    if readonly:
        _reg_label = _reg_labels[settings.do_registration] if 0 <= settings.do_registration <= 2 else "?"
        imgui.text(f"  do_registration = {_reg_label}")
    else:
        imgui.set_next_item_width(hello_imgui.em_size(8))
        _reg_idx = settings.do_registration if 0 <= settings.do_registration <= 2 else 1
        _changed, _new_idx = imgui.combo("do_registration##panel", _reg_idx, _reg_labels)
        if _changed:
            settings.do_registration = _new_idx
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
        imgui.text(f"  [{'x' if settings.do_detection else ' '}] do_detection")
    else:
        _, settings.do_detection = imgui.checkbox("do_detection##panel", settings.do_detection)
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
        imgui.text(f"{settings.diameter_y:.1f}, {settings.diameter_x:.1f}")
    else:
        _, settings.diameter_y = imgui.input_float(
            "##diam_y_panel", settings.diameter_y, 0.5, 2.0, "%.1f"
        )
        imgui.same_line()
        _, settings.diameter_x = imgui.input_float(
            "##diam_x_panel", settings.diameter_x, 0.5, 2.0, "%.1f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Cell diameter (dy, dx) px")

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

    imgui.text("  neuropil_coefficient")
    imgui.same_line(hello_imgui.em_size(14))
    imgui.set_next_item_width(hello_imgui.em_size(6))
    if readonly:
        imgui.text(f"{settings.neuropil_coefficient:.2f}")
    else:
        _, settings.neuropil_coefficient = imgui.input_float(
            "##neuc_panel", settings.neuropil_coefficient, 0.05, 0.1, "%.2f"
        )
    imgui.same_line()
    imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.6, 1.0), "Neuropil coefficient")

    if readonly:
        imgui.text(f"  [{'x' if settings.do_deconvolution else ' '}] do_deconvolution")
    else:
        _, settings.do_deconvolution = imgui.checkbox(
            "do_deconvolution##panel", settings.do_deconvolution
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
    Suite2p pipeline processing settings — mirrors upstream
    `suite2p.default_settings()`.

    Attribute names match upstream canonical names. The dataclass is kept
    flat for GUI ergonomics; `to_dict()` produces the upstream-shaped
    nested dict (run / io / registration / detection / ... ).

    Paths, plane counts, and I/O config live on `Suite2pDB`. Fork/mbo-only
    fields (keep_raw, dff_*, 1P registration, etc.) live on
    `MboSuite2pExtras`.
    """

    # top-level
    torch_device: str = "cuda"  # passed through; _assign_torch_device falls back to cpu on allocation failure
    tau: float = 1.0  # upstream default
    fs: float = 10.0  # upstream default
    diameter_y: float = 12.0  # upstream default [12., 12.]
    diameter_x: float = 12.0

    # run section
    do_registration: int = 1  # 0=skip, 1=run, 2=force
    do_regmetrics: bool = True
    do_detection: bool = True  # was fork's "roidetect"
    do_deconvolution: bool = True  # was fork's "spikedetect"
    multiplane_parallel: bool = False

    # io section
    combined: bool = True
    save_mat: bool = False
    save_NWB: bool = False  # upstream naming
    save_ops_orig: bool = True  # drives whether ops.npy is written per plane
    delete_bin: bool = False
    move_bin: bool = False

    # registration section
    align_by_chan: int = 1  # user-facing 1/2; serialized as bool align_by_chan2
    nimg_init: int = 400  # upstream default (fork was 300)
    maxregshift: float = 0.1
    do_bidiphase: bool = False
    bidiphase: float = 0.0
    reg_batch_size: int = 100  # upstream registration.batch_size (fork conflated with extraction)
    nonrigid: bool = True
    maxregshiftNR: int = 5
    block_size_y: int = 128
    block_size_x: int = 128
    smooth_sigma_time: float = 0.0
    smooth_sigma: float = 1.15
    spatial_taper: float = 3.45  # upstream default; 1P override lives on MboSuite2pExtras
    th_badframes: float = 1.0
    norm_frames: bool = True
    snr_thresh: float = 1.2
    subpixel: int = 10
    two_step_registration: bool = False
    reg_tif: bool = False
    reg_tif_chan2: bool = False

    # detection section (algorithm is derived from sparse_mode + anatomical_only at serialize time)
    sparse_mode: bool = True  # user-visible per spec
    anatomical_only: int = 0  # user-visible per spec; 0..4; non-zero => cellpose
    denoise: bool = False
    det_block_size_y: int = 64
    det_block_size_x: int = 64
    nbins: int = 5000  # was fork's "nbinned"
    highpass_time: int = 100  # was fork's "high_pass"
    threshold_scaling: float = 1.0
    max_overlap: float = 0.75
    soma_crop: bool = True
    chan2_threshold: float = 0.25  # was fork's "chan2_thres" (default differs)
    cellpose_chan2: bool = False

    # detection.sparsery_settings
    highpass_neuropil: int = 25
    max_ROIs: int = 5000
    spatial_scale: int = 0  # upstream default 0=auto
    active_percentile: float = 0.0

    # detection.sourcery_settings
    connected: bool = True
    max_iterations: int = 20
    smooth_masks: bool = False  # upstream default False (fork was True)
    spatial_hp_detect: int = 25  # fork-only (fork puts this under detection) — kept here for GUI parity

    # detection.cellpose_settings
    cellpose_model: str = "cpsam"
    cellpose_img: str = "max_proj / meanImg"
    highpass_spatial: float = 0.0  # was fork's "spatial_hp_cp"
    flow_threshold: float = 0.4  # upstream default (fork was 0)
    cellprob_threshold: float = 0.0  # upstream default (fork was -6)

    # classification section
    classifier_path: str = ""
    use_builtin_classifier: bool = False
    preclassify: float = 0.0
    skip_classification: bool = False
    accept_all_cells: bool = False

    # extraction section
    snr_threshold: float = 0.0
    extract_batch_size: int = 500
    neuropil_extract: bool = True
    neuropil_coefficient: float = 0.7  # was fork's "neucoeff"
    inner_neuropil_radius: int = 2
    min_neuropil_pixels: int = 350
    lam_percentile: int = 50
    allow_overlap: bool = False
    circular_neuropil: bool = False

    # dcnv_preprocess section
    baseline: str = "maximin"  # "maximin" | "prctile" | "constant" (upstream spellings)
    win_baseline: float = 60.0
    sig_baseline: float = 10.0
    prctile_baseline: float = 8.0

    def _derived_algorithm(self) -> str:
        if self.anatomical_only and int(self.anatomical_only) > 0:
            return "cellpose"
        return "sparsery" if self.sparse_mode else "sourcery"

    def to_dict(self) -> dict:
        """Return the upstream-shaped nested settings dict."""
        return {
            "torch_device": self.torch_device,
            "tau": self.tau,
            "fs": self.fs,
            "diameter": [float(self.diameter_y), float(self.diameter_x)],
            "run": {
                "do_registration": int(self.do_registration),
                "do_regmetrics": self.do_regmetrics,
                "do_detection": self.do_detection,
                "do_deconvolution": self.do_deconvolution,
                "multiplane_parallel": self.multiplane_parallel,
            },
            "io": {
                "combined": self.combined,
                "save_mat": self.save_mat,
                "save_NWB": self.save_NWB,
                "save_ops_orig": self.save_ops_orig,
                "delete_bin": self.delete_bin,
                "move_bin": self.move_bin,
            },
            "registration": {
                "align_by_chan2": bool(self.align_by_chan == 2),
                "nimg_init": self.nimg_init,
                "maxregshift": self.maxregshift,
                "do_bidiphase": self.do_bidiphase,
                "bidiphase": self.bidiphase,
                "batch_size": self.reg_batch_size,
                "nonrigid": self.nonrigid,
                "maxregshiftNR": self.maxregshiftNR,
                "block_size": (int(self.block_size_y), int(self.block_size_x)),
                "smooth_sigma_time": self.smooth_sigma_time,
                "smooth_sigma": self.smooth_sigma,
                "spatial_taper": self.spatial_taper,
                "th_badframes": self.th_badframes,
                "norm_frames": self.norm_frames,
                "snr_thresh": self.snr_thresh,
                "subpixel": self.subpixel,
                "two_step_registration": self.two_step_registration,
                "reg_tif": self.reg_tif,
                "reg_tif_chan2": self.reg_tif_chan2,
            },
            "detection": {
                "algorithm": self._derived_algorithm(),
                "denoise": self.denoise,
                "block_size": (int(self.det_block_size_y), int(self.det_block_size_x)),
                "nbins": self.nbins,
                "highpass_time": self.highpass_time,
                "threshold_scaling": self.threshold_scaling,
                "max_overlap": self.max_overlap,
                "soma_crop": self.soma_crop,
                "chan2_threshold": self.chan2_threshold,
                "cellpose_chan2": self.cellpose_chan2,
                "sparsery_settings": {
                    "highpass_neuropil": self.highpass_neuropil,
                    "max_ROIs": self.max_ROIs,
                    "spatial_scale": self.spatial_scale,
                    "active_percentile": self.active_percentile,
                },
                "sourcery_settings": {
                    "connected": self.connected,
                    "max_iterations": self.max_iterations,
                    "smooth_masks": self.smooth_masks,
                },
                "cellpose_settings": {
                    "cellpose_model": self.cellpose_model,
                    "img": self.cellpose_img,
                    "highpass_spatial": self.highpass_spatial,
                    "flow_threshold": self.flow_threshold,
                    "cellprob_threshold": self.cellprob_threshold,
                },
            },
            "classification": {
                "classifier_path": self.classifier_path,
                "use_builtin_classifier": self.use_builtin_classifier,
                "preclassify": self.preclassify,
                "skip_classification": self.skip_classification,
                "accept_all_cells": self.accept_all_cells,
            },
            "extraction": {
                "snr_threshold": self.snr_threshold,
                "batch_size": self.extract_batch_size,
                "neuropil_extract": self.neuropil_extract,
                "neuropil_coefficient": self.neuropil_coefficient,
                "inner_neuropil_radius": self.inner_neuropil_radius,
                "min_neuropil_pixels": self.min_neuropil_pixels,
                "lam_percentile": self.lam_percentile,
                "allow_overlap": self.allow_overlap,
                "circular_neuropil": self.circular_neuropil,
            },
            "dcnv_preprocess": {
                "baseline": self.baseline,
                "win_baseline": self.win_baseline,
                "sig_baseline": self.sig_baseline,
                "prctile_baseline": self.prctile_baseline,
            },
        }

    def to_file(self, filepath):
        """Save settings to settings.npy (upstream-shaped nested dict)."""
        np.save(filepath, _convert_paths_to_strings(self.to_dict()), allow_pickle=True)


@dataclass
class Suite2pDB:
    """
    Suite2p input/output database — mirrors upstream `suite2p.default_db()`.
    These are paths, plane counts, and I/O flags that identify *what* to
    process (not *how*).
    """

    data_path: list | None = None  # list of directories containing input files
    look_one_level_down: bool = False
    input_format: str = "tif"  # "tif" | "h5" | "nwb" | "bruker" | "movie" | "dcimg"
    keep_movie_raw: bool = False
    nplanes: int = 1
    nrois: int = 1
    nchannels: int = 1
    swap_order: bool = False
    functional_chan: int = 1  # moved here from settings
    subfolders: list | None = None
    save_path0: str = ""
    fast_disk: str = ""
    save_folder: str = "suite2p"
    h5py_key: str = "data"
    nwb_driver: str = ""
    nwb_series: str = ""
    force_sktiff: bool = False
    bruker_bidirectional: bool = False

    def __post_init__(self):
        if self.data_path is None:
            self.data_path = []
        if self.subfolders is None:
            self.subfolders = []

    def to_dict(self) -> dict:
        return {
            "data_path": list(self.data_path or []),
            "look_one_level_down": self.look_one_level_down,
            "input_format": self.input_format,
            "keep_movie_raw": self.keep_movie_raw,
            "nplanes": self.nplanes,
            "nrois": self.nrois,
            "nchannels": self.nchannels,
            "swap_order": self.swap_order,
            "functional_chan": self.functional_chan,
            "subfolders": list(self.subfolders or []),
            "save_path0": self.save_path0,
            "fast_disk": self.fast_disk,
            "save_folder": self.save_folder,
            "h5py_key": self.h5py_key,
            "nwb_driver": self.nwb_driver,
            "nwb_series": self.nwb_series,
            "force_sktiff": self.force_sktiff,
            "bruker_bidirectional": self.bruker_bidirectional,
        }

    def to_file(self, filepath):
        """Save db to db.npy (upstream-shaped flat dict)."""
        np.save(filepath, _convert_paths_to_strings(self.to_dict()), allow_pickle=True)


@dataclass
class MboSuite2pExtras:
    """
    Fork/mbo-specific fields that have no upstream home. Kept on a
    dedicated object so `Suite2pSettings` can stay in lockstep with
    upstream's schema.

    These are consumed by `lbm_suite2p_python.run_plane`'s own kwargs or
    by mbo's pre/post-processing (dF/F, target_timepoints clipping).
    """

    # lbm_suite2p_python.run_plane kwargs
    keep_raw: bool = False
    keep_reg: bool = True
    force_reg: bool = False
    force_detect: bool = False
    dff_window_size: int = 300
    dff_percentile: int = 20
    dff_smooth_window: int = 0

    # mbo-side timepoint / plane selection
    target_timepoints: int = -1
    frames_include: int = -1

    # gui-only display
    aspect: float = 1.0
    report_time: bool = True

    # fork-only registration flags (no upstream equivalent)
    force_refImg: bool = False
    pad_fft: bool = False
    chan2_file: str = ""

    # 1P-specific (fork-only)
    do_1Preg: bool = False
    spatial_hp_reg: int = 42
    pre_smooth: float = 0.0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()  # type: ignore
        }


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

            # fallback: if multiple files loaded but each is single-plane natively,
            # treat the list of files as a volumetric stack (used when reloading Suite2p output)
            if isinstance(self.fpath, (list, tuple)) and len(self.fpath) > 1:
                if num_planes == 1:
                    num_planes = len(self.fpath)

            if hasattr(data, "num_color_channels"):
                num_channels = data.num_color_channels
            elif hasattr(data, "_num_color_channels"):
                num_channels = data._num_color_channels
            # fallback: detect channels from dims/shape for 5D data
            if num_channels <= 1 and data.ndim == 5:
                from mbo_utilities.arrays.features import get_dims
                dims = get_dims(data)
                if dims is not None and len(dims) >= 5 and dims[1] == "C":
                    num_channels = data.shape[1]
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

    # always ensure _selected_planes stays purely synced to the current z slicing logic
    if num_planes > 1:
        self._selected_planes = set(range(self._s2p_z_start, self._s2p_z_stop + 1, self._s2p_z_step))
    else:
        self._selected_planes = {1}

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

        # (logic moved to unconditionally sync in _init_s2p_selection_state)

        # update s2p.target_timepoints from selection
        if tp_parsed:
            self.s2p_extras.target_timepoints = tp_parsed.count

        imgui.spacing()
        imgui.spacing()
        if imgui.button("Close", imgui.ImVec2(80, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


def _draw_data_options_content(self):
    """Draw data options content showing settings that affect Suite2p processing."""

    INPUT_WIDTH = 100
    has_phase_support = getattr(self, "has_raster_scan_support", False)
    nz = getattr(self, "nz", 1)

    # check if data already has z-registration applied
    already_registered = False
    arrays = self._get_data_arrays() if hasattr(self, "_get_data_arrays") else []
    if arrays and hasattr(arrays[0], "metadata"):
        already_registered = arrays[0].metadata.get("apply_shift", False)

    is_raw = getattr(self, "is_mbo_scan", False)
    has_z_reg = nz > 1 and is_raw and not already_registered

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

    # axial z-registration - only for raw scanimage data
    if nz > 1 and is_raw and not already_registered:
        if has_phase_support:
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Axial Registration")
        imgui.spacing()

        if has_z_reg:
            reg_z_val = getattr(self, "_register_z", False)
            reg_changed, reg_value = imgui.checkbox("Register Z-Planes Axially", reg_z_val)
            set_tooltip(
                "Compute per-plane rigid shifts via phase correlation and\n"
                "apply them on write. Corrects z-drift between adjacent planes.\n"
                "Uses GPU (cupy) if available, otherwise CPU."
            )
            if reg_changed:
                self._register_z = reg_value

            if self._register_z:
                if not hasattr(self, "_axial_max_frames"):
                    self._axial_max_frames = 200
                if not hasattr(self, "_axial_max_reg_xy"):
                    self._axial_max_reg_xy = 150

                imgui.indent(16)
                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max frames", self._axial_max_frames, 50, 100
                )
                if _changed:
                    self._axial_max_frames = max(10, _val)
                set_tooltip(
                    "Frames subsampled (evenly spaced) for the time-mean used\n"
                    "to compute plane shifts. 200 is usually plenty."
                )

                imgui.set_next_item_width(80)
                _changed, _val = imgui.input_int(
                    "Max shift (px)", self._axial_max_reg_xy, 10, 50
                )
                if _changed:
                    self._axial_max_reg_xy = max(1, _val)
                set_tooltip(
                    "Max shift search radius in pixels. Default 150."
                )
                imgui.unindent(16)
        else:
            imgui.begin_disabled()
            imgui.checkbox("Register Z-Planes Axially", False)
            imgui.end_disabled()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Requires multi-plane raw data",
            )

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

    # self.s2p is a lazy property: returns None when HAS_SUITE2P is False
    # or when the lazy import failed. bail early so the later rows
    # (`self.s2p.do_registration`, etc.) don't raise AttributeError
    # mid-table and leave imgui's table scope open.
    if self.s2p is None:
        imgui.text_colored(
            imgui.ImVec4(1.0, 0.7, 0.2, 1.0),
            "Suite2p is not installed."
        )
        imgui.text("Install with:")
        imgui.text_colored(
            imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
            "uv pip install mbo_utilities[suite2p]"
        )
        return

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

    imgui.same_line()
    if imgui.button("Metadata"):
        self._saveas_popup_open = True
        self._saveas_select_metadata_tab = True
    set_tooltip("Edit dataset metadata prior to processing")

    # --- Device badge (torch_device from Main settings) ---------------------
    # show the current torch device + its real-world availability so users
    # see at a glance whether the run will actually use GPU. cached per
    # frame to avoid re-probing torch.cuda every frame.
    imgui.same_line()
    if not hasattr(self, "_device_probe_cache"):
        self._device_probe_cache = {}
    device_setting = getattr(self.s2p, "torch_device", "cuda")
    probe_key = device_setting
    if probe_key not in self._device_probe_cache:
        actual = device_setting
        reason = ""
        try:
            import torch
            if device_setting == "cuda":
                if not torch.cuda.is_available():
                    actual = "cpu"
                    reason = "CUDA unavailable at runtime; will fall back to CPU"
            elif device_setting == "mps":
                mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                if not mps_ok:
                    actual = "cpu"
                    reason = "MPS unavailable at runtime; will fall back to CPU"
        except ImportError:
            actual = "cpu"
            reason = "torch not installed"
        self._device_probe_cache[probe_key] = (actual, reason)
    actual_device, probe_reason = self._device_probe_cache[probe_key]

    gpu_like = actual_device in ("cuda", "mps")
    if actual_device == device_setting:
        label = f"[{device_setting.upper()}]"
        color = imgui.ImVec4(0.3, 0.85, 0.3, 1.0) if gpu_like else imgui.ImVec4(0.75, 0.75, 0.75, 1.0)
        tip = (
            f"Compute device: {device_setting} (GPU accelerated)"
            if gpu_like else
            "Compute device: cpu (no GPU acceleration)"
        ) + "\nChange via Run → Main → Settings → Torch Device"
    else:
        label = f"[{device_setting.upper()}→{actual_device.upper()}]"
        color = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
        tip = f"{probe_reason}\nChange via Run → Main → Settings → Torch Device"
    imgui.text_colored(color, label)
    set_tooltip(tip)

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
        run_process(self)

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

    # --- Main settings (closure, drawn inside the popup block below
    # alongside Registration / ROI Detection / etc., so it follows the
    # same begin_popup_modal lifecycle as every other settings panel) ---
    def draw_main_settings():
        # --- Torch device (top-level upstream setting) ----------------------
        device_options = ["cuda", "cpu", "mps"]
        current_device_idx = (
            device_options.index(self.s2p.torch_device)
            if self.s2p.torch_device in device_options
            else 0
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        device_changed, selected_device_idx = imgui.combo(
            "Torch Device", current_device_idx, device_options
        )
        if device_changed:
            self.s2p.torch_device = device_options[selected_device_idx]
            # badge probes torch.cuda once and caches; drop that cache so
            # the badge reflects the new setting on the next frame
            if hasattr(self, "_device_probe_cache"):
                self._device_probe_cache.clear()
        set_tooltip(
            "GPU device for registration / detection / extraction / dcnv.\n"
            "'cuda' falls back to 'cpu' at runtime if allocation fails."
        )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Tau / fs / denoise
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.tau = imgui.input_float("Tau (s)", self.s2p.tau)
        set_tooltip(
            "Calcium indicator decay timescale in seconds.\n"
            "GCaMP6f=0.7, GCaMP6m=1.0-1.3 (LBM default), GCaMP6s=1.25-1.5"
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.fs = imgui.input_float("Sampling Rate (Hz)", self.s2p.fs)
        set_tooltip("Per-plane sampling rate in Hz; drives baseline window sizing.")
        _, self.s2p.denoise = imgui.checkbox("Denoise Movie", self.s2p.denoise)
        set_tooltip("Denoise binned movie before cell detection.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # processing control options
        _, self.s2p_extras.keep_raw = imgui.checkbox("Keep Raw Binary", self.s2p_extras.keep_raw)
        set_tooltip("Keep data_raw.bin after processing (uses disk space)")
        _, self.s2p_extras.keep_reg = imgui.checkbox("Keep Registered Binary", self.s2p_extras.keep_reg)
        set_tooltip("Keep data.bin after processing (useful for QC)")
        _, self.s2p_extras.force_reg = imgui.checkbox("Force Re-registration", self.s2p_extras.force_reg)
        set_tooltip("Force re-registration even if already processed")
        _, self.s2p_extras.force_detect = imgui.checkbox("Force Re-detection", self.s2p_extras.force_detect)
        set_tooltip("Force ROI detection even if stat.npy exists")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # dF/F settings
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "\u0394F/F")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p_extras.dff_window_size = imgui.input_int("Window", self.s2p_extras.dff_window_size)
        set_tooltip("Frames for rolling percentile baseline (default: 300)")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p_extras.dff_percentile = imgui.input_int("Percentile", self.s2p_extras.dff_percentile)
        set_tooltip("Percentile for baseline F\u2080 estimation (default: 20)")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p_extras.dff_smooth_window = imgui.input_int("Smooth", self.s2p_extras.dff_smooth_window)
        set_tooltip("Smooth \u0394F/F trace with rolling window (0 = disabled)")

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
        num_planes_main = 1
        n_selected_main = 1
        try:
            if hasattr(self, "image_widget") and self.image_widget.data:
                data = self.image_widget.data[0]
                if hasattr(data, "num_planes"):
                    num_planes_main = data.num_planes
                elif data.ndim == 4:
                    num_planes_main = data.shape[1]
                elif data.ndim == 5:
                    num_planes_main = data.shape[2]
            n_selected_main = len(getattr(self, "_selected_planes", {1}))
        except Exception:
            pass

        if num_planes_main > 1 and n_selected_main > 1:
            _, self._parallel_processing = imgui.checkbox(
                "Parallel plane processing", self._parallel_processing
            )
            set_tooltip("Process multiple planes simultaneously (uses more memory)")
            if self._parallel_processing:
                imgui.set_next_item_width(INPUT_WIDTH)
                _, self._max_parallel_jobs = imgui.input_int(
                    "Max parallel jobs", self._max_parallel_jobs, step=1
                )
                self._max_parallel_jobs = max(1, min(self._max_parallel_jobs, n_selected_main))

    # --- Registration ---
    def draw_registration_settings():
        # do_registration is int 0/1/2 in upstream: 0=skip, 1=run, 2=force re-run.
        # Exposed as three radio buttons on the same row.
        imgui.text("Run Registration:")
        imgui.same_line()
        if imgui.radio_button("Skip##do_reg", self.s2p.do_registration == 0):
            self.s2p.do_registration = 0
        imgui.same_line()
        if imgui.radio_button("Run##do_reg", self.s2p.do_registration == 1):
            self.s2p.do_registration = 1
        imgui.same_line()
        if imgui.radio_button("Force##do_reg", self.s2p.do_registration == 2):
            self.s2p.do_registration = 2
        set_tooltip(
            "0=skip registration entirely\n"
            "1=run if not already done (default)\n"
            "2=force re-run even if ops.npy reports registration done"
        )

        imgui.begin_disabled(self.s2p.do_registration == 0)

        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.align_by_chan = imgui.input_int(
            "Align by Channel", self.s2p.align_by_chan
        )
        set_tooltip("Channel index used for alignment (1-based). "
                    "Serialized to settings['registration']['align_by_chan2'] = (align_by_chan == 2).")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nimg_init = imgui.input_int("Initial Frames", self.s2p.nimg_init)
        set_tooltip("Number of frames used to build the reference image.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.reg_batch_size = imgui.input_int("Batch Size", self.s2p.reg_batch_size)
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
        _, self.s2p_db.keep_movie_raw = imgui.checkbox(
            "Keep Raw Movie", self.s2p_db.keep_movie_raw
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
        _, self.s2p_extras.force_refImg = imgui.checkbox("Force refImg", self.s2p_extras.force_refImg)
        set_tooltip("Use stored reference image instead of recomputing.")
        _, self.s2p_extras.pad_fft = imgui.checkbox("Pad FFT", self.s2p_extras.pad_fft)
        set_tooltip("Pad image for FFT registration to reduce edge artifacts.")

        imgui.spacing()
        imgui.text("Channel 2 File:")
        imgui.push_text_wrap_pos(imgui.get_content_region_avail().x)
        imgui.text(self.s2p_extras.chan2_file if self.s2p_extras.chan2_file else "(none)")
        imgui.pop_text_wrap_pos()
        if imgui.button("Browse##chan2"):
            default_dir = str(get_last_dir("suite2p_chan2") or pathlib.Path().home())
            res = pfd.open_file("Select channel 2 file", default_dir)
            if res and res.result():
                self.s2p_extras.chan2_file = res.result()[0]
                set_last_dir("suite2p_chan2", res.result()[0])
        set_tooltip("Path to channel 2 binary file for cross-channel registration.")

        if imgui.tree_node("1-Photon Registration"):
            _, self.s2p_extras.do_1Preg = imgui.checkbox(
                "Enable 1P Registration", self.s2p_extras.do_1Preg
            )
            set_tooltip("Apply high-pass filtering and tapering for 1-photon data.")

            imgui.begin_disabled(not self.s2p_extras.do_1Preg)
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p_extras.spatial_hp_reg = imgui.input_int(
                "Spatial HP Window", self.s2p_extras.spatial_hp_reg
            )
            set_tooltip(
                "Window size for spatial high-pass filtering before registration."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p_extras.pre_smooth = imgui.input_float(
                "Pre-smooth Sigma", self.s2p_extras.pre_smooth
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

            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.block_size_y = imgui.input_int(
                "Block Height", self.s2p.block_size_y
            )
            set_tooltip(
                "Block height for non-rigid registration (power of 2/3 recommended)."
            )
            imgui.set_next_item_width(INPUT_WIDTH)
            _, self.s2p.block_size_x = imgui.input_int(
                "Block Width", self.s2p.block_size_x
            )
            set_tooltip(
                "Block width for non-rigid registration (power of 2/3 recommended)."
            )

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

        imgui.end_disabled()  # closes do_registration == 0 gate

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

        # Diameter is a 2-float list [dy, dx] upstream. Show as two scalars with
        # a lock that keeps them synced on edit.
        if not hasattr(self, "_s2p_diameter_lock"):
            self._s2p_diameter_lock = True
        imgui.set_next_item_width(INPUT_WIDTH)
        dy_changed, dy = imgui.input_float("Diameter Y (px)", self.s2p.diameter_y)
        set_tooltip(
            "Expected cell diameter (Y axis) in pixels. Passed to Cellpose.\n"
            "Upstream default: 12."
        )
        imgui.set_next_item_width(INPUT_WIDTH)
        dx_changed, dx = imgui.input_float("Diameter X (px)", self.s2p.diameter_x)
        set_tooltip(
            "Expected cell diameter (X axis) in pixels. Passed to Cellpose.\n"
            "Upstream default: 12."
        )
        _, self._s2p_diameter_lock = imgui.checkbox(
            "Lock Y/X", self._s2p_diameter_lock
        )
        set_tooltip("Keep diameter_y and diameter_x in sync when editing either.")
        if dy_changed:
            self.s2p.diameter_y = dy
            if self._s2p_diameter_lock:
                self.s2p.diameter_x = dy
        if dx_changed:
            self.s2p.diameter_x = dx
            if self._s2p_diameter_lock:
                self.s2p.diameter_y = dx
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
        _, self.s2p.highpass_spatial = imgui.input_float(
            "Spatial HP (Cellpose)", self.s2p.highpass_spatial
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
            _, self.s2p_db.functional_chan = imgui.input_int(
                "Functional Channel", self.s2p_db.functional_chan
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
        _, self.s2p.highpass_time = imgui.input_int("High-Pass Window", self.s2p.highpass_time)
        set_tooltip("Running mean subtraction window for temporal high-pass filtering.")
        _, self.s2p.smooth_masks = imgui.checkbox("Smooth Masks", self.s2p.smooth_masks)
        set_tooltip("Smooth masks in the final ROI detection pass.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.nbins = imgui.input_int("Max Binned Frames", self.s2p.nbins)
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
        _, self.s2p.neuropil_coefficient = imgui.input_float(
            "Neuropil Coefficient", self.s2p.neuropil_coefficient
        )
        set_tooltip(
            "Neuropil coefficient for all ROIs (F_corrected = F - coeff * F_neu)."
        )

        # Baseline method as combo box. Upstream spelling is "prctile"
        # (not "constant_percentile"); the lbm_suite2p_python db_settings
        # helper round-trips it to the fork's "constant_prctile" literal
        # at pipeline entry so the fork's dcnv.preprocess still branches.
        baseline_options = ["maximin", "constant", "prctile"]
        # accept legacy values silently; normalize to upstream spelling
        _legacy_map = {
            "constant_percentile": "prctile",
            "constant_prctile": "prctile",
        }
        if self.s2p.baseline in _legacy_map:
            self.s2p.baseline = _legacy_map[self.s2p.baseline]
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
            "maximin: moving baseline with min/max filters.\n"
            "constant: minimum of Gaussian-filtered trace.\n"
            "prctile: percentile of trace (controlled by 'Baseline Percentile')."
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
        set_tooltip("Percentile of trace for 'prctile' baseline method.")

    # --- Output Settings ---
    def draw_output_settings():
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.preclassify = imgui.input_float(
            "Preclassify Threshold", self.s2p.preclassify
        )
        set_tooltip("Probability threshold to apply classifier before extraction.")
        _, self.s2p.save_NWB = imgui.checkbox("Save NWB", self.s2p.save_NWB)
        _, self.s2p.save_mat = imgui.checkbox("Save MATLAB File", self.s2p.save_mat)
        set_tooltip("Export results to Fall.mat for MATLAB analysis.")
        _, self.s2p.combined = imgui.checkbox(
            "Combine Across Planes", self.s2p.combined
        )
        set_tooltip("Combine per-plane results into one GUI-loadable folder.")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p_extras.aspect = imgui.input_float("Aspect Ratio", self.s2p_extras.aspect)
        set_tooltip("um/pixel ratio X/Y for correct GUI aspect display.")
        _, self.s2p_extras.report_time = imgui.checkbox("Report Timing", self.s2p_extras.report_time)
        set_tooltip("Return timing dictionary for each processing stage.")

        # Channel 2 settings
        imgui.spacing()
        imgui.separator()
        imgui.text("Channel 2:")
        imgui.set_next_item_width(INPUT_WIDTH)
        _, self.s2p.chan2_threshold = imgui.input_float(
            "Chan2 Detection Threshold", self.s2p.chan2_threshold
        )
        set_tooltip("Threshold for calling ROI detected on channel 2.")

    # === PIPELINE SETTINGS TABLE ===
    # tabular layout: [checkbox/empty] | [label] | [settings button]
    # wrapped in try/finally: draw_run_tab catches exceptions from
    # pipeline.draw(), so any attribute error in a row (e.g. self.s2p
    # returning None) would otherwise leave end_table uncalled and
    # surface as "Missing EndTable()" on the outer imgui.end().
    table_flags = imgui.TableFlags_.sizing_fixed_fit | imgui.TableFlags_.no_borders_in_body
    if imgui.begin_table("pipeline_settings_table", 3, table_flags):
        try:
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
                _popup_states["main_settings"] = True
                imgui.open_popup("Main##main_settings")
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
            # do_registration is an int 0/1/2 upstream; the table-level control
            # is a simple on/off toggle that flips between 0 and 1 (user can
            # pick 2=Force inside the Registration popup).
            imgui.table_next_row()
            imgui.table_next_column()
            _reg_on = self.s2p.do_registration != 0
            _reg_changed, _reg_on = imgui.checkbox("##reg_cb", _reg_on)
            if _reg_changed:
                self.s2p.do_registration = 1 if _reg_on else 0
            set_tooltip("Enable/disable motion registration (open popup for Skip/Run/Force)")
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
            _, self.s2p.do_detection = imgui.checkbox("##roi_cb", self.s2p.do_detection)
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
            _, self.s2p.do_deconvolution = imgui.checkbox("##spike_cb", self.s2p.do_deconvolution)
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

            # --- Draw all popups INSIDE the table/if-block ---
            # imgui's open_popup → begin_popup_modal chain only fires when
            # both live under the same parent scope. Moving the popup
            # begins outside `if imgui.begin_table(...)` makes the open_popup
            # calls (which happen inside the if) unreachable from the popup
            # begins (which happen outside), and the modals silently never
            # appear. Keep them in-scope; imgui handles popup windows as
            # their own layer regardless of the surrounding table.

            # Main settings popup
            imgui.set_next_window_size(imgui.ImVec2(350, 0), imgui.Cond_.first_use_ever)
            opened, visible = imgui.begin_popup_modal(
                "Main##main_settings",
                p_open=True,
                flags=imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.always_auto_resize,
            )
            if opened:
                try:
                    if not visible:
                        _popup_states["main_settings"] = False
                        imgui.close_current_popup()
                    else:
                        draw_main_settings()
                        imgui.spacing()
                        imgui.separator()
                        if imgui.button("Close##main", imgui.ImVec2(80, 0)):
                            _popup_states["main_settings"] = False
                            imgui.close_current_popup()
                finally:
                    imgui.end_popup()

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
        finally:
            imgui.end_table()


def _build_channel_dirname(self, channel: int) -> str:
    """build dimension-tagged dirname for a single channel run."""
    from mbo_utilities.arrays.features import DimensionTag, TAG_REGISTRY

    tags = []
    # channel first (varies across sibling dirs)
    tags.append(DimensionTag(TAG_REGISTRY["C"], start=channel, stop=None, step=1))
    # z-planes
    z_start = getattr(self, "_s2p_z_start", 1)
    z_stop = getattr(self, "_s2p_z_stop", 1)
    z_step = getattr(self, "_s2p_z_step", 1)
    tags.append(DimensionTag(TAG_REGISTRY["Z"], start=z_start, stop=z_stop, step=z_step))
    # timepoints last (same across sibling dirs)
    tp = self._s2p_tp_parsed
    if tp and tp.final_indices:
        tp_start = tp.final_indices[0] + 1
        tp_stop = tp.final_indices[-1] + 1
        tags.append(DimensionTag(TAG_REGISTRY["T"], start=tp_start, stop=tp_stop, step=1))
    return "_".join(tag.to_string() for tag in tags)


def run_process(self):
    """Runs the selected processing pipeline."""
    if self._current_pipeline != "suite2p":
        if self._current_pipeline == "masknmf":
            self.logger.info("Running MaskNMF pipeline (not yet implemented).")
        else:
            self.logger.error(f"Unknown pipeline selected: {self._current_pipeline}")
        return

    self.logger.debug(f"suite2p settings: {self.s2p}")
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

            # self.fpath is the array's canonical source_path — a single
            # Path that imread can use to reconstruct the full volume.
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
            # always pass channel for multi-channel source data (5D needs _ChannelView)
            has_channels = getattr(self, "_s2p_last_num_channels", 1) > 1

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
                    "num_timepoints": self.s2p_extras.target_timepoints,
                    # upstream-shaped pair; the subprocess entry point flattens
                    # these into ops via lbm_suite2p_python.db_settings. `ops`
                    # is intentionally omitted — settings+db is the canonical
                    # config path going forward.
                    "settings": self.s2p.to_dict(),
                    "db": self.s2p_db.to_dict(),
                    "fix_phase": self._s2p_fix_phase,
                    "use_fft": self._s2p_use_fft,
                    "channel": channel if (multi_channel or has_channels) else None,
                    # User-set metadata from the GUI metadata editor
                    # (lives on parent._custom_metadata, NOT on the source
                    # file). The worker merges this into ops before
                    # invoking lbm_suite2p_python.pipeline so the user's
                    # z_step / dx / dy / fs reach ops.npy.
                    "custom_metadata": dict(getattr(self, "_custom_metadata", {})),
                    # User's timepoint selection (0-based, full list).
                    # The subprocess uses this with OutputMetadata to
                    # reactively scale fs based on the timepoint stride.
                    "tp_indices": (
                        list(self._s2p_tp_parsed.final_indices)
                        if getattr(self, "_s2p_tp_parsed", None) is not None
                        else None
                    ),
                    # FULL list of selected planes (0-based) — for
                    # reactive dz scaling via OutputMetadata.
                    "selected_planes_0based": [p - 1 for p in sorted(selected_planes)],
                    # axial registration
                    "register_z": getattr(self, "_register_z", False),
                    "max_frames": getattr(self, "_axial_max_frames", 200),
                    "max_reg_xy": getattr(self, "_axial_max_reg_xy", 150),
                    "s2p_settings": {
                        "keep_raw": self.s2p_extras.keep_raw,
                        "keep_reg": self.s2p_extras.keep_reg,
                        "force_reg": self.s2p_extras.force_reg,
                        "force_detect": self.s2p_extras.force_detect,
                        "dff_window_size": self.s2p_extras.dff_window_size,
                        "dff_percentile": self.s2p_extras.dff_percentile,
                        "dff_smooth_window": self.s2p_extras.dff_smooth_window,
                    },
                }

                description = f"Suite2p: {len(selected_planes)} plane(s)"
                if multi_channel or has_channels:
                    description += f" ch{channel}"
                if roi:
                    description += f" ROI {roi}"

                pid = pm.spawn(
                    task_type="suite2p",
                    args=worker_args,
                    description=description,
                    output_path=output_dir,
                )

                if not pid:
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
            has_channels = getattr(self, "_s2p_last_num_channels", 1) > 1

            s2p_path = getattr(self, "_s2p_outdir", "") or getattr(
                self, "_saveas_outdir", ""
            )

            # Pre-extract shared state (upstream-shaped dicts for settings/db)
            s2p_settings_dict = self.s2p.to_dict()
            s2p_db_dict = self.s2p_db.to_dict()
            target_timepoints = self.s2p_extras.target_timepoints
            keep_raw = self.s2p_extras.keep_raw
            keep_reg = self.s2p_extras.keep_reg
            force_reg = self.s2p_extras.force_reg
            force_detect = self.s2p_extras.force_detect
            dff_window_size = self.s2p_extras.dff_window_size
            dff_percentile = self.s2p_extras.dff_percentile
            dff_smooth_window = self.s2p_extras.dff_smooth_window
            fix_phase = getattr(self, "_s2p_fix_phase", False)
            use_fft = getattr(self, "_s2p_use_fft", False)

            if not s2p_path:
                from mbo_utilities.preferences import get_mbo_dirs
                from mbo_utilities.file_io import get_last_savedir_path
                last_savedir = get_last_savedir_path()
                s2p_path = str(Path(last_savedir) if last_savedir else get_mbo_dirs()["data"])

            fpath_str = str(self.fpath) if self.fpath else ""

            # Check if parallel processing is enabled
            use_parallel = getattr(self, "_parallel_processing", False)
            max_jobs = getattr(self, "_max_parallel_jobs", 2)

            # Snapshot the user's selections so the worker can rebuild
            # the OutputMetadata reactive layer (fs/dz reactively scaled
            # by the timepoint and z-plane stride). Without these, the
            # worker would fall back to source values and the resulting
            # ops.npy reports the wrong fs/dz when the user struds the
            # selection.
            tp_indices = (
                list(self._s2p_tp_parsed.final_indices)
                if getattr(self, "_s2p_tp_parsed", None) is not None
                else None
            )
            selected_planes_0based = [p - 1 for p in sorted(selected_planes)]

            # Build list of configuration dicts for each job to completely decouple GUI state
            jobs = []
            for i, _arr in enumerate(self.image_widget.data):
                for channel in selected_channels:
                    if multi_channel:
                        base_out = Path(s2p_path) / _build_channel_dirname(self, channel)
                    else:
                        base_out = Path(s2p_path)

                    for z_plane in sorted(selected_planes):
                        current_z = z_plane - 1

                        if len(self.image_widget.graphics) > 1:
                            plane_dir = base_out / f"plane{z_plane:02d}_roi{i + 1:02d}"
                            roi = i + 1
                        else:
                            plane_dir = base_out / f"plane{z_plane:02d}_stitched"
                            roi = None

                        import copy as _copy
                        config = {
                            "arr": _arr,
                            "arr_idx": i,
                            "z_plane": current_z,
                            "plane": z_plane,
                            "channel": channel if (multi_channel or has_channels) else None,
                            "base_out": base_out,
                            "plane_dir": plane_dir,
                            "roi": roi,
                            "s2p_settings_dict": _copy.deepcopy(s2p_settings_dict),
                            "s2p_db_dict": _copy.deepcopy(s2p_db_dict),
                            "target_timepoints": target_timepoints,
                            "fpath": fpath_str,
                            "keep_raw": keep_raw,
                            "keep_reg": keep_reg,
                            "force_reg": force_reg,
                            "force_detect": force_detect,
                            "dff_window_size": dff_window_size,
                            "dff_percentile": dff_percentile,
                            "dff_smooth_window": dff_smooth_window,
                            "fix_phase": fix_phase,
                            "use_fft": use_fft,
                            # User-set metadata (e.g. dz from the metadata
                            # editor) lives on parent._custom_metadata, NOT
                            # on arr.metadata. Snapshot it here so the
                            # worker can merge it before computing voxel
                            # size — otherwise the user's z_step is
                            # silently dropped and the source-file dz
                            # (often None for LBM, 1.0 default otherwise)
                            # ends up in ops.npy.
                            "custom_metadata": dict(getattr(self, "_custom_metadata", {})),
                            # User's timepoint selection — 0-based final
                            # indices from TimeSelection. None means all.
                            "tp_indices": tp_indices,
                            # FULL list of all selected planes (0-based)
                            # — needed by OutputMetadata to compute the
                            # z-step factor reactively, even though each
                            # job processes only one plane at a time.
                            "selected_planes_0based": selected_planes_0based,
                            "logger": self.logger
                        }

                        # Fix Issue 2: restrict Suite2p to single internal process if using ThreadPoolExecutor
                        if use_parallel:
                            # num_workers isn't part of upstream's schema but the
                            # fork reads it off the flat ops dict; inject into
                            # settings via an extra key that survives the
                            # flatten (db_settings_to_ops preserves unknown
                            # top-level keys).
                            config["num_workers_override"] = 0

                        jobs.append(config)

            if use_parallel and len(jobs) > 1:
                # Parallel processing with limited concurrency
                from concurrent.futures import ThreadPoolExecutor

                def run_parallel():
                    self.logger.info(
                        f"Starting parallel processing with max {max_jobs} concurrent jobs..."
                    )
                    # Fix Issue 3: track executor via 'self' which can handle gracefully closing out tasks on GUI shutdown
                    executor = ThreadPoolExecutor(max_workers=max_jobs)
                    self._active_executor = executor
                    try:
                        futures = {}
                        for job_idx, config in enumerate(jobs):
                            future = executor.submit(_run_plane_worker_thread, config)
                            futures[future] = (config, job_idx)

                        from concurrent.futures import as_completed
                        for future in as_completed(futures):
                            config, job_idx = futures[future]
                            try:
                                future.result()
                                desc = f"Plane {config['plane']}"
                                if config['channel'] is not None:
                                    desc += f" ch{config['channel']}"
                                self.logger.info(
                                    f"{desc} completed ({job_idx + 1}/{len(jobs)})"
                                )
                            except Exception as e:
                                self.logger.exception(
                                    f"Error processing plane {config['plane']}: {e}"
                                )
                    finally:
                        executor.shutdown(wait=False)
                        if getattr(self, "_active_executor", None) is executor:
                            self._active_executor = None
                    self.logger.info("Suite2p parallel processing complete.")

                threading.Thread(target=run_parallel, daemon=True).start()
            else:
                # Sequential processing in a single background thread
                def run_all_planes_sequential():
                    for job_idx, config in enumerate(jobs):
                        desc = f"plane {config['plane']}"
                        if config['channel'] is not None:
                            desc += f" ch{config['channel']}"
                        self.logger.info(
                            f"Processing {desc} ({job_idx + 1}/{len(jobs)})..."
                        )
                        try:
                            _run_plane_worker_thread(config)
                        except Exception as e:
                            self.logger.exception(
                                f"Error processing {desc}: {e}"
                            )
                    self.logger.info("Suite2p processing complete.")

                threading.Thread(target=run_all_planes_sequential, daemon=True).start()


def _run_plane_worker_thread(config):
    """
    Decoupled pure worker function for processing planes on a thread.
    Takes a pre-configured dict of snapshot variables to prevent GUI data racing.
    """
    if not _check_lsp_available():
        if config["logger"]:
            config["logger"].error("lbm_suite2p_python is not installed.")
        return

    arr = config["arr"]
    arr_idx = config["arr_idx"]
    current_z = config["z_plane"]
    plane = config["plane"]
    channel = config["channel"]
    base_out = config["base_out"]
    plane_dir = config["plane_dir"]
    roi = config["roi"]

    if not base_out.exists():
        base_out.mkdir(parents=True, exist_ok=True)

    ops_path = plane_dir / "ops.npy"
    lazy_mdata = getattr(arr, "metadata", {}).copy()

    # Merge GUI-set custom metadata (e.g. dz from the metadata editor)
    # into lazy_mdata BEFORE computing voxel size, so the OutputMetadata
    # reactive layer sees the user's value rather than the source file's.
    custom_metadata = config.get("custom_metadata") or {}
    if custom_metadata:
        lazy_mdata.update(custom_metadata)

    Lx = arr.shape[-1]
    Ly = arr.shape[-2]

    from mbo_utilities.metadata import OutputMetadata, get_param

    # Build the output metadata via the reactive layer. fs scales by
    # the timepoint stride, dz scales by the z-plane stride, and every
    # alias (num_timepoints/T/nt/...) gets updated consistently.
    # Anything done by hand here would just re-introduce the kind of
    # alias-drift bugs we just spent a week chasing.
    tp_indices = config.get("tp_indices")
    selected_planes_0based = config.get("selected_planes_0based")

    selections = {}
    if tp_indices is not None:
        selections["T"] = list(tp_indices)
    if selected_planes_0based is not None:
        selections["Z"] = list(selected_planes_0based)

    # source shape/dims must come from the lazy array's 5D contract,
    # not from arr.shape (which may be natural-rank for TiffArray).
    source_shape = tuple(arr.shape5d) if hasattr(arr, "shape5d") else None
    source_dims = ("T", "C", "Z", "Y", "X")

    # Log raw source values BEFORE scaling — critical diagnostic when
    # the user reports a wrong fs/dz in the output.
    raw_src_fs = get_param(lazy_mdata, "fs")
    raw_src_dz = get_param(lazy_mdata, "dz")
    config["logger"].info(
        f"_run_plane_worker_thread: source fs={raw_src_fs}, dz={raw_src_dz}, "
        f"plane={plane}"
    )

    if raw_src_fs is None:
        config["logger"].warning(
            f"_run_plane_worker_thread: source metadata for plane {plane} has "
            f"NO fs field — reactive scaling cannot compute the output rate. "
            f"ops.npy fs will fall through to lbm_suite2p_python's default "
            f"(10 Hz). To fix: set fs via the metadata editor before running, "
            f"or fix the source TIFF metadata."
        )

    out_meta = OutputMetadata(
        source=lazy_mdata,
        source_shape=source_shape,
        source_dims=source_dims,
        selections=selections,
    )

    # to_dict() gives reactively-scaled fs/dz/dx/dy and consistent
    # timepoint aliases. Add only the per-plane bookkeeping on top.
    md = out_meta.to_dict()

    # Strip fs/dz keys if they're None — otherwise `defaults.update(md)`
    # below would clobber the lbm default with None, which then leaks
    # into ops.npy as either None or write_ops's hardcoded fs=10
    # fallback. Removing the key surfaces the missing-data signal
    # explicitly to the user (they get None in ops.npy, not a fake 10).
    for key in ("fs", "dz"):
        if key in md and md[key] is None:
            md.pop(key)

    md["process_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    md["original_file"] = config["fpath"]
    md["roi_index"] = arr_idx
    md["z_index"] = current_z
    md["plane"] = plane
    md["Ly"] = Ly
    md["Lx"] = Lx
    md["ops_path"] = str(ops_path)
    md["save_path"] = str(plane_dir)
    md["raw_file"] = str((plane_dir / "data_raw.bin").resolve())

    config["logger"].info(
        f"_run_plane_worker_thread: applied reactive metadata -> "
        f"fs={md.get('fs')}, dz={md.get('dz')} "
        f"(t-stride from {len(tp_indices) if tp_indices else 0} indices, "
        f"z-stride from {len(selected_planes_0based) if selected_planes_0based else 0} planes)"
    )

    # Build upstream-shaped (settings, db) dicts from the GUI dataclasses.
    # settings is nested; db is flat. The lbm_suite2p_python patch flattens
    # these to a legacy ops dict at run_plane entry and writes db.npy /
    # settings.npy siblings for every ops.npy write.
    settings_dict = config["s2p_settings_dict"]
    db_dict = config["s2p_db_dict"]

    # Merge reactive metadata (fs/dz/dx/dy plus per-plane bookkeeping)
    # into the right halves. Keys that map to upstream go into the
    # appropriate dict; anything unknown falls through via extra_ops.
    extra_ops: dict = {}
    _settings_top = {"fs", "tau", "diameter"}
    _db_keys = {
        "nplanes", "nchannels", "data_path", "save_path0", "fast_disk",
        "save_folder", "h5py_key", "functional_chan", "nrois",
    }
    for key, value in md.items():
        if key == "fs":
            settings_dict["fs"] = value
        elif key in _settings_top:
            settings_dict[key] = value
        elif key in _db_keys:
            db_dict[key] = value
        else:
            # voxel size (dx/dy/dz), timepoint aliases, bookkeeping
            # (process_timestamp, original_file, roi_index, Ly, Lx,
            # ops_path, save_path, raw_file) — upstream schema has no
            # home so park on the flat ops dict via lbm's merge helper.
            extra_ops[key] = value

    if channel is not None:
        db_dict["functional_chan"] = 1
        # align_by_chan2 False (we're aligning by channel 1)
        settings_dict.setdefault("registration", {})["align_by_chan2"] = False

    # num_workers override (set when parallel plane processing is on) —
    # upstream doesn't have this key, but the fork reads it off ops.
    if "num_workers_override" in config:
        extra_ops["num_workers"] = config["num_workers_override"]

    from mbo_utilities.writer import imwrite

    plane_dir.mkdir(parents=True, exist_ok=True)

    # Pass `frames=` (1-based) to imwrite when the user has a stride
    # selection. Without this, only the COUNT was honored — the actual
    # stride was silently dropped and ops.npy ended up with raw source
    # fs even though the writer thought it was truncating.
    if tp_indices:
        frames_arg = [i + 1 for i in tp_indices]
    else:
        frames_arg = None

    # imwrite still expects a flat metadata dict. Flatten our (db, settings)
    # pair so the bin-writer's ops header stays consistent.
    try:
        from lbm_suite2p_python.db_settings import db_settings_to_ops
        metadata_flat = db_settings_to_ops(db_dict, settings_dict)
    except ImportError:
        metadata_flat = {**db_dict, **extra_ops}
    metadata_flat.update(extra_ops)

    imwrite(
        arr,
        plane_dir,
        ext=".bin",
        overwrite=True,
        register_z=False,
        planes=plane,
        channels=[channel] if channel is not None else None,
        output_name="data_raw.bin",
        roi=roi,
        metadata=metadata_flat,
        frames=frames_arg,
        num_frames=config["target_timepoints"] if frames_arg is None else None,
        fix_phase=config["fix_phase"],
        use_fft=config["use_fft"],
    )

    from lbm_suite2p_python import run_plane

    raw_file = plane_dir / "data_raw.bin"

    # Write db.npy / settings.npy up-front (the lbm patch also writes them
    # on every ops.npy save, but producing them BEFORE run_plane starts
    # gives the user something to inspect even if the pipeline crashes).
    try:
        import numpy as _np
        _np.save(plane_dir / "db.npy", db_dict, allow_pickle=True)
        _np.save(plane_dir / "settings.npy", settings_dict, allow_pickle=True)
    except Exception as _e:
        config["logger"].warning(f"Could not pre-write db.npy/settings.npy: {_e}")

    config["logger"].info(
        f"Suite2p run — do_detection: {settings_dict.get('run', {}).get('do_detection')}, "
        f"force_detect: {config['force_detect']}"
    )
    config["logger"].info(
        f"Detection params — algorithm: {settings_dict.get('detection', {}).get('algorithm')}, "
        f"diameter: {settings_dict.get('diameter')}"
    )

    try:
        run_plane(
            input_path=raw_file,
            save_path=plane_dir,
            db=db_dict,
            settings=settings_dict,
            ops=extra_ops if extra_ops else None,
            keep_raw=config["keep_raw"],
            keep_reg=config["keep_reg"],
            force_reg=config["force_reg"],
            force_detect=config["force_detect"],
            dff_window_size=config["dff_window_size"],
            dff_percentile=config["dff_percentile"],
            dff_smooth_window=config["dff_smooth_window"] if config["dff_smooth_window"] > 0 else None,
        )
        config["logger"].info(
            f"Suite2p processing complete for plane {current_z}, roi {arr_idx}. Results in {plane_dir}"
        )

        stat_file = plane_dir / "stat.npy"
        if stat_file.exists():
            config["logger"].info("Detection succeeded - stat.npy created")
        else:
            config["logger"].warning(
                "Detection did not run - stat.npy not found. Check Suite2p output logs."
            )
    except Exception as e:
        config["logger"].exception(
            f"Suite2p processing failed for plane {current_z}, roi {arr_idx}: {e}"
        )

