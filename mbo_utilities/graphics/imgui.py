import webbrowser
from pathlib import Path
from typing import Literal
import threading

from icecream import ic

import numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift
from skimage.registration import phase_cross_correlation

from imgui_bundle import (
    imgui,
    hello_imgui,
    imgui_ctx,
    implot,
    portable_file_dialogs as pfd,
)

from mbo_utilities.assembly import save_as
from mbo_utilities.file_io import (
    Scan_MBO,
    SAVE_AS_TYPES,
    _get_mbo_dirs,
    read_scan, to_lazy_array,
)
from mbo_utilities.graphics._widgets import set_tooltip, checkbox_with_tooltip
from mbo_utilities.graphics.gui_logger import GuiLogger
from mbo_utilities.graphics.pipeline_widgets import draw_tab_process
from mbo_utilities.graphics.progress_bar import draw_zstats_progress, draw_saveas_progress
from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings

try:
    import cupy as cp  # noqa
    from cusignal import (
        register_translation,
    )  # GPU version of phase_cross_correlation # noqa

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    register_translation = phase_cross_correlation  # noqa

import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
# from .pipeline_widgets import draw_pipeline_section

REGION_TYPES = ["Full FOV", "Sub-FOV"]
USER_PIPELINES = ["suite2p", "masknmf"]

def apply_phase_offset(frame: np.ndarray, offset: float) -> np.ndarray:
    result = frame.copy()
    rows = result[1::2, :]
    f = np.fft.fftn(rows)
    fshift = fourier_shift(f, (0, offset))
    result[1::2, :] = np.fft.ifftn(fshift).real
    return result


def compute_phase_offset(
    frame: np.ndarray, upsample: int = 10, exclude_center_px: int = 0
) -> float:
    if frame.ndim == 3:
        frame = np.mean(frame, axis=0)
    _, w = frame.shape

    frame = frame.astype(np.float32)

    cx = w // 2
    keep_left = slice(None, cx - exclude_center_px)
    keep_right = slice(cx + exclude_center_px, None)

    pre = frame[::2]
    post = frame[1::2]
    m = min(pre.shape[0], post.shape[0])
    pre_crop = np.concatenate([pre[:m, keep_left], pre[:m, keep_right]], axis=1)
    post_crop = np.concatenate([post[:m, keep_left], post[:m, keep_right]], axis=1)

    ic(pre_crop)
    ic(post_crop)

    # transfer to GPU
    if HAS_CUPY:
        pre_gpu = cp.asarray(pre_crop)
        post_gpu = cp.asarray(post_crop)
        shift, _, _ = register_translation(
            cp.asarray(pre_gpu), cp.asarray(post_gpu), upsample_factor=upsample
        )
    else:
        shift, _, _ = phase_cross_correlation(
            pre_crop,  # noqa
            post_crop,  # noqa
            upsample_factor=upsample,  # noqa
        )

    return float(shift[1])


def _save_as(path, **kwargs):
    """
    read scan from path for threading
    there must be a better way to do this
    """
    scan = read_scan(path)
    save_as(scan, **kwargs)


class PreviewDataWidget(EdgeWindow):
    def __init__(
        self,
        iw: fpl.ImageWidget,
        fpath: str | None = None,
        size: int = 350,
        location: Literal["bottom", "right"] = "right",
        title: str = "Data Preview",
    ):
        super().__init__(figure=iw.figure, size=size, location=location, title=title)

        if implot.get_current_context() is None:
            implot.create_context()

        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        self.s2p = Suite2pSettings()
        self.debug_panel = GuiLogger()

        self._selected_pipelines = None

        self.show_debug_panel = False
        self.show_tool_about = False
        self.show_tool_style_editor = False
        self.show_tool_id_stack_tool = False
        self.show_tool_debug_log = False
        self.show_tool_metrics = False
        self.font_size = 12

        self._show_theme_window = False
        self._zstats = None
        self._zstats_done = None
        self._zstats_progress = None
        self._zstats_current_z = None
        self._open_save_popup = None
        self._show_debug_panel = None

        self.max_offset = 8
        self.image_widget = iw
        self.shape = self.image_widget.data[0].shape

        if len(self.shape) == 4:
            self.nz = self.shape[1]
        elif len(self.shape) == 3:
            self.nz = 1

        self._gaussian_sigma = 0
        self._current_offset = 0.0
        self._window_size = 1

        self._phase_upsample = 20
        self._auto_update = False
        self._proj = "mean"

        self._planes_str = str(getattr(self, "_planes_str", ""))

        # Combo boxes
        self._region = str(getattr(self, "_region", "Full FOV"))
        self._region_idx = REGION_TYPES.index(self._region) if self._region in REGION_TYPES else 0

        self._ext = str(getattr(self, "_ext", ".tiff"))
        self._ext_idx = SAVE_AS_TYPES.index(".tiff")

        self._selected_planes = set()
        self._overwrite = True
        self._fix_phase = True
        self._debug = False

        if isinstance(self.image_widget.data[0], Scan_MBO):
            self.is_mbo_scan = True
        else:
            self.is_mbo_scan = False

        for subplot in self.image_widget.figure:
            subplot.toolbar = False

        self.image_widget._image_widget_sliders._loop = True  # noqa

        self._saveas_done = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0
        self._saveas_outdir = str(getattr(self, "_save_dir", ""))
        self._saveas_total = 0

        self._zstats_meansub_progress = 0.0
        self._zstats_means = None
        self._zstats_show_subtracted = dict()
        self._zstats_stats_thread = None
        self._zstats_progress = 0.0
        self._zstats_current_mean_z = None

        threading.Thread(target=self.compute_zstats, daemon=True).start()

    @property
    def gaussian_sigma(self):
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value):
        if value > 0:
            self._gaussian_sigma = value
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, value):
        if value != self._proj:
            if value == "mean-sub":
                self.image_widget.frame_apply = {0: self._combined_frame_apply}
            else:
                self.image_widget.window_funcs["t"].func = getattr(np, value)
            self._proj = value

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        self.image_widget.window_funcs["t"].window_size = value
        self._window_size = value

    @property
    def current_offset(self):
        return self._current_offset

    @current_offset.setter
    def current_offset(self, value):
        if value == self._current_offset or value > self.max_offset:
            return
        self._current_offset = value

    @property
    def phase_upsample(self):
        return self._phase_upsample

    @phase_upsample.setter
    def phase_upsample(self, value):
        if value > 0:
            self._phase_upsample = value
            self.image_widget.frame_apply = {0: self._combined_frame_apply}

    @property
    def auto_update(self):
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        if self._auto_update == value:
            # no change
            return
        if value:
            self.image_widget.add_event_handler(self.calculate_offset, "current_index")
            self.image_widget.frame_apply = {0: self._combined_frame_apply}
        if not value:
            self.image_widget.remove_event_handler(self.calculate_offset)
        self._auto_update = value
        self.calculate_offset()

    def update(self):
        # Top Menu Bar
        cflags: imgui.ChildFlags = (
            imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # noqa
        )
        wflags: imgui.WindowFlags = imgui.WindowFlags_.menu_bar  # noqa
        with imgui_ctx.begin_child("menu", window_flags=wflags, child_flags=cflags):
            if imgui.begin_menu_bar():
                if imgui.begin_menu("File", True):
                    if imgui.menu_item(
                        "Save as", "Ctrl+S", p_selected=False, enabled=self.is_mbo_scan
                    )[0]:
                        self._open_save_popup = True
                    imgui.end_menu()
                if imgui.begin_menu("Docs", True):
                    if imgui.menu_item(
                        "Open Docs", "Ctrl+I", p_selected=False, enabled=True
                    )[0]:
                        webbrowser.open(
                            "https://millerbrainobservatory.github.io/mbo_utilities/"
                        )
                    imgui.end_menu()
                if imgui.begin_menu("Settings", True):
                    imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Tools")
                    _, self.show_tool_style_editor = imgui.menu_item(
                        "Style Editor", "", self.show_tool_style_editor, True
                    )
                    _, self.show_debug_panel = imgui.menu_item(
                        "Debug Panel",
                        "",
                        p_selected=self.show_debug_panel,
                        enabled=True,
                    )
                    _, self.show_tool_about = imgui.menu_item(
                        "About MBO", "", self.show_tool_about, True
                    )
                    imgui.end_menu()
            imgui.end_menu_bar()

        if not hasattr(self, "show_tool_metrics"):
            self.show_tool_metrics = False
            self.show_tool_debug_log = False
            self.show_tool_id_stack_tool = False
            self.show_tool_style_editor = False
            self.show_tool_about = False

        # (accessible from the "Tools" menu)
        if self.show_tool_metrics:
            imgui.show_metrics_window(self.show_tool_metrics)
        if self.show_tool_debug_log:
            imgui.show_debug_log_window(self.show_tool_debug_log)
        if self.show_tool_id_stack_tool:
            imgui.show_id_stack_tool_window(self.show_tool_id_stack_tool)
        if self.show_tool_style_editor:
            _, self.show_tool_style_editor = imgui.begin(
                "Style Editor", self.show_tool_style_editor
            )
            imgui.show_style_editor()
            imgui.end()
        if self.show_tool_about:
            imgui.show_about_window(self.show_tool_about)

        if self.show_debug_panel:
            self.debug_panel.draw()

        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))   # noqa
                self.draw_preview_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()

            if self._zstats_done and imgui.begin_tab_item("Summary Stats")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))  # noqa
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))   # noqa
                self.draw_stats_section()
                imgui.pop_style_var()
                imgui.pop_style_var()
                imgui.end_tab_item()

            if imgui.begin_tab_item("Process")[0]:
                draw_tab_process(self)
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def draw_suite2p_settings(self):
        with imgui_ctx.begin_child("Suite2p Settings"):
            imgui.text("Quick-Run Suite2p Pipeline Options")
            imgui.separator()
            imgui.checkbox("Run registration", True)
            imgui.checkbox("Run cell detection", True)
            imgui.slider_int("Threshold", 30, 0, 100)
            if imgui.button("Run"):
                self.debug_panel.log("info", "Running Suite2p pipeline...")
                self.debug_panel.log("info", "Suite2p pipeline completed.")

    def draw_stats_section(self):
        if not getattr(self, "_zstats_done", False):
            return

        z_vals = np.arange(len(self._zstats["mean"]))
        mean_vals = np.array(self._zstats["mean"])
        std_vals = np.array(self._zstats["std"])
        snr_vals = np.array(self._zstats["snr"])

        # imgui.set_cursor_pos_y(hello_imgui.em_size(1))  # push content toward top
        imgui.text_colored(imgui.ImVec4(0.8, 1.0, 0.2, 1.0), "Z-Plane Summary Stats")

        cflags: imgui.ChildFlags = (
            imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # noqa
        )
        with imgui_ctx.begin_child(
            "##Summary", size=imgui.ImVec2(0, 0), child_flags=cflags
        ):
            if imgui.begin_table(
                "zstats", 4, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg  # noqa
            ):
                for col in ["Z", "Mean", "Std", "SNR"]:
                    imgui.table_setup_column(col, imgui.TableColumnFlags_.width_stretch)  # noqa
                imgui.table_headers_row()
                for i in range(len(z_vals)):
                    imgui.table_next_row()
                    for val in (z_vals[i], mean_vals[i], std_vals[i], snr_vals[i]):
                        imgui.table_next_column()
                        imgui.text(f"{val:.2f}")
                imgui.end_table()

        imgui.separator()
        with imgui_ctx.begin_child(
            "##Plots", size=imgui.ImVec2(0, 0), child_flags=cflags
        ):
            imgui.text("Z-plane Signal: Mean ± Std")

            z_vals = np.arange(len(self._zstats["mean"]), dtype=np.float32)
            mean_vals = np.array(self._zstats["mean"], dtype=np.float32)
            std_vals = np.array(self._zstats["std"], dtype=np.float32)

            if implot.begin_plot("Z-Plane Signal", size=imgui.ImVec2(-1, 300)):
                implot.plot_error_bars("Mean ± Std", z_vals, mean_vals, std_vals)
                implot.plot_line("Mean", z_vals, mean_vals)
                implot.end_plot()

    def draw_preview_section(self):
        cflags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize  # noqa
        with imgui_ctx.begin_child(
            "##PreviewChild",
            imgui.ImVec2(0, 0),
            cflags,
        ):
            if getattr(self, "_open_save_popup", False):
                imgui.open_popup("Save As")
                self._open_save_popup = False

            if imgui.begin_popup_modal("Save As")[0]:

                # Directory + Ext
                imgui.set_next_item_width(hello_imgui.em_size(25))
                # TODO: make _save_dir a property to expand ~
                _, self._saveas_outdir = imgui.input_text(
                    "Save Dir", str(Path(self._saveas_outdir).expanduser().resolve()), 256
                )
                imgui.same_line()
                if imgui.button("Browse"):
                    home = Path().home()
                    res = pfd.select_folder(str(home))
                    if res:
                        self._saveas_outdir = res.result()

                imgui.set_next_item_width(hello_imgui.em_size(25))
                _, self._ext_idx = imgui.combo("Ext", self._ext_idx, SAVE_AS_TYPES)
                self._ext = SAVE_AS_TYPES[self._ext_idx]

                # Options Section
                imgui.separator()
                imgui.text("Options")

                self._overwrite = checkbox_with_tooltip(
                    "Overwrite", self._overwrite, "Replace any existing output files."
                )
                self._fix_phase = checkbox_with_tooltip(
                    "Fix Phase",
                    self._fix_phase,
                    "Apply scan-phase correction to interleaved lines.",
                )
                self._debug = checkbox_with_tooltip(
                    "Debug",
                    self._debug,
                    "Run with debugging, settings -> debug to view the outputs.",
                )

                # Z-plane selection
                imgui.separator()
                imgui.text("Select z-planes to save")

                try:
                    num_planes = self.image_widget.data[0].num_channels  # noqa
                except Exception as e:
                    num_planes = 1
                    hello_imgui.log(
                        hello_imgui.LogLevel.error,
                        f"Could not read number of planes: {e}",
                    )

                if imgui.button("All"):
                    self._selected_planes = set(range(num_planes))
                imgui.same_line()
                if imgui.button("None"):
                    self._selected_planes = set()

                imgui.columns(2, borders=False)
                for i in range(num_planes):
                    imgui.push_id(i)
                    selected = i in self._selected_planes
                    _, selected = imgui.checkbox(f"Plane {i + 1}", selected)
                    if selected:
                        self._selected_planes.add(i)
                    else:
                        self._selected_planes.discard(i)
                    imgui.pop_id()
                    imgui.next_column()
                imgui.columns(1)

                imgui.separator()
                if imgui.button("Save", imgui.ImVec2(100, 0)):
                    if not self._saveas_outdir:
                        self._saveas_outdir = _get_mbo_dirs()["base"].joinpath("data")
                    try:
                        save_planes = [p + 1 for p in self._selected_planes]
                        self._saveas_total = len(save_planes)
                        save_kwargs = {
                            "path": self.fpath,
                            "savedir": self._saveas_outdir,
                            "planes": save_planes,
                            "overwrite": self._overwrite,
                            "fix_phase": self._fix_phase,
                            "debug": self._debug,
                            "ext": self._ext,
                            "save_phase_png": False,
                            "target_chunk_mb": 20,
                            "progress_callback": lambda frac,
                            current_plane: self.gui_progress_callback(
                                frac, current_plane
                            ),
                        }
                        self.debug_panel.log("info", f"Saving planes {save_planes}")
                        self.debug_panel.log(
                            "info", f"Saving to {self._saveas_outdir} as {self._ext}"
                        )
                        threading.Thread(target=_save_as, kwargs=save_kwargs, daemon=True).start()
                        imgui.close_current_popup()
                    except Exception as e:
                        hello_imgui.log(hello_imgui.LogLevel.error, f"Save failed: {e}")
                imgui.same_line()
                if imgui.button("Cancel"):
                    imgui.close_current_popup()

                imgui.end_popup()

            # Section: Window Functions
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
            imgui.separator()

            imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(2, 2))  # noqa

            imgui.begin_group()
            options = ["mean", "max", "std"]
            disabled_label = (
                "mean-sub (pending)" if not self._zstats_done else "mean-sub"
            )
            options.append(disabled_label)

            current_display_idx = options.index(
                self.proj if self._proj != "mean-sub" else disabled_label
            )

            imgui.set_next_item_width(hello_imgui.em_size(15))
            proj_changed, selected_display_idx = imgui.combo(
                "Projection", current_display_idx, options
            )
            set_tooltip(
                "Choose projection method over the sliding window: “mean” (average), “max” (peak), “std” (variance), or “mean-sub” (background-subtracted mean, recommended for motion preview)."
            )

            if proj_changed:
                selected_label = options[selected_display_idx]
                if selected_label == "mean-sub (pending)":
                    pass  # ignore user click while disabled
                else:
                    self.proj = selected_label
                    if self.proj == "mean-sub":
                        self.image_widget.frame_apply = {0: self._combined_frame_apply}
                    else:
                        self.image_widget.window_funcs["t"].func = getattr(
                            np, self.proj
                        )

            # Window size for projections
            imgui.set_next_item_width(hello_imgui.em_size(15))
            winsize_changed, new_winsize = imgui.input_int(
                "Window Size", self.window_size, step=1, step_fast=2
            )
            set_tooltip(
                "Size of the temporal window (in frames) used for projection."
                " E.g. a value of 3 averages over 3 consecutive frames."
            )
            if winsize_changed and new_winsize > 0:
                self.window_size = new_winsize
                self.debug_panel.log("info", f"New Window Size: {new_winsize}")

            # Gaussian Filter
            imgui.set_next_item_width(hello_imgui.em_size(15))
            gaussian_changed, new_gaussian_sigma = imgui.slider_float(
                label="sigma",
                v=self.gaussian_sigma,
                v_min=0.0,
                v_max=20.0,
            )
            set_tooltip(
                "Apply a Gaussian blur to the preview image. Sigma is in pixels; larger values yield stronger smoothing."
            )
            if gaussian_changed:
                self.gaussian_sigma = new_gaussian_sigma
                self.debug_panel.log(
                    "info", f"New gaussian sigma: {new_gaussian_sigma}"
                )

            imgui.end_group()

            imgui.pop_style_var()

            # Section: Scan-phase Correction
            imgui.spacing()
            imgui.separator()
            imgui.text_colored(
                imgui.ImVec4(0.8, 0.6, 1.0, 1.0), "Scan-Phase Correction"
            )
            imgui.separator()
            imgui.begin_group()

            imgui.text("Offset")
            imgui.same_line()
            imgui.text_colored(
                imgui.ImVec4(1.0, 0.5, 0.0, 1.0), f"{self.current_offset:.3f}"
            )

            imgui.set_next_item_width(hello_imgui.em_size(10))
            upsample_changed, upsample_val = imgui.input_int(
                "Upsample", self._phase_upsample, step=1, step_fast=2
            )
            set_tooltip(
                "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment."
            )
            if upsample_changed:
                self.phase_upsample = max(1, upsample_val)
                self.debug_panel.log("info", f"New upsample: {upsample_val}")

            imgui.set_next_item_width(hello_imgui.em_size(10))
            max_offset_changed, max_offset = imgui.input_int(
                "max-offset", self.max_offset, step=1, step_fast=2
            )
            set_tooltip(
                "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
            )
            if max_offset_changed:
                self.max_offset = max(1, max_offset)
                self.debug_panel.log("info", f"New max-offset: {max_offset}")

            auto_changed, new_auto_update = imgui.checkbox(
                "Auto Update", self.auto_update
            )
            set_tooltip(
                "When enabled, automatically recompute phase correction whenever any parameter changes."
            )
            if auto_changed:
                self.auto_update = new_auto_update

            if imgui.button("Apply", imgui.ImVec2(0, 0)):
                self.debug_panel.log("debug", f"Calculating offset")
                self.calculate_offset()
                self.image_widget.frame_apply = {0: self._combined_frame_apply}
            set_tooltip(
                "Run the phase-correction algorithm now using the current settings.",
                _show_mark=False,
            )
            imgui.same_line()
            if imgui.button("Reset", imgui.ImVec2(0, 0)):
                self.debug_panel.log("debug", f"Reset offset")
                self.current_offset = 0
                self.image_widget.frame_apply = {0: self._combined_frame_apply}

            set_tooltip("Reset the computed scan-phase offset back to zero.")

            imgui.end_group()
            imgui.separator()

        imgui.separator()
        draw_zstats_progress(self)
        draw_saveas_progress(self)

    def get_raw_frame(self):
        data = self.image_widget.data[0]
        idx = self.image_widget.current_index

        if data.ndim == 4:  # TZXY
            t = idx.get("t", 0)
            z = idx.get("z", 0)
            return data[t, z, :, :]
        elif data.ndim == 3:  # TXY
            t = idx.get("t", 0)
            return data[t, :, :]
        elif data.ndim == 2:  # XY
            return data
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

    def gui_progress_callback(self, fraction, current_plane):
        """Callback for save_as progress updates."""
        self._saveas_current_index = current_plane
        self._saveas_progress = fraction
        self._saveas_done = fraction >= 1.0

    def _combined_frame_apply(self, frame: np.ndarray) -> np.ndarray:
        """alter final frame only once, in ImageWidget.frame_apply"""
        if self._current_offset:
            frame = apply_phase_offset(frame, self.current_offset)
        if self._gaussian_sigma > 0:
            frame = gaussian_filter(frame, sigma=self.gaussian_sigma)
        if self.proj == "mean-sub" and self._zstats_means is not None:
            if self.shape == 4:
                z = self.image_widget.current_index["z"]
                frame = frame - self._zstats_means[z]
            else:
                frame = frame - self._zstats_means[0]
        return frame

    def calculate_offset(self, ev=None):
        """Get the current frame, calculate the offset"""
        frame = self.get_raw_frame()
        self.debug_panel.log("debug", f"Calculating offset")
        ofs = compute_phase_offset(frame, upsample=self._phase_upsample)
        self.current_offset = ofs
        self.debug_panel.log("debug", f"Offset: {self.current_offset:.3f}")

    def compute_zstats(self):
        if self.fpath:
            data, fpath = to_lazy_array(self.fpath)

            # Ensure data is always 4D (T, Z, X, Y)
            if data.ndim == 3:
                data = data[:, np.newaxis, :, :]

            self._zstats = {"mean": [], "std": [], "snr": []}
            self._zstats_progress = 0.0
            self._zstats_done = False

            means = []
            for z in range(self.nz):
                stack = data[:, z, :, :].astype(np.float32)
                mean_img = np.mean(stack, axis=0)
                std_img = np.std(stack, axis=0)
                snr_img = np.divide(mean_img, std_img + 1e-5, where=(std_img > 1e-5))

                self._zstats["mean"].append(np.mean(mean_img))
                self._zstats["std"].append(np.mean(std_img))
                self._zstats["snr"].append(np.mean(snr_img))

                means.append(mean_img)
                self._zstats_current_z = z
                self._zstats_progress = (z + 1) / self.nz
                self._zstats_meansub_progress = (z + 1) / self.nz

            self._zstats_done = True
            self._zstats_means = np.stack(means)
            self.debug_panel.log("info", "Z-stats and mean-sub completed")
        else:
            return
