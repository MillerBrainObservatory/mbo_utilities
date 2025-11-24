"""
suite2p pipeline widget.

combines processing configuration and results viewing in one widget
with a toggle to switch between modes.
"""

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs as pfd, hello_imgui, implot

from mbo_utilities.graphics.widgets.pipelines._base import PipelineWidget
from mbo_utilities.graphics._widgets import set_tooltip
from mbo_utilities.graphics._availability import HAS_SUITE2P

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget

# check if lbm_suite2p_python is available
try:
    from lbm_suite2p_python.run_lsp import run_plane, run_plane_bin
    from lbm_suite2p_python import load_planar_results
    HAS_LSP = True
except ImportError:
    HAS_LSP = False
    run_plane = None
    run_plane_bin = None
    load_planar_results = None


class Suite2pPipelineWidget(PipelineWidget):
    """suite2p processing and results widget."""

    name = "Suite2p"
    is_available = HAS_SUITE2P and HAS_LSP
    install_command = "uv pip install mbo_utilities[suite2p]"

    def __init__(self, parent: "PreviewDataWidget"):
        super().__init__(parent)

        # import settings from existing module
        from mbo_utilities.graphics.pipeline_widgets import Suite2pSettings
        self.settings = Suite2pSettings()

        # config state
        self._saveas_outdir = ""
        self._install_error = False
        self._frames_initialized = False
        self._last_max_frames = 1000
        self._selected_planes = set()
        self._show_plane_popup = False
        self._parallel_processing = False
        self._max_parallel_jobs = 2
        self._savepath_flash_start = None
        self._show_savepath_popup = False

        # results state
        self._ops_path = None
        self._stat = None
        self._F = None
        self._Fneu = None
        self._spks = None
        self._iscell = None
        self._cellprob = None
        self._ops = None
        self._selected_cell = 0
        self._show_trace = True  # default to showing trace
        self._sync_with_viewer = False
        self._show_spks = False
        self._show_fneu = False

    def draw_config(self) -> None:
        """draw suite2p configuration ui."""
        # import the existing draw function
        from mbo_utilities.graphics.pipeline_widgets import draw_section_suite2p

        # temporarily set attributes on parent for compatibility
        self.parent._saveas_outdir = self._saveas_outdir
        self.parent._install_error = self._install_error
        self.parent._frames_initialized = self._frames_initialized
        self.parent._last_max_frames = self._last_max_frames
        self.parent._selected_planes = self._selected_planes
        self.parent._show_plane_popup = self._show_plane_popup
        self.parent._parallel_processing = self._parallel_processing
        self.parent._max_parallel_jobs = self._max_parallel_jobs
        self.parent._s2p_savepath_flash_start = self._savepath_flash_start
        self.parent._s2p_show_savepath_popup = self._show_savepath_popup
        self.parent._current_pipeline = "suite2p"

        # draw the config
        draw_section_suite2p(self.parent)

        # sync back
        self._saveas_outdir = self.parent._saveas_outdir
        self._install_error = self.parent._install_error
        self._frames_initialized = getattr(self.parent, '_frames_initialized', False)
        self._last_max_frames = getattr(self.parent, '_last_max_frames', 1000)
        self._selected_planes = getattr(self.parent, '_selected_planes', set())
        self._show_plane_popup = getattr(self.parent, '_show_plane_popup', False)
        self._parallel_processing = getattr(self.parent, '_parallel_processing', False)
        self._max_parallel_jobs = getattr(self.parent, '_max_parallel_jobs', 2)
        self._savepath_flash_start = getattr(self.parent, '_s2p_savepath_flash_start', None)
        self._show_savepath_popup = getattr(self.parent, '_s2p_show_savepath_popup', False)

    def draw_results(self) -> None:
        """draw suite2p results viewer ui."""
        INPUT_WIDTH = 120

        # sync save path from parent (might have been set in config mode)
        self._saveas_outdir = getattr(self.parent, '_saveas_outdir', self._saveas_outdir)

        # determine starting directory for file dialog
        start_dir = str(Path.home())
        if self._saveas_outdir:
            start_dir = self._saveas_outdir

        # file selection - look for stat.npy like suite2p gui
        if imgui.button("Load Results"):
            result = pfd.open_file(
                "Select stat.npy",
                start_dir,
                ["stat.npy", "*.npy"]
            )
            if result and result.result():
                stat_file = Path(result.result()[0])
                if stat_file.name == "stat.npy":
                    try:
                        self._load_results(str(stat_file))
                    except Exception as e:
                        self.parent.logger.error(f"Error loading results: {e}")
                else:
                    self.parent.logger.error("Please select stat.npy file")

        if self._ops_path:
            imgui.same_line()
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 1.0, 1.0),
                str(self._ops_path.name)
            )
            set_tooltip(str(self._ops_path))

            if self._F is not None and len(self._F) > 0:
                num_cells = len(self._F)
                num_frames = len(self._F[0]) if len(self._F) > 0 else 0

                imgui.spacing()
                imgui.text(f"{num_cells} cells, {num_frames} frames")

                # cell selector
                imgui.set_next_item_width(INPUT_WIDTH)
                changed, self._selected_cell = imgui.slider_int(
                    "Cell",
                    self._selected_cell,
                    0,
                    num_cells - 1
                )

                # cell probability
                if self._cellprob is not None and self._selected_cell < len(self._cellprob):
                    cell_prob = self._cellprob[self._selected_cell]
                    imgui.same_line()
                    imgui.text(f"(prob: {cell_prob:.2f})")

                # trace options
                _, self._show_trace = imgui.checkbox("Show Trace", self._show_trace)
                if self._show_trace:
                    imgui.same_line()
                    _, self._show_fneu = imgui.checkbox("Fneu", self._show_fneu)
                    imgui.same_line()
                    _, self._show_spks = imgui.checkbox("Spks", self._show_spks)

                # sync with viewer
                _, self._sync_with_viewer = imgui.checkbox("Sync with viewer", self._sync_with_viewer)
                set_tooltip("Sync current frame marker with the image viewer's time slider")

                # handle sync - update viewer frame when clicking on plot
                if self._sync_with_viewer and changed:
                    # could update z-plane based on which plane this cell is from
                    pass

                # trace plot
                if self._show_trace:
                    trace_f = self._F[self._selected_cell]
                    xs = np.arange(len(trace_f), dtype=np.float64)

                    avail = imgui.get_content_region_avail()
                    plot_height = min(200, avail.y - 20) if avail.y > 50 else 150

                    plot_flags = implot.Flags_.crosshairs
                    if implot.begin_plot(f"##trace_{self._selected_cell}", imgui.ImVec2(-1, plot_height), plot_flags):
                        implot.setup_axes("Frame", "Fluorescence (a.u.)")
                        implot.setup_axis_limits(implot.ImAxis_.x1, 0, len(trace_f), implot.Cond_.once)

                        # plot F trace
                        ys_f = trace_f.astype(np.float64)
                        implot.plot_line("F", xs, ys_f)

                        # plot Fneu if enabled
                        if self._show_fneu and self._Fneu is not None:
                            ys_fneu = self._Fneu[self._selected_cell].astype(np.float64)
                            implot.plot_line("Fneu", xs, ys_fneu)

                        # plot spikes if enabled
                        if self._show_spks and self._spks is not None:
                            ys_spks = self._spks[self._selected_cell].astype(np.float64)
                            # scale spikes to be visible alongside F
                            if ys_spks.max() > 0:
                                scale = ys_f.max() / ys_spks.max() * 0.5
                                ys_spks = ys_spks * scale
                            implot.plot_line("Spks", xs, ys_spks)

                        # draw current frame marker if synced
                        if self._sync_with_viewer:
                            try:
                                iw = self.parent.image_widget
                                names = iw._slider_dim_names or ()
                                if "t" in names:
                                    current_t = iw.indices["t"]
                                    implot.drag_line_x(0, current_t, imgui.ImVec4(1, 1, 0, 0.8))
                            except Exception:
                                pass

                        implot.end_plot()
        else:
            imgui.spacing()
            imgui.text_disabled("No results loaded")

    def _load_results(self, stat_path: str) -> None:
        """load suite2p results from stat.npy directory."""
        stat_path = Path(stat_path)
        result_dir = stat_path.parent
        self._ops_path = result_dir

        self.parent.logger.info(f"Loading results from: {result_dir}")

        # load files directly to avoid issues with load_planar_results
        def load_npy(name):
            p = result_dir / name
            if p.exists():
                return np.load(p, allow_pickle=True)
            return None

        self._stat = load_npy("stat.npy")
        self._F = load_npy("F.npy")
        self._Fneu = load_npy("Fneu.npy")
        self._spks = load_npy("spks.npy")
        iscell_data = load_npy("iscell.npy")

        # iscell.npy is typically (n_cells, 2) where col 0 is bool, col 1 is probability
        if iscell_data is not None:
            if iscell_data.ndim == 2 and iscell_data.shape[1] >= 2:
                self._iscell = iscell_data[:, 0].astype(bool)
                self._cellprob = iscell_data[:, 1]
            else:
                self._iscell = iscell_data.astype(bool)
                self._cellprob = None
        else:
            self._iscell = None
            self._cellprob = None

        self._ops = load_npy("ops.npy")
        if self._ops is not None:
            self._ops = self._ops.item() if self._ops.ndim == 0 else self._ops

        if self._F is not None and len(self._F) > 0:
            self._selected_cell = 0
            self.parent.logger.info(f"Loaded {len(self._F)} cells, {self._F.shape[1]} frames")
