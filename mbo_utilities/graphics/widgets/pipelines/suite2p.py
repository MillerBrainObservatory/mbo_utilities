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
from mbo_utilities.graphics.diagnostics_widget import DiagnosticsWidget

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

        # results state - use DiagnosticsWidget for comprehensive ROI analysis
        self._diagnostics_widget = DiagnosticsWidget()

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
        """Draw suite2p results viewer using DiagnosticsWidget."""
        try:
            self._diagnostics_widget.draw()
        except Exception as e:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")
            import traceback
            traceback.print_exc()
