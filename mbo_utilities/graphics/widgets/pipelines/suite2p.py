"""
suite2p pipeline widget.

combines processing configuration with a button to view trace quality statistics
in a separate popup window.
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
        self._saveas_outdir = ""  # for save_as dialog
        self._s2p_outdir = ""  # for suite2p run/load (separate from save_as)
        self._install_error = False
        self._frames_initialized = False
        self._last_max_frames = 1000
        self._selected_planes = set()
        self._show_plane_popup = False
        self._parallel_processing = False
        self._max_parallel_jobs = 2
        self._savepath_flash_start = None
        self._show_savepath_popup = False

        # diagnostics popup state
        self._diagnostics_widget = DiagnosticsWidget()
        self._show_diagnostics_popup = False
        self._diagnostics_popup_open = False
        self._folder_dialog = None

    def draw_config(self) -> None:
        """draw suite2p configuration ui."""
        from mbo_utilities.graphics.pipeline_widgets import draw_section_suite2p

        self._draw_diagnostics_button()
        imgui.separator()
        imgui.spacing()

        # sync widget state to parent before drawing
        # ONLY set parent values if parent doesn't already have a value set
        # This prevents overwriting values set by the Browse dialog
        if self._saveas_outdir and not getattr(self.parent, '_saveas_outdir', ''):
            self.parent._saveas_outdir = self._saveas_outdir
        if self._s2p_outdir and not getattr(self.parent, '_s2p_outdir', ''):
            self.parent._s2p_outdir = self._s2p_outdir
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

        draw_section_suite2p(self.parent)

        # sync back from parent - always read latest values
        # use parent value if set, otherwise keep widget value
        parent_saveas = getattr(self.parent, '_saveas_outdir', '')
        parent_s2p = getattr(self.parent, '_s2p_outdir', '')
        if parent_saveas:
            self._saveas_outdir = parent_saveas
        if parent_s2p:
            self._s2p_outdir = parent_s2p
        self._install_error = self.parent._install_error
        self._frames_initialized = getattr(self.parent, '_frames_initialized', False)
        self._last_max_frames = getattr(self.parent, '_last_max_frames', 1000)
        self._selected_planes = getattr(self.parent, '_selected_planes', set())
        self._show_plane_popup = getattr(self.parent, '_show_plane_popup', False)
        self._parallel_processing = getattr(self.parent, '_parallel_processing', False)
        self._max_parallel_jobs = getattr(self.parent, '_max_parallel_jobs', 2)
        self._savepath_flash_start = getattr(self.parent, '_s2p_savepath_flash_start', None)
        self._show_savepath_popup = getattr(self.parent, '_s2p_show_savepath_popup', False)

        # Draw diagnostics popup window (managed separately from config)
        self._draw_diagnostics_popup()

    def _draw_diagnostics_button(self):
        """Draw the button to open trace quality statistics popup."""
        if imgui.button("Load trace quality statistics"):
            self._folder_dialog = pfd.select_folder(
                "Select plane folder with suite2p results",
                str(Path.home())
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Load suite2p results (F.npy, stat.npy, iscell.npy) to view\n"
                "ROI traces, dF/F, SNR, compactness, and other quality metrics."
            )

    def _draw_diagnostics_popup(self):
        """Draw the diagnostics popup window if open."""
        # Check if folder dialog has a result
        if self._folder_dialog is not None and self._folder_dialog.ready():
            result = self._folder_dialog.result()
            if result:
                try:
                    self._diagnostics_widget.load_results(Path(result))
                    self._show_diagnostics_popup = True
                except Exception as e:
                    print(f"Error loading results: {e}")
            self._folder_dialog = None

        if self._show_diagnostics_popup:
            self._diagnostics_popup_open = True
            imgui.open_popup("Trace Quality Statistics")
            self._show_diagnostics_popup = False

        # Set popup size
        viewport = imgui.get_main_viewport()
        popup_width = min(1200, viewport.size.x * 0.9)
        popup_height = min(800, viewport.size.y * 0.85)
        imgui.set_next_window_size(imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            "Trace Quality Statistics",
            p_open=True if self._diagnostics_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings
        )

        if opened:
            if not visible:
                # User closed the popup via X button
                self._diagnostics_popup_open = False
                imgui.close_current_popup()
            else:
                # Draw the diagnostics content
                try:
                    self._diagnostics_widget.draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")

                # Close button at bottom
                imgui.spacing()
                imgui.separator()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._diagnostics_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()
