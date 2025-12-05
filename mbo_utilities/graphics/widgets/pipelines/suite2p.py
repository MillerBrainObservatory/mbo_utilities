"""
suite2p pipeline widget.

combines processing configuration with a button to view trace quality statistics
in a separate popup window.
"""

from typing import TYPE_CHECKING, Optional
from pathlib import Path
import time

import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.graphics.widgets.pipelines._base import PipelineWidget
from mbo_utilities.graphics._availability import HAS_SUITE2P
from mbo_utilities.graphics.diagnostics_widget import DiagnosticsWidget
from mbo_utilities.graphics.grid_search_viewer import GridSearchViewer
from mbo_utilities.preferences import get_last_dir, set_last_dir

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget

# Apply PySide6 compatibility fix for suite2p GUI BEFORE any suite2p imports
# suite2p's RangeSlider uses self.NoTicks which doesn't exist in PySide6
try:
    from PySide6.QtWidgets import QSlider
    if not hasattr(QSlider, 'NoTicks'):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks
except ImportError:
    QSlider = None
    pass  # PySide6 not available

# check if lbm_suite2p_python is available
try:
    from lbm_suite2p_python import load_planar_results
    HAS_LSP = True
except ImportError:
    HAS_LSP = False
    run_plane = None
    load_planar_results = None


class Suite2pPipelineWidget(PipelineWidget):
    """suite2p processing and results widget."""

    name = "Suite2p"
    is_available = HAS_SUITE2P and HAS_LSP
    install_command = "uv pip install mbo_utilities[all]"

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
        self._file_dialog = None

        # grid search viewer state
        self._grid_search_widget = GridSearchViewer()
        self._show_grid_search_popup = False
        self._grid_search_popup_open = False
        self._grid_search_dialog = None

        # suite2p GUI integration
        self._suite2p_window = None
        self._last_suite2p_ichosen = None
        self._last_poll_time = 0.0
        self._poll_interval = 0.1  # 100ms polling interval

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

        # Poll suite2p for selection changes
        self._poll_suite2p_selection()

        # Draw popup windows (managed separately from config)
        self._draw_diagnostics_popup()
        self._draw_grid_search_popup()

    def _draw_diagnostics_button(self):
        """Draw buttons to load diagnostics and grid search results."""
        if imgui.button("Load stat.npy"):
            default_dir = str(get_last_dir("suite2p_stat") or Path.home())
            self._file_dialog = pfd.open_file(
                "Select stat.npy file",
                default_dir,
                ["stat.npy files", "stat.npy"],
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Load a stat.npy file to view ROI diagnostics.\n"
                "Opens suite2p GUI synced with the diagnostics popup.\n"
                "Click cells in suite2p to update the diagnostics view."
            )

        imgui.same_line()

        # disable button if dialog is already open
        dialog_pending = self._grid_search_dialog is not None
        if dialog_pending:
            imgui.begin_disabled()

        if imgui.button("Grid Search..."):
            if self._grid_search_dialog is None:
                default_dir = str(get_last_dir("grid_search") or Path.home())
                self._grid_search_dialog = pfd.select_folder(
                    "Select grid search results folder", default_dir
                )

        if dialog_pending:
            imgui.end_disabled()

        if imgui.is_item_hovered():
            if dialog_pending:
                imgui.set_tooltip("Waiting for folder selection...")
            else:
                imgui.set_tooltip(
                    "Load grid search results to compare parameter combinations.\n"
                    "Select a folder containing subfolders for each parameter set,\n"
                    "each with suite2p/plane0/ containing the results."
                )

    def _poll_suite2p_selection(self):
        """Poll suite2p window for selection changes."""
        current_time = time.time()
        if current_time - self._last_poll_time < self._poll_interval:
            return
        self._last_poll_time = current_time

        if self._suite2p_window is None:
            return

        # Check if window was closed by user - isVisible() returns False after close
        try:
            if not self._suite2p_window.isVisible():
                self._suite2p_window = None
                self._last_suite2p_ichosen = None
                return
        except RuntimeError:
            # Window was deleted (Qt object wrapped C++ deleted)
            self._suite2p_window = None
            self._last_suite2p_ichosen = None
            return

        if not hasattr(self._suite2p_window, 'loaded') or not self._suite2p_window.loaded:
            return

        # Get current selection from suite2p
        ichosen = getattr(self._suite2p_window, 'ichosen', None)

        # Check if selection changed
        if ichosen != self._last_suite2p_ichosen:
            self._last_suite2p_ichosen = ichosen
            self._on_suite2p_cell_selected(ichosen)

    def _on_suite2p_cell_selected(self, cell_idx: int):
        """Handle cell selection in suite2p GUI.

        Parameters
        ----------
        cell_idx : int
            Index of the selected cell in suite2p
        """
        if cell_idx is None:
            return

        # Update diagnostics widget selection
        # Map the global cell index to visible index if showing only cells
        visible = self._diagnostics_widget.visible_indices
        if len(visible) > 0:
            # Find where cell_idx is in visible indices
            matches = np.where(visible == cell_idx)[0]
            if len(matches) > 0:
                self._diagnostics_widget.selected_roi = int(matches[0])

    def _draw_diagnostics_popup(self):
        """Draw the diagnostics popup window if open."""
        # Check if file dialog has a result
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result and len(result) > 0:
                stat_path = Path(result[0])
                # Save the directory for next time
                set_last_dir("suite2p_stat", stat_path)
                if stat_path.name == "stat.npy" and stat_path.exists():
                    try:
                        # Load results from the directory containing stat.npy
                        plane_dir = stat_path.parent
                        self._diagnostics_widget.load_results(plane_dir)
                        self._show_diagnostics_popup = True

                        # Open suite2p GUI with this stat file
                        self._open_suite2p_gui(stat_path)
                    except Exception as e:
                        print(f"Error loading results: {e}")
                else:
                    print(f"Please select a stat.npy file, got: {stat_path.name}")
            self._file_dialog = None

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

    def _open_suite2p_gui(self, statfile: Path):
        """Open suite2p GUI with the given stat.npy file.

        Positions both the MBO GUI and suite2p side-by-side, each taking
        half the screen width and full available height.

        Parameters
        ----------
        statfile : Path
            Path to stat.npy file
        """
        try:
            from suite2p.gui.gui2p import MainWindow as Suite2pMainWindow
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import QRect

            # Create suite2p window
            self._suite2p_window = Suite2pMainWindow(statfile=str(statfile))

            # Get screen geometry for side-by-side layout
            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()

                # Split screen in half - each window gets 50% width
                half_width = screen_w // 2

                # Leave margin for title bar and taskbar
                margin_top = 30
                margin_bottom = 10
                win_height = screen_h - margin_top - margin_bottom

                # Suite2p goes on the RIGHT half
                s2p_x = screen_x + half_width
                s2p_y = screen_y + margin_top
                s2p_width = half_width

                # Set suite2p geometry and ensure it's resizable
                self._suite2p_window.setGeometry(QRect(s2p_x, s2p_y, s2p_width, win_height))

                # Set minimum size to allow shrinking (suite2p default min size is too large)
                self._suite2p_window.setMinimumSize(400, 300)

                # Try to reposition the MBO imgui window to the LEFT half
                self._reposition_mbo_window(screen_x, screen_y + margin_top, half_width, win_height)

            self._suite2p_window.show()
            # Ensure window has normal state (not maximized/fullscreen)
            self._suite2p_window.showNormal()
            self._last_suite2p_ichosen = None

        except ImportError as e:
            print(f"Could not open suite2p GUI: {e}")
        except Exception as e:
            print(f"Error opening suite2p GUI: {e}")

    def _reposition_mbo_window(self, x: int, y: int, width: int, height: int):
        """Reposition the MBO window to the specified geometry.

        Uses the Qt canvas from fastplotlib's ImageWidget to find and
        reposition the parent window.

        Parameters
        ----------
        x, y : int
            Window position
        width, height : int
            Window size
        """
        try:
            from PySide6.QtCore import QRect

            # Access the canvas through the parent widget hierarchy
            # parent -> PreviewDataWidget -> image_widget -> figure -> canvas
            if hasattr(self.parent, 'image_widget'):
                canvas = self.parent.image_widget.figure.canvas
                # Get the top-level window containing the canvas
                if hasattr(canvas, 'window'):
                    # rendercanvas provides window() method
                    window = canvas.window()
                    if window:
                        window.setGeometry(QRect(x, y, width, height))
                        return
                # Fallback: traverse Qt parent hierarchy to find top-level window
                widget = canvas
                while widget is not None:
                    if widget.isWindow():
                        widget.setGeometry(QRect(x, y, width, height))
                        return
                    widget = widget.parent()
        except Exception as e:
            # Silently fail - window positioning is not critical
            print(f"Could not reposition MBO window: {e}")

    @property
    def suite2p_window(self):
        """Access to the suite2p GUI window if open."""
        return self._suite2p_window

    def _draw_grid_search_popup(self):
        """Draw the grid search viewer popup window if open."""
        # Check if folder dialog has a result
        if self._grid_search_dialog is not None and self._grid_search_dialog.ready():
            result = self._grid_search_dialog.result()
            if result:
                try:
                    set_last_dir("grid_search", result)
                    self._grid_search_widget.load_results(Path(result))
                    self._show_grid_search_popup = True
                except Exception as e:
                    print(f"Error loading grid search results: {e}")
            self._grid_search_dialog = None

        if self._show_grid_search_popup:
            self._grid_search_popup_open = True
            imgui.open_popup("Grid Search Results")
            self._show_grid_search_popup = False

        # Set popup size
        viewport = imgui.get_main_viewport()
        popup_width = min(1200, viewport.size.x * 0.9)
        popup_height = min(800, viewport.size.y * 0.85)
        imgui.set_next_window_size(imgui.ImVec2(popup_width, popup_height), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            "Grid Search Results",
            p_open=True if self._grid_search_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings
        )

        if opened:
            if not visible:
                self._grid_search_popup_open = False
                imgui.close_current_popup()
            else:
                try:
                    self._grid_search_widget.draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")

                imgui.spacing()
                imgui.separator()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._grid_search_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()

    def cleanup(self):
        """Clean up resources when widget is destroyed.

        Should be called when the parent GUI is closing to ensure
        proper cleanup of Qt windows and other resources.
        """
        # Close suite2p window if open
        if self._suite2p_window is not None:
            try:
                self._suite2p_window.close()
            except (RuntimeError, AttributeError):
                pass  # Window already deleted
            self._suite2p_window = None
            self._last_suite2p_ichosen = None

        # Clear file dialogs (they are async and may be pending)
        self._file_dialog = None
        self._grid_search_dialog = None
