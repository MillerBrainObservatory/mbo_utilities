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

from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.preferences import get_last_dir, set_last_dir

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


# lazy availability check cache
_HAS_LSP: bool | None = None


def _check_lsp_available() -> bool:
    """check if lbm_suite2p_python is available (lazy, cached)."""
    global _HAS_LSP
    if _HAS_LSP is None:
        try:
            import lbm_suite2p_python
            _HAS_LSP = True
        except ImportError:
            _HAS_LSP = False
    return _HAS_LSP


def _patch_pyqt6_slider():
    """apply PyQt6 compatibility fix for suite2p GUI (call before opening GUI)."""
    try:
        from PyQt6.QtWidgets import QSlider
        if not hasattr(QSlider, 'NoTicks'):
            QSlider.NoTicks = QSlider.TickPosition.NoTicks
    except ImportError:
        pass


class Suite2pPipelineWidget(PipelineWidget):
    """suite2p processing and results widget."""

    name = "Suite2p"
    install_command = "uv pip install mbo_utilities[all]"

    @property
    def is_available(self) -> bool:
        """check availability lazily to avoid slow imports at module load."""
        return HAS_SUITE2P and _check_lsp_available()

    def __init__(self, parent: "PreviewDataWidget"):
        super().__init__(parent)

        # import settings from existing module
        from mbo_utilities.gui.pipeline_widgets import Suite2pSettings
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

        # diagnostics popup state (lazy init)
        self._diagnostics_widget = None
        self._show_diagnostics_popup = False
        self._diagnostics_popup_open = False
        self._file_dialog = None

        # grid search viewer state (lazy init)
        self._grid_search_widget = None
        self._show_grid_search_popup = False
        self._grid_search_popup_open = False
        self._grid_search_dialog = None

        # external GUI integration (suite2p or cellpose)
        self._external_gui_window = None
        self._external_gui_type = None  # "suite2p" or "cellpose"
        self._last_suite2p_ichosen = None
        self._last_poll_time = 0.0
        self._poll_interval = 0.1  # 100ms polling interval

        # gui choice: 0 = suite2p, 1 = cellpose
        self._gui_choice = 0

    def _get_diagnostics_widget(self):
        """get diagnostics widget (lazy init)."""
        if self._diagnostics_widget is None:
            from mbo_utilities.gui.diagnostics_widget import DiagnosticsWidget
            self._diagnostics_widget = DiagnosticsWidget()
        return self._diagnostics_widget

    def _get_grid_search_widget(self):
        """get grid search widget (lazy init)."""
        if self._grid_search_widget is None:
            from mbo_utilities.gui.grid_search_viewer import GridSearchViewer
            self._grid_search_widget = GridSearchViewer()
        return self._grid_search_widget

    def _check_suite2p_gui(self) -> bool:
        """check if suite2p GUI is available (requires rastermap)."""
        # always re-check using find_spec (fast, no actual import)
        import importlib.util
        # suite2p GUI requires rastermap
        return (
            importlib.util.find_spec("suite2p.gui") is not None
            and importlib.util.find_spec("rastermap") is not None
        )

    def _check_cellpose_gui(self) -> bool:
        """check if cellpose GUI is available."""
        # always re-check using find_spec (fast, no actual import)
        import importlib.util
        return importlib.util.find_spec("cellpose.gui") is not None

    def draw_config(self) -> None:
        """draw suite2p configuration ui."""
        from mbo_utilities.gui.pipeline_widgets import draw_section_suite2p

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
            imgui.set_tooltip("Load a stat.npy file to view ROI diagnostics.")

        # gui choice radio buttons
        imgui.same_line()
        imgui.text("Open in:")
        imgui.same_line()

        # suite2p option
        suite2p_available = self._check_suite2p_gui()
        if not suite2p_available:
            imgui.begin_disabled()
        if imgui.radio_button("suite2p", self._gui_choice == 0):
            self._gui_choice = 0
        if not suite2p_available:
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("suite2p GUI requires rastermap.\nInstall with: uv pip install rastermap")

        imgui.same_line()

        # cellpose option
        cellpose_available = self._check_cellpose_gui()
        if not cellpose_available:
            imgui.begin_disabled()
        if imgui.radio_button("cellpose", self._gui_choice == 1):
            self._gui_choice = 1
        if not cellpose_available:
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("cellpose GUI not available.\nInstall with: uv pip install cellpose[gui]")

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
        # only poll for suite2p gui (cellpose doesn't sync selections)
        if self._external_gui_type != "suite2p" or self._external_gui_window is None:
            return

        current_time = time.time()
        if current_time - self._last_poll_time < self._poll_interval:
            return
        self._last_poll_time = current_time

        # check if window was closed by user
        try:
            if not self._external_gui_window.isVisible():
                self._external_gui_window = None
                self._external_gui_type = None
                self._last_suite2p_ichosen = None
                return
        except RuntimeError:
            # window was deleted (Qt object wrapped C++ deleted)
            self._external_gui_window = None
            self._external_gui_type = None
            self._last_suite2p_ichosen = None
            return

        if not hasattr(self._external_gui_window, 'loaded') or not self._external_gui_window.loaded:
            return

        # get current selection from suite2p
        ichosen = getattr(self._external_gui_window, 'ichosen', None)

        # check if selection changed
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
        diag = self._get_diagnostics_widget()
        visible = diag.visible_indices
        if len(visible) > 0:
            # Find where cell_idx is in visible indices
            matches = np.where(visible == cell_idx)[0]
            if len(matches) > 0:
                diag.selected_roi = int(matches[0])

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
                        self._get_diagnostics_widget().load_results(plane_dir)
                        self._show_diagnostics_popup = True

                        # Open external GUI based on user choice (optional)
                        self._open_external_gui(stat_path)
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
                    self._get_diagnostics_widget().draw()
                except Exception as e:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {e}")

                # Close button at bottom
                imgui.spacing()
                imgui.separator()
                if imgui.button("Close", imgui.ImVec2(100, 0)):
                    self._diagnostics_popup_open = False
                    imgui.close_current_popup()

            imgui.end_popup()

    def _open_external_gui(self, statfile: Path):
        """Open external GUI (suite2p or cellpose) based on user choice.

        Parameters
        ----------
        statfile : Path
            Path to stat.npy file
        """
        if self._gui_choice == 0:
            if self._check_suite2p_gui():
                self._open_suite2p_gui(statfile)
            else:
                print("suite2p GUI not available (requires rastermap). Install with: uv pip install rastermap")
        elif self._gui_choice == 1:
            if self._check_cellpose_gui():
                self._open_cellpose_gui(statfile)
            else:
                print("cellpose GUI not available. Install with: uv pip install cellpose[gui]")

    def _open_suite2p_gui(self, statfile: Path):
        """Open suite2p GUI with the given stat.npy file."""
        try:
            # apply PyQt6 compatibility fix before importing suite2p GUI
            _patch_pyqt6_slider()
            from suite2p.gui.gui2p import MainWindow as Suite2pMainWindow
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QRect

            self._external_gui_window = Suite2pMainWindow(statfile=str(statfile))
            self._external_gui_type = "suite2p"

            # position windows side-by-side
            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()
                half_width = screen_w // 2
                margin_top = 30
                margin_bottom = 10
                win_height = screen_h - margin_top - margin_bottom

                self._external_gui_window.setGeometry(QRect(
                    screen_x + half_width,
                    screen_y + margin_top,
                    half_width,
                    win_height
                ))
                self._external_gui_window.setMinimumSize(400, 300)
                self._reposition_mbo_window(screen_x, screen_y + margin_top, half_width, win_height)

            self._external_gui_window.show()
            self._external_gui_window.showNormal()
            self._last_suite2p_ichosen = None

        except ImportError as e:
            print(f"Could not open suite2p GUI: {e}")
        except Exception as e:
            print(f"Error opening suite2p GUI: {e}")

    def _open_cellpose_gui(self, statfile: Path):
        """Open cellpose GUI with the stat.npy converted to masks."""
        try:
            import tempfile
            import tifffile
            # try lbm_suite2p_python first (primary source), fallback to mbo_utilities.analysis
            try:
                from lbm_suite2p_python import stat_to_masks
            except ImportError:
                from mbo_utilities.analysis import stat_to_masks
            from cellpose.gui.gui import MainW
            from cellpose.gui import io as cellpose_io
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QRect

            # patch QCheckBox for Qt5/Qt6 compatibility
            from mbo_utilities.analysis import _patch_qt_checkbox
            _patch_qt_checkbox()

            # load stat.npy and convert to masks for cellpose
            plane_dir = statfile.parent
            stat = np.load(statfile, allow_pickle=True)

            # get image shape from ops.npy, or use meanImg if available
            ops_file = plane_dir / "ops.npy"
            image = None
            if ops_file.exists():
                ops = np.load(ops_file, allow_pickle=True).item()
                Ly, Lx = ops.get("Ly", 512), ops.get("Lx", 512)
                # use meanImg if available for better visualization
                if "meanImg" in ops:
                    image = ops["meanImg"].astype(np.float32)
            else:
                # fallback: estimate from stat
                Ly = max(s["ypix"].max() for s in stat) + 1
                Lx = max(s["xpix"].max() for s in stat) + 1

            if image is None:
                image = np.zeros((Ly, Lx), dtype=np.float32)

            masks = stat_to_masks(stat, (Ly, Lx))

            # save image to temp file (cellpose expects file path)
            temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tifffile.imwrite(temp_file.name, image)
            temp_file.close()

            # create cellpose MainW directly (avoid creating new QApplication)
            self._external_gui_window = MainW(image=temp_file.name)
            self._external_gui_type = "cellpose"

            # load masks into the gui
            self._external_gui_window.masks = masks
            self._external_gui_window.ncells = int(masks.max())
            # trigger redraw with masks
            if hasattr(self._external_gui_window, 'draw_layer'):
                self._external_gui_window.draw_layer()

            # position window side-by-side
            screen = QApplication.primaryScreen()
            if screen:
                screen_geom = screen.availableGeometry()
                screen_x = screen_geom.x()
                screen_y = screen_geom.y()
                screen_w = screen_geom.width()
                screen_h = screen_geom.height()
                half_width = screen_w // 2
                margin_top = 30
                margin_bottom = 10
                win_height = screen_h - margin_top - margin_bottom

                self._external_gui_window.setGeometry(QRect(
                    screen_x + half_width,
                    screen_y + margin_top,
                    half_width,
                    win_height
                ))
                self._reposition_mbo_window(screen_x, screen_y + margin_top, half_width, win_height)

            self._external_gui_window.show()
            print(f"Opened cellpose GUI with {len(stat)} ROIs")

        except ImportError as e:
            print(f"Could not open cellpose GUI: {e}")
        except Exception as e:
            print(f"Error opening cellpose GUI: {e}")

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
            from PyQt6.QtCore import QRect

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
        if self._external_gui_type == "suite2p":
            return self._external_gui_window
        return None

    @property
    def external_gui_window(self):
        """Access to the external GUI window (suite2p or cellpose) if open."""
        return self._external_gui_window

    def _draw_grid_search_popup(self):
        """Draw the grid search viewer popup window if open."""
        # Check if folder dialog has a result
        if self._grid_search_dialog is not None and self._grid_search_dialog.ready():
            result = self._grid_search_dialog.result()
            if result:
                try:
                    set_last_dir("grid_search", result)
                    self._get_grid_search_widget().load_results(Path(result))
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
                    self._get_grid_search_widget().draw()
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
