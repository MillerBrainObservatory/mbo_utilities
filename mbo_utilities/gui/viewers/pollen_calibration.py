"""
Pollen calibration viewer for LBM beamlet calibration.

This viewer handles pollen calibration data (ZCYX) and provides:
- Info panel showing beamlet count, cavities, z-step, pixel size
- Automatic background calibration (detection + analysis)
- Manual calibration mode: click through beamlets in the viewer
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import threading

import numpy as np
from scipy.ndimage import uniform_filter1d

from imgui_bundle import imgui, implot, portable_file_dialogs as pfd

from . import BaseViewer
from mbo_utilities.gui.panels import DebugPanel, ProcessPanel, MetadataPanel
from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.metadata import get_param
from mbo_utilities.metadata.scanimage import (
    get_lbm_ai_sources,
    get_z_step_size,
)

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["PollenCalibrationViewer"]


def get_cavity_indices(metadata: dict, nc: int) -> dict:
    """Get cavity A and cavity B channel indices from LBM metadata."""
    from mbo_utilities.metadata.scanimage import is_lbm_stack

    result = {
        "cavity_a": [],
        "cavity_b": [],
        "is_lbm": False,
        "num_cavities": 1,
    }

    if not is_lbm_stack(metadata):
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    result["is_lbm"] = True
    ai_sources = get_lbm_ai_sources(metadata)

    if not ai_sources:
        half = nc // 2
        result["cavity_a"] = list(range(half))
        result["cavity_b"] = list(range(half, nc))
        return result

    sorted_sources = sorted(ai_sources.keys())

    if len(sorted_sources) >= 1:
        cavity_a_channels = ai_sources.get(sorted_sources[0], [])
        result["cavity_a"] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_a_channels])

    if len(sorted_sources) >= 2:
        cavity_b_channels = ai_sources.get(sorted_sources[1], [])
        result["cavity_b"] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_b_channels])
        result["num_cavities"] = 2

    return result


class PollenCalibrationViewer(BaseViewer):
    """
    Viewer for pollen calibration data (ZCYX).

    This viewer is specialized for LBM beamlet calibration using
    pollen grain stacks. It provides:
    - Automatic bead detection and calibration
    - Interactive manual calibration mode
    - Results visualization and comparison
    """

    name = "Pollen Calibration Viewer"

    # Default beam order for 30-channel system
    DEFAULT_ORDER_30 = [
        0, 4, 5, 6, 7, 8, 1, 9, 10, 11, 12, 13, 14, 15,
        2, 16, 17, 18, 19, 20, 21, 3, 22, 23, 24, 25, 26, 27, 28, 29
    ]

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)

        # Pollen calibration has its own specialized UI
        # so we don't use the generic feature system
        self._features = []

        # Pollen-specific state
        self._z_step_um = 1.0
        self._pixel_size_um = 1.0

        self._cavity_info = None
        self._beam_order = None

        # Auto calibration state
        self._status = "Initializing..."
        self._progress = 0.0
        self._processing = False
        self._done = False
        self._error = None
        self._initialized = False

        # Separate results for auto and manual
        self._results_auto = None
        self._results_manual = None

        # Results viewing state
        self._saved_images: list[Path] = []
        self._show_figures_popup = False
        self._figures_popup_mode = "auto"
        self._current_figure_idx = 0
        self._calibration_data_auto = None   # cached H5 data for auto mode
        self._calibration_data_manual = None  # cached H5 data for manual mode

        # Manual calibration state
        self._manual_mode = False
        self._manual_channel_idx = 0  # Current beamlet index in order
        self._manual_positions = []   # User-clicked positions [(x, y), ...]
        self._manual_z_indices = []   # Best z for each position
        self._click_handler = None
        self._vol = None  # Cached volume for manual mode
        self._num_channels = None  # Number of channels (set during manual mode)
        self._max_projections = None  # Max projections for viewing
        self._original_metadata = None  # Store metadata before replacing with numpy array

        # External file loading state
        self._external_h5_dialog = None  # pfd file dialog for loading external H5
        self._loaded_external = None     # External calibration data dict
        self._existing_h5_files = []     # H5 files found in current directory
        self._original_data_array = None  # Store original array for restoration

        # Drag detection state for click handler
        self._pointer_down_pos = None  # (x, y) on pointer_down
        self._drag_threshold = 5.0     # pixels moved to consider it a drag

        # Only initialize panels when not using legacy delegation
        if parent is None:
            self._panels["debug"] = DebugPanel(self)
            self._panels["processes"] = ProcessPanel(self)
            self._panels["metadata"] = MetadataPanel(self)
            self._setup_logging()

    @property
    def data(self):
        """Access the loaded data arrays."""
        if self.parent is not None:
            return self.parent.image_widget.data if self.parent.image_widget else None
        return self.image_widget.data if self.image_widget else None

    def get_metadata(self) -> dict:
        """Get metadata, using stored original if in manual mode."""
        if self._original_metadata:
            return self._original_metadata
        arr = self._get_array()
        return getattr(arr, "metadata", {}) if arr is not None else {}

    @property
    def logger(self):
        """Access the GUI logger."""
        if self.parent is not None:
            return self.parent.logger
        import logging
        return logging.getLogger("mbo_utilities")

    def _setup_logging(self) -> None:
        """Set up log handler to route to debug panel."""
        try:
            import logging
            from mbo_utilities.gui.panels.debug_log import GuiLogHandler
            handler = GuiLogHandler(self._panels["debug"])
            logging.getLogger("mbo_utilities").addHandler(handler)
        except Exception:
            pass

    def _init_from_data(self):
        """Initialize calibration parameters from loaded data."""
        try:
            data = self.data
            if data is None:
                return
            arr = data[0]
            if arr is None:
                return
        except (TypeError, IndexError):
            return

        metadata = getattr(arr, "metadata", {})

        z_step = get_z_step_size(metadata)
        if z_step is not None:
            self._z_step_um = z_step
        else:
            self._z_step_um = get_param(metadata, "dz", default=1.0)

        # get pixel size from metadata (already calculated in microns)
        pixel_res = get_param(metadata, "pixel_resolution", default=None)
        if pixel_res is not None:
            # pixel_resolution is (dx, dy) tuple in microns
            self._pixel_size_um = float(pixel_res[0]) if hasattr(pixel_res, '__getitem__') else float(pixel_res)
        else:
            # fallback: use fov_um if available, otherwise warn and use default
            fov_um = get_param(metadata, "fov_um", default=None)
            if fov_um is not None and arr.ndim >= 2:
                nx = arr.shape[-1]
                fov_x = fov_um[0] if hasattr(fov_um, '__getitem__') else fov_um
                self._pixel_size_um = float(fov_x) / nx
            elif arr.ndim >= 2:
                # last resort: use default 600um FOV with zoom
                zoom = get_param(metadata, "zoom_factor", default=1.0)
                nx = arr.shape[-1]
                self._pixel_size_um = 600.0 / zoom / nx
                self.logger.warning("pixel_resolution not in metadata, using default FOV=600um")

        nc = getattr(arr, "num_channels", arr.shape[1] if arr.ndim >= 2 else 1)
        self._cavity_info = get_cavity_indices(metadata, nc)

        if nc == 30:
            self._beam_order = self.DEFAULT_ORDER_30.copy()
        else:
            self._beam_order = list(range(nc))

        self._initialized = True

    def _get_array(self):
        """Safely get the first data array."""
        try:
            data = self.data
            if data is None:
                return None
            return data[0]
        except (TypeError, IndexError):
            return None

    def _get_fpath(self):
        """Get the file path."""
        parent_fpath = self.parent.fpath if self.parent is not None else self.fpath
        if isinstance(parent_fpath, (list, tuple)):
            parent_fpath = parent_fpath[0] if parent_fpath else None
        return Path(parent_fpath) if parent_fpath else None

    @property
    def num_beamlets(self) -> int:
        """Get number of beamlets (channels) - shape[1] for ZCYX data."""
        arr = self._get_array()
        if arr is None:
            return 0
        # For ZCYX data: shape is (Z, C, Y, X), so channels is shape[1]
        if hasattr(arr, "num_beamlets"):
            return arr.num_beamlets
        if hasattr(arr, "num_channels"):
            return arr.num_channels
        # Fallback: for 4D ZCYX, channels is dim 1
        if arr.ndim == 4:
            return arr.shape[1]
        return 1

    @property
    def num_z_planes(self) -> int:
        arr = self._get_array()
        if arr is None:
            return 0
        return getattr(arr, "num_zplanes", arr.shape[0])

    def draw(self) -> None:
        """Draw the pollen calibration UI."""
        if self.parent is not None:
            # Legacy mode: full calibration UI in sidebar
            self._draw_calibration_ui()
        else:
            # New mode: panel-based with menu bar
            self.draw_menu_bar()
            self._draw_calibration_ui()
            for panel in self._panels.values():
                panel.draw()

        # Always draw popup (works in both modes)
        if self._show_figures_popup:
            self._draw_figures_popup()

    def _draw_calibration_ui(self) -> None:
        """Draw the main calibration UI."""
        if not self._initialized:
            self._init_from_data()
            if self._initialized and not self._processing and not self._done:
                self._start_auto_calibration()

        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

        imgui.dummy(imgui.ImVec2(0, 5))

        # Info section
        self._draw_info_section()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Manual or auto mode
        if self._manual_mode:
            self._draw_manual_mode()
        else:
            # Load previous section
            self._draw_load_previous_section()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            self._draw_auto_status()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            self._draw_manual_button()

        imgui.pop_style_var()
        imgui.pop_style_var()

    def _draw_info_section(self):
        """Draw info panel with meaningful calibration metrics."""
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Pollen Calibration")
        imgui.spacing()

        arr = self._get_array()
        if arr is not None:
            # Shape info (compact)
            imgui.text(f"Shape: {arr.shape}")
            set_tooltip("(Z-piezo, Channels, Y, X)", show_mark=False)

            # Core parameters with tooltips
            imgui.text(f"Z-step: {self._z_step_um:.2f} um")
            set_tooltip("Piezo z-step size between each slice (from stackZStepSize)", show_mark=False)
            imgui.text(f"Pixel: {self._pixel_size_um:.3f} um")
            set_tooltip("Pixel size in microns (from metadata pixel_resolution)", show_mark=False)
        else:
            imgui.text_disabled("No data loaded")

    def _draw_auto_status(self):
        """Draw automatic calibration status and results."""
        # Show processing status
        if self._processing:
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.8, 1.0), "Processing...")
            imgui.text(self._status)
            imgui.progress_bar(self._progress, imgui.ImVec2(-1, 0))
            imgui.spacing()
        elif self._error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {self._error}")
            imgui.spacing()

        # Check if we have any results
        has_auto = self._results_auto is not None
        has_manual = self._results_manual is not None

        if has_auto or has_manual:
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.9, 1.0), "Results")
            imgui.spacing()

            # Buttons - mode selection is in the popup
            if imgui.button("Open Folder"):
                # Open folder for whichever result exists (prefer manual)
                mode = "manual" if has_manual else "auto"
                self._open_output_folder(mode)
            imgui.same_line()
            if imgui.button("View Figures"):
                # Default to auto if available, otherwise manual
                mode = "auto" if has_auto else "manual"
                self._open_all_graphs(mode)

        elif not self._processing and not self._error:
            imgui.text_disabled("Waiting...")

    def _draw_load_previous_section(self):
        """Draw UI for loading previous calibration results."""
        # Scan for existing H5 files if not already done
        if not self._existing_h5_files:
            self._scan_existing_h5_files()

        # Auto-load first existing result if available and not already loaded
        if self._existing_h5_files and not self._loaded_external:
            self._load_h5_file(str(self._existing_h5_files[0]))

        # Show loaded results status
        if self._loaded_external:
            mode = self._loaded_external.get("calibration_mode", "previous")
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 0.6, 1.0), f"Loaded: {mode}")
            h5_path = self._loaded_external.get("h5_path", "")
            if h5_path:
                name = Path(h5_path).name
                imgui.text_disabled(name[:30] + "..." if len(name) > 30 else name)

    def _scan_existing_h5_files(self):
        """Scan current file's directory for existing pollen H5 files."""
        self._existing_h5_files = []
        fpath = self._get_fpath()
        if fpath is None:
            return

        parent_dir = fpath.parent
        if not parent_dir.exists():
            return

        # Find pollen calibration H5 files
        for h5_file in parent_dir.glob("*_pollen.h5"):
            self._existing_h5_files.append(h5_file)

        # Sort by modification time (newest first)
        self._existing_h5_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    def _load_h5_file(self, path: str):
        """Load calibration data from an H5 file."""
        from mbo_utilities.gui._pollen_analysis import extract_calibration_summary

        summary = extract_calibration_summary(path)
        if summary:
            self._loaded_external = summary
            self._loaded_external["h5_path"] = path
            self.logger.info(f"Loaded calibration from: {path}")
        else:
            self.logger.error(f"Failed to load calibration from: {path}")

    def _draw_manual_button(self):
        """Draw manual calibration button."""
        # Centered header with help
        avail_w = imgui.get_content_region_avail().x
        text = "Interactive Calibration"
        text_w = imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_w - 20) / 2)
        imgui.text_colored(imgui.ImVec4(0.8, 0.6, 0.2, 1.0), text)
        # set_tooltip adds its own (?) mark
        set_tooltip(
            "Manual pollen calibration mode.\n\n"
            "Click on the same pollen bead in each beamlet.\n"
            "The viewer will show one beamlet at a time.\n\n"
            "Navigation:\n"
            "  - Drag to pan the view\n"
            "  - Scroll wheel to zoom\n"
            "  - Single click (no drag) to mark bead position"
        )

        imgui.spacing()

        # Centered button
        btn_w = 180
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - btn_w) / 2)
        if imgui.button("Start", imgui.ImVec2(btn_w, 0)):
            self._start_manual_mode()

    def _draw_manual_mode(self):
        """Draw manual calibration UI."""
        imgui.text_colored(imgui.ImVec4(0.2, 0.8, 0.4, 1.0), "Manual Calibration")
        imgui.spacing()

        # Use actual channel count from max projections
        nc = self._max_projections.shape[0] if self._max_projections is not None else self.num_beamlets
        current = self._manual_channel_idx
        channel = self._beam_order[current] if current < len(self._beam_order) else current
        num_marked = len(self._manual_positions)

        imgui.text(f"Beamlet {current + 1}/{nc} (ch {channel})")
        imgui.text(f"Marked: {num_marked}")

        # Progress bar
        progress = num_marked / nc if nc > 0 else 0
        imgui.progress_bar(progress, imgui.ImVec2(-1, 0), f"{num_marked}/{nc}")
        imgui.spacing()

        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Click on pollen bead")
        imgui.spacing()

        # Navigation buttons - compact, no fixed width
        imgui.button("Prev") and self._manual_prev()
        imgui.same_line()
        imgui.button("Skip") and self._manual_skip()
        imgui.same_line()
        imgui.button("Next") and self._manual_next()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Finish button - always available if we have some positions
        if num_marked > 0:
            if num_marked >= nc:
                imgui.text_colored(imgui.ImVec4(0.3, 1.0, 0.3, 1.0), "All done!")
            imgui.button("Finish") and self._finish_manual_calibration()
            imgui.spacing()

        imgui.button("Cancel") and self._cancel_manual_mode()

    def _start_manual_mode(self):
        """Start interactive manual calibration."""
        arr = self._get_array()
        if arr is None:
            self.logger.error("No data for manual calibration")
            return

        self._manual_mode = True
        self._manual_channel_idx = 0
        self._manual_positions = []
        self._manual_z_indices = []

        # Store original data array and metadata before we replace viewer data
        self._original_data_array = arr
        self._original_metadata = getattr(arr, "metadata", {})

        # Load volume into memory for click analysis
        self.logger.info("Loading volume for manual calibration...")
        self._vol = np.asarray(arr[:]).astype(np.float32)
        self._vol -= self._vol.mean()

        # For ZCYX data: shape is (Z, C, Y, X)
        # Z = piezo positions (typically many, e.g. 224)
        # C = beamlet channels (typically fewer, e.g. 14)
        self.logger.info(f"Volume shape: {self._vol.shape} (Z, C, Y, X)")
        nz, nc, _ny, _nx = self._vol.shape
        self.logger.info(f"Z-planes: {nz}, Channels: {nc}")

        # Store channel count for UI
        self._num_channels = nc

        # Compute max projections over Z for each channel -> (C, Y, X)
        # This creates a 3D array we can navigate by channel
        self._max_projections = self._vol.max(axis=0)
        self.logger.info(f"Max projections shape: {self._max_projections.shape}")

        # Replace viewer data with max projections
        # Note: This replaces with numpy array - metadata viewer handles this gracefully
        self.image_widget.data[0] = self._max_projections

        # Reset to first channel
        if self.image_widget.n_sliders > 0:
            self.image_widget.indices = [0]

        # Show first beamlet
        self._show_beamlet(0)

        # Add click handler to the figure
        self._setup_click_handler()

        self.logger.info("Manual calibration started. Click on pollen beads.")

    def _setup_click_handler(self):
        """Set up click event handler on the image with drag detection."""
        if self.image_widget is None:
            return

        # Get the subplot and add click handler
        try:
            subplot = self.image_widget.figure[0, 0]

            def on_pointer_down(ev):
                """Track pointer down position for drag detection."""
                if not self._manual_mode:
                    return
                self._pointer_down_pos = (ev.x, ev.y)

            def on_click(ev):
                if not self._manual_mode:
                    return

                # Check if this was a drag (mouse moved significantly)
                if self._pointer_down_pos is not None:
                    dx = abs(ev.x - self._pointer_down_pos[0])
                    dy = abs(ev.y - self._pointer_down_pos[1])
                    if dx > self._drag_threshold or dy > self._drag_threshold:
                        # This was a drag, not a click - skip marking
                        self._pointer_down_pos = None
                        return

                self._pointer_down_pos = None

                # Map screen coords to world coords
                try:
                    world_pos = subplot.map_screen_to_world((ev.x, ev.y))
                    if world_pos is not None:
                        x, y = world_pos[0], world_pos[1]
                        self._handle_click(x, y)
                except Exception as e:
                    self.logger.exception(f"Click mapping error: {e}")

            subplot.renderer.add_event_handler(on_pointer_down, "pointer_down")
            subplot.renderer.add_event_handler(on_click, "click")
            self._click_handler = on_click
            self.logger.info("Click handler registered with drag detection")

        except Exception as e:
            self.logger.exception(f"Failed to set up click handler: {e}")

    def _handle_click(self, x, y):
        """Handle a click event during manual calibration."""
        if not self._manual_mode or self._vol is None:
            return

        _nz, nc, ny, nx = self._vol.shape
        current = self._manual_channel_idx

        # Don't process clicks if we're already past the last beamlet
        if current >= nc:
            self.logger.info("All beamlets already marked. Click 'Finish'.")
            return

        channel = self._beam_order[current] if current < len(self._beam_order) else current

        # Clamp to image bounds
        x = max(0, min(nx - 1, x))
        y = max(0, min(ny - 1, y))

        self.logger.info(f"Beamlet {current + 1}: clicked at ({x:.1f}, {y:.1f})")

        # Find best z at this position
        ix, iy = round(x), round(y)
        patch_size = 10
        y0 = max(0, iy - patch_size)
        y1 = min(ny, iy + patch_size + 1)
        x0 = max(0, ix - patch_size)
        x1 = min(nx, ix + patch_size + 1)

        patch = self._vol[:, channel, y0:y1, x0:x1]
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
        best_z = int(np.argmax(smoothed))

        # Store position
        if current < len(self._manual_positions):
            self._manual_positions[current] = (x, y)
            self._manual_z_indices[current] = best_z
        else:
            self._manual_positions.append((x, y))
            self._manual_z_indices.append(best_z)

        # Auto-advance to next beamlet (but don't go past nc-1)
        if current < nc - 1:
            self._manual_channel_idx = current + 1
            self._show_beamlet(self._manual_channel_idx)
        else:
            # We just marked the last one - auto-finish
            self.logger.info("All beamlets marked! Starting calibration...")
            self._finish_manual_calibration()

    def _show_beamlet(self, idx):
        """Show a single beamlet's max projection in the viewer."""
        if self._max_projections is None or self.image_widget is None:
            return

        nc = self._max_projections.shape[0]
        channel = self._beam_order[idx] if idx < len(self._beam_order) else idx

        if channel >= nc:
            self.logger.error(f"Channel {channel} out of range ({nc} channels)")
            return

        # Navigate to this channel using ImageWidget indices
        # _max_projections is (C, Y, X) with 1 slider for C
        try:
            if self.image_widget.n_sliders > 0:
                self.image_widget.indices = [channel]

        except Exception as e:
            self.logger.exception(f"Failed to update display: {e}")

    def _manual_prev(self):
        """Go to previous beamlet."""
        if self._manual_channel_idx > 0:
            self._manual_channel_idx -= 1
            self._show_beamlet(self._manual_channel_idx)

    def _manual_next(self):
        """Go to next beamlet."""
        nc = self.num_beamlets
        if self._manual_channel_idx < nc - 1:
            self._manual_channel_idx += 1
            self._show_beamlet(self._manual_channel_idx)

    def _manual_skip(self):
        """Skip current beamlet (use center as position)."""
        if self._vol is None:
            return

        _nz, _nc, ny, nx = self._vol.shape
        current = self._manual_channel_idx
        channel = self._beam_order[current] if current < len(self._beam_order) else current

        # Use center
        x, y = nx / 2, ny / 2

        # Find best z at center
        patch_size = 10
        iy, ix = int(ny / 2), int(nx / 2)
        y0 = max(0, iy - patch_size)
        y1 = min(ny, iy + patch_size + 1)
        x0 = max(0, ix - patch_size)
        x1 = min(nx, ix + patch_size + 1)

        patch = self._vol[:, channel, y0:y1, x0:x1]
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
        best_z = int(np.argmax(smoothed))

        if current < len(self._manual_positions):
            self._manual_positions[current] = (x, y)
            self._manual_z_indices[current] = best_z
        else:
            self._manual_positions.append((x, y))
            self._manual_z_indices.append(best_z)

        self.logger.info(f"Beamlet {current + 1}: skipped (using center)")
        self._manual_next()

    def _cancel_manual_mode(self):
        """Cancel manual calibration."""
        self._manual_mode = False
        self._manual_positions = []
        self._manual_z_indices = []
        self._vol = None
        self._num_channels = None
        self._max_projections = None

        # Restore original data view before clearing references
        self._restore_original_view()

        # Clear stored references after restore
        self._original_metadata = None
        self._original_data_array = None
        self.logger.info("Manual calibration cancelled")

    def _restore_original_view(self):
        """Restore the original full data view."""
        if self.image_widget is None:
            return

        # Use stored original array if available
        arr = self._original_data_array if self._original_data_array is not None else self._get_array()
        if arr is None:
            return

        try:
            # Reload original lazy array
            self.image_widget.data[0] = arr

            # Reset indices to start
            if self.image_widget.n_sliders > 0:
                self.image_widget.indices = [0] * self.image_widget.n_sliders

            self.image_widget.figure[0, 0].auto_scale()
            self.logger.info("Restored original data view")
        except Exception as e:
            self.logger.exception(f"Failed to restore view: {e}")

    def _finish_manual_calibration(self):
        """Complete manual calibration and run analysis."""
        self._manual_mode = False

        positions = self._manual_positions
        z_indices = self._manual_z_indices

        if len(positions) < self.num_beamlets:
            self.logger.warning(f"Only {len(positions)} positions marked, expected {self.num_beamlets}")

        self.logger.info(f"Running calibration with {len(positions)} marked positions...")

        # Restore view before running background calibration
        self._restore_original_view()

        # Clear max projections (no longer needed)
        self._max_projections = None
        self._num_channels = None

        # Run calibration in background
        self._processing = True
        self._status = "Running calibration..."

        threading.Thread(
            target=self._run_calibration_with_positions,
            args=(positions, z_indices),
            daemon=True
        ).start()

    def _run_calibration_with_positions(self, positions, z_indices):
        """Run calibration using manually marked positions."""
        try:
            arr = self._get_array()
            if arr is None:
                raise ValueError("No data")

            fpath = self._get_fpath()
            if fpath is None:
                fpath = Path("calibration")

            vol = self._vol if self._vol is not None else np.asarray(arr[:]).astype(np.float32)
            if self._vol is None:
                vol -= vol.mean()

            nz, nc, ny, nx = vol.shape

            from mbo_utilities.gui._pollen_analysis import (
                correct_scan_phase,
                analyze_power_vs_z,
                analyze_z_positions,
                fit_exp_decay,
                plot_z_spacing,
                calibrate_xy,
                plot_beamlet_grid,
            )

            self._progress = 0.2
            # Use stored metadata (arr may be numpy array now)
            metadata = self._original_metadata if self._original_metadata else getattr(arr, "metadata", {})
            vol, _ = correct_scan_phase(vol, fpath, self._z_step_um, metadata, mode="manual")

            self._progress = 0.3
            plot_beamlet_grid(vol, self._beam_order, fpath, mode="manual")

            self._progress = 0.4
            Iz, III = self._extract_traces(vol, positions, z_indices)
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])

            self._progress = 0.5
            ZZ, zoi, pp = analyze_power_vs_z(Iz, fpath, self._z_step_um, self._beam_order, nc, mode="manual")

            self._progress = 0.6
            analyze_z_positions(ZZ, zoi, self._beam_order, fpath, self._cavity_info, mode="manual")

            self._progress = 0.7
            fit_exp_decay(ZZ, zoi, self._beam_order, fpath, pp, self._cavity_info, self._z_step_um, nz, mode="manual")

            self._progress = 0.8
            plot_z_spacing(ZZ, zoi, self._beam_order, fpath, mode="manual")

            self._progress = 0.9
            dx = dy = self._pixel_size_um
            calibrate_xy(xs, ys, III, fpath, dx, dy, nx, ny, self._cavity_info, mode="manual")

            self._progress = 1.0
            self._done = True
            self._results_manual = {
                "output_dir": str(fpath.parent),
                "h5_file": str(fpath.with_name(fpath.stem + "_pollen.h5")),
                "mode": "manual",
            }
            self.logger.info(f"Manual calibration complete! Results saved to {fpath.parent}")

        except Exception as e:
            self.logger.exception(f"Calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False
            self._vol = None
            self._original_metadata = None
            self._original_data_array = None

    # === Auto calibration methods ===

    def _start_auto_calibration(self):
        """Start automatic background calibration."""
        # Check if previous results exist
        self._scan_existing_h5_files()
        if self._existing_h5_files:
            # Load previous results instead of running calibration
            self._load_h5_file(str(self._existing_h5_files[0]))
            self._done = True
            self._status = "Loaded previous results"
            self.logger.info("Found previous calibration results, skipping auto calibration")
            return

        self._processing = True
        self._progress = 0.0
        self._done = False
        self._error = None
        self._results = None
        self._status = "Starting..."

        threading.Thread(target=self._auto_calibration_worker, daemon=True).start()

    def _auto_calibration_worker(self):
        """Background worker for automatic calibration."""
        try:
            arr = self._get_array()
            if arr is None:
                raise ValueError("No data loaded")

            fpath = self._get_fpath()
            if fpath is None:
                fpath = Path("calibration")

            self._status = "Loading data..."
            self._progress = 0.1
            vol = np.asarray(arr[:]).astype(np.float32)
            vol -= vol.mean()
            nz, nc, ny, nx = vol.shape

            self._status = "Detecting beads..."
            self._progress = 0.2
            positions, z_indices = self._detect_beads(vol)
            self.logger.info(f"Detected {len(positions)} bead positions")

            self._status = "Running calibration..."
            self._progress = 0.4

            from mbo_utilities.gui._pollen_analysis import (
                correct_scan_phase,
                analyze_power_vs_z,
                analyze_z_positions,
                fit_exp_decay,
                plot_z_spacing,
                calibrate_xy,
                plot_beamlet_grid,
            )

            vol, _ = correct_scan_phase(vol, fpath, self._z_step_um, arr.metadata, mode="auto")
            self._progress = 0.5

            plot_beamlet_grid(vol, self._beam_order, fpath, mode="auto")
            self._progress = 0.6

            Iz, III = self._extract_traces(vol, positions, z_indices)
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])

            ZZ, zoi, pp = analyze_power_vs_z(Iz, fpath, self._z_step_um, self._beam_order, nc, mode="auto")
            self._progress = 0.7

            analyze_z_positions(ZZ, zoi, self._beam_order, fpath, self._cavity_info, mode="auto")

            fit_exp_decay(ZZ, zoi, self._beam_order, fpath, pp, self._cavity_info, self._z_step_um, nz, mode="auto")
            self._progress = 0.8

            plot_z_spacing(ZZ, zoi, self._beam_order, fpath, mode="auto")
            self._progress = 0.9

            dx = dy = self._pixel_size_um
            calibrate_xy(xs, ys, III, fpath, dx, dy, nx, ny, self._cavity_info, mode="auto")

            self._progress = 1.0
            self._done = True
            self._results_auto = {
                "output_dir": str(fpath.parent),
                "h5_file": str(fpath.with_name(fpath.stem + "_pollen.h5")),
                "mode": "auto",
            }
            self.logger.info(f"Auto calibration complete! Results saved to {fpath.parent}")

        except Exception as e:
            self.logger.exception(f"Auto calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False

    def _detect_beads(self, vol):
        """Detect bead positions by tracking a reference bead across channels.

        Uses the same approach as manual mode: find a bead in the first channel,
        then track that same bead in other channels using cross-correlation.

        Returns positions and z_indices in beam order (matching manual calibration).
        """
        from scipy.signal import correlate2d

        _nz, nc, ny, nx = vol.shape
        positions = []
        z_indices = []
        patch_size = 10
        template_size = 25  # larger template for better matching

        # get max projections for all channels (same as manual mode shows)
        max_projs = vol.max(axis=0)  # (nc, ny, nx)

        # find reference bead in first channel (beam order)
        first_channel = self._beam_order[0] if len(self._beam_order) > 0 else 0
        ref_img = max_projs[first_channel]

        # find brightest region in reference channel
        threshold = np.percentile(ref_img, 90)
        mask = ref_img > threshold
        if mask.sum() > 0:
            yy, xx = np.where(mask)
            weights = ref_img[mask]
            ref_cx = np.average(xx, weights=weights)
            ref_cy = np.average(yy, weights=weights)
        else:
            peak_idx = np.argmax(ref_img)
            ref_cy, ref_cx = np.unravel_index(peak_idx, ref_img.shape)

        # extract reference template
        ix, iy = round(ref_cx), round(ref_cy)
        t_y0 = max(0, iy - template_size)
        t_y1 = min(ny, iy + template_size + 1)
        t_x0 = max(0, ix - template_size)
        t_x1 = min(nx, ix + template_size + 1)
        template = ref_img[t_y0:t_y1, t_x0:t_x1].copy()

        # normalize template for correlation
        template = template - template.mean()
        template_std = template.std()
        if template_std > 0:
            template = template / template_std

        # track bead in each channel using cross-correlation
        for idx in range(nc):
            channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
            img = max_projs[channel]

            if idx == 0:
                # first channel uses reference position directly
                cx, cy = ref_cx, ref_cy
            else:
                # cross-correlate template with this channel's image
                img_norm = img - img.mean()
                img_std = img.std()
                if img_std > 0:
                    img_norm = img_norm / img_std

                # use normalized cross-correlation
                corr = correlate2d(img_norm, template, mode='same')

                # find peak in correlation map
                peak_idx = np.argmax(corr)
                cy, cx = np.unravel_index(peak_idx, corr.shape)

                # refine with centroid around peak
                p_y0 = max(0, cy - 5)
                p_y1 = min(ny, cy + 6)
                p_x0 = max(0, cx - 5)
                p_x1 = min(nx, cx + 6)
                peak_region = corr[p_y0:p_y1, p_x0:p_x1]

                if peak_region.max() > 0:
                    # weighted centroid refinement
                    peak_region = peak_region - peak_region.min()
                    if peak_region.sum() > 0:
                        yy, xx = np.mgrid[0:peak_region.shape[0], 0:peak_region.shape[1]]
                        cx = p_x0 + np.average(xx, weights=peak_region)
                        cy = p_y0 + np.average(yy, weights=peak_region)

            positions.append((float(cx), float(cy)))

            # find best Z for this position
            ix, iy = round(cx), round(cy)
            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, channel, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode="nearest")
            best_z = int(np.argmax(smoothed))
            z_indices.append(best_z)

        return positions, z_indices

    def _extract_traces(self, vol, positions, z_indices):
        """Extract intensity traces and patches."""
        _nz, _nc, ny, nx = vol.shape
        Iz = []
        III = []
        patch_size = 10

        for idx, (x, y) in enumerate(positions):
            channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
            ix, iy = round(x), round(y)

            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, channel, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch, size=3, axis=1, mode="nearest")
            smoothed = uniform_filter1d(smoothed, size=3, axis=2, mode="nearest")
            trace = smoothed.max(axis=(1, 2))
            Iz.append(trace)

            best_z = z_indices[idx] if idx < len(z_indices) else 0
            III.append(vol[best_z, channel, y0:y1, x0:x1])

        Iz = np.vstack(Iz) if Iz else np.zeros((0, vol.shape[0]))

        if III:
            max_h = max(im.shape[0] for im in III)
            max_w = max(im.shape[1] for im in III)
            pads = [
                np.pad(im, ((0, max_h - im.shape[0]), (0, max_w - im.shape[1])), mode="constant")
                for im in III
            ]
            III = np.stack(pads, axis=-1)
        else:
            III = np.zeros((2 * patch_size + 1, 2 * patch_size + 1, 0))

        return Iz, III

    def _scan_saved_images(self, mode: str = "auto"):
        """Scan output directory for saved images of a specific mode."""
        self._saved_images = []
        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        out_dir = results.get("output_dir", "")
        if not out_dir:
            return

        out_path = Path(out_dir)
        if not out_path.exists():
            return

        # Find images matching the mode prefix (pollen_auto_* or pollen_manual_*)
        prefix = f"pollen_{mode}_"
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            for img in out_path.glob(ext):
                if img.name.startswith(prefix):
                    self._saved_images.append(img)

        # Sort by name
        self._saved_images.sort(key=lambda p: p.name)

    def _open_image(self, path: Path):
        """Open an image in the system default viewer."""
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                import os
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            self.logger.exception(f"Failed to open image: {e}")

    def _open_output_folder(self, mode: str = "auto"):
        """Open the output folder in file explorer."""
        import subprocess
        import sys

        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        out_dir = results.get("output_dir", "")
        if not out_dir:
            return

        try:
            if sys.platform == "win32":
                import os
                os.startfile(out_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", out_dir], check=False)
            else:
                subprocess.run(["xdg-open", out_dir], check=False)
        except Exception as e:
            self.logger.exception(f"Failed to open folder: {e}")

    def _open_all_graphs(self, mode: str = "auto"):
        """Show figures popup with implot graphs."""
        self._figures_popup_mode = mode
        self._show_figures_popup = True

    def _load_calibration_data(self, mode: str):
        """Load calibration data from H5 file for plotting."""
        from mbo_utilities.gui._pollen_analysis import extract_calibration_summary

        # Check cache first
        if mode == "auto" and self._calibration_data_auto is not None:
            return self._calibration_data_auto
        if mode == "manual" and self._calibration_data_manual is not None:
            return self._calibration_data_manual

        # Find H5 file
        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            # Try loaded external
            if self._loaded_external and self._loaded_external.get("h5_path"):
                h5_path = self._loaded_external["h5_path"]
            else:
                return None
        else:
            h5_path = results.get("h5_file", "")

        if not h5_path:
            return None

        data = extract_calibration_summary(h5_path, mode=mode)
        if data:
            if mode == "auto":
                self._calibration_data_auto = data
            else:
                self._calibration_data_manual = data

        return data

    def _draw_figures_popup(self):
        """Draw popup window with calibration plots using implot."""
        io = imgui.get_io()
        screen_w, screen_h = io.display_size.x, io.display_size.y
        win_w, win_h = min(600, screen_w * 0.5), min(550, screen_h * 0.7)

        imgui.set_next_window_pos(
            imgui.ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(imgui.ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)

        flags = imgui.WindowFlags_.no_collapse
        expanded, opened = imgui.begin("Calibration Results", True, flags)
        if not opened:
            self._show_figures_popup = False
            imgui.end()
            return

        if expanded:
            # Load data for both modes
            data_auto = self._load_calibration_data("auto")
            data_manual = self._load_calibration_data("manual")

            has_auto = data_auto is not None and "xs_um" in data_auto
            has_manual = data_manual is not None and "xs_um" in data_manual

            if not has_auto and not has_manual:
                imgui.text_disabled("No calibration data available")
                if imgui.button("Open Saved Figures"):
                    self._scan_saved_images(self._figures_popup_mode)
                    for img_path in self._saved_images:
                        self._open_image(img_path)
                imgui.end()
                return

            # Colors for auto/manual - match pollen_analysis.py
            color_auto = imgui.ImVec4(0.0, 0.75, 1.0, 1.0)   # cyan
            color_manual = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)  # green

            # Legend
            imgui.text("Legend:")
            imgui.same_line()
            imgui.text_colored(color_auto, "Auto")
            imgui.same_line()
            imgui.text_colored(color_manual, "Manual")
            imgui.spacing()

            # XY Positions plot
            plot_h = 220
            if implot.begin_plot("XY Positions (um)", imgui.ImVec2(-1, plot_h)):
                implot.setup_axes("X (um)", "Y (um)")
                implot.setup_axis_limits(implot.ImAxis_.x1, -150, 150, implot.Cond_.once)
                implot.setup_axis_limits(implot.ImAxis_.y1, -150, 150, implot.Cond_.once)

                # Plot auto data
                if has_auto:
                    xs_auto = np.asarray(data_auto["xs_um"], dtype=np.float64)
                    ys_auto = np.asarray(data_auto["ys_um"], dtype=np.float64)
                    implot.push_style_color(implot.Col_.marker_fill, color_auto)
                    implot.push_style_var(implot.StyleVar_.marker_size, 6.0)
                    implot.set_next_marker_style(implot.Marker_.circle)
                    implot.plot_scatter("Auto", xs_auto, ys_auto)
                    implot.pop_style_var()
                    implot.pop_style_color()

                # Plot manual data
                if has_manual:
                    xs_manual = np.asarray(data_manual["xs_um"], dtype=np.float64)
                    ys_manual = np.asarray(data_manual["ys_um"], dtype=np.float64)
                    implot.push_style_color(implot.Col_.marker_fill, color_manual)
                    implot.push_style_var(implot.StyleVar_.marker_size, 6.0)
                    implot.set_next_marker_style(implot.Marker_.square)
                    implot.plot_scatter("Manual", xs_manual, ys_manual)
                    implot.pop_style_var()
                    implot.pop_style_color()

                implot.end_plot()

            imgui.spacing()

            # X/Y offsets bar chart
            if implot.begin_plot("XY Offsets (um)", imgui.ImVec2(-1, plot_h)):
                implot.setup_axes("Beam #", "Offset (um)")

                # Determine beam count
                n_beams = 0
                if has_auto and "diffx" in data_auto:
                    n_beams = max(n_beams, len(data_auto["diffx"]))
                if has_manual and "diffx" in data_manual:
                    n_beams = max(n_beams, len(data_manual["diffx"]))

                if n_beams > 0:
                    beam_nums = np.arange(1, n_beams + 1, dtype=np.float64)
                    bar_width = 0.35

                    # Auto X offsets
                    if has_auto and "diffx" in data_auto:
                        diffx_auto = np.asarray(data_auto["diffx"], dtype=np.float64)
                        implot.push_style_color(implot.Col_.fill, color_auto)
                        implot.plot_bars("dX Auto", beam_nums - bar_width/2, diffx_auto, bar_width)
                        implot.pop_style_color()

                    # Manual X offsets
                    if has_manual and "diffx" in data_manual:
                        diffx_manual = np.asarray(data_manual["diffx"], dtype=np.float64)
                        implot.push_style_color(implot.Col_.fill, color_manual)
                        implot.plot_bars("dX Manual", beam_nums + bar_width/2, diffx_manual, bar_width)
                        implot.pop_style_color()

                implot.end_plot()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Summary stats
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.9, 1.0), "Summary:")
            if has_auto:
                rms_dx = data_auto.get("rms_dx")
                rms_dy = data_auto.get("rms_dy")
                if rms_dx is not None and rms_dy is not None:
                    imgui.text_colored(color_auto, f"  Auto RMS: dX={rms_dx:.2f}um, dY={rms_dy:.2f}um")

            if has_manual:
                rms_dx = data_manual.get("rms_dx")
                rms_dy = data_manual.get("rms_dy")
                if rms_dx is not None and rms_dy is not None:
                    imgui.text_colored(color_manual, f"  Manual RMS: dX={rms_dx:.2f}um, dY={rms_dy:.2f}um")

            imgui.spacing()

            # Open full figures button
            if imgui.button("Open All Saved Figures"):
                # Open figures for both modes if available
                for mode in ["auto", "manual"]:
                    self._scan_saved_images(mode)
                    for img_path in self._saved_images:
                        self._open_image(img_path)

        imgui.end()

    def draw_menu_bar(self) -> None:
        """Render the menu bar."""
        if imgui.begin_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Open File", "Ctrl+O")[0]:
                    pass
                imgui.end_menu()

            if imgui.begin_menu("View"):
                if imgui.menu_item("Metadata", "M")[0]:
                    self._panels["metadata"].toggle()
                if imgui.menu_item("Debug Log")[0]:
                    self._panels["debug"].toggle()
                imgui.end_menu()

            if imgui.begin_menu("Help"):
                if imgui.menu_item("Documentation")[0]:
                    import webbrowser
                    webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/")
                imgui.end_menu()

            imgui.end_menu_bar()

    def on_data_loaded(self) -> None:
        """Reinitialize when new data is loaded."""
        self._init_from_data()
        self._done = False
        self._error = None
        self._results_auto = None
        self._results_manual = None
        self._manual_mode = False
        self._saved_images = []
        self._show_figures_popup = False
        self._calibration_data_auto = None
        self._calibration_data_manual = None

        if self._initialized and not self._processing:
            self._start_auto_calibration()

    def cleanup(self) -> None:
        """Clean up resources when viewer closes."""
        self._vol = None
        self._num_channels = None
        self._max_projections = None
        self._original_metadata = None
        self._original_data_array = None
        self._manual_mode = False
        self._pointer_down_pos = None
        self._calibration_data_auto = None
        self._calibration_data_manual = None
        super().cleanup()
