"""
Pollen calibration widget for LBM beamlet calibration.

This widget handles pollen calibration data (ZCYX) and provides:
- Info panel showing beamlet count, cavities, z-step, pixel size
- Automatic background calibration (detection + analysis)
- Manual calibration mode: click through beamlets in the viewer
"""

from pathlib import Path
from typing import TYPE_CHECKING
import threading

import numpy as np
from scipy.ndimage import uniform_filter1d

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.gui.main_widgets._base import MainWidget
from mbo_utilities.gui._widgets import set_tooltip
from mbo_utilities.metadata import get_param
from mbo_utilities.metadata.scanimage import (
    get_lbm_ai_sources,
    get_z_step_size,
)

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


def get_cavity_indices(metadata: dict, nc: int) -> dict:
    """Get cavity A and cavity B channel indices from LBM metadata."""
    from mbo_utilities.metadata.scanimage import is_lbm_stack

    result = {
        'cavity_a': [],
        'cavity_b': [],
        'is_lbm': False,
        'num_cavities': 1,
    }

    if not is_lbm_stack(metadata):
        half = nc // 2
        result['cavity_a'] = list(range(half))
        result['cavity_b'] = list(range(half, nc))
        return result

    result['is_lbm'] = True
    ai_sources = get_lbm_ai_sources(metadata)

    if not ai_sources:
        half = nc // 2
        result['cavity_a'] = list(range(half))
        result['cavity_b'] = list(range(half, nc))
        return result

    sorted_sources = sorted(ai_sources.keys())

    if len(sorted_sources) >= 1:
        cavity_a_channels = ai_sources.get(sorted_sources[0], [])
        result['cavity_a'] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_a_channels])

    if len(sorted_sources) >= 2:
        cavity_b_channels = ai_sources.get(sorted_sources[1], [])
        result['cavity_b'] = sorted([ch - 1 if ch > 0 else ch for ch in cavity_b_channels])
        result['num_cavities'] = 2

    return result


class PollenCalibrationWidget(MainWidget):
    """
    Main widget for pollen calibration data.

    Handles ZCYX pollen stacks for LBM beamlet calibration.
    Runs automatic calibration in background, provides manual mode.
    """

    name = "Pollen Calibration"

    # Default beam order for 30-channel system
    DEFAULT_ORDER_30 = [
        0, 4, 5, 6, 7, 8, 1, 9, 10, 11, 12, 13, 14, 15,
        2, 16, 17, 18, 19, 20, 21, 3, 22, 23, 24, 25, 26, 27, 28, 29
    ]

    def __init__(self, parent: "PreviewDataWidget"):
        super().__init__(parent)

        # Pollen-specific state
        self._z_step_um = 1.0
        self._pixel_size_um = 1.0
        self._fov_um = 600.0
        self._zoom = 1.0

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

        # Image preview state
        self._saved_images: list[Path] = []
        self._show_image_popup = False
        self._image_popup_open = False
        self._image_popup_mode = None  # Which mode's images to show

        # Results table state
        self._show_results_table = False
        self._results_table_open = False
        self._calibration_data_auto = None
        self._calibration_data_manual = None

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

        metadata = getattr(arr, 'metadata', {})

        z_step = get_z_step_size(metadata)
        if z_step is not None:
            self._z_step_um = z_step
        else:
            self._z_step_um = get_param(metadata, "dz", default=1.0)

        self._zoom = get_param(metadata, "zoom_factor", default=1.0)
        if arr.ndim >= 2:
            nx = arr.shape[-1]
            self._pixel_size_um = self._fov_um / self._zoom / nx

        nc = getattr(arr, 'num_channels', arr.shape[1] if arr.ndim >= 2 else 1)
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
        parent_fpath = self.parent.fpath
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
        if hasattr(arr, 'num_beamlets'):
            return arr.num_beamlets
        if hasattr(arr, 'num_channels'):
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
        return getattr(arr, 'num_zplanes', arr.shape[0])

    def draw(self) -> None:
        """Draw the pollen calibration UI."""
        if not self._initialized:
            self._init_from_data()
            if self._initialized and not self._processing and not self._done:
                self._start_auto_calibration()

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
            self._draw_auto_status()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            self._draw_manual_button()

    def _draw_info_section(self):
        """Draw info panel."""
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Pollen Calibration")
        imgui.spacing()

        arr = self._get_array()
        if arr is not None:
            imgui.text(f"Shape: {arr.shape} (Z, C, Y, X)")
            imgui.text(f"Beamlets: {self.num_beamlets}")
            imgui.text(f"Z-planes: {self.num_z_planes}")
            if self._cavity_info and self._cavity_info['is_lbm']:
                imgui.text(f"Cavities: {self._cavity_info['num_cavities']}")
        else:
            imgui.text_disabled("No data loaded")

    def _draw_auto_status(self):
        """Draw automatic calibration status and results for both modes."""
        # Show processing status
        if self._processing:
            imgui.text_colored(imgui.ImVec4(0.6, 0.6, 0.8, 1.0), "Processing...")
            imgui.text(self._status)
            imgui.progress_bar(self._progress, imgui.ImVec2(-1, 0))
            imgui.spacing()
        elif self._error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.3, 0.3, 1.0), f"Error: {self._error}")
            imgui.spacing()

        # Auto calibration results
        if self._results_auto:
            imgui.text_colored(imgui.ImVec4(0.4, 0.7, 1.0, 1.0), "Auto Results")
            if imgui.button("Graphs##auto"):
                self._image_popup_mode = "auto"
                self._show_image_popup = True
                self._scan_saved_images("auto")
            imgui.same_line()
            if imgui.button("Table##auto"):
                self._show_results_table = True
                self._load_calibration_data("auto")
            imgui.spacing()

        # Manual calibration results
        if self._results_manual:
            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.6, 1.0), "Manual Results")
            if imgui.button("Graphs##manual"):
                self._image_popup_mode = "manual"
                self._show_image_popup = True
                self._scan_saved_images("manual")
            imgui.same_line()
            if imgui.button("Table##manual"):
                self._show_results_table = True
                self._load_calibration_data("manual")
            imgui.spacing()

        # Comparison button - appears when both auto and manual are done
        if self._results_auto and self._results_manual:
            imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.4, 1.0), "Comparison")
            if imgui.button("Auto vs Manual"):
                self._show_comparison()
            imgui.spacing()

        # Show waiting if nothing is done yet
        if not self._processing and not self._results_auto and not self._results_manual and not self._error:
            imgui.text_disabled("Waiting...")

        # Draw popups
        self._draw_image_popup()
        self._draw_results_table()

    def _draw_manual_button(self):
        """Draw manual calibration button."""
        imgui.text_colored(imgui.ImVec4(0.8, 0.6, 0.2, 1.0), "Interactive Mode")
        imgui.spacing()

        if imgui.button("Start Manual Calibration", imgui.ImVec2(-1, 0)):
            self._start_manual_mode()

        set_tooltip(
            "Click on the same pollen bead in each beamlet.\n"
            "The viewer will show one beamlet at a time."
        )

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

        # Store original metadata before we replace viewer data with numpy array
        self._original_metadata = getattr(arr, 'metadata', {})

        # Load volume into memory for click analysis
        self.logger.info("Loading volume for manual calibration...")
        self._vol = np.asarray(arr[:]).astype(np.float32)
        self._vol -= self._vol.mean()

        # For ZCYX data: shape is (Z, C, Y, X)
        # Z = piezo positions (typically many, e.g. 224)
        # C = beamlet channels (typically fewer, e.g. 14)
        self.logger.info(f"Volume shape: {self._vol.shape} (Z, C, Y, X)")
        nz, nc, ny, nx = self._vol.shape
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
        """Set up click event handler on the image."""
        if self.image_widget is None:
            return

        # Get the subplot and add click handler
        try:
            subplot = self.image_widget.figure[0, 0]

            def on_click(ev):
                if not self._manual_mode:
                    return

                # Map screen coords to world coords
                try:
                    world_pos = subplot.map_screen_to_world((ev.x, ev.y))
                    if world_pos is not None:
                        x, y = world_pos[0], world_pos[1]
                        self._handle_click(x, y)
                except Exception as e:
                    self.logger.error(f"Click mapping error: {e}")

            subplot.renderer.add_event_handler(on_click, "click")
            self._click_handler = on_click
            self.logger.info("Click handler registered")

        except Exception as e:
            self.logger.error(f"Failed to set up click handler: {e}")

    def _handle_click(self, x, y):
        """Handle a click event during manual calibration."""
        if not self._manual_mode or self._vol is None:
            return

        nz, nc, ny, nx = self._vol.shape
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
        ix, iy = int(round(x)), int(round(y))
        patch_size = 10
        y0 = max(0, iy - patch_size)
        y1 = min(ny, iy + patch_size + 1)
        x0 = max(0, ix - patch_size)
        x1 = min(nx, ix + patch_size + 1)

        patch = self._vol[:, channel, y0:y1, x0:x1]
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode='nearest')
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
            self.logger.error(f"Failed to update display: {e}")

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

        nz, nc, ny, nx = self._vol.shape
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
        smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode='nearest')
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
        self._original_metadata = None

        # Restore original data view
        self._restore_original_view()
        self.logger.info("Manual calibration cancelled")

    def _restore_original_view(self):
        """Restore the original full data view."""
        arr = self._get_array()
        if arr is None or self.image_widget is None:
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
            self.logger.error(f"Failed to restore view: {e}")

    def _finish_manual_calibration(self):
        """Complete manual calibration and run analysis."""
        self._manual_mode = False

        positions = self._manual_positions
        z_indices = self._manual_z_indices

        if len(positions) < self.num_beamlets:
            self.logger.warning(f"Only {len(positions)} positions marked, expected {self.num_beamlets}")

        self.logger.info(f"Running calibration with {len(positions)} marked positions...")

        # Restore view
        self._restore_original_view()

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

            from mbo_utilities.gui.main_widgets._pollen_analysis import (
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
            metadata = self._original_metadata if self._original_metadata else getattr(arr, 'metadata', {})
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
                'output_dir': str(fpath.parent),
                'h5_file': str(fpath.with_name(fpath.stem + "_pollen_manual.h5")),
                'mode': 'manual',
            }
            self.logger.info(f"Manual calibration complete! Results saved to {fpath.parent}")

            # Auto-generate comparison if auto results are also available
            self._generate_comparison_if_ready()

        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False
            self._vol = None

    # === Auto calibration methods ===

    def _start_auto_calibration(self):
        """Start automatic background calibration."""
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

            from mbo_utilities.gui.main_widgets._pollen_analysis import (
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
                'output_dir': str(fpath.parent),
                'h5_file': str(fpath.with_name(fpath.stem + "_pollen_auto.h5")),
                'mode': 'auto',
            }
            self.logger.info(f"Auto calibration complete! Results saved to {fpath.parent}")

        except Exception as e:
            self.logger.error(f"Auto calibration failed: {e}")
            self._error = str(e)
        finally:
            self._processing = False

    def _detect_beads(self, vol):
        """Detect bead positions in each channel."""
        nz, nc, ny, nx = vol.shape
        positions = []
        z_indices = []

        for c in range(nc):
            img = vol[:, c, :, :].max(axis=0)
            threshold = np.percentile(img, 95)
            mask = img > threshold

            if mask.sum() > 0:
                yy, xx = np.where(mask)
                weights = img[mask]
                cx = np.average(xx, weights=weights)
                cy = np.average(yy, weights=weights)
            else:
                cx, cy = nx / 2, ny / 2

            positions.append((cx, cy))

            ix, iy = int(round(cx)), int(round(cy))
            patch_size = 10
            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, c, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch.max(axis=(1, 2)), size=3, mode='nearest')
            best_z = int(np.argmax(smoothed))
            z_indices.append(best_z)

        return positions, z_indices

    def _extract_traces(self, vol, positions, z_indices):
        """Extract intensity traces and patches."""
        nz, nc, ny, nx = vol.shape
        Iz = []
        III = []
        patch_size = 10

        for idx, (x, y) in enumerate(positions):
            channel = self._beam_order[idx] if idx < len(self._beam_order) else idx
            ix, iy = int(round(x)), int(round(y))

            y0 = max(0, iy - patch_size)
            y1 = min(ny, iy + patch_size + 1)
            x0 = max(0, ix - patch_size)
            x1 = min(nx, ix + patch_size + 1)

            patch = vol[:, channel, y0:y1, x0:x1]
            smoothed = uniform_filter1d(patch, size=3, axis=1, mode='nearest')
            smoothed = uniform_filter1d(smoothed, size=3, axis=2, mode='nearest')
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

        out_dir = results.get('output_dir', '')
        if not out_dir:
            return

        out_path = Path(out_dir)
        if not out_path.exists():
            return

        # Find images matching the mode prefix (pollen_auto_* or pollen_manual_*)
        prefix = f"pollen_{mode}_"
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.svg'):
            for img in out_path.glob(ext):
                if img.name.startswith(prefix):
                    self._saved_images.append(img)

        # Sort by name
        self._saved_images.sort(key=lambda p: p.name)

    def _draw_image_popup(self):
        """Draw popup window listing saved images."""
        if self._show_image_popup:
            self._image_popup_open = True
            mode_label = "Manual" if self._image_popup_mode == "manual" else "Auto"
            imgui.open_popup(f"Outputs ({mode_label})")
            self._show_image_popup = False

        mode_label = "Manual" if self._image_popup_mode == "manual" else "Auto"
        imgui.set_next_window_size(imgui.ImVec2(400, 400), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            f"Outputs ({mode_label})",
            p_open=True if self._image_popup_open else None,
            flags=imgui.WindowFlags_.no_saved_settings
        )

        if opened:
            if not visible:
                self._image_popup_open = False
                imgui.close_current_popup()
                imgui.end_popup()
                return

            imgui.text(f"Found {len(self._saved_images)} {mode_label.lower()} images")
            imgui.separator()
            imgui.spacing()

            # Scrollable list of images
            if imgui.begin_child("image_list", imgui.ImVec2(0, -30)):
                for img_path in self._saved_images:
                    # Show filename without prefix for cleaner display
                    display_name = img_path.name.replace(f"pollen_{self._image_popup_mode}_", "")
                    if imgui.button(display_name, imgui.ImVec2(-1, 0)):
                        self._open_image(img_path)
                imgui.end_child()

            imgui.spacing()
            if imgui.button("Open Folder"):
                self._open_output_folder(self._image_popup_mode)
            imgui.same_line()
            if imgui.button("Close"):
                self._image_popup_open = False
                imgui.close_current_popup()

            imgui.end_popup()

    def _open_image(self, path: Path):
        """Open an image in the system default viewer."""
        import subprocess
        import sys

        try:
            if sys.platform == 'win32':
                import os
                os.startfile(str(path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(path)])
            else:
                subprocess.run(['xdg-open', str(path)])
        except Exception as e:
            self.logger.error(f"Failed to open image: {e}")

    def _open_output_folder(self, mode: str = "auto"):
        """Open the output folder in file explorer."""
        import subprocess
        import sys

        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        out_dir = results.get('output_dir', '')
        if not out_dir:
            return

        try:
            if sys.platform == 'win32':
                import os
                os.startfile(out_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', out_dir])
            else:
                subprocess.run(['xdg-open', out_dir])
        except Exception as e:
            self.logger.error(f"Failed to open folder: {e}")

    def _show_comparison(self):
        """Generate and show comparison plot between auto and manual results."""
        fpath = self._get_fpath()
        if fpath is None:
            self.logger.error("No file path for comparison")
            return

        from mbo_utilities.gui.main_widgets._pollen_analysis import plot_comparison

        success = plot_comparison(fpath)
        if success:
            # Open the comparison image
            comparison_path = fpath.with_name("pollen_comparison.png")
            if comparison_path.exists():
                self._open_image(comparison_path)
        else:
            self.logger.warning("Could not generate comparison - both modes required")

    def _generate_comparison_if_ready(self):
        """Generate comparison plot if both auto and manual results are available."""
        if not self._results_auto or not self._results_manual:
            return

        fpath = self._get_fpath()
        if fpath is None:
            return

        from mbo_utilities.gui.main_widgets._pollen_analysis import plot_comparison

        try:
            plot_comparison(fpath)
            self.logger.info("Generated auto vs manual comparison plot")
        except Exception as e:
            self.logger.error(f"Failed to generate comparison: {e}")

    def _load_calibration_data(self, mode: str = "auto"):
        """Load calibration data from HDF5 file for results table."""
        import h5py

        # Store which mode we're displaying
        self._results_table_mode = mode

        results = self._results_auto if mode == "auto" else self._results_manual
        if not results:
            return

        h5_file = results.get('h5_file', '')
        if not h5_file or not Path(h5_file).exists():
            return

        try:
            with h5py.File(h5_file, 'r') as f:
                data = {
                    'num_beamlets': f.attrs.get('num_planes', 0),
                    'z_step_um': f.attrs.get('z_step_um', 0),
                    'is_lbm': f.attrs.get('is_lbm', False),
                    'num_cavities': f.attrs.get('num_cavities', 1),
                    'mode': mode,
                }

                # Load arrays
                if 'diffx' in f:
                    data['diffx'] = f['diffx'][:]
                if 'diffy' in f:
                    data['diffy'] = f['diffy'][:]
                if 'xs_um' in f:
                    data['xs_um'] = f['xs_um'][:]
                if 'ys_um' in f:
                    data['ys_um'] = f['ys_um'][:]
                if 'scan_corrections' in f:
                    data['scan_corrections'] = f['scan_corrections'][:]
                if 'cavity_a_channels' in f:
                    data['cavity_a'] = f['cavity_a_channels'][:]
                if 'cavity_b_channels' in f:
                    data['cavity_b'] = f['cavity_b_channels'][:]

                # Store in appropriate slot
                if mode == "auto":
                    self._calibration_data_auto = data
                else:
                    self._calibration_data_manual = data

                self.logger.info(f"Loaded {mode} calibration data: {len(data)} fields")

        except Exception as e:
            self.logger.error(f"Failed to load calibration data: {e}")

    def _draw_results_table(self):
        """Draw popup window with calibration results table."""
        if self._show_results_table:
            self._results_table_open = True
            mode = getattr(self, '_results_table_mode', 'auto')
            mode_label = "Manual" if mode == "manual" else "Auto"
            imgui.open_popup(f"Results ({mode_label})")
            self._show_results_table = False

        mode = getattr(self, '_results_table_mode', 'auto')
        mode_label = "Manual" if mode == "manual" else "Auto"
        imgui.set_next_window_size(imgui.ImVec2(500, 450), imgui.Cond_.first_use_ever)

        opened, visible = imgui.begin_popup_modal(
            f"Results ({mode_label})",
            p_open=True if self._results_table_open else None,
            flags=imgui.WindowFlags_.no_saved_settings
        )

        if opened:
            if not visible:
                self._results_table_open = False
                imgui.close_current_popup()
                imgui.end_popup()
                return

            # Get data for current mode
            data = self._calibration_data_auto if mode == "auto" else self._calibration_data_manual
            if data is None:
                imgui.text("No calibration data available")
                if imgui.button("Close"):
                    self._results_table_open = False
                    imgui.close_current_popup()
                imgui.end_popup()
                return

            # Summary section with mode indicator
            color = imgui.ImVec4(0.4, 0.7, 1.0, 1.0) if mode == "auto" else imgui.ImVec4(0.4, 1.0, 0.6, 1.0)
            imgui.text_colored(color, f"{mode_label} Calibration Summary")
            imgui.separator()
            imgui.text(f"Beamlets: {data.get('num_beamlets', 'N/A')}")
            z_step = data.get('z_step_um', 0)
            imgui.text(f"Z-step: {z_step:.2f} um" if z_step else "Z-step: N/A")
            imgui.text(f"LBM: {'Yes' if data.get('is_lbm') else 'No'}")
            imgui.text(f"Cavities: {data.get('num_cavities', 1)}")
            imgui.spacing()

            # Beamlet table
            imgui.text_colored(imgui.ImVec4(0.6, 0.8, 0.6, 1.0), "Per-Beamlet Values")
            imgui.separator()

            # Table with scroll
            table_flags = (
                imgui.TableFlags_.borders |
                imgui.TableFlags_.row_bg |
                imgui.TableFlags_.scroll_y |
                imgui.TableFlags_.resizable
            )

            if imgui.begin_table("beamlet_table", 5, table_flags, imgui.ImVec2(0, 250)):
                # Headers
                imgui.table_setup_column("Beam", imgui.TableColumnFlags_.width_fixed, 45)
                imgui.table_setup_column("X (um)", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_setup_column("Y (um)", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_setup_column("dX (um)", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_setup_column("dY (um)", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_setup_scroll_freeze(0, 1)
                imgui.table_headers_row()

                # Data rows
                xs = data.get('xs_um', [])
                ys = data.get('ys_um', [])
                dx = data.get('diffx', [])
                dy = data.get('diffy', [])
                n_rows = max(len(xs), len(ys), len(dx), len(dy)) if any([len(xs), len(ys), len(dx), len(dy)]) else 0

                for i in range(n_rows):
                    imgui.table_next_row()

                    imgui.table_next_column()
                    imgui.text(f"{i + 1}")

                    imgui.table_next_column()
                    if i < len(xs):
                        imgui.text(f"{xs[i]:.1f}")

                    imgui.table_next_column()
                    if i < len(ys):
                        imgui.text(f"{ys[i]:.1f}")

                    imgui.table_next_column()
                    if i < len(dx):
                        imgui.text(f"{dx[i]:.1f}")

                    imgui.table_next_column()
                    if i < len(dy):
                        imgui.text(f"{dy[i]:.1f}")

                imgui.end_table()

            imgui.spacing()

            # Buttons row
            if imgui.button("View Graphs"):
                self._image_popup_mode = mode
                self._show_image_popup = True
                self._scan_saved_images(mode)
            imgui.same_line()
            results = self._results_auto if mode == "auto" else self._results_manual
            if imgui.button("Open H5 File"):
                h5_file = results.get('h5_file', '') if results else ''
                if h5_file:
                    self._open_image(Path(h5_file))
            imgui.same_line()
            if imgui.button("Close"):
                self._results_table_open = False
                imgui.close_current_popup()

            imgui.end_popup()

    def on_data_loaded(self) -> None:
        """Reinitialize when new data is loaded."""
        self._init_from_data()
        self._done = False
        self._error = None
        self._results_auto = None
        self._results_manual = None
        self._manual_mode = False
        self._saved_images = []
        self._calibration_data_auto = None
        self._calibration_data_manual = None

        if self._initialized and not self._processing:
            self._start_auto_calibration()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._vol = None
        self._num_channels = None
        self._max_projections = None
        self._original_metadata = None
        self._calibration_data_auto = None
        self._calibration_data_manual = None
        self._manual_mode = False
