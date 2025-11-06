"""Tests for graphics module with screenshot generation."""

import os
import sys

import numpy as np
import pytest

# Skip all tests on Windows or if running in CI without offscreen rendering
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or os.environ.get("RENDERCANVAS_FORCE_OFFSCREEN") != "1",
    reason="Graphics tests require offscreen rendering (Linux CI only)",
)


def test_wgpu_backend_available():
    """Test that wgpu backend is properly configured."""
    from tests.test_utils import get_wgpu_backend

    backend = get_wgpu_backend()
    print(f"WGPU backend: {backend}")
    assert backend != "unknown"


class TestSingleZPlaneGUI:
    """Tests for single z-plane data visualization."""

    @pytest.fixture
    def single_zplane_data(self):
        """Create single z-plane test data."""
        from tests.test_utils import create_test_data_single_zplane

        return create_test_data_single_zplane(shape=(50, 256, 256))

    def test_preview_tab_visible(self, single_zplane_data):
        """Test that Preview tab is visible for single z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def check_tabs(gui, iw):
            # For single z-plane, nz should be 1
            assert gui.nz == 1, f"Expected nz=1, got {gui.nz}"
            # Preview tab should be available
            assert hasattr(gui, "draw_preview_section")

        run_gui_with_screenshot(
            single_zplane_data,
            "single_zplane_preview_tab.png",
            gui_action=check_tabs,
            frames=10,
        )

    def test_summary_stats_tab(self, single_zplane_data):
        """Test Summary Stats tab for single z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def show_stats_tab(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Check that stats were computed
            assert all(gui._zstats_done), "Stats computation did not complete"
            assert len(gui._zstats) > 0
            assert "mean" in gui._zstats[0]

        run_gui_with_screenshot(
            single_zplane_data,
            "single_zplane_summary_stats.png",
            gui_action=show_stats_tab,
            frames=30,
        )

    def test_process_tab_visible(self, single_zplane_data):
        """Test that Process tab is visible for single z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def check_process_tab(gui, iw):
            # For single z-plane, Process tab should be available
            assert gui.nz == 1

        run_gui_with_screenshot(
            single_zplane_data,
            "single_zplane_process_tab.png",
            gui_action=check_process_tab,
            frames=10,
        )

    def test_signal_quality_metrics_table(self, single_zplane_data):
        """Test signal quality metrics table for single z-plane."""
        from tests.test_utils import run_gui_with_screenshot

        def verify_metrics(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Verify stats structure
            assert all(gui._zstats_done), "Stats computation did not complete"
            stats = gui._zstats[0]
            assert "mean" in stats
            assert "std" in stats
            assert "snr" in stats
            assert len(stats["mean"]) == 1  # Single z-plane

        run_gui_with_screenshot(
            single_zplane_data,
            "single_zplane_metrics_table.png",
            gui_action=verify_metrics,
            frames=30,
        )

    def test_bar_chart_visualization(self, single_zplane_data):
        """Test bar chart visualization for single z-plane."""
        from tests.test_utils import run_gui_with_screenshot

        def verify_chart(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Select stats section
            gui._selected_array = 0

        run_gui_with_screenshot(
            single_zplane_data,
            "single_zplane_bar_chart.png",
            gui_action=verify_chart,
            frames=30,
        )


class TestMultiZPlaneGUI:
    """Tests for multi z-plane data visualization."""

    @pytest.fixture
    def multi_zplane_data(self):
        """Create multi z-plane test data."""
        from tests.test_utils import create_test_data_multi_zplane

        return create_test_data_multi_zplane(shape=(50, 5, 256, 256))

    def test_only_summary_tab_visible(self, multi_zplane_data):
        """Test that only Summary Stats tab is visible for multi z-plane."""
        from tests.test_utils import run_gui_with_screenshot

        def check_tabs(gui, iw):
            # For multi z-plane, nz should be > 1
            assert gui.nz > 1, f"Expected nz>1, got {gui.nz}"

        run_gui_with_screenshot(
            multi_zplane_data,
            "multi_zplane_summary_only.png",
            gui_action=check_tabs,
            frames=10,
        )

    def test_zplane_profile_plot(self, multi_zplane_data):
        """Test z-plane profile plot for multi z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def verify_plot(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Check stats for multiple z-planes
            assert all(gui._zstats_done), "Stats computation did not complete"
            stats = gui._zstats[0]
            assert len(stats["mean"]) == gui.nz, "Stats should match number of z-planes"

        run_gui_with_screenshot(
            multi_zplane_data,
            "multi_zplane_profile_plot.png",
            gui_action=verify_plot,
            frames=30,
        )

    def test_combined_view_multi_roi(self, multi_zplane_data):
        """Test combined view for multi z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def show_combined(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Select combined view (last option)
            gui._selected_array = len(gui._zstats)

        run_gui_with_screenshot(
            multi_zplane_data,
            "multi_zplane_combined_view.png",
            gui_action=show_combined,
            frames=30,
        )

    def test_zplane_table_display(self, multi_zplane_data):
        """Test z-plane table display for multi z-plane data."""
        from tests.test_utils import run_gui_with_screenshot

        def verify_table(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            assert all(gui._zstats_done), "Stats computation did not complete"
            stats = gui._zstats[0]
            # Should have multiple z-planes
            assert len(stats["mean"]) > 1

        run_gui_with_screenshot(
            multi_zplane_data,
            "multi_zplane_table.png",
            gui_action=verify_table,
            frames=30,
        )


class TestGUIMenuItems:
    """Tests for GUI menu items and interactions."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        from tests.test_utils import create_test_data_single_zplane

        return create_test_data_single_zplane(shape=(50, 256, 256))

    def test_menu_bar_present(self, test_data):
        """Test that menu bar is present."""
        from tests.test_utils import run_gui_with_screenshot

        def check_menu(gui, iw):
            # Menu should be drawn by draw_menu function
            assert hasattr(gui, "show_debug_panel")
            assert hasattr(gui, "show_scope_window")

        run_gui_with_screenshot(
            test_data,
            "gui_menu_bar.png",
            gui_action=check_menu,
            frames=10,
        )

    def test_debug_panel_toggle(self, test_data):
        """Test debug panel toggle."""
        from tests.test_utils import run_gui_with_screenshot

        def toggle_debug(gui, iw):
            initial_state = gui.show_debug_panel
            gui.show_debug_panel = not initial_state
            # Render a few frames
            for _ in range(5):
                iw.figure.canvas.draw()

        run_gui_with_screenshot(
            test_data,
            "gui_debug_panel.png",
            gui_action=toggle_debug,
            frames=15,
        )

    def test_array_selector(self, test_data):
        """Test array selector functionality."""
        from tests.test_utils import run_gui_with_screenshot

        def test_selector(gui, iw):
            # Manually compute stats (threading is disabled in tests)
            # Call the internal method directly to avoid threading
            for idx, arr in enumerate(iw.data, start=1):
                gui._compute_zstats_single_array(idx, arr)

            # Try selecting different arrays
            initial_selection = gui._selected_array
            gui._selected_array = 0
            assert gui._selected_array == 0

        run_gui_with_screenshot(
            test_data,
            "gui_array_selector.png",
            gui_action=test_selector,
            frames=30,
        )


class TestGUIWindowFunctions:
    """Tests for window function controls."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        from tests.test_utils import create_test_data_single_zplane

        return create_test_data_single_zplane(shape=(50, 256, 256))

    def test_projection_selector(self, test_data):
        """Test projection method selector."""
        from tests.test_utils import run_gui_with_screenshot

        def test_projections(gui, iw):
            # Test projection options
            projections = ["mean", "max", "std"]
            for proj in projections:
                gui._proj = proj
                iw.figure.canvas.draw()

        run_gui_with_screenshot(
            test_data,
            "gui_projection_selector.png",
            gui_action=test_projections,
            frames=15,
        )

    def test_window_size_slider(self, test_data):
        """Test window size slider."""
        from tests.test_utils import run_gui_with_screenshot

        def test_window_size(gui, iw):
            # Test different window sizes
            gui._window_size = 5
            iw.figure.canvas.draw()

        run_gui_with_screenshot(
            test_data,
            "gui_window_size.png",
            gui_action=test_window_size,
            frames=10,
        )


class TestPhaseCorrection:
    """Tests for phase correction controls."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        from tests.test_utils import create_test_data_single_zplane

        return create_test_data_single_zplane(shape=(50, 256, 256))

    def test_phase_correction_controls(self, test_data):
        """Test phase correction control visibility."""
        from tests.test_utils import run_gui_with_screenshot

        def check_controls(gui, iw):
            # Check phase correction attributes
            assert hasattr(gui, "_fix_phase")
            assert hasattr(gui, "_use_fft")
            assert hasattr(gui, "_max_offset")
            assert hasattr(gui, "_phase_upsample")

        run_gui_with_screenshot(
            test_data,
            "gui_phase_correction.png",
            gui_action=check_controls,
            frames=10,
        )


@pytest.mark.parametrize("data_shape", [(50, 256, 256), (50, 3, 256, 256)])
def test_gui_initialization(data_shape):
    """Test GUI initialization with different data shapes."""
    from tests.test_utils import (
        create_test_data_multi_zplane,
        create_test_data_single_zplane,
    )

    if len(data_shape) == 3:
        data = create_test_data_single_zplane(shape=data_shape)
    else:
        data = create_test_data_multi_zplane(shape=data_shape)

    import fastplotlib as fpl

    from mbo_utilities.graphics.imgui import PreviewDataWidget

    # Create ImageWidget
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (800, 600)},
        graphic_kwargs={"vmin": 0, "vmax": 4000},
        window_funcs={"t": (np.mean, 0)},
    )

    # Add PreviewDataWidget
    gui = PreviewDataWidget(
        iw=iw,
        fpath=None,
        size=300,
        threading_enabled=False,
    )

    # Verify initialization
    assert gui.image_widget == iw
    if len(data_shape) == 3:
        assert gui.nz == 1
    else:
        assert gui.nz == data_shape[1]

    # Close widget (handle offscreen mode)
    try:
        iw.close()
    except AttributeError:
        if hasattr(iw.figure, "canvas"):
            iw.figure.canvas.close()
