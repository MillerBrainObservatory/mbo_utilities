"""Tests for to_video export functionality."""

import numpy as np
import pytest
from pathlib import Path

import mbo_utilities as mbo
from mbo_utilities import imread, to_video


class TestToVideoSynthetic:
    """Test to_video with synthetic data."""

    def test_basic_export(self, tmp_path, synthetic_3d_data):
        """Basic 3D array export."""
        out = tmp_path / "test.mp4"
        result = to_video(synthetic_3d_data, out, fps=30, max_frames=10)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_speed_factor(self, tmp_path, synthetic_3d_data):
        """Speed factor increases playback rate."""
        out = tmp_path / "fast.mp4"
        result = to_video(synthetic_3d_data, out, fps=30, speed_factor=10, max_frames=10)
        assert result.exists()

    def test_quality_options(self, tmp_path, synthetic_3d_data):
        """Quality enhancement options."""
        out = tmp_path / "quality.mp4"
        result = to_video(
            synthetic_3d_data,
            out,
            fps=30,
            max_frames=10,
            temporal_smooth=3,
            spatial_smooth=0.5,
            gamma=0.8,
            vmin_percentile=2,
            vmax_percentile=98,
        )
        assert result.exists()

    def test_4d_array_plane_selection(self, tmp_path, synthetic_3d_data):
        """4D array with plane selection."""
        arr_4d = np.stack([synthetic_3d_data] * 3, axis=1)
        out = tmp_path / "plane1.mp4"
        result = to_video(arr_4d, out, fps=30, plane=1, max_frames=10)
        assert result.exists()


@pytest.mark.skipif(
    not Path(r"D:\demo\raw").exists(),
    reason="Demo data not available"
)
class TestToVideoDemo:
    """Quick validation with demo data."""

    def test_demo_tiff_export(self, tmp_path):
        """Export demo TIFF to video."""
        demo_path = Path(r"D:\demo\raw")
        tiffs = list(demo_path.glob("*.tif"))
        assert tiffs, f"No TIFFs found in {demo_path}"

        arr = imread(tiffs[0])
        out = tmp_path / "demo_preview.mp4"

        result = to_video(
            arr,
            out,
            fps=30,
            speed_factor=10,
            max_frames=100,
            quality=7,
        )
        assert result.exists()
        assert result.stat().st_size > 1000


@pytest.mark.skipif(
    not Path(r"D:\example_extraction\zarr").exists(),
    reason="Zarr data not available"
)
class TestToVideoZarr:
    """Full quality test with zarr data."""

    def test_zarr_single_plane(self, tmp_path):
        """Export single zarr plane."""
        zarr_path = Path(r"D:\example_extraction\zarr\plane01_stitched.zarr")
        arr = imread(zarr_path)
        out = tmp_path / "zarr_single.mp4"

        result = to_video(
            arr,
            out,
            fps=30,
            speed_factor=5,
            temporal_smooth=3,
            gamma=0.8,
            quality=10,
            max_frames=500,
        )
        assert result.exists()
        assert result.stat().st_size > 5000

    def test_zarr_folder_multiplane(self, tmp_path):
        """Export from zarr folder (multiple planes)."""
        zarr_folder = Path(r"D:\example_extraction\zarr")
        arr = imread(zarr_folder)
        out = tmp_path / "zarr_folder.mp4"

        result = to_video(
            arr,
            out,
            fps=30,
            speed_factor=5,
            plane=0,
            temporal_smooth=5,
            spatial_smooth=0.5,
            gamma=0.8,
            vmin_percentile=1,
            vmax_percentile=99.5,
            quality=10,
            max_frames=1000,
        )
        assert result.exists()
        assert result.stat().st_size > 10000
