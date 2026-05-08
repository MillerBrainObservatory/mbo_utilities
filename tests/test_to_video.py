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

    def test_temporal_mode_modes(self, tmp_path, synthetic_3d_data):
        """temporal_mode supports mean/max/std and rejects bad values."""
        for mode in ("mean", "max", "std"):
            out = tmp_path / f"tm_{mode}.mp4"
            assert to_video(
                synthetic_3d_data, out, fps=30, max_frames=10,
                temporal_smooth=3, temporal_mode=mode,
            ).exists()
        with pytest.raises(ValueError):
            to_video(
                synthetic_3d_data, tmp_path / "bad.mp4", fps=30, max_frames=5,
                temporal_smooth=3, temporal_mode="median",
            )

    def test_mean_subtract(self, tmp_path, synthetic_3d_data):
        """mean_subtract subtracts a 2D image; bad shape raises."""
        arr = np.asarray(synthetic_3d_data)
        mean_img = arr.mean(axis=0)
        out = tmp_path / "ms.mp4"
        assert to_video(
            arr, out, fps=30, max_frames=10, mean_subtract=mean_img,
        ).exists()
        with pytest.raises(ValueError):
            to_video(
                arr, tmp_path / "ms_bad.mp4", fps=30, max_frames=5,
                mean_subtract=np.zeros((4, 4)),
            )

    def test_scalebar(self, tmp_path, synthetic_3d_data):
        """scalebar with explicit pixel_size_um produces a non-empty mp4."""
        out = tmp_path / "scalebar.mp4"
        result = to_video(
            synthetic_3d_data, out, fps=30, max_frames=10,
            scalebar=True, pixel_size_um=0.65,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    @staticmethod
    def _expected_bar_rect(h: int, w: int):
        """Mirror _draw_scalebar's layout to know where the bar lives in pixel space."""
        import cv2
        bar_px = max(2, int(round(w * 0.10)))
        font_scale = max(0.3, min(0.6, h / 900.0))
        (_text_w, text_h), _ = cv2.getTextSize(
            "0.0 um", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        bar_h = max(2, int(round(h * 0.010)))
        gap = max(2, int(round(h * 0.012)))
        pad_x = max(4, int(round(w * 0.025)))
        pad_y = max(4, int(round(h * 0.020)))
        outline_pad = 2
        label_baseline_y = h - pad_y - outline_pad
        label_top_y = label_baseline_y - text_h
        bar_bottom_y = label_top_y - gap
        bar_top_y = bar_bottom_y - bar_h
        return bar_top_y, bar_bottom_y, pad_x, pad_x + bar_px, label_baseline_y

    def test_scalebar_writes_white_pixels(self, tmp_path):
        """Scalebar must leave white pixels in the exact bar rectangle."""
        import imageio.v3 as iio
        rng = np.random.default_rng(0)
        h_in, w_in = 200, 400
        arr = (np.full((10, h_in, w_in), 100.0, dtype=np.float32)
               + rng.standard_normal((10, h_in, w_in)).astype(np.float32) * 5)
        out = tmp_path / "scalebar_pixels.mp4"
        to_video(
            arr, out, fps=30, max_frames=10, quality="visually lossless",
            scalebar=True, pixel_size_um=0.65, vmin=80, vmax=120,
        )
        first = iio.imread(out)[0]
        h, w = first.shape[:2]
        y0, y1, x0, x1, baseline = self._expected_bar_rect(h, w)
        bar = first[y0:y1, x0:x1]
        assert bar.mean() > 240, (
            f"bar interior at y=[{y0}:{y1}], x=[{x0}:{x1}] should be near-white, "
            f"got mean={bar.mean():.1f}"
        )
        # label baseline must sit inside the frame (no bottom clipping)
        assert baseline < h, f"label baseline {baseline} >= frame height {h}"
        # bar must sit inside the frame (no left clipping)
        assert x0 >= 0 and y0 >= 0, f"bar starts at ({x0}, {y0}), out of bounds"

    def test_scalebar_skips_without_pixel_size(self, tmp_path, synthetic_3d_data):
        """scalebar without pixel size silently skips the overlay and still writes."""
        out = tmp_path / "scalebar_skip.mp4"
        result = to_video(
            synthetic_3d_data, out, fps=30, max_frames=5, scalebar=True,
        )
        assert result.exists()

    def test_scalebar_reads_metadata_pixel_resolution(self, tmp_path):
        """Regression: ScanImage-style arrays expose pixel size via
        metadata['pixel_resolution'] = (dx, dy), not an arr.dx attribute."""
        import imageio.v3 as iio

        class _ArrayWithMetadata:
            """Minimal lazy-array stand-in: shape + dtype + metadata + indexing."""
            def __init__(self, data, metadata):
                self._data = data
                self.metadata = metadata
                self.shape = data.shape
                self.dtype = data.dtype
                self.ndim = data.ndim
            def __getitem__(self, key):
                return self._data[key]
            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._data, dtype=dtype) if dtype else np.asarray(self._data)
            def __len__(self):
                return len(self._data)

        rng = np.random.default_rng(0)
        h_in, w_in = 200, 400
        raw = (np.full((10, h_in, w_in), 100.0, dtype=np.float32)
               + rng.standard_normal((10, h_in, w_in)).astype(np.float32) * 5)
        # ScanImage style: tuple under 'pixel_resolution', no .dx attribute
        wrapped = _ArrayWithMetadata(raw, {"pixel_resolution": (0.65, 0.65)})

        out = tmp_path / "scalebar_md.mp4"
        to_video(
            wrapped, out, fps=30, max_frames=10, quality="visually lossless",
            scalebar=True, vmin=80, vmax=120,
        )
        first = iio.imread(out)[0]
        h, w = first.shape[:2]
        y0, y1, x0, x1, _ = self._expected_bar_rect(h, w)
        bar = first[y0:y1, x0:x1]
        assert bar.mean() > 240, (
            f"scalebar pixels missing — pixel_size lookup did not find "
            f"metadata['pixel_resolution']; bar.mean()={bar.mean():.1f}"
        )

    def test_time_overlay(self, tmp_path, synthetic_3d_data):
        """time_overlay produces a non-empty mp4 (cv2 putText does not crash)."""
        out = tmp_path / "overlay.mp4"
        result = to_video(
            synthetic_3d_data, out, fps=30, max_frames=15, time_overlay=True,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    @pytest.mark.parametrize("quality", ["preview", "high", "visually lossless", "lossless"])
    @pytest.mark.parametrize("ext", ["mp4", "mov"])
    def test_quality_x_ext(self, tmp_path, synthetic_3d_data, quality, ext):
        """Each quality preset must produce a non-empty file in mp4 and mov."""
        out = tmp_path / f"q_{quality.replace(' ', '_')}.{ext}"
        result = to_video(synthetic_3d_data, out, fps=30, max_frames=15, quality=quality)
        assert result.exists()
        assert result.stat().st_size > 0, f"empty output for {quality} {ext}"

    @pytest.mark.parametrize("quality", ["visually lossless", "lossless"])
    @pytest.mark.parametrize("ext", ["mp4", "mov"])
    def test_realistic_size_lossless(self, tmp_path, quality, ext):
        """Reproduce the user's setup: 448x550, 60 frames, libx264 lossless."""
        rng = np.random.default_rng(0)
        arr = (rng.random((60, 448, 550), dtype=np.float32) * 1000 + 200).astype(np.float32)
        arr += rng.standard_normal(arr.shape, dtype=np.float32) * 80
        out = tmp_path / f"realistic_{quality.replace(' ', '_')}.{ext}"
        result = to_video(arr, out, fps=30, max_frames=60, quality=quality)
        size = result.stat().st_size
        print(f"\n  {quality!r:20s} .{ext}: {size} bytes")
        assert result.exists()
        assert size > 1000, f"suspiciously tiny output for {quality} {ext}: {size} bytes"

    def test_rawvideo_in_mp4_rejected(self, tmp_path, synthetic_3d_data):
        """rawvideo cannot be muxed into .mp4 — should fail fast with a clear error."""
        with pytest.raises(ValueError, match="rawvideo"):
            to_video(
                synthetic_3d_data, tmp_path / "raw.mp4", fps=30, max_frames=5,
                codec="rawvideo",
            )


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
