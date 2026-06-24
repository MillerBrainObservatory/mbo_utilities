"""
MP4 array reader/writer.

Provides MP4Array for reading .mp4 files back as lazy grayscale (T, Y, X)
arrays and for encoding source arrays to .mp4. The single-file encoder
(`to_video`) and the per-(z, channel) writer (`MP4Array.write_video`) live
here; `mbo_utilities._writers.to_video` is a thin shim re-exporting this one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from mbo_utilities import log
from mbo_utilities.arrays._base import _imwrite_base, ReductionMixin, Shape5DMixin
from mbo_utilities.pipeline_registry import PipelineInfo, register_pipeline

logger = log.get("arrays.mp4")

_MP4_INFO = PipelineInfo(
    name="mp4",
    description="MP4 video files",
    input_patterns=["**/*.mp4"],
    output_patterns=["**/*.mp4"],
    input_extensions=["mp4"],
    output_extensions=["mp4"],
    marker_files=[],
    category="reader",
)
register_pipeline(_MP4_INFO)


VIDEO_QUALITY_PRESETS = ("preview", "high", "visually lossless", "lossless")


def _resolve_pixel_size_um(data) -> float | None:
    """Canonical dx (µm/px) for a scalebar, or None.

    Uses ``data.dx`` for a LazyArray, else ``get_param`` on a ``.metadata``
    dict — both resolve every alias incl. the ScanImage pixel_resolution
    tuple. Plain ndarrays have neither and yield None.
    """
    dx = getattr(data, "dx", None)
    if dx is None:
        md = getattr(data, "metadata", None)
        if isinstance(md, dict):
            from mbo_utilities.metadata import get_param

            dx = get_param(md, "dx")
    try:
        dx = float(dx)
    except (TypeError, ValueError):
        return None
    return dx if dx > 0 else None


def _format_overlay_time(t_seconds: float) -> str:
    if t_seconds < 60:
        return f"{t_seconds:5.1f}s"
    m, s = divmod(t_seconds, 60)
    return f"{int(m)}m {s:4.1f}s"


def _draw_scalebar(frame_rgb: np.ndarray, dx_um: float) -> None:
    """Draw a scalebar in the bottom-left with the label centered below the bar.

    Bar width is exactly 10% of the frame so multiplying the label by 10
    gives the full-frame width. The whole block (bar + gap + label + outline)
    is laid out from the bottom up and clamped to stay inside the frame, so
    the label never gets cut off on the bottom or the left.
    """
    import cv2
    h, w = frame_rgb.shape[:2]
    bar_px = max(2, int(round(w * 0.10)))
    bar_um = bar_px * dx_um

    text = f"{bar_um:.3g} um"
    # smaller font than before — h/900 instead of h/600, capped at 0.6
    font_scale = max(0.3, min(0.6, h / 900.0))
    font_thickness = 1
    (text_w, text_h), text_baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness,
    )

    bar_h = max(2, int(round(h * 0.010)))
    gap = max(2, int(round(h * 0.012)))  # space between bar and label
    pad_x = max(4, int(round(w * 0.025)))
    pad_y = max(4, int(round(h * 0.020)))
    outline_pad = font_thickness + 1  # the +2 outline added per putText call

    # lay out from the bottom up so the label baseline sits inside the frame
    label_baseline_y = h - pad_y - outline_pad
    label_top_y = label_baseline_y - text_h
    bar_bottom_y = label_top_y - gap
    bar_top_y = bar_bottom_y - bar_h

    bar_x0 = pad_x
    bar_x1 = bar_x0 + bar_px

    # text centered on bar, but never off the left edge
    text_x = bar_x0 + (bar_px - text_w) // 2
    text_x = max(pad_x, text_x)
    # also keep right edge inside the frame
    text_x = min(text_x, w - pad_x - text_w)

    # black backing under the bar for contrast on bright cmaps
    cv2.rectangle(
        frame_rgb,
        (bar_x0 - 1, bar_top_y - 1), (bar_x1 + 1, bar_bottom_y + 1),
        (0, 0, 0), -1,
    )
    cv2.rectangle(
        frame_rgb,
        (bar_x0, bar_top_y), (bar_x1, bar_bottom_y),
        (255, 255, 255), -1,
    )

    cv2.putText(
        frame_rgb, text, (text_x, label_baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (0, 0, 0), font_thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb, text, (text_x, label_baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), font_thickness, cv2.LINE_AA,
    )


def _draw_time_overlay(frame_rgb: np.ndarray, t_seconds: float) -> None:
    """Draw an MM:SS / X.Xs clock on `frame_rgb` in-place (top-left)."""
    import cv2
    h = frame_rgb.shape[0]
    text = _format_overlay_time(t_seconds)
    scale = max(0.4, min(1.5, h / 600.0))
    thickness = max(1, int(round(scale * 1.6)))
    pos = (max(6, int(scale * 12)), max(18, int(scale * 28)))
    cv2.putText(
        frame_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )

# yuv420p is left to imageio's default — adding -pix_fmt here collides with
# imageio's own injected -pix_fmt and triggers a "Multiple -pix_fmt" ffmpeg
# warning. yuv420p is mathematically lossless for grayscale input since R=G=B
# means chroma is identically zero (subsampling zero is still zero) and is the
# only browser-compatible chroma layout.
# veryslow + yuv444p was tried and crashed the bundled imageio-ffmpeg binary
# mid-stream (broken pipe). slow gives ~95% of the compression efficiency at a
# fraction of the memory/time and is rock-solid.
# "lossless" intentionally uses crf=8, not crf=0: -crf 0 enables libx264's
# lossless mode which produces a non-standard High Profile stream that Windows
# Photos / Chrome / native QuickTime refuse to display (file is created but
# won't open). crf=8 stays inside standard High Profile so every player works,
# and for 8-bit output it is mathematically lossless within the LSB.
_X264_PRESET_TABLE = {
    "preview":           {"crf": 23, "preset": "medium", "tune": None},
    "high":              {"crf": 18, "preset": "slow",   "tune": None},
    "visually lossless": {"crf": 14, "preset": "slow",   "tune": None},
    "lossless":          {"crf":  8, "preset": "slow",   "tune": "psnr"},
}

_MPEG4_QSCALE_TABLE = {
    "preview": 8,
    "high": 4,
    "visually lossless": 2,
    "lossless": 1,
}


def _resolve_quality_preset(quality: str | int) -> str:
    """Normalize quality (string preset name or legacy 1-10 int) to a preset key."""
    if isinstance(quality, str):
        key = quality.strip().lower().replace("_", " ")
        if key not in _X264_PRESET_TABLE:
            raise ValueError(
                f"Unknown quality preset {quality!r}. "
                f"Expected one of {VIDEO_QUALITY_PRESETS}."
            )
        return key
    q = int(quality)
    if q <= 3:
        return "preview"
    if q <= 7:
        return "high"
    if q <= 9:
        return "visually lossless"
    return "lossless"


def _build_video_output_params(codec: str, quality: str | int) -> list[str]:
    """Build ffmpeg output_params for (codec, quality preset)."""
    preset = _resolve_quality_preset(quality)
    if codec in ("libx264", "libx265"):
        cfg = _X264_PRESET_TABLE[preset]
        params = ["-crf", str(cfg["crf"]), "-preset", cfg["preset"]]
        if cfg.get("tune"):
            params.extend(["-tune", cfg["tune"]])
        return params
    if codec == "mpeg4":
        return ["-qscale:v", str(_MPEG4_QSCALE_TABLE[preset])]
    if codec == "rawvideo":
        return []
    return ["-q:v", str(_MPEG4_QSCALE_TABLE[preset])]


def to_video(
    data,
    output_path,
    fps: int = 30,
    speed_factor: float = 1.0,
    plane: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.5,
    temporal_smooth: int = 0,
    spatial_smooth: float = 0,
    gamma: float = 1.0,
    cmap: str | None = None,
    quality: str | int = "visually lossless",
    codec: str = "libx264",
    max_frames: int | None = None,
    mean_subtract: np.ndarray | None = None,
    temporal_mode: str = "mean",
    time_overlay: bool = False,
    scalebar: bool = False,
    pixel_size_um: float | None = None,
    channel: int = 0,
    frame_indices: list | None = None,
):
    """
    Export array data to an mp4 video file.

    Works with 3D (T, Y, X), 4D (T, Z, Y, X), or 5D (T, C, Z, Y, X) arrays,
    including lazy arrays. Frames are read one at a time via the array's
    indexing, so lazy zarr/tiff/bin sources stream instead of materializing
    the whole stack.

    Parameters
    ----------
    data : array-like
        3D/4D/5D array. Supports lazy arrays.
    output_path : str or Path
        Output video path (.mp4).
    fps : int, default 30
        Base frame rate of the recording.
    speed_factor : float, default 1.0
        Playback speed multiplier. speed_factor=10 plays 10x faster (all frames
        included, just faster playback). Use this to show cell stability quickly.
    plane : int, optional
        For 4D/5D arrays, which z-plane to export (0-indexed). If None, plane 0.
    channel : int, default 0
        For 5D arrays, which channel to export (0-indexed).
    frame_indices : list[int], optional
        Explicit 0-based frame indices to export, in order. If None, all frames.
    vmin : float, optional
        Min value for intensity scaling. If None, uses vmin_percentile.
    vmax : float, optional
        Max value for intensity scaling. If None, uses vmax_percentile.
    vmin_percentile : float, default 1.0
        Percentile for auto vmin calculation. Lower = darker blacks.
    vmax_percentile : float, default 99.5
        Percentile for auto vmax calculation. Lower = brighter highlights.
    temporal_smooth : int, default 0
        Rolling-window size in frames. Reduces flicker/noise.
        0 = disabled, 3-7 = subtle, 10+ = heavy. The aggregation applied to
        the window is controlled by `temporal_mode`.
    temporal_mode : {"mean", "max", "std"}, default "mean"
        How the rolling window is aggregated. "mean" smooths flicker, "max"
        emphasizes transient activity, "std" highlights variance/noise.
    mean_subtract : ndarray, optional
        2D image (Y, X) subtracted from every frame before contrast scaling.
        Useful for removing static structure to highlight dynamics.
    time_overlay : bool, default False
        Draw a clock in the top-left of every frame showing elapsed *recording*
        time (frame_index / fps). Independent of speed_factor — the overlay
        always reflects the source recording duration, so a sped-up clip ticks
        through real seconds faster on screen.
    scalebar : bool, default False
        Draw a scalebar at exactly 10% of frame width in the bottom-left, with
        a "NN um" label. Multiply the label by 10 to read the full-frame width.
        Requires `pixel_size_um` (taken from `data.dx` automatically when the
        input array exposes the VoxelSize feature).
    pixel_size_um : float, optional
        Pixel size in micrometers (X). Read from `data.dx` if not provided.
        If neither source has a positive value, scalebar is silently skipped
        with a warning.
    spatial_smooth : float, default 0
        Gaussian blur sigma (pixels). Reduces pixel noise.
        0 = disabled, 0.5-1.0 = subtle, 2+ = heavy blur.
    gamma : float, default 1.0
        Gamma correction. <1 = brighter midtones, >1 = darker midtones.
        0.7-0.8 often looks good for calcium imaging.
    cmap : str, optional
        Matplotlib colormap name (e.g., "viridis", "gray", "hot").
        If None, outputs grayscale.
    quality : str or int, default "visually lossless"
        Quality preset. One of:
          - "preview"           crf 23, preset medium, yuv420p (small/fast)
          - "high"              crf 18, preset slow,    yuv420p
          - "visually lossless" crf 14, preset veryslow, yuv444p
          - "lossless"          crf 0,  preset veryslow, yuv444p (math-lossless)
        Ints 1-10 are mapped to presets for backwards compatibility
        (1-3 -> preview, 4-7 -> high, 8-9 -> visually lossless, 10 -> lossless).
    codec : str, default "libx264"
        Video codec. "libx264" for mp4 (best compatibility). For mpeg4 the
        preset is mapped to -qscale:v; lossless mode is not supported there.
    max_frames : int, optional
        Limit number of frames to export. If None, exports all frames.

    Returns
    -------
    Path
        Path to the created video file.

    Examples
    --------
    >>> from mbo_utilities import imread, to_video
    >>> arr = imread("data.tif")

    >>> # Quick preview at 10x speed (good for checking stability)
    >>> to_video(arr, "preview.mp4", speed_factor=10)

    >>> # High-quality export for website
    >>> to_video(arr, "movie.mp4", fps=30, speed_factor=5,
    ...          temporal_smooth=3, gamma=0.8, quality=10)

    >>> # Export specific z-plane from 4D data
    >>> to_video(arr, "plane3.mp4", plane=3, speed_factor=10)

    >>> # With colormap and custom intensity range
    >>> to_video(arr, "movie.mp4", cmap="viridis", vmin=100, vmax=2000)
    """
    import imageio
    from scipy.ndimage import gaussian_filter

    output_path = Path(output_path)

    ext = output_path.suffix.lower()
    if codec == "rawvideo" and ext != ".mkv":
        raise ValueError(
            f"codec='rawvideo' is not supported in {ext} containers — "
            f"use .mkv (or pick codec='libx264' for .mp4)."
        )

    # resolve scalebar pixel size before frame reads. arr.dx (LazyArray) or
    # get_param on a metadata dict both resolve every alias incl. the
    # ScanImage pixel_resolution tuple.
    if scalebar and pixel_size_um is None:
        pixel_size_um = _resolve_pixel_size_um(data)
    if scalebar and (pixel_size_um is None or pixel_size_um <= 0):
        logger.warning(
            f"scalebar requested but pixel_size_um is unavailable "
            f"(resolved value: {pixel_size_um!r}); skipping."
        )
        scalebar = False
    elif scalebar:
        logger.info(f"Scalebar enabled with pixel_size_um={pixel_size_um}")

    # read frames lazily from the source array (no upfront full-load). branch
    # the per-frame accessor on natural rank so zarr/tiff/bin arrays stream one
    # frame per read instead of materializing the whole stack.
    if not hasattr(data, "shape"):
        data = np.asarray(data)
    shape = tuple(data.shape)
    ndim = len(shape)
    if ndim not in (3, 4, 5):
        raise ValueError(f"to_video expects 3D/4D/5D data, got {ndim}D {shape}")

    plane_idx = plane if plane is not None else 0
    n_planes = shape[2] if ndim == 5 else (shape[1] if ndim == 4 else 1)
    if plane_idx >= n_planes:
        raise ValueError(f"plane={plane_idx} but array only has {n_planes} planes")
    height, width = shape[-2], shape[-1]

    def _read_frame(t):
        if ndim == 3:
            f = data[t]
        elif ndim == 4:
            f = data[t, plane_idx]
        else:
            f = data[t, channel, plane_idx]
        return np.asarray(f, dtype=np.float32)

    # resolve frame iteration order — one lazy read per frame
    if frame_indices is not None:
        order = [int(x) for x in frame_indices]
    else:
        order = list(range(shape[0]))
    if max_frames is not None:
        order = order[:max_frames]
    n_frames = len(order)
    logger.info(
        f"Exporting plane {plane_idx}: {n_frames} frames, {height}x{width}"
    )

    # Calculate output fps based on speed factor
    output_fps = int(fps * speed_factor)
    duration = n_frames / output_fps

    logger.info(
        f"Writing {n_frames} frames at {output_fps} fps "
        f"(speed_factor={speed_factor}x, duration={duration:.1f}s)"
    )

    # Determine intensity range from sample frames. When mean_subtract is on,
    # percentile must be computed on subtracted samples so contrast matches
    # what gets rendered.
    if vmin is None or vmax is None:
        n_samples = min(50, n_frames)
        sample_positions = np.linspace(0, n_frames - 1, n_samples, dtype=int)
        samples = []
        for pos in sample_positions:
            frame = _read_frame(order[pos])
            if mean_subtract is not None:
                frame = frame - np.asarray(mean_subtract, dtype=np.float32)
            samples.append(frame)
        sample_stack = np.stack(samples)

        if vmin is None:
            vmin = float(np.percentile(sample_stack, vmin_percentile))
        if vmax is None:
            vmax = float(np.percentile(sample_stack, vmax_percentile))

    logger.info(f"Intensity range: [{vmin:.1f}, {vmax:.1f}]")

    # Setup colormap if requested
    if cmap is not None:
        try:
            import matplotlib.pyplot as plt

            colormap = plt.get_cmap(cmap)
        except ImportError:
            logger.warning("matplotlib not available, using grayscale")
            colormap = None
    else:
        colormap = None

    output_params = _build_video_output_params(codec, quality)
    logger.info(f"ffmpeg output params: {' '.join(output_params)}")

    _temporal_aggregators = {
        "mean": lambda buf: np.mean(buf, axis=0),
        "max":  lambda buf: np.max(buf, axis=0),
        "std":  lambda buf: np.std(buf, axis=0),
    }
    if temporal_mode not in _temporal_aggregators:
        raise ValueError(
            f"temporal_mode={temporal_mode!r} not in {list(_temporal_aggregators)}"
        )
    aggregate_window = _temporal_aggregators[temporal_mode]
    frame_buffer = [] if temporal_smooth > 0 else None

    mean_sub_2d = None
    if mean_subtract is not None:
        mean_sub_2d = np.asarray(mean_subtract, dtype=np.float32)
        if mean_sub_2d.shape != (height, width):
            raise ValueError(
                f"mean_subtract shape {mean_sub_2d.shape} != frame ({height}, {width})"
            )

    writer = imageio.get_writer(
        str(output_path),
        fps=output_fps,
        codec=codec,
        macro_block_size=2,
        output_params=output_params,
    )

    try:
        for i in tqdm(range(n_frames), desc="Writing video", unit="frames"):
            frame = _read_frame(order[i])

            if mean_sub_2d is not None:
                frame = frame - mean_sub_2d

            if temporal_smooth > 0:
                frame_buffer.append(frame)
                if len(frame_buffer) > temporal_smooth:
                    frame_buffer.pop(0)
                frame = aggregate_window(frame_buffer)

            if spatial_smooth > 0:
                frame = gaussian_filter(frame, sigma=spatial_smooth)

            # Normalize to 0-1
            frame = np.clip((frame - vmin) / (vmax - vmin), 0, 1)

            # Gamma correction
            if gamma != 1.0:
                frame = np.power(frame, gamma)

            # Convert to RGB
            if colormap is not None:
                # Apply colormap (returns RGBA)
                frame_rgb = (colormap(frame)[:, :, :3] * 255).astype(np.uint8)
            else:
                # Grayscale -> RGB
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_rgb = np.stack([frame_uint8] * 3, axis=-1)

            if time_overlay or scalebar:
                # cv2 draws in-place; ensure the array is contiguous and writable
                frame_rgb = np.ascontiguousarray(frame_rgb)
                if time_overlay:
                    _draw_time_overlay(frame_rgb, i / fps)
                if scalebar:
                    _draw_scalebar(frame_rgb, pixel_size_um)

            writer.append_data(frame_rgb)
    finally:
        writer.close()

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Video saved to {output_path} ({file_size_mb:.1f} MB)")
    return output_path


class MP4Array(ReductionMixin, Shape5DMixin):
    """
    Lazy reader for .mp4 video files.

    Frames are read on demand via imageio and collapsed to grayscale, so the
    array presents as (T, Y, X) uint8. Encoding lives in the module-level
    `to_video` (single file) and `MP4Array.write_video` (per z/channel).

    Parameters
    ----------
    filenames : Path or str
        Path to the .mp4 file.
    metadata : dict, optional
        Extra metadata to attach (fps is auto-detected from the container).

    Examples
    --------
    >>> arr = MP4Array("movie.mp4")
    >>> arr.shape
    (300, 512, 512)
    >>> frame = arr[0]
    """

    def __init__(self, filenames: Path | str, metadata: dict | None = None):
        import imageio

        self.filenames = [Path(filenames)]
        path = self.filenames[0]
        self._reader = imageio.get_reader(str(path))
        meta = self._reader.get_meta_data()

        fps = meta.get("fps")
        self._fps = float(fps) if fps else None

        try:
            nframes = int(self._reader.count_frames())
        except Exception:
            nframes = int(meta.get("nframes", 0) or 0)
            if not np.isfinite(nframes) or nframes <= 0:
                duration = meta.get("duration") or 0
                nframes = int((self._fps or 0) * duration)

        first = np.asarray(self._reader.get_data(0))
        height, width = first.shape[0], first.shape[1]
        self._dtype = np.dtype(first.dtype)
        self._raw_shape = (nframes, height, width)
        self._target_dtype = None

        self._metadata = dict(metadata) if metadata else {}
        if self._fps:
            self._metadata.setdefault("fs", self._fps)
            self._metadata.setdefault("fps", self._fps)

    @classmethod
    def can_open(cls, path: Path | str) -> bool:
        return Path(path).suffix.lower() == ".mp4"

    @property
    def shape(self) -> tuple[int, int, int]:
        # MP4Array stays 3D (T, Y, X); it is not part of the 5D dispatch set.
        return self._raw_shape

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        s = self._raw_shape  # always (T, Y, X)
        return (s[0], 1, 1, s[1], s[2])

    @property
    def ndim(self) -> int:
        return len(self._raw_shape)

    @property
    def dtype(self):
        return self._target_dtype if self._target_dtype is not None else self._dtype

    def astype(self, dtype, copy=True):
        """Set target dtype for lazy conversion on data access."""
        self._target_dtype = np.dtype(dtype)
        return self

    def __len__(self) -> int:
        return self.shape[0]

    def _read_gray(self, i: int) -> np.ndarray:
        frame = np.asarray(self._reader.get_data(int(i)))
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        return frame.astype(self._dtype)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0] if len(key) > 0 else slice(None)
        rest = key[1:]

        n = self.shape[0]
        if isinstance(t_key, (int, np.integer)):
            t_list = [int(t_key) if t_key >= 0 else n + int(t_key)]
            squeeze_t = True
        elif isinstance(t_key, slice):
            t_list = list(range(*t_key.indices(n)))
            squeeze_t = False
        elif isinstance(t_key, (list, tuple, np.ndarray)):
            t_list = [int(t) if t >= 0 else n + int(t) for t in t_key]
            squeeze_t = False
        else:
            raise TypeError(f"unsupported time index {t_key!r}")

        if t_list:
            data = np.stack([self._read_gray(t) for t in t_list])
        else:
            data = np.empty((0, *self.shape[1:]), dtype=self._dtype)

        if rest:
            data = data[(slice(None),) + rest]
        if squeeze_t:
            data = data[0]
        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        return data

    def __array__(self, dtype=None, copy=None):
        # return first frame for fast histogram/preview (no full load)
        data = self._read_gray(0)
        if self._target_dtype is not None:
            data = data.astype(self._target_dtype)
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self) -> None:
        """Close the underlying video reader."""
        try:
            self._reader.close()
        except Exception:
            pass

    def _imwrite(
        self,
        outpath: Path | str,
        overwrite=False,
        target_chunk_mb=50,
        ext=".tiff",
        progress_callback=None,
        debug=None,
        planes=None,
        **kwargs,
    ):
        """Write MP4Array to disk in various formats."""
        return _imwrite_base(
            self,
            outpath,
            planes=planes,
            ext=ext,
            overwrite=overwrite,
            target_chunk_mb=target_chunk_mb,
            progress_callback=progress_callback,
            debug=debug,
            **kwargs,
        )

    @staticmethod
    def write_video(
        arr,
        outpath: Path,
        metadata: dict | None = None,
        planes: list | None = None,
        frames: list | None = None,
        channels: list | None = None,
        ext: str = "mp4",
        overwrite: bool = False,
        output_suffix: str | None = None,
        progress_callback=None,
        show_progress: bool = True,
        fps: int = 30,
        speed_factor: float = 1.0,
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_percentile: float = 1.0,
        vmax_percentile: float = 99.5,
        temporal_smooth: int = 0,
        spatial_smooth: float = 0.0,
        gamma: float = 1.0,
        cmap: str | None = None,
        quality: str | int = "visually lossless",
        codec: str = "libx264",
        mean_subtract_stack: np.ndarray | None = None,
        temporal_mode: str = "mean",
        time_overlay: bool = False,
        scalebar: bool = False,
    ) -> Path:
        """
        Write one video file per (z-plane, channel) from a source array.

        Filename pattern: zplaneNN_tpN-tpN_chNN_<suffix>.<ext> — Z first because
        each file is a single plane; channel suffix is always present. Frames
        stream lazily out of `arr` (the (c, z) slice is read one frame at a
        time inside `to_video`).

        `mean_subtract_stack` is an optional (C, Z, Y, X) array of per-channel,
        per-plane mean images; the (c, z) slice is forwarded to `to_video`.
        """
        from mbo_utilities.arrays.features import (
            OutputFilename,
            DimensionTag,
            TAG_REGISTRY,
        )

        outpath = Path(outpath)
        outpath.mkdir(parents=True, exist_ok=True)
        ext_clean = ext.lower().lstrip(".")

        s5 = arr._shape5d()
        num_planes = s5[2]
        # C axis = num_views for IsoView (cameras), else num_color_channels
        num_channels = getattr(arr, "num_views", None) or getattr(
            arr, "num_color_channels", s5[1]
        )
        nframes_total = s5[0]

        planes_0idx = [p - 1 for p in planes] if planes else list(range(num_planes))
        channels_0idx = [c - 1 for c in channels] if channels else list(range(num_channels))

        if frames:
            frame_indices_0 = [f - 1 for f in frames]
        else:
            frame_indices_0 = None

        suffix = output_suffix.lstrip("_") if output_suffix else "movie"

        # carry dx through to to_video for the scalebar.
        pixel_size_um = None
        if scalebar:
            pixel_size_um = _resolve_pixel_size_um(arr)
            logger.info(f"Scalebar pixel size: {pixel_size_um!r}")

        t_tag = DimensionTag.from_dim_size(TAG_REGISTRY["T"], nframes_total, frames)

        total_files = len(planes_0idx) * len(channels_0idx)
        file_idx = 0

        for plane_idx in planes_0idx:
            for c_idx in channels_0idx:
                z_tag = DimensionTag.from_dim_size(TAG_REGISTRY["Z"], num_planes, [plane_idx + 1])
                c_tag = DimensionTag.from_dim_size(TAG_REGISTRY["C"], num_channels, [c_idx + 1])

                filename = OutputFilename([z_tag, c_tag, t_tag], suffix=suffix).build(f".{ext_clean}")
                target = outpath / filename

                if target.exists() and not overwrite:
                    logger.warning(f"File {target} exists. Skipping write.")
                    file_idx += 1
                    continue

                logger.info(
                    f"Writing video {file_idx + 1}/{total_files}: {target.name}"
                )
                mean_img = None
                if mean_subtract_stack is not None:
                    mean_img = mean_subtract_stack[c_idx, plane_idx]

                to_video(
                    arr,
                    target,
                    fps=fps,
                    speed_factor=speed_factor,
                    plane=plane_idx,
                    channel=c_idx,
                    frame_indices=frame_indices_0,
                    vmin=vmin,
                    vmax=vmax,
                    vmin_percentile=vmin_percentile,
                    vmax_percentile=vmax_percentile,
                    temporal_smooth=temporal_smooth,
                    spatial_smooth=spatial_smooth,
                    gamma=gamma,
                    cmap=cmap,
                    quality=quality,
                    codec=codec,
                    mean_subtract=mean_img,
                    temporal_mode=temporal_mode,
                    time_overlay=time_overlay,
                    scalebar=scalebar,
                    pixel_size_um=pixel_size_um,
                )

                file_idx += 1
                if progress_callback:
                    progress_callback(file_idx / total_files, target.name)

        if progress_callback:
            progress_callback(1.0, "Complete")

        return outpath
