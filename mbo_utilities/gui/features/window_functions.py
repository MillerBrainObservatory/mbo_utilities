"""
Window and spatial function features.

Controls for projection type, window size, gaussian blur, and mean subtraction.
These are fundamental features that apply to most time-series data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from . import BaseFeature

if TYPE_CHECKING:
    from ..viewers import BaseViewer

__all__ = ["WindowFunctionFeature", "SpatialFunctionFeature"]


def _set_tooltip(text: str, show_mark: bool = True) -> None:
    """Set tooltip on hover. Helper to avoid circular import."""
    if show_mark:
        imgui.same_line()
        imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class WindowFunctionFeature(BaseFeature):
    """
    Projection type and temporal window size controls.

    Controls:
    - Projection type (mean/max/std) for temporal aggregation
    - Window size for sliding window operations
    """

    name = "Window Functions"
    priority = 10  # Show first

    @classmethod
    def is_supported(cls, viewer: "BaseViewer") -> bool:
        """Always supported for any viewer with an image widget."""
        return hasattr(viewer, "image_widget") and viewer.image_widget is not None

    def draw(self) -> None:
        """Draw window functions controls."""
        viewer = self.viewer

        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
        imgui.spacing()

        # Projection type combo (temporal operations only)
        options = ["mean", "max", "std"]

        current_proj = getattr(viewer, "proj", "mean")
        current_display_idx = options.index(current_proj) if current_proj in options else 0

        imgui.set_next_item_width(hello_imgui.em_size(6))
        proj_changed, selected_display_idx = imgui.combo(
            "Projection", current_display_idx, options
        )
        _set_tooltip(
            "Choose projection method over the sliding window:\n\n"
            ' "mean" (average)\n'
            ' "max" (peak)\n'
            ' "std" (variance)'
        )

        if proj_changed:
            viewer.proj = options[selected_display_idx]

        # Window size
        window_size = getattr(viewer, "window_size", 10)
        imgui.set_next_item_width(hello_imgui.em_size(6))
        winsize_changed, new_winsize = imgui.input_int(
            "Window Size", window_size, step=1, step_fast=2
        )
        _set_tooltip(
            "Size of the temporal window (in frames) used for projection."
            " E.g. a value of 3 averages over 3 consecutive frames."
        )
        if winsize_changed and new_winsize > 0:
            viewer.window_size = new_winsize


class SpatialFunctionFeature(BaseFeature):
    """
    Gaussian blur and mean subtraction controls.

    Controls:
    - Gaussian sigma for spatial smoothing
    - Mean subtraction toggle (requires z-stats)
    """

    name = "Spatial Functions"
    priority = 11  # After window functions

    @classmethod
    def is_supported(cls, viewer: "BaseViewer") -> bool:
        """Always supported for any viewer with an image widget."""
        return hasattr(viewer, "image_widget") and viewer.image_widget is not None

    def draw(self) -> None:
        """Draw spatial functions controls."""
        viewer = self.viewer

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Spatial Functions")
        imgui.spacing()

        # Gaussian sigma
        gaussian_sigma = getattr(viewer, "gaussian_sigma", 0.0)
        imgui.set_next_item_width(hello_imgui.em_size(6))
        gaussian_changed, new_sigma = imgui.input_float(
            "Gaussian Sigma", gaussian_sigma, step=0.1, step_fast=1.0, format="%.1f"
        )
        _set_tooltip(
            "Apply a Gaussian blur to the preview image. "
            "Sigma is in pixels; larger values yield stronger smoothing."
        )
        if gaussian_changed:
            viewer.gaussian_sigma = max(0.0, new_sigma)

        # Mean subtraction checkbox
        zstats_done = getattr(viewer, "_zstats_done", [])
        zstats_ready = all(zstats_done) if zstats_done else False

        if not zstats_ready:
            imgui.begin_disabled()

        mean_subtraction = getattr(viewer, "mean_subtraction", False)
        mean_sub_changed, mean_sub_value = imgui.checkbox(
            "Mean Subtraction", mean_subtraction
        )

        if not zstats_ready:
            _set_tooltip("Mean subtraction requires z-stats to be computed first (in progress...)")
            imgui.end_disabled()
        else:
            _set_tooltip(
                "Subtract the mean image from each frame. "
                "Useful for visualizing activity changes."
            )

        if mean_sub_changed and zstats_ready:
            viewer.mean_subtraction = mean_sub_value
