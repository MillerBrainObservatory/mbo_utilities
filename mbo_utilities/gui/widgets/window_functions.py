"""
window functions widget.

controls for projection type and window size (temporal operations).
always shows for any data with temporal dimension.
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui._widgets import set_tooltip

if TYPE_CHECKING:
    from mbo_utilities.gui.imgui import PreviewDataWidget


class WindowFunctionsWidget(Widget):
    """ui widget for window functions (projection, window size)."""

    name = "Window Functions"
    priority = 10  # show first

    @classmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """always supported for any array with temporal dimension."""
        return True

    def draw(self) -> None:
        """draw window functions controls."""
        parent = self.parent

        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
        imgui.spacing()

        # projection type combo (temporal operations only)
        options = ["mean", "max", "std"]

        current_display_idx = options.index(parent.proj) if parent.proj in options else 0

        imgui.set_next_item_width(hello_imgui.em_size(6))
        proj_changed, selected_display_idx = imgui.combo(
            "Projection", current_display_idx, options
        )
        set_tooltip(
            "Choose projection method over the sliding window:\n\n"
            ' "mean" (average)\n'
            ' "max" (peak)\n'
            ' "std" (variance)'
        )

        if proj_changed:
            parent.proj = options[selected_display_idx]

        # window size
        imgui.set_next_item_width(hello_imgui.em_size(6))
        winsize_changed, new_winsize = imgui.input_int(
            "Window Size", parent.window_size, step=1, step_fast=2
        )
        set_tooltip(
            "Size of the temporal window (in frames) used for projection."
            " E.g. a value of 3 averages over 3 consecutive frames."
        )
        if winsize_changed and new_winsize > 0:
            parent.window_size = new_winsize


class SpatialFunctionsWidget(Widget):
    """ui widget for spatial functions (gaussian blur, mean subtraction)."""

    name = "Spatial Functions"
    priority = 11  # show after window functions

    @classmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """always supported."""
        return True

    def draw(self) -> None:
        """draw spatial functions controls."""
        parent = self.parent

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Spatial Functions")
        imgui.spacing()

        # gaussian sigma
        imgui.set_next_item_width(hello_imgui.em_size(6))
        gaussian_changed, new_sigma = imgui.input_float(
            "Gaussian Sigma", parent.gaussian_sigma, step=0.1, step_fast=1.0, format="%.1f"
        )
        set_tooltip(
            "Apply a Gaussian blur to the preview image. Sigma is in pixels; larger values yield stronger smoothing."
        )
        if gaussian_changed:
            parent.gaussian_sigma = max(0.0, new_sigma)

        # mean subtraction checkbox
        zstats_ready = all(parent._zstats_done)
        if not zstats_ready:
            imgui.begin_disabled()

        mean_sub_changed, mean_sub_value = imgui.checkbox(
            "Mean Subtraction", parent.mean_subtraction
        )
        if not zstats_ready:
            set_tooltip("Mean subtraction requires z-stats to be computed first (in progress...)")
            imgui.end_disabled()
        else:
            set_tooltip(
                "Subtract the mean image from each frame. Useful for visualizing activity changes."
            )

        if mean_sub_changed and zstats_ready:
            parent.mean_subtraction = mean_sub_value
