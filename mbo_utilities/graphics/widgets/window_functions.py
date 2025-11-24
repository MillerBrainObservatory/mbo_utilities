"""
window functions widget.

controls for projection type, window size, and gaussian blur.
always shows for any data with temporal dimension.
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.graphics.widgets._base import Widget
from mbo_utilities.graphics._widgets import set_tooltip

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget


class WindowFunctionsWidget(Widget):
    """ui widget for window functions (projection, window size, gaussian)."""

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
        imgui.separator()
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
        imgui.spacing()

        # projection type combo
        options = ["mean", "max", "std"]
        disabled_label = (
            "mean-sub (pending)" if not all(parent._zstats_done) else "mean-sub"
        )
        options.append(disabled_label)

        current_display_idx = options.index(
            parent.proj if parent._proj != "mean-sub" else disabled_label
        )

        imgui.set_next_item_width(hello_imgui.em_size(6))
        proj_changed, selected_display_idx = imgui.combo(
            "Projection", current_display_idx, options
        )
        set_tooltip(
            "Choose projection method over the sliding window:\n\n"
            ' "mean" (average)\n'
            ' "max" (peak)\n'
            ' "std" (variance)\n'
            ' "mean-sub" (mean-subtracted).'
        )

        if proj_changed:
            selected_label = options[selected_display_idx]
            if selected_label == "mean-sub (pending)":
                pass  # don't change if pending
            else:
                parent.proj = selected_label

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
