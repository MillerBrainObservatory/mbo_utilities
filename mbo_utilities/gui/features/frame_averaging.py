"""
Frame averaging feature for piezo stacks.

Shows UI for toggling frame averaging on piezo stack data
that has framesPerSlice > 1 and was not pre-averaged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from . import BaseFeature

if TYPE_CHECKING:
    from ..viewers import BaseViewer

__all__ = ["FrameAveragingFeature"]


def _set_tooltip(text: str, show_mark: bool = True) -> None:
    """Set tooltip on hover. Helper to avoid circular import."""
    if show_mark:
        imgui.same_line()
        imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class FrameAveragingFeature(BaseFeature):
    """
    Frame averaging controls for piezo stacks.

    Shows:
    - Frames per slice info
    - Pre-averaging status
    - Toggle for software frame averaging
    """

    name = "Frame Averaging"
    priority = 55  # After raster scan (50)

    @classmethod
    def is_supported(cls, viewer: "BaseViewer") -> bool:
        """Show only for piezo arrays that can average frames."""
        arrays = viewer._get_data_arrays()
        for arr in arrays:
            # Check if this is a PiezoArray with averaging capability
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                return True
        return False

    def draw(self) -> None:
        """Draw frame averaging controls."""
        viewer = self.viewer
        arrays = viewer._get_data_arrays()

        # Find first piezo array
        piezo_arr = None
        for arr in arrays:
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                piezo_arr = arr
                break

        if piezo_arr is None:
            return

        imgui.spacing()
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Frame Averaging"
        )
        imgui.spacing()

        # Show acquisition info
        fps = piezo_arr.frames_per_slice
        log_avg = piezo_arr.log_average_factor

        imgui.text(f"Frames per slice: {fps}")
        _set_tooltip(
            "Number of frames acquired at each z-slice position "
            "(from si.hStackManager.framesPerSlice)."
        )

        if log_avg > 1:
            # Data was pre-averaged at acquisition
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
                f"Pre-averaged at acquisition (factor: {log_avg})"
            )
            _set_tooltip(
                "Frames were averaged during acquisition before saving. "
                "No additional software averaging needed."
            )
        elif fps > 1:
            # Can average
            can_avg = piezo_arr.can_average
            current_avg = piezo_arr.average_frames

            changed, new_value = imgui.checkbox("Average frames per slice", current_avg)
            _set_tooltip(
                f"Average {fps} frames at each z-slice to produce one frame per slice. "
                "This reduces temporal resolution but improves SNR."
            )

            if changed and can_avg:
                piezo_arr.average_frames = new_value
                # Refresh the display
                if hasattr(viewer, "_refresh_image_widget"):
                    viewer._refresh_image_widget()

            if current_avg:
                # Show shape change info
                orig_t = piezo_arr.num_frames
                new_t = piezo_arr.shape[0]
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
                    f"Shape: {orig_t} frames -> {new_t} volumes"
                )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Single frame per slice (no averaging)"
            )

        imgui.separator()
