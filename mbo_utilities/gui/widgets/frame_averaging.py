"""
frame averaging widget for piezo stacks.

shows ui for toggling frame averaging on piezo stack data
that has framesPerSlice > 1 and was not pre-averaged.
"""

from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui._imgui_helpers import set_tooltip


class FrameAveragingWidget(Widget):
    """ui widget for piezo stack frame averaging controls."""

    name = "Frame Averaging"
    priority = 55  # after raster scan (50)

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        """Show only for piezo arrays that can average frames."""
        arrays = parent._get_data_arrays()
        for arr in arrays:
            # check if this is a PiezoArray with averaging capability
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                return True
        return False

    def draw(self) -> None:
        """Draw frame averaging controls."""
        parent = self.parent
        arrays = parent._get_data_arrays()

        # find first piezo array
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

        # show acquisition info
        fps = piezo_arr.frames_per_slice
        log_avg = piezo_arr.log_average_factor

        imgui.text(f"Frames per slice: {fps}")
        set_tooltip(
            "Number of frames acquired at each z-slice position "
            "(from si.hStackManager.framesPerSlice)."
        )

        if log_avg > 1:
            # data was pre-averaged at acquisition
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
                f"Pre-averaged at acquisition (factor: {log_avg})"
            )
            set_tooltip(
                "Frames were averaged during acquisition before saving. "
                "No additional software averaging needed."
            )
        elif fps > 1:
            # can average
            can_avg = piezo_arr.can_average
            current_avg = piezo_arr.average_frames

            changed, new_value = imgui.checkbox("Average frames per slice", current_avg)
            set_tooltip(
                f"Average {fps} frames at each z-slice to produce one frame per slice. "
                "This reduces temporal resolution but improves SNR."
            )

            if changed and can_avg:
                piezo_arr.average_frames = new_value
                # refresh the display
                parent._refresh_image_widget()

            if current_avg:
                # show shape change info
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
