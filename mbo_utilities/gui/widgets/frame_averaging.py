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
        """show only for piezo arrays that can average frames."""
        arrays = parent._get_data_arrays()
        for arr in arrays:
            # check if this is a PiezoArray with averaging capability
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                return True
        return False

    def draw(self) -> None:
        """draw frame averaging controls."""
        parent = self.parent
        arrays = parent._get_data_arrays()

        # find piezo arrays with averaging support
        piezo_arrays = []
        for i, arr in enumerate(arrays):
            if hasattr(arr, "frames_per_slice") and hasattr(arr, "can_average"):
                piezo_arrays.append((i, arr))

        if not piezo_arrays:
            return

        # use first piezo array for display
        arr_idx, piezo_arr = piezo_arrays[0]

        imgui.spacing()
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Piezo Stack Info"
        )
        imgui.spacing()

        # show volumetric structure info
        num_volumes = getattr(piezo_arr, "num_volumes", piezo_arr.shape[0])
        num_slices = getattr(piezo_arr, "num_slices", 1)
        fps = piezo_arr.frames_per_slice
        log_avg = piezo_arr.log_average_factor
        current_avg = getattr(piezo_arr, "average_frames", False)

        # current frame dimension (changes based on averaging)
        current_frame_dim = piezo_arr.shape[1]

        imgui.text(f"Volumes: {num_volumes}")
        set_tooltip("number of volumes (T dimension)")

        imgui.text(f"Z-slices: {num_slices}")
        set_tooltip("number of z-slices per volume (from hStackManager.numSlices)")

        imgui.text(f"Frames per slice: {fps}")
        set_tooltip(
            "number of frames acquired at each z-slice position "
            "(from si.hStackManager.framesPerSlice)."
        )

        # show current slider dimension
        imgui.text(f"Current frames: {current_frame_dim}")
        if current_avg or log_avg > 1:
            set_tooltip("slider shows averaged z-slices")
        else:
            set_tooltip(f"slider shows all {num_slices} x {fps} = {current_frame_dim} raw frames")

        if log_avg > 1:
            # data was pre-averaged at acquisition
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
                f"Pre-averaged (factor: {log_avg})"
            )
            set_tooltip(
                "frames were averaged during acquisition before saving. "
                "no additional software averaging needed."
            )
        elif fps > 1:
            # can average
            can_avg = piezo_arr.can_average

            changed, new_value = imgui.checkbox("Average frames", current_avg)
            set_tooltip(
                f"average {fps} frames at each z-slice to improve SNR. "
                "this averages on-the-fly when reading data."
            )

            if changed and can_avg:
                # toggle averaging on the array
                piezo_arr.average_frames = new_value
                parent.logger.info(
                    f"Frame averaging {'enabled' if new_value else 'disabled'}"
                )
                # notify viewer that shape changed so sliders update
                if hasattr(parent, "_on_data_shape_changed"):
                    parent._on_data_shape_changed()

            if current_avg:
                imgui.text_colored(
                    imgui.ImVec4(0.6, 0.8, 0.6, 1.0),
                    "Averaging enabled"
                )
        else:
            imgui.text_colored(
                imgui.ImVec4(0.6, 0.6, 0.6, 1.0),
                "Single frame per slice"
            )

        imgui.separator()
