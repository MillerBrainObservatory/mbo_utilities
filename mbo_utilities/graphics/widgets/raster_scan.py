"""
raster scan phase correction widget.

shows ui for bidirectional scan phase correction on data
that has scanimage metadata (metadata.get('si')).
"""

from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.graphics.widgets._base import Widget
from mbo_utilities.graphics._widgets import set_tooltip

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget


def has_scanimage_metadata(parent: "PreviewDataWidget") -> bool:
    """check if any data array has scanimage metadata."""
    try:
        for arr in parent.image_widget.data:
            if hasattr(arr, 'metadata'):
                md = arr.metadata
                if md and md.get('si'):
                    return True
    except Exception:
        pass
    return False


class RasterScanWidget(Widget):
    """ui widget for raster scan phase correction controls."""

    name = "Scan-Phase Correction"
    priority = 50

    @classmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """show only if data has scanimage metadata."""
        return has_scanimage_metadata(parent)

    def draw(self) -> None:
        """draw raster scan phase correction controls."""
        parent = self.parent

        imgui.spacing()
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction"
        )
        imgui.separator()

        # fix phase checkbox
        imgui.set_next_item_width(hello_imgui.em_size(10))
        phase_changed, phase_value = imgui.checkbox("Fix Phase", parent.fix_phase)
        set_tooltip(
            "Enable to apply scan-phase correction which shifts every other line/row of pixels "
            "to maximize correlation between these rows."
        )
        if phase_changed:
            parent.fix_phase = phase_value

        # fft subpixel checkbox
        imgui.set_next_item_width(hello_imgui.em_size(10))
        fft_changed, fft_value = imgui.checkbox("Sub-Pixel (slower)", parent.use_fft)
        set_tooltip(
            "Use FFT-based sub-pixel registration (slower but more accurate)."
        )
        if fft_changed:
            parent.use_fft = fft_value

        # display current offsets
        current_offsets = parent.current_offset
        for i, ofs in enumerate(current_offsets):
            max_abs_offset = abs(ofs)
            display_text = f"{ofs:.3f}"

            imgui.text(f"graphic {i + 1}: ")
            imgui.same_line()

            if max_abs_offset > parent.max_offset:
                imgui.push_style_color(
                    imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                )
                imgui.text(display_text)
                imgui.pop_style_color()
            else:
                imgui.text(display_text)

        # upsample factor
        imgui.set_next_item_width(hello_imgui.em_size(5))
        upsample_changed, upsample_val = imgui.input_int(
            "Upsample", parent.phase_upsample, step=1, step_fast=2
        )
        set_tooltip(
            "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment."
        )
        if upsample_changed:
            parent.phase_upsample = max(1, upsample_val)

        # border exclusion
        imgui.set_next_item_width(hello_imgui.em_size(5))
        border_changed, border_val = imgui.input_int(
            "Exclude border-px", parent.border, step=1, step_fast=2
        )
        set_tooltip(
            "Number of pixels to exclude from the edges of the image when computing the scan-phase offset."
        )
        if border_changed:
            parent.border = max(0, border_val)

        # max offset
        imgui.set_next_item_width(hello_imgui.em_size(5))
        max_offset_changed, max_offset_val = imgui.input_int(
            "max-offset", parent.max_offset, step=1, step_fast=2
        )
        set_tooltip(
            "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
        )
        if max_offset_changed:
            parent.max_offset = max(1, max_offset_val)

        imgui.separator()
