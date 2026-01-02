"""
Raster scan phase correction feature.

Shows UI for bidirectional scan phase correction on data
that supports phase correction (has fix_phase and use_fft attributes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from . import BaseFeature

if TYPE_CHECKING:
    from ..viewers import BaseViewer

__all__ = ["RasterScanFeature"]


def _set_tooltip(text: str, show_mark: bool = True) -> None:
    """Set tooltip on hover. Helper to avoid circular import."""
    if show_mark:
        imgui.same_line()
        imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class RasterScanFeature(BaseFeature):
    """
    Bidirectional scan phase correction controls.

    Controls:
    - Fix phase toggle
    - Sub-pixel (FFT) toggle
    - Upsample factor
    - Border exclusion
    - Max offset
    - Current offset display per graphic
    """

    name = "Scan-Phase Correction"
    priority = 50

    @classmethod
    def is_supported(cls, viewer: "BaseViewer") -> bool:
        """Show only if data array supports phase correction."""
        # Check cached property if available
        if hasattr(viewer, "has_raster_scan_support"):
            return viewer.has_raster_scan_support

        # Otherwise check data arrays directly
        from .._protocols import supports_raster_scan

        arrays = viewer._get_data_arrays()
        return any(supports_raster_scan(arr) for arr in arrays)

    def draw(self) -> None:
        """Draw raster scan phase correction controls."""
        viewer = self.viewer

        imgui.spacing()
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction"
        )
        imgui.spacing()

        # Fix phase checkbox
        fix_phase = getattr(viewer, "fix_phase", False)
        imgui.set_next_item_width(hello_imgui.em_size(10))
        phase_changed, phase_value = imgui.checkbox("Fix Phase", fix_phase)
        _set_tooltip(
            "Enable to apply scan-phase correction which shifts every other line/row of pixels "
            "to maximize correlation between these rows."
        )
        if phase_changed:
            viewer.fix_phase = phase_value

        # FFT subpixel checkbox
        use_fft = getattr(viewer, "use_fft", True)
        imgui.set_next_item_width(hello_imgui.em_size(10))
        fft_changed, fft_value = imgui.checkbox("Sub-Pixel (slower)", use_fft)
        _set_tooltip(
            "Use FFT-based sub-pixel registration (slower but more accurate)."
        )
        if fft_changed:
            viewer.use_fft = fft_value

        # Display current offsets
        current_offsets = getattr(viewer, "current_offset", [])
        max_offset = getattr(viewer, "max_offset", 10)

        for i, ofs in enumerate(current_offsets):
            max_abs_offset = abs(ofs)
            display_text = f"{ofs:.3f}"

            imgui.text(f"graphic {i + 1}: ")
            imgui.same_line()

            if max_abs_offset > max_offset:
                imgui.push_style_color(
                    imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                )
                imgui.text(display_text)
                imgui.pop_style_color()
            else:
                imgui.text(display_text)

        # Upsample factor
        phase_upsample = getattr(viewer, "phase_upsample", 10)
        imgui.set_next_item_width(hello_imgui.em_size(5))
        upsample_changed, upsample_val = imgui.input_int(
            "Upsample", phase_upsample, step=1, step_fast=2
        )
        _set_tooltip(
            "Phase-correction upsampling factor: interpolates the image by this integer factor "
            "to improve subpixel alignment."
        )
        if upsample_changed:
            viewer.phase_upsample = max(1, upsample_val)

        # Border exclusion
        border = getattr(viewer, "border", 10)
        imgui.set_next_item_width(hello_imgui.em_size(5))
        border_changed, border_val = imgui.input_int(
            "Exclude border-px", border, step=1, step_fast=2
        )
        _set_tooltip(
            "Number of pixels to exclude from the edges of the image "
            "when computing the scan-phase offset."
        )
        if border_changed:
            viewer.border = max(0, border_val)

        # Max offset
        imgui.set_next_item_width(hello_imgui.em_size(5))
        max_offset_changed, max_offset_val = imgui.input_int(
            "max-offset", max_offset, step=1, step_fast=2
        )
        _set_tooltip(
            "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
        )
        if max_offset_changed:
            viewer.max_offset = max(1, max_offset_val)

        imgui.separator()
