"""
Modular UI sections for PreviewDataWidget.

Each section is a standalone class that can be conditionally added based on
data/processor capabilities detected via protocols.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from imgui_bundle import imgui, hello_imgui

from mbo_utilities.graphics._widgets import set_tooltip
from mbo_utilities.graphics._protocols import (
    supports_raster_scan,
    SupportsRasterScan,
)

if TYPE_CHECKING:
    from mbo_utilities.graphics.imgui import PreviewDataWidget


class UISection(ABC):
    """Base class for modular UI sections."""

    # Human-readable name for the section
    name: str = "Section"

    # Priority for ordering (lower = higher priority, rendered first)
    priority: int = 100

    def __init__(self, parent: "PreviewDataWidget"):
        self.parent = parent

    @classmethod
    @abstractmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """
        Check if this section should be shown for the given widget.

        Override this to check capabilities of data/processors.
        """
        ...

    @abstractmethod
    def draw(self) -> None:
        """Draw the UI section."""
        ...


class RasterScanSection(UISection):
    """UI section for raster scan phase correction controls."""

    name = "Scan-Phase Correction"
    priority = 50

    @classmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """
        Show this section only if raster scan support is available.
        """
        return parent.has_raster_scan_support

    def draw(self) -> None:
        """Draw raster scan phase correction controls."""
        parent = self.parent

        imgui.spacing()
        imgui.separator()
        imgui.text_colored(
            imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Scan-Phase Correction"
        )

        imgui.separator()
        imgui.begin_group()

        imgui.set_next_item_width(hello_imgui.em_size(10))
        phase_changed, phase_value = imgui.checkbox("Fix Phase", parent.fix_phase)
        set_tooltip(
            "Enable to apply scan-phase correction which shifts every other line/row of pixels "
            "to maximize correlation between these rows."
        )
        if phase_changed:
            parent.fix_phase = phase_value

        imgui.set_next_item_width(hello_imgui.em_size(10))
        fft_changed, fft_value = imgui.checkbox("Sub-Pixel (slower)", parent.use_fft)
        set_tooltip(
            "Use FFT-based sub-pixel registration (slower but more accurate)."
        )
        if fft_changed:
            parent.use_fft = fft_value

        # Display current offsets
        current_offsets = parent.current_offset
        imgui.columns(2, "offsets", False)
        for i, ofs in enumerate(current_offsets):
            max_abs_offset = abs(ofs)

            imgui.text(f"graphic {i + 1}:")
            imgui.next_column()

            display_text = f"{ofs:.3f}"

            if max_abs_offset > parent.max_offset:
                imgui.push_style_color(
                    imgui.Col_.text, imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                )
                imgui.text(display_text)
                imgui.pop_style_color()
            else:
                imgui.text(display_text)

            imgui.next_column()
        imgui.columns(1)

        imgui.set_next_item_width(hello_imgui.em_size(5))
        upsample_changed, upsample_val = imgui.input_int(
            "Upsample", parent.phase_upsample, step=1, step_fast=2
        )
        set_tooltip(
            "Phase-correction upsampling factor: interpolates the image by this integer factor to improve subpixel alignment."
        )
        if upsample_changed:
            parent.phase_upsample = max(1, upsample_val)

        imgui.set_next_item_width(hello_imgui.em_size(5))
        border_changed, border_val = imgui.input_int(
            "Exclude border-px", parent.border, step=1, step_fast=2
        )
        set_tooltip(
            "Number of pixels to exclude from the edges of the image when computing the scan-phase offset."
        )
        if border_changed:
            parent.border = max(0, border_val)

        imgui.set_next_item_width(hello_imgui.em_size(5))
        max_offset_changed, max_offset_val = imgui.input_int(
            "max-offset", parent.max_offset, step=1, step_fast=2
        )
        set_tooltip(
            "Maximum allowed pixel shift (in pixels) when estimating the scan-phase offset."
        )
        if max_offset_changed:
            parent.max_offset = max(1, max_offset_val)

        imgui.end_group()
        imgui.separator()


class WindowFunctionsSection(UISection):
    """UI section for window functions (projection, window size, gaussian)."""

    name = "Window Functions"
    priority = 10  # Show first

    @classmethod
    def is_supported(cls, parent: "PreviewDataWidget") -> bool:
        """Always supported for any array with temporal dimension."""
        return True

    def draw(self) -> None:
        """Draw window functions controls."""
        parent = self.parent

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "Window Functions")
        imgui.spacing()

        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(2, 2))
        imgui.begin_group()

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
            " \"mean\" (average)\n"
            " \"max\" (peak)\n"
            " \"std\" (variance)\n"
            " \"mean-sub\" (mean-subtracted)."
        )

        if proj_changed:
            selected_label = options[selected_display_idx]
            if selected_label == "mean-sub (pending)":
                pass
            else:
                parent.proj = selected_label

        # Window size for projections (temporal dimension)
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

        # Gaussian Filter
        imgui.set_next_item_width(hello_imgui.em_size(6))
        gaussian_changed, new_sigma = imgui.input_float(
            "Gaussian Sigma", parent.gaussian_sigma, step=0.1, step_fast=1.0, format="%.1f"
        )
        set_tooltip(
            "Apply a Gaussian blur to the preview image. Sigma is in pixels; larger values yield stronger smoothing."
        )
        if gaussian_changed:
            parent.gaussian_sigma = max(0.0, new_sigma)

        imgui.end_group()

        imgui.pop_style_var()


# Registry of all available UI sections
UI_SECTIONS: list[type[UISection]] = [
    WindowFunctionsSection,
    RasterScanSection,
]


def get_supported_sections(parent: "PreviewDataWidget") -> list[UISection]:
    """
    Get all UI sections that are supported for the given widget.

    Returns instantiated sections sorted by priority.
    """
    supported = []
    for section_cls in UI_SECTIONS:
        if section_cls.is_supported(parent):
            supported.append(section_cls(parent))

    # Sort by priority (lower = first)
    supported.sort(key=lambda s: s.priority)
    return supported


def draw_all_sections(parent: "PreviewDataWidget", sections: list[UISection]) -> None:
    """Draw all supported UI sections."""
    for section in sections:
        section.draw()
