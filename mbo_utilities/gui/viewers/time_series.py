"""Time series viewer for standard calcium imaging data.

Renders the Preview / Signal Quality / Run tab bar by delegating to
the parent PreviewDataWidget, which owns the actual state and draw
methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui, imgui_ctx

from . import BaseViewer

if TYPE_CHECKING:
    from fastplotlib.widgets import ImageWidget

__all__ = ["TimeSeriesViewer"]


class TimeSeriesViewer(BaseViewer):
    """Viewer for time-series calcium imaging data (TZYX)."""

    name = "Time Series Viewer"

    def __init__(
        self,
        image_widget: ImageWidget,
        fpath: str | list[str],
        parent=None,
        **kwargs,
    ):
        super().__init__(image_widget, fpath, parent=parent, **kwargs)
        self._has_pipeline: bool | None = None

    def draw(self) -> None:
        """Draw the main tab bar, delegating to parent PreviewDataWidget."""
        if imgui.begin_tab_bar("MainPreviewTabs"):
            if imgui.begin_tab_item("Preview")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    self.parent.draw_preview_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()

            imgui.begin_disabled(not all(self.parent._zstats_done))
            if imgui.begin_tab_item("Signal Quality")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    with imgui_ctx.begin_child("##StatsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
                        self.parent.draw_stats_section()
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            imgui.end_disabled()

            if self._has_pipeline is None:
                from mbo_utilities.gui.widgets.pipelines import any_pipeline_available
                self._has_pipeline = any_pipeline_available()
            if not self._has_pipeline:
                imgui.begin_disabled()
            if imgui.begin_tab_item("Run")[0]:
                imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
                imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))
                try:
                    from mbo_utilities.gui.widgets.pipelines import draw_run_tab
                    draw_run_tab(self.parent)
                finally:
                    imgui.pop_style_var(2)
                imgui.end_tab_item()
            if not self._has_pipeline:
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        "Suite2p not installed.\n"
                        'Install with: uv pip install "mbo_utilities[suite2p]"'
                    )
            imgui.end_tab_bar()
