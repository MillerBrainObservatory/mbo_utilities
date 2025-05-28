import time
from collections import defaultdict

from imgui_bundle import (
    imgui,
    hello_imgui,
)

_progress_state = defaultdict(lambda: {
    "hide_time": None,
    "is_showing_done": False,
    "done_shown_once": False,
    "done_cleared": False,
})

def draw_progress(
    key: str,
    current_index: int,
    total_count: int,
    percent_complete: float,
    running_text: str = "Processing",
    done_text: str = "Completed",
    done: bool = False,
    custom_text: str | None = None,
):
    state = _progress_state[key]
    now = time.time()

    if done and not state["done_shown_once"]:
        state["hide_time"] = now + 3
        state["is_showing_done"] = True
        state["done_shown_once"] = True
        state["done_cleared"] = False

    elif not done:
        state["hide_time"] = None
        state["is_showing_done"] = False
        state["done_shown_once"] = False
        state["done_cleared"] = False

    if state["is_showing_done"] and state["hide_time"] and now >= state["hide_time"]:
        state["hide_time"] = None
        state["is_showing_done"] = False
        state["done_cleared"] = True
        return

    if not done and state["done_cleared"]:
        return  # prevent flashing previous bar

    # Set position
    bar_height = hello_imgui.em_size(1.4)
    window_height = imgui.get_window_height()
    bar_y = window_height - bar_height - imgui.get_style().item_spacing.y * 2
    imgui.set_cursor_pos_y(bar_y)

    # Choose bar style
    p = min(max(percent_complete, 0.0), 1.0)
    w = imgui.get_content_region_avail().x
    h = bar_height

    bar_color = imgui.ImVec4(0.0, 0.8, 0.0, 1.0) if state["is_showing_done"] else imgui.ImVec4(0.2, 0.5, 0.9, 1.0)
    if state["is_showing_done"]:
        text = done_text
    elif custom_text:
        text = custom_text
    elif current_index is not None and total_count is not None:
        text = f"{running_text} {current_index + 1} of {total_count} [{int(p * 100)}%]"
    else:
        text = f"{running_text} [{int(p * 100)}%]"

    imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)
    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))
    imgui.progress_bar(p, imgui.ImVec2(w, h), "")
    imgui.begin_group()

    if text:
        ts = imgui.calc_text_size(text)
        y = imgui.get_cursor_pos_y() - h + (h - ts.y) / 2
        x = (w - ts.x) / 2
        imgui.set_cursor_pos_y(y)
        imgui.set_cursor_pos_x(x)
        imgui.text_colored(imgui.ImVec4(1, 1, 1, 1), text)

    imgui.pop_style_var()
    imgui.pop_style_color()
    imgui.end_group()

def draw_saveas_progress(self):
    key = "saveas"
    state = _progress_state[key]

    if state["done_cleared"]:
        return  # don't draw anything anymore

    if state["is_showing_done"]:
        draw_progress(
            key=key,
            current_index=self._saveas_current_index,
            total_count=self._saveas_total,
            percent_complete=self._saveas_progress,
            running_text="Saving",
            done_text="Completed",
            done=True,
        )
    elif 0.0 < self._saveas_progress < 1.0:
        draw_progress(
            key=key,
            current_index=self._saveas_current_index,
            total_count=self._saveas_total,
            percent_complete=self._saveas_progress,
            running_text="Saving",
            custom_text=f"Saving z-plane {self._saveas_current_index} [{int(self._saveas_progress * 100)}%]",
        )

def draw_zstats_progress(self):
    key = "zstats"
    state = _progress_state[key]

    if state["done_cleared"]:
        return  # don't draw anything anymore

    draw_progress(
        key=key,
        current_index=self._zstats_current_z,
        total_count=self.nz,
        percent_complete=self._zstats_meansub_progress,
        running_text="Computing stats for plane(s)",
        done_text="Z-stats complete",
        done=self._zstats_done,
    )
