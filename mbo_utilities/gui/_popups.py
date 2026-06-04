"""
Popup windows and dialogs.

This module contains popup windows for tools, scope inspector,
metadata viewer, and process console.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from imgui_bundle import imgui, imgui_ctx, ImVec2

from mbo_utilities.gui._imgui_helpers import PopupAutoSize, begin_popup_size
from mbo_utilities.gui._metadata import draw_metadata_inspector
from mbo_utilities.gui._options_popup import _ensure_gpu_list
from mbo_utilities.gui.panels.debug_log import draw_scope
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.preferences import get_gpu_index


_SYS_TITLE = imgui.ImVec4(0.5, 0.8, 1.0, 1.0)
_SYS_LABEL = imgui.ImVec4(0.7, 0.7, 0.72, 1.0)
_SYS_ACCENT = imgui.ImVec4(0.95, 0.85, 0.45, 1.0)

_LOG_INFO = imgui.ImVec4(0.86, 0.86, 0.86, 1.0)
_LOG_ORANGE = imgui.ImVec4(1.0, 0.6, 0.2, 1.0)
_LOG_RED = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
_LOG_LEVEL_COLORS = {
    "DEBUG": _LOG_ORANGE,
    "INFO": _LOG_INFO,
    "WARNING": _LOG_ORANGE,
    "ERROR": _LOG_RED,
    "CRITICAL": _LOG_RED,
}

_MIN_LOG_BOX_H = 80.0


def _log_line_style(line: str) -> tuple[Any, str]:
    """Color and display text for a worker log line.

    Worker logs are ``asctime | name | levelname | message``. Color by
    level and drop the level field; lines without a level render white.
    """
    parts = line.split(" | ", 3)
    if len(parts) == 4:
        color = _LOG_LEVEL_COLORS.get(parts[2].strip())
        if color is not None:
            return color, f"{parts[0]} | {parts[1].strip()} | {parts[3]}"
    return _LOG_INFO, line


def _gpu_usage_str(d: dict) -> str:
    """One-line ``name · util% · used/total GB · tempC`` for a GPU device."""
    parts = [d.get("name", "?")]
    util = d.get("util_pct")
    if util is not None:
        parts.append(f"{util:.0f}% util")
    total = d.get("total_mb") or 0
    used = d.get("used_mb") or 0
    if total:
        parts.append(f"{used / 1024:.1f}/{total / 1024:.1f} GB")
    temp = d.get("temp_c")
    if temp is not None:
        parts.append(f"{temp:.0f}C")
    return " · ".join(parts)


def _draw_system_info_header(parent: Any) -> None:
    """Compact system-capacity header for the Process Console.

    Shows: CPU cores, live available/total RAM, the selected render GPU
    adapter (from preferences), and one live-usage entry per physical GPU.
    Live RAM is cheap to probe each frame (~µs syscall); per-GPU usage comes
    from nvidia-smi and is refreshed on a throttle (~1.5s), not every frame.
    """
    if not imgui.collapsing_header("System", imgui.TreeNodeFlags_.default_open):
        return

    _ensure_gpu_list(parent)

    # per-GPU live usage from nvidia-smi. throttled — nvidia-smi is a
    # subprocess, far too costly to run every frame.
    now = time.monotonic()
    if (not hasattr(parent, "_sys_gpu_devices")
            or now - getattr(parent, "_sys_gpu_last_refresh", 0.0) >= 1.5):
        from mbo_utilities.gpu import gpu_devices
        try:
            parent._sys_gpu_devices = gpu_devices()
        except Exception:
            parent._sys_gpu_devices = []
        parent._sys_gpu_last_refresh = now
    gpu_devices_live = parent._sys_gpu_devices

    # CPU + RAM via psutil (live RAM each frame).
    try:
        import psutil
        cpu_phys = psutil.cpu_count(logical=False)
        cpu_log = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        cpu_str = (
            f"{cpu_phys}p / {cpu_log}t"
            if cpu_phys else f"{cpu_log or '?'}"
        )
        ram_str = f"{vm.available/1024**3:.1f} / {vm.total/1024**3:.1f} GB"
    except ImportError:
        import os as _os
        cpu_str = f"{_os.cpu_count() or '?'}"
        ram_str = "?"

    adapters = getattr(parent, "_options_gpu_adapters", []) or []

    # Selected adapter resolved from persisted preferences. -1 == auto.
    sel_idx = get_gpu_index()
    if 0 <= sel_idx < len(adapters):
        info_d = getattr(adapters[sel_idx], "info", {}) or {}
        sel_name = info_d.get("device", "?")
        sel_backend = info_d.get("backend_type", "")
        selected_str = (
            f"{sel_name} [{sel_backend}]" if sel_backend else sel_name
        )
    else:
        selected_str = "auto (wgpu picks)"

    # Deduplicate physical GPUs by device name; skip software/CPU adapters
    # so the list shows only real hardware the user can render with.
    seen: set[str] = set()
    gpu_names: list[str] = []
    for a in adapters:
        info_d = getattr(a, "info", {}) or {}
        name = info_d.get("device", "?")
        adapter_type = info_d.get("adapter_type", "?")
        if adapter_type in ("CPU", "Unknown"):
            continue
        if name not in seen:
            seen.add(name)
            gpu_names.append(name)
    gpus_str = ", ".join(gpu_names) if gpu_names else "none detected"

    imgui.spacing()

    if imgui.begin_table(
        "##sysinfo_table", 2, imgui.TableFlags_.sizing_fixed_fit
    ):
        imgui.table_setup_column(
            "k", imgui.TableColumnFlags_.width_fixed, 140
        )
        imgui.table_setup_column(
            "v", imgui.TableColumnFlags_.width_stretch
        )

        def _row(k: str, v: str, value_color: Any = None) -> None:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_colored(_SYS_LABEL, k)
            imgui.table_next_column()
            if value_color is not None:
                imgui.text_colored(value_color, v)
            else:
                imgui.text(v)

        _row("CPU cores:", cpu_str)
        _row("RAM (avail/total):", ram_str)
        _row("GPU (selected):", selected_str, value_color=_SYS_ACCENT)
        if gpu_devices_live:
            for d in gpu_devices_live:
                _row(f"GPU {d['index']}:", _gpu_usage_str(d))
        else:
            _row("GPU (available):", gpus_str)

        imgui.end_table()

    imgui.spacing()
    imgui.spacing()


def draw_tools_popups(parent: Any):
    """Draw independent popup windows (Scope, Debug, Metadata)."""
    if parent.show_scope_window:
        size = begin_popup_size()
        imgui.set_next_window_size(size, imgui.Cond_.first_use_ever)
        _, parent.show_scope_window = imgui.begin(
            "Scope Inspector",
            parent.show_scope_window,
        )
        draw_scope()
        imgui.end()

    if parent.show_metadata_viewer:
        # use absolute screen positioning so window is visible even when widget collapsed
        io = imgui.get_io()
        screen_w, screen_h = io.display_size.x, io.display_size.y
        win_w, win_h = min(600, screen_w * 0.5), min(500, screen_h * 0.6)
        # center on screen
        imgui.set_next_window_pos(
            ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
            imgui.Cond_.first_use_ever,
        )
        imgui.set_next_window_size(ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)
        _, parent.show_metadata_viewer = imgui.begin(
            "Metadata Viewer",
            parent.show_metadata_viewer,
        )
        if parent.image_widget and parent.image_widget.data:
            data_arr = parent.image_widget.data[0]
            # Check if data has metadata (numpy arrays don't)
            if hasattr(data_arr, "metadata"):
                metadata = data_arr.metadata
                draw_metadata_inspector(metadata, data_array=data_arr)
            else:
                imgui.text("No metadata available")
                imgui.text(f"Data type: {type(data_arr).__name__}")
                if hasattr(data_arr, "shape"):
                    imgui.text(f"Shape: {data_arr.shape}")
        else:
            imgui.text("No data loaded")
        imgui.end()


def draw_process_console_popup(parent: Any):
    """Draw popup showing active tasks and background processes."""
    if not hasattr(parent, "_show_process_console"):
        parent._show_process_console = False
    if not hasattr(parent, "_process_console_size"):
        parent._process_console_size = ImVec2(500, 350)
    if not hasattr(parent, "_process_console_content_h"):
        parent._process_console_content_h = 0.0
    if not hasattr(parent, "_process_console_grow_to"):
        parent._process_console_grow_to = None
    if not hasattr(parent, "_proc_log_fixed_h"):
        parent._proc_log_fixed_h = 0.0
    if not hasattr(parent, "_proc_log_count"):
        parent._proc_log_count = 0
    if not hasattr(parent, "_process_console_sizer"):
        parent._process_console_sizer = PopupAutoSize(
            "Process Console", auto_resize=False
        )

    if parent._show_process_console:
        parent._process_console_sizer.before_open()
        imgui.open_popup("Process Console")
        parent._show_process_console = False

    work = imgui.get_main_viewport().work_size
    max_w = max(400.0, work.x - 40.0)
    max_h = max(300.0, work.y - 40.0)

    imgui.set_next_window_size(parent._process_console_size, imgui.Cond_.appearing)
    imgui.set_next_window_size_constraints(imgui.ImVec2(350, 200), imgui.ImVec2(max_w, max_h))

    # grow to fit content (requested last frame); width preserved, height bounded
    if parent._process_console_grow_to is not None:
        imgui.set_next_window_size(
            imgui.ImVec2(parent._process_console_size.x, parent._process_console_grow_to),
            imgui.Cond_.always,
        )
        parent._process_console_grow_to = None

    # use resizable modal (no auto_resize flag)
    opened, visible = imgui.begin_popup_modal(
        "Process Console",
        p_open=True,
        flags=imgui.WindowFlags_.none,
    )

    if opened:
        if not visible:
            imgui.close_current_popup()
        else:
            # save current size for next time
            parent._process_console_size = imgui.get_window_size()

            # System info header — pinned above the scrollable list so it
            # stays visible regardless of how many processes are queued.
            _draw_system_info_header(parent)

            pm = get_process_manager()
            pm.cleanup_finished()
            running = pm.get_running()

            from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items
            progress_items = _get_active_progress_items(parent)

            # recompute available height AFTER the header so the scroll
            # area gets exactly what's left, minus footer space.
            avail = imgui.get_content_region_avail()
            content_height = avail.y - 35  # space for separator + close button

            # split leftover height evenly among expanded log boxes so they
            # fill the area. uses last frame's non-box height and box count
            # (stable on resize) against this frame's available height.
            n_boxes = parent._proc_log_count
            if n_boxes > 0:
                log_fill_h = max(_MIN_LOG_BOX_H, (content_height - parent._proc_log_fixed_h) / n_boxes)
            else:
                log_fill_h = _MIN_LOG_BOX_H
            expanded_boxes = 0
            sum_box_h = 0.0

            # scrollable content area
            if imgui.begin_child("##ProcessContent", ImVec2(0, content_height), imgui.ChildFlags_.none):
                # active tasks section
                if progress_items:
                    imgui.text_colored(_SYS_TITLE, f"Active Tasks ({len(progress_items)})")
                    imgui.separator()
                    imgui.spacing()

                    for item in progress_items:
                        pct = int(item["progress"] * 100)
                        imgui.push_text_wrap_pos(0.0)
                        if item.get("done", False):
                            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[Done] {item['text']}")
                        else:
                            imgui.text(f"{item['text']}")
                        imgui.pop_text_wrap_pos()

                        # progress bar with percentage overlay
                        imgui.progress_bar(item["progress"], ImVec2(-1, 0), f"{pct}%")
                        imgui.spacing()

                    if running:
                        imgui.spacing()

                # background processes section
                if running:
                    imgui.text_colored(_SYS_TITLE, f"Background Processes ({len(running)})")
                    imgui.separator()
                    imgui.spacing()

                    for i, proc in enumerate(running):
                        if i > 0:
                            imgui.separator()
                            imgui.spacing()
                        box_h = _draw_process_entry(pm, proc, log_fill_h)
                        if box_h > 0.0:
                            expanded_boxes += 1
                            sum_box_h += box_h

                # empty state
                if not running and not progress_items:
                    imgui.spacing()
                    imgui.text_disabled("No active tasks or background processes.")

                # natural height of everything drawn, for next-frame window grow
                parent._process_console_content_h = imgui.get_cursor_pos_y()
                imgui.end_child()

            # remember the non-box height and box count for next frame's split
            parent._proc_log_fixed_h = parent._process_console_content_h - sum_box_h
            parent._proc_log_count = expanded_boxes

            # enlarge window to fit content, bounded by the main window.
            # skip on frames where a log box was just toggled (box count
            # changed): the measured height is transiently off by one box.
            win_h = imgui.get_window_height()
            target_h = min(win_h - content_height + parent._process_console_content_h, max_h)
            if target_h > win_h + 1.0 and expanded_boxes == n_boxes:
                parent._process_console_grow_to = target_h

            # footer actions, centered
            imgui.separator()
            imgui.spacing()

            finished = [p for p in running if not p.is_alive()]
            close_w = 80.0
            dismiss_w = 150.0
            spacing = imgui.get_style().item_spacing.x
            total_w = close_w + (dismiss_w + spacing if finished else 0.0)
            imgui.set_cursor_pos_x((imgui.get_window_width() - total_w) * 0.5)
            if imgui.button("Close", ImVec2(close_w, 0)):
                imgui.close_current_popup()
            if finished:
                imgui.same_line()
                if imgui.button(f"Dismiss finished ({len(finished)})", ImVec2(dismiss_w, 0)):
                    for p in finished:
                        pm._processes.pop(p.pid, None)
                    pm._save()

        imgui.end_popup()


def _draw_process_entry(pm: Any, proc: Any, log_fill_h: float = 0.0) -> float:
    """Draw a single process entry in the console.

    ``log_fill_h`` is the height for an expanded log box. Returns the box
    height used (0.0 when the log is collapsed or absent).
    """
    imgui.push_id(f"proc_{proc.pid}")
    box_h = 0.0

    # status indicator + description
    if proc.status == "error":
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "[ERR]")
    elif proc.status == "completed":
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), "[OK]")
    else:
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "[...]")

    imgui.same_line()

    # description wraps to the full content width
    imgui.push_text_wrap_pos(0.0)
    imgui.text(proc.description)
    imgui.pop_text_wrap_pos()

    # meta line: PID/elapsed on the left, actions right-aligned
    has_copy = bool(proc.output_path and Path(proc.output_path).is_file())
    action_label = "Kill" if proc.is_alive() else "Dismiss"
    style = imgui.get_style()

    def _btn_w(label: str) -> float:
        return imgui.calc_text_size(label).x + style.frame_padding.x * 2.0

    actions_w = _btn_w(action_label)
    if has_copy:
        actions_w += style.item_spacing.x + _btn_w("Copy")

    imgui.text_disabled(f"PID {proc.pid} · {proc.elapsed_str()}")
    imgui.same_line()
    right_x = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - actions_w
    imgui.set_cursor_pos_x(max(imgui.get_cursor_pos_x(), right_x))

    if proc.is_alive():
        if imgui.small_button("Kill"):
            pm.kill(proc.pid)
    else:
        if imgui.small_button("Dismiss"):
            pm._processes.pop(proc.pid, None)
            pm._save()

    if has_copy:
        imgui.same_line()
        if imgui.small_button("Copy"):
            try:
                with open(proc.output_path, encoding="utf-8") as f:
                    imgui.set_clipboard_text(f.read())
            except Exception:
                pass

    # error message
    if proc.status == "error" and proc.status_message:
        imgui.push_text_wrap_pos(0)
        imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.6, 1.0), f"  {proc.status_message}")
        imgui.pop_text_wrap_pos()

    # collapsible log output
    if proc.output_path and Path(proc.output_path).is_file():
        tree_open = imgui.tree_node(f"Log Output##proc_{proc.pid}")
        if tree_open:
            try:
                lines = proc.tail_log(500)
                box_h = max(_MIN_LOG_BOX_H, log_fill_h)

                child_flags = imgui.ChildFlags_.borders
                # begin_child always needs end_child, regardless of return value
                imgui.begin_child(f"##log_{proc.pid}", ImVec2(-1, box_h), child_flags)
                # only auto-scroll when user is already pinned to the bottom;
                # otherwise scrolling up would be overridden every frame
                at_bottom = imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 1.0
                line_h = imgui.get_text_line_height()
                imgui.push_text_wrap_pos(0.0)
                for line in lines:
                    color, display = _log_line_style(line.strip())
                    imgui.text_colored(color, display)
                    # wrapped line spans >1 row: add a small gap before the next
                    if imgui.get_item_rect_size().y > line_h + 1.0:
                        imgui.spacing()
                imgui.pop_text_wrap_pos()
                if at_bottom:
                    imgui.set_scroll_here_y(1.0)
                imgui.end_child()
            finally:
                imgui.tree_pop()

    imgui.spacing()
    imgui.pop_id()
    return box_h
