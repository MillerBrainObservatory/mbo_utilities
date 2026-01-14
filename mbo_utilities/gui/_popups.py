"""
Popup windows and dialogs.

This module contains popup windows for tools, scope inspector,
metadata viewer, and process console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui, imgui_ctx, ImVec2

from mbo_utilities.gui._imgui_helpers import begin_popup_size
from mbo_utilities.gui._metadata import draw_metadata_inspector
from mbo_utilities.gui.panels.debug_log import draw_scope
from mbo_utilities.gui.widgets.process_manager import get_process_manager


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

    if parent._show_process_console:
        imgui.open_popup("Process Console")
        parent._show_process_console = False

    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(parent._process_console_size, imgui.Cond_.appearing)
    imgui.set_next_window_size_constraints(imgui.ImVec2(350, 200), imgui.ImVec2(1200, 800))

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

            pm = get_process_manager()
            pm.cleanup_finished()
            running = pm.get_running()

            from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items
            progress_items = _get_active_progress_items(parent)

            # calculate content area (leave space for close button)
            avail = imgui.get_content_region_avail()
            content_height = avail.y - 35  # space for separator + close button

            # scrollable content area
            if imgui.begin_child("##ProcessContent", ImVec2(0, content_height), imgui.ChildFlags_.none):
                # active tasks section
                if progress_items:
                    imgui.text_colored(imgui.ImVec4(0.5, 0.8, 1.0, 1.0), "Active Tasks")
                    imgui.separator()
                    imgui.spacing()

                    for item in progress_items:
                        pct = int(item["progress"] * 100)
                        if item.get("done", False):
                            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[Done] {item['text']}")
                        else:
                            imgui.text(f"{item['text']}")

                        # progress bar with percentage overlay
                        imgui.progress_bar(item["progress"], ImVec2(-1, 0), f"{pct}%")
                        imgui.spacing()

                    if running:
                        imgui.spacing()

                # background processes section
                if running:
                    imgui.text_colored(imgui.ImVec4(0.5, 0.8, 1.0, 1.0), "Background Processes")
                    imgui.separator()
                    imgui.spacing()

                    for proc in running:
                        _draw_process_entry(pm, proc)

                # empty state
                if not running and not progress_items:
                    imgui.spacing()
                    imgui.text_disabled("No active tasks or background processes.")

                imgui.end_child()

            # footer with close button
            imgui.separator()
            imgui.spacing()
            btn_width = 80
            imgui.set_cursor_pos_x((imgui.get_window_width() - btn_width) * 0.5)
            if imgui.button("Close", ImVec2(btn_width, 0)):
                imgui.close_current_popup()

        imgui.end_popup()


def _draw_process_entry(pm: Any, proc: Any) -> None:
    """Draw a single process entry in the console."""
    imgui.push_id(f"proc_{proc.pid}")

    # status indicator + description
    if proc.status == "error":
        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "[ERR]")
    elif proc.status == "completed":
        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), "[OK]")
    else:
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "[...]")

    imgui.same_line()

    # wrap description text
    avail_width = imgui.get_content_region_avail().x - 80  # leave space for buttons
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + avail_width)
    imgui.text(proc.description)
    imgui.pop_text_wrap_pos()

    # info line with buttons
    imgui.text_disabled(f"PID {proc.pid} | {proc.elapsed_str()}")

    imgui.same_line()

    # action buttons
    if proc.is_alive():
        if imgui.small_button("Kill"):
            pm.kill(proc.pid)
    else:
        if imgui.small_button("Dismiss"):
            if proc.pid in pm._processes:
                del pm._processes[proc.pid]
                pm._save()

    # copy log button
    if proc.output_path and Path(proc.output_path).is_file():
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
        if imgui.tree_node(f"Log Output##proc_{proc.pid}"):
            lines = proc.tail_log(30)
            line_height = imgui.get_text_line_height_with_spacing()
            max_height = 180
            content_h = min(len(lines) * line_height + 8, max_height) if lines else line_height + 8

            child_flags = imgui.ChildFlags_.borders
            if imgui.begin_child(f"##log_{proc.pid}", ImVec2(-1, content_h), child_flags):
                for line in lines:
                    line_stripped = line.strip()
                    # color code log lines
                    if "error" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                    elif "warning" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                    elif "success" in line_stripped.lower() or "complete" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), line_stripped)
                    else:
                        imgui.text(line_stripped)
                # auto-scroll to bottom
                imgui.set_scroll_here_y(1.0)
                imgui.end_child()
            imgui.tree_pop()

    imgui.spacing()
    imgui.pop_id()


def draw_background_processes_section(parent: Any):
    """Draw listing of background processes and their logs."""
    pm = get_process_manager()
    pm.cleanup_finished()  # clean up dead processes
    running = pm.get_running()

    if not running:
        imgui.text_disabled("No background processes running.")
        return

    imgui.text_colored(
        imgui.ImVec4(0.9, 0.8, 0.3, 1.0),
        f"{len(running)} active process(es):"
    )
    imgui.separator()
    imgui.spacing()

    for proc in running:
        imgui.push_id(f"proc_{proc.pid}")

        # process description
        imgui.bullet()

        # Color code status
        if proc.status == "error":
            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"[ERROR] {proc.description}")
        elif proc.status == "completed":
            imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[DONE] {proc.description}")
        else:
            imgui.text(proc.description)

        # details
        imgui.indent()
        imgui.text_disabled(f"PID: {proc.pid} | Started: {proc.elapsed_str()}")

        # last log line
        last_line = proc.get_last_log_line()
        if last_line:
            if len(last_line) > 100:
                last_line = last_line[:97] + "..."
            imgui.text_colored(imgui.ImVec4(0.5, 0.7, 1.0, 0.8), f"> {last_line}")

        # Buttons (Kill/Dismiss and Show console)
        if imgui.small_button("Show Console"):
            parent._viewing_process_pid = proc.pid

        imgui.same_line()

        if proc.is_alive():
            if imgui.small_button("Kill"):
                if pm.kill(proc.pid):
                    parent.logger.info(f"Killed process {proc.pid}")
                    if parent._viewing_process_pid == proc.pid:
                        parent._viewing_process_pid = None
                else:
                    parent.logger.warning(f"Failed to kill process {proc.pid}")
        elif imgui.small_button("Dismiss") and proc.pid in pm._processes:
            if parent._viewing_process_pid == proc.pid:
                parent._viewing_process_pid = None
            del pm._processes[proc.pid]
            pm._save()

        imgui.unindent()
        imgui.spacing()
        imgui.pop_id()

    # console output area for selected process
    if parent._viewing_process_pid is not None:
        # find the process
        v_proc = next((p for p in running if p.pid == parent._viewing_process_pid), None)
        if v_proc:
            imgui.dummy(imgui.ImVec2(0, 10))
            imgui.text_colored(imgui.ImVec4(0.3, 0.6, 1.0, 1.0), f"Console: {v_proc.description} (PID {v_proc.pid})")
            imgui.same_line(imgui.get_content_region_avail().x - 20)
            if imgui.small_button("x##close_console"):
                parent._viewing_process_pid = None

            # tail log
            lines = v_proc.tail_log(30)
            # Calculate height to fit content, with a max height and scrollbar when needed
            line_height = imgui.get_text_line_height_with_spacing()
            content_height = len(lines) * line_height + 10  # padding
            max_height = 250
            console_height = min(content_height, max_height) if lines else line_height + 10
            if imgui.begin_child("##proc_console", imgui.ImVec2(0, console_height), imgui.ChildFlags_.borders):
                for line in lines:
                    line_stripped = line.strip()
                    if "error" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                    elif "warning" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                    elif "success" in line_stripped.lower() or "complete" in line_stripped.lower():
                        imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), line_stripped)
                    else:
                        imgui.text(line_stripped)
                # auto-scroll
                imgui.set_scroll_here_y(1.0)
                imgui.end_child()
        else:
            parent._viewing_process_pid = None

    imgui.spacing()
    if running:
        any_alive = any(p.is_alive() for p in running)
        if any_alive:
            if imgui.button("Kill All Processes", imgui.ImVec2(-1, 0)):
                killed = pm.kill_all()
                parent.logger.info(f"Killed {killed} processes")
        elif imgui.button("Clear Finished Processes", imgui.ImVec2(-1, 0)):
            to_remove = [p.pid for p in running if not p.is_alive()]
            for pid in to_remove:
                del pm._processes[pid]
            pm._save()
