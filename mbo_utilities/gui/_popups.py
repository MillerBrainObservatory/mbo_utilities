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
    """Draw popup showing process outputs and debug logs."""
    if not hasattr(parent, "_show_process_console"):
        parent._show_process_console = False

    if parent._show_process_console:
        imgui.open_popup("Process Console")
        parent._show_process_console = False

    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size_constraints(imgui.ImVec2(400, 100), imgui.ImVec2(900, 600))

    if imgui.begin_popup_modal("Process Console", flags=imgui.WindowFlags_.always_auto_resize)[0]:
        pm = get_process_manager()
        pm.cleanup_finished()
        running = pm.get_running()

        # Use tabs instead of collapsible headers
        if imgui.begin_tab_bar("ProcessConsoleTabs"):
            # Tab 1: Processes
            if imgui.begin_tab_item("Processes")[0]:
                with imgui_ctx.begin_child("##BGTasksContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.auto_resize_y):
                    from mbo_utilities.gui.widgets.progress_bar import _get_active_progress_items
                    progress_items = _get_active_progress_items(parent)

                    if progress_items:
                        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), f"Active Tasks ({len(progress_items)})")
                        imgui.separator()
                        for item in progress_items:
                            pct = int(item["progress"] * 100)
                            if item.get("done", False):
                                imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"{item['text']} [Done]")
                            else:
                                imgui.text(f"{item['text']} [{pct}%]")

                            imgui.progress_bar(item["progress"], imgui.ImVec2(-1, 0), "")
                            imgui.spacing()
                        imgui.separator()
                        imgui.spacing()

                    if not running and not progress_items:
                        imgui.text_disabled("No active tasks or background processes.")
                    elif running:
                        if progress_items:
                            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), f"Background Processes ({len(running)})")
                            imgui.separator()

                        for proc in running:
                            imgui.push_id(f"proc_{proc.pid}")
                            imgui.bullet()

                            # Color code status
                            if proc.status == "error":
                                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), f"[ERROR] {proc.description}")
                            elif proc.status == "completed":
                                imgui.text_colored(imgui.ImVec4(0.4, 1.0, 0.4, 1.0), f"[DONE] {proc.description}")
                            else:
                                imgui.text(proc.description)

                            imgui.indent()
                            imgui.text_disabled(f"PID: {proc.pid} | Started: {proc.elapsed_str()}")

                            # Move Kill button here (next to PID) if active
                            if proc.is_alive():
                                imgui.same_line()
                                if imgui.small_button(f"Kill##{proc.pid}"):
                                    pm.kill(proc.pid)

                            # Show error message prominently if status is error
                            if proc.status == "error" and proc.status_message:
                                imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.6, 1.0), f"Error: {proc.status_message}")

                            # Show process output (tail of log)
                            if proc.output_path and Path(proc.output_path).is_file():
                                if imgui.tree_node(f"Output##proc_{proc.pid}"):
                                    lines = proc.tail_log(20)
                                    # Calculate height to fit content, max 150px
                                    line_height = imgui.get_text_line_height_with_spacing()
                                    output_content_height = len(lines) * line_height + 10
                                    output_height = min(output_content_height, 150) if lines else line_height + 10
                                    if imgui.begin_child(f"##proc_output_{proc.pid}", imgui.ImVec2(0, output_height), imgui.ChildFlags_.borders):
                                        for line in lines:
                                            line_stripped = line.strip()
                                            if "error" in line_stripped.lower():
                                                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), line_stripped)
                                            elif "warning" in line_stripped.lower():
                                                imgui.text_colored(imgui.ImVec4(1.0, 0.8, 0.2, 1.0), line_stripped)
                                            else:
                                                imgui.text(line_stripped)
                                        imgui.end_child()
                                    imgui.tree_pop()

                            # Control buttons (Dismiss / Copy)
                            if not proc.is_alive():
                                if imgui.small_button(f"Dismiss##{proc.pid}"):
                                    if proc.pid in pm._processes:
                                        del pm._processes[proc.pid]
                                        pm._save()
                                imgui.same_line()

                            # Copy Log Button (always available if log exists)
                            if proc.output_path and Path(proc.output_path).is_file():
                                if imgui.small_button(f"Copy Log##{proc.pid}"):
                                    try:
                                        with open(proc.output_path, encoding="utf-8") as f:
                                            full_log = f.read()
                                        imgui.set_clipboard_text(full_log)
                                    except Exception:
                                        pass

                            imgui.unindent()
                            imgui.spacing()
                            imgui.pop_id()
                imgui.end_tab_item()

            # Tab 2: System Logs
            if imgui.begin_tab_item("System Logs")[0]:
                with imgui_ctx.begin_child("##SysLogsContent", imgui.ImVec2(0, 0), imgui.ChildFlags_.auto_resize_y):
                    parent.debug_panel.draw()
                imgui.end_tab_item()

            imgui.end_tab_bar()

        # Close button
        imgui.separator()
        if imgui.button("Close", imgui.ImVec2(100, 0)):
            imgui.close_current_popup()

        imgui.end_popup()


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
