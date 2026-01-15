"""
embedded help viewer with markdown rendering.

renders markdown docs shipped with the package in imgui popups.
uses a simple custom renderer since imgui_md requires font setup
that only happens when using immapp.run() (not fastplotlib).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui, ImVec2


# cached doc content
_doc_cache: dict[str, str] = {}

# available docs: (display_name, filename)
DOCS = [
    ("Quick Start", "gui_quickstart.md"),
    ("Keybinds", "keybinds.md"),
    ("File Formats", "file_formats.md"),
]


def get_docs_dir() -> Path:
    """get path to embedded docs directory."""
    return Path(__file__).parent.parent / "assets" / "docs"


def load_doc(filename: str) -> str:
    """load markdown doc from assets, with caching."""
    if filename not in _doc_cache:
        doc_path = get_docs_dir() / filename
        if doc_path.exists():
            _doc_cache[filename] = doc_path.read_text(encoding="utf-8")
        else:
            _doc_cache[filename] = f"*Document not found: {filename}*"
    return _doc_cache[filename]


def clear_doc_cache() -> None:
    """clear cached docs (useful for dev/reload)."""
    _doc_cache.clear()


def draw_help_popup(parent: Any) -> None:
    """draw help viewer popup with markdown rendering."""
    if not hasattr(parent, "_show_help_popup"):
        parent._show_help_popup = False
    if not hasattr(parent, "_help_selected_doc"):
        parent._help_selected_doc = 0

    if parent._show_help_popup:
        imgui.open_popup("Help##HelpViewer")
        parent._show_help_popup = False

    # center popup on screen
    io = imgui.get_io()
    screen_w, screen_h = io.display_size.x, io.display_size.y
    win_w, win_h = min(650, screen_w * 0.7), min(550, screen_h * 0.7)
    imgui.set_next_window_pos(
        ImVec2((screen_w - win_w) / 2, (screen_h - win_h) / 2),
        imgui.Cond_.appearing,
    )
    imgui.set_next_window_size(ImVec2(win_w, win_h), imgui.Cond_.first_use_ever)
    imgui.set_next_window_size_constraints(ImVec2(400, 300), ImVec2(1200, 900))

    opened, visible = imgui.begin_popup_modal(
        "Help##HelpViewer",
        p_open=True,
        flags=imgui.WindowFlags_.none,
    )

    if opened:
        if not visible:
            imgui.close_current_popup()
        else:
            # doc selector tabs
            if imgui.begin_tab_bar("##HelpTabs"):
                for i, (name, filename) in enumerate(DOCS):
                    if imgui.begin_tab_item(name)[0]:
                        parent._help_selected_doc = i
                        imgui.end_tab_item()
                imgui.end_tab_bar()

            imgui.separator()
            imgui.spacing()

            # content area
            avail = imgui.get_content_region_avail()
            content_height = avail.y - 35  # space for close button

            if imgui.begin_child("##HelpContent", ImVec2(0, content_height), imgui.ChildFlags_.borders):
                _, filename = DOCS[parent._help_selected_doc]
                content = load_doc(filename)
                _render_markdown(content)
                imgui.end_child()

            # close button
            imgui.separator()
            imgui.spacing()
            btn_width = 80
            imgui.set_cursor_pos_x((imgui.get_window_width() - btn_width) * 0.5)
            if imgui.button("Close", ImVec2(btn_width, 0)):
                imgui.close_current_popup()

        imgui.end_popup()


def _render_markdown(content: str) -> None:
    """render markdown content with basic formatting."""
    in_code_block = False
    in_table = False

    for line in content.split("\n"):
        stripped = line.strip()

        # code blocks
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            imgui.text_colored(imgui.ImVec4(0.7, 0.9, 0.7, 1.0), f"  {line}")
            continue

        # empty lines
        if not stripped:
            in_table = False
            imgui.spacing()
            continue

        # headers
        if stripped.startswith("# "):
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(1.0, 0.9, 0.4, 1.0), stripped[2:])
            imgui.separator()
            continue
        if stripped.startswith("## "):
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(0.6, 0.85, 1.0, 1.0), stripped[3:])
            continue
        if stripped.startswith("### "):
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.8, 1.0), stripped[4:])
            continue

        # tables
        if stripped.startswith("|"):
            # skip separator rows
            if stripped.replace("|", "").replace("-", "").replace(" ", "") == "":
                continue
            in_table = True
            # parse table cells
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if cells:
                row_text = "  ".join(f"{c:<20}" for c in cells)
                imgui.text(row_text)
            continue

        # bullet points
        if stripped.startswith("- **") and "**:" in stripped:
            # definition style: - **term**: description
            parts = stripped[2:].split("**:", 1)
            term = parts[0].replace("**", "")
            desc = parts[1].strip() if len(parts) > 1 else ""
            imgui.bullet()
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(0.9, 0.9, 0.5, 1.0), term + ":")
            imgui.same_line()
            imgui.text(desc)
            continue
        if stripped.startswith("- "):
            imgui.bullet_text(stripped[2:])
            continue

        # regular text
        imgui.text_wrapped(stripped)
