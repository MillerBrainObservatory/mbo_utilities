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


import re

# colors for the renderer
_C_NORMAL = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_C_BOLD = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)         # brighter = "bold" surrogate
_C_CODE_INLINE = imgui.ImVec4(0.95, 0.75, 0.55, 1.0)
_C_CODE_BLOCK = imgui.ImVec4(0.7, 0.9, 0.7, 1.0)
_C_H1 = imgui.ImVec4(1.0, 0.9, 0.4, 1.0)
_C_H2 = imgui.ImVec4(0.6, 0.85, 1.0, 1.0)
_C_H3 = imgui.ImVec4(0.85, 0.85, 0.85, 1.0)
_C_BULLET_TERM = imgui.ImVec4(0.9, 0.9, 0.5, 1.0)

# Tokenizer for inline markdown spans:
#   **bold** | `code` | plain
_INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def _render_inline(text: str, base_color: imgui.ImVec4 = _C_NORMAL) -> None:
    """Render a single line of text with inline **bold** and `code` spans,
    wrapping at the right edge of the content region.

    imgui has no native rich text and the help viewer doesn't load a
    bold-weight font, so 'bold' is rendered with a brighter tint than the
    base color. Inline `code` uses an orange-ish accent.

    Wrapping is manual because `same_line(0, 0)` (used to compose spans
    inline) blocks imgui's normal text-wrap. We tokenize each span into
    words + whitespace, measure cumulative width with `calc_text_size`,
    and emit a virtual newline (no `same_line`) when the next word would
    overflow the available region.
    """
    # 1. split into colored spans
    spans = []  # list of (text, color)
    for part in _INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            spans.append((part[2:-2], _C_BOLD))
        elif part.startswith("`") and part.endswith("`"):
            spans.append((part[1:-1], _C_CODE_INLINE))
        else:
            spans.append((part, base_color))

    # 2. tokenize each span into word + whitespace pieces; keep colors
    tokens = []  # list of (text, color, is_space)
    for span_text, color in spans:
        # match runs of whitespace OR runs of non-whitespace
        for m in re.finditer(r"\s+|\S+", span_text):
            piece = m.group(0)
            tokens.append((piece, color, piece.isspace()))

    if not tokens:
        return

    # 3. emit tokens with manual wrapping
    avail = imgui.get_content_region_avail().x
    start_x = imgui.get_cursor_pos_x()
    cur_x = start_x
    first_on_line = True
    space_pad = 4.0  # imgui's default item spacing slack

    for piece, color, is_space in tokens:
        # leading whitespace at the start of a wrapped line is dropped
        if first_on_line and is_space:
            continue
        w = imgui.calc_text_size(piece).x
        # would this token overflow? if so, break to a new line first.
        # never break before a whitespace (it just gets dropped).
        if not first_on_line and (cur_x + w) > (start_x + avail - space_pad):
            if is_space:
                continue  # don't emit trailing whitespace at end of line
            # force a newline by NOT calling same_line — the next text
            # call lands on the next line at the natural cursor.
            cur_x = start_x
            first_on_line = True

        if not first_on_line:
            imgui.same_line(0, 0)
        imgui.text_colored(color, piece)
        cur_x += w
        first_on_line = False


_TABLE_COUNTER = [0]


def _flush_table(rows: list, has_header: bool) -> None:
    """Render a buffered set of pipe-table rows using imgui.begin_table
    so columns actually line up.

    `rows` is a list of lists of cell strings. `has_header` is True iff
    the markdown source had a `|---|...|` separator after the first row;
    in that case the first row gets the header style.
    """
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    _TABLE_COUNTER[0] += 1
    table_id = f"##help_table_{_TABLE_COUNTER[0]}"
    flags = (
        imgui.TableFlags_.sizing_fixed_fit
        | imgui.TableFlags_.no_borders_in_body
        | imgui.TableFlags_.no_host_extend_x
    )
    if not imgui.begin_table(table_id, n_cols, flags):
        return
    try:
        for col in range(n_cols):
            imgui.table_setup_column(
                f"col{col}", imgui.TableColumnFlags_.width_fixed
            )
        start = 0
        if has_header:
            imgui.table_next_row()
            for col, cell in enumerate(rows[0]):
                imgui.table_set_column_index(col)
                # headers in white (bold surrogate)
                imgui.push_style_color(imgui.Col_.text, _C_BOLD)
                _render_inline(cell, base_color=_C_BOLD)
                imgui.pop_style_color()
            start = 1
        for row in rows[start:]:
            imgui.table_next_row()
            for col, cell in enumerate(row):
                imgui.table_set_column_index(col)
                _render_inline(cell)
    finally:
        imgui.end_table()


def _render_markdown(content: str) -> None:
    """render markdown content with basic formatting."""
    in_code_block = False
    table_rows: list[list[str]] = []
    table_has_header = False

    def _maybe_flush_table():
        nonlocal table_rows, table_has_header
        if table_rows:
            _flush_table(table_rows, table_has_header)
            table_rows = []
            table_has_header = False

    for line in content.split("\n"):
        stripped = line.strip()

        # code blocks
        if stripped.startswith("```"):
            _maybe_flush_table()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            # preserve leading whitespace and wrap long lines.
            imgui.push_style_color(imgui.Col_.text, _C_CODE_BLOCK)
            try:
                imgui.push_text_wrap_pos(0.0)
                try:
                    imgui.text_unformatted(line if line else " ")
                finally:
                    imgui.pop_text_wrap_pos()
            finally:
                imgui.pop_style_color()
            continue

        # tables: collect rows; render via imgui.begin_table on flush.
        if stripped.startswith("|"):
            sep_only = (
                stripped.replace("|", "")
                .replace("-", "")
                .replace(":", "")
                .replace(" ", "")
                == ""
            )
            if sep_only:
                # this is the header separator. mark the previous row
                # (already buffered) as the header.
                if table_rows:
                    table_has_header = True
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if cells:
                table_rows.append(cells)
            continue
        # any non-table line ends the current table
        _maybe_flush_table()

        # empty lines
        if not stripped:
            imgui.spacing()
            continue

        # headers
        if stripped.startswith("# "):
            imgui.spacing()
            imgui.text_colored(_C_H1, stripped[2:])
            imgui.separator()
            continue
        if stripped.startswith("## "):
            imgui.spacing()
            imgui.text_colored(_C_H2, stripped[3:])
            continue
        if stripped.startswith("### "):
            imgui.spacing()
            imgui.text_colored(_C_H3, stripped[4:])
            continue

        # bullet points (definition style: - **term**: description)
        if stripped.startswith("- **") and "**:" in stripped:
            parts = stripped[2:].split("**:", 1)
            term = parts[0].replace("**", "")
            desc = parts[1].strip() if len(parts) > 1 else ""
            imgui.bullet()
            imgui.same_line()
            imgui.text_colored(_C_BULLET_TERM, term + ":")
            if desc:
                imgui.same_line()
                _render_inline(desc)
            continue
        if stripped.startswith("- "):
            imgui.bullet()
            imgui.same_line()
            _render_inline(stripped[2:])
            continue

        # regular paragraph — wrap and apply inline spans
        _render_inline(stripped)

    # end-of-document flush
    _maybe_flush_table()
