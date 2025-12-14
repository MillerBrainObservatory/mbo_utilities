import shutil
from pathlib import Path

import imgui_bundle

import mbo_utilities as mbo
from imgui_bundle import (
    hello_imgui,
    imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
)
from mbo_utilities.graphics._widgets import set_tooltip
from mbo_utilities.file_io import get_package_assets_path
from mbo_utilities.preferences import (
    get_default_open_dir,
    get_last_dir,
    set_last_dir,
    add_recent_file,
    get_gui_preference,
    set_gui_preference,
)


def setup_imgui():
    """set up hello_imgui assets folder, copying package assets to user config."""
    package_assets = get_package_assets_path()
    user_assets = Path(mbo.get_mbo_dirs()["base"]) / "imgui" / "assets"

    # copy package assets to user config directory
    user_assets.mkdir(parents=True, exist_ok=True)
    if package_assets.is_dir():
        shutil.copytree(package_assets, user_assets, dirs_exist_ok=True)

    # also copy imgui_bundle default fonts as fallback
    fonts_dst = user_assets / "fonts"
    fonts_dst.mkdir(parents=True, exist_ok=True)
    (user_assets / "static").mkdir(parents=True, exist_ok=True)

    fonts_src = Path(imgui_bundle.__file__).parent / "assets" / "fonts"
    for p in fonts_src.rglob("*"):
        if p.is_file():
            d = fonts_dst / p.relative_to(fonts_src)
            if not d.exists():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)

    # ensure roboto fonts exist for markdown rendering
    roboto_dir = fonts_dst / "Roboto"
    roboto_dir.mkdir(parents=True, exist_ok=True)
    required = [
        roboto_dir / "Roboto-Regular.ttf",
        roboto_dir / "Roboto-Bold.ttf",
        roboto_dir / "Roboto-RegularItalic.ttf",
        fonts_dst / "fontawesome-webfont.ttf",
    ]
    fallback = next((t for t in roboto_dir.glob("*.ttf")), None)
    for need in required:
        if not need.exists() and fallback and fallback.exists():
            need.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, need)

    # set hello_imgui assets folder (icon.png must be in assets/app_settings/)
    hello_imgui.set_assets_folder(str(user_assets))


# setup_imgui()


# colors - high contrast dark theme for better visibility
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)  # Darker blue for better text contrast
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)  # Pure white for maximum visibility
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)  # Lighter dim text
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)  # Lighter secondary buttons
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)


def push_button_style(primary=True):
    if primary:
        imgui.push_style_color(imgui.Col_.button, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_ACCENT_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_ACCENT_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    else:
        imgui.push_style_color(imgui.Col_.button, COL_SECONDARY)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_SECONDARY_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_SECONDARY_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)


def pop_button_style():
    imgui.pop_style_var(2)
    imgui.pop_style_color(4)


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        # Load saved GUI preferences
        self._widget_enabled = get_gui_preference("widget_enabled", True)
        self.metadata_only = get_gui_preference("metadata_only", False)
        self.split_rois = get_gui_preference("split_rois", False)
        # Get default directory for file dialogs
        self._default_dir = str(get_default_open_dir())

    @property
    def widget_enabled(self):
        return self._widget_enabled

    @widget_enabled.setter
    def widget_enabled(self, value):
        self._widget_enabled = value

    def _save_gui_preferences(self):
        """Save current GUI preferences to disk."""
        set_gui_preference("widget_enabled", self._widget_enabled)
        set_gui_preference("metadata_only", self.metadata_only)
        set_gui_preference("split_rois", self.split_rois)

    def render(self):
        # global style - high contrast for visibility
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.37, 0.6))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.22, 0.22, 0.23, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.28, 0.28, 0.29, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)

        win_w = imgui.get_window_width()
        win_h = imgui.get_window_height()

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0)):
            imgui.push_id("pfd")

            # header
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            title = "Miller Brain Observatory"
            title_sz = imgui.calc_text_size(title)
            imgui.set_cursor_pos_x((win_w - title_sz.x) * 0.5)
            imgui.text_colored(COL_ACCENT, title)

            subtitle = "Data Preview & Utilities"
            sub_sz = imgui.calc_text_size(subtitle)
            imgui.set_cursor_pos_x((win_w - sub_sz.x) * 0.5)
            imgui.text_colored(COL_TEXT_DIM, subtitle)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # action buttons
            btn_w = hello_imgui.em_size(16)
            btn_h = hello_imgui.em_size(1.8)
            btn_x = (win_w - btn_w) * 0.5

            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Open File(s)", imgui.ImVec2(btn_w, btn_h)):
                self._open_multi = pfd.open_file(
                    "Select files",
                    self._default_dir,
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin",
                     "All Files", "*"],
                    pfd.opt.multiselect
                )
            pop_button_style()
            set_tooltip("Select one or more image files")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            imgui.set_cursor_pos_x(btn_x)
            push_button_style(primary=True)
            if imgui.button("Select Folder", imgui.ImVec2(btn_w, btn_h)):
                self._select_folder = pfd.select_folder("Select folder", self._default_dir)
            pop_button_style()
            set_tooltip("Select folder with image data")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

            # calculate dynamic card height based on available space
            # fixed heights: header ~3em, buttons ~4.5em, quit ~2em, padding ~2em
            fixed_height = hello_imgui.em_size(12)
            available_for_card = win_h - fixed_height
            # content needs ~14em to fit without scrollbar
            content_height = hello_imgui.em_size(14)
            # use available space, but cap at content height (no need to be bigger)
            card_h = min(available_for_card, content_height)
            # minimum height to show at least something useful
            card_h = max(card_h, hello_imgui.em_size(6))

            # formats section - dynamic width based on window
            card_w = min(hello_imgui.em_size(24), win_w - hello_imgui.em_size(2))
            card_x = (win_w - card_w) * 0.5
            imgui.set_cursor_pos_x(card_x)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6.0)

            # only show scrollbar if content doesn't fit
            needs_scroll = card_h < content_height
            child_flags = imgui.ChildFlags_.borders
            window_flags = imgui.WindowFlags_.none if needs_scroll else (imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)

            with imgui_ctx.begin_child("##formats", size=imgui.ImVec2(card_w, card_h), child_flags=child_flags, window_flags=window_flags):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.4))

                imgui.text_colored(COL_ACCENT, "Supported Formats")
                imgui.same_line()
                push_button_style(primary=False)
                if imgui.small_button("docs"):
                    import webbrowser
                    webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/array_types.html")
                pop_button_style()

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

                # table with array types
                table_flags = (
                    imgui.TableFlags_.borders_inner_v
                    | imgui.TableFlags_.row_bg
                    | imgui.TableFlags_.sizing_stretch_same
                )
                if imgui.begin_table("##array_types", 2, table_flags):
                    imgui.table_setup_column("Format")
                    imgui.table_setup_column("Extensions")
                    imgui.table_headers_row()

                    array_types = [
                        ("ScanImage", ".tif, .tiff"),
                        ("TIFF", ".tif, .tiff"),
                        ("Zarr", ".zarr/"),
                        ("HDF5", ".h5, .hdf5"),
                        ("Suite2p", ".bin, ops.npy"),
                        ("NumPy", ".npy"),
                        ("NWB", ".nwb"),
                    ]
                    for name, ext in array_types:
                        imgui.table_next_row()
                        imgui.table_next_column()
                        imgui.text(name)
                        imgui.table_next_column()
                        imgui.text_colored(COL_TEXT_DIM, ext)
                    imgui.end_table()

                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.separator()
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

                imgui.text_colored(COL_ACCENT, "Options")
                _, self._widget_enabled = imgui.checkbox("Preview widget", self._widget_enabled)
                _, self.split_rois = imgui.checkbox("Separate mROIs", self.split_rois)
                _, self.metadata_only = imgui.checkbox("Metadata only", self.metadata_only)

                imgui.unindent(hello_imgui.em_size(0.4))
            imgui.pop_style_var()
            imgui.pop_style_color()

            # file/folder completion
            if self._open_multi and self._open_multi.ready():
                self.selected_path = self._open_multi.result()
                if self.selected_path:
                    for p in (self.selected_path if isinstance(self.selected_path, list) else [self.selected_path]):
                        add_recent_file(p, file_type="file")
                        set_last_dir("open_file", p)
                    self._save_gui_preferences()
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_multi = None
            if self._select_folder and self._select_folder.ready():
                self.selected_path = self._select_folder.result()
                if self.selected_path:
                    add_recent_file(self.selected_path, file_type="folder")
                    set_last_dir("open_folder", self.selected_path)
                    self._save_gui_preferences()
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._select_folder = None

            # quit button - inline at bottom
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            qsz = imgui.ImVec2(hello_imgui.em_size(5), hello_imgui.em_size(1.5))
            imgui.set_cursor_pos_x(win_w - qsz.x - hello_imgui.em_size(1.0))
            push_button_style(primary=False)
            if imgui.button("Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


if __name__ == "__main__":
    pass
