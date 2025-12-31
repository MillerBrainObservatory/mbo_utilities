from pathlib import Path
import threading
import math

from imgui_bundle import (
    hello_imgui,
    imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
    icons_fontawesome_6 as fa,
)
from mbo_utilities.gui._widgets import set_tooltip
from mbo_utilities.gui import _setup  # triggers setup on import
from mbo_utilities.preferences import (
    get_default_open_dir,
    get_last_dir,
    set_last_dir,
    add_recent_file,
    get_gui_preference,
    set_gui_preference,
)
from mbo_utilities.gui.upgrade_manager import UpgradeManager, CheckStatus, UpgradeStatus
from mbo_utilities.install import check_installation, Status

# re-export for backwards compatibility
setup_imgui = _setup.setup_imgui
__all__ = ["setup_imgui", "FileDialog"]

# dark theme
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)
COL_OK = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)
COL_WARN = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
COL_ERR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
COL_NA = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)


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


def icon_button(icon: str, label: str, size: imgui.ImVec2, tooltip: str = "") -> bool:
    """
    Draw a styled icon button with MBO theme.

    Dark gray background with blue outline, blue icon+text, hover effect.

    Parameters
    ----------
    icon : str
        FontAwesome 6 icon character (e.g., fa.ICON_FA_FOLDER_OPEN)
    label : str
        Button label text
    size : imgui.ImVec2
        Button size
    tooltip : str
        Tooltip text shown on hover

    Returns
    -------
    bool
        True if button was clicked
    """
    # Style: dark gray bg, blue border, blue text
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.18, 0.18, 0.20, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.22, 0.22, 0.25, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.15, 0.17, 1.0))
    imgui.push_style_color(imgui.Col_.text, COL_ACCENT)
    imgui.push_style_color(imgui.Col_.border, COL_ACCENT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.5)

    # Combine icon and label
    button_text = f"{icon}  {label}"
    clicked = imgui.button(button_text, size)

    # Show tooltip on hover
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)

    imgui.pop_style_var(2)
    imgui.pop_style_color(5)

    return clicked


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        self._widget_enabled = True
        self.metadata_only = False
        self.split_rois = False
        self._default_dir = str(get_default_open_dir())
        # upgrade manager - auto-check on startup
        self.upgrade_manager = UpgradeManager(enabled=True)
        self.upgrade_manager.check_for_upgrade()
        # cached install status (computed on first render)
        self._install_status = None
        self._check_thread = None
        
        # GUI Modes
        self.gui_modes = ["Standard Viewer", "Napari", "Cellpose", "Suite2p"]
        self.selected_mode_index = 0

    @property
    def widget_enabled(self):
        return self._widget_enabled

    @widget_enabled.setter
    def widget_enabled(self, value):
        self._widget_enabled = value

    def _save_gui_preferences(self):
        pass

    def _get_feature(self, name: str):
        """get feature status by name from install status."""
        if self._install_status is None:
            return None
        for f in self._install_status.features:
            if f.name == name:
                return f
        return None

    def _draw_version_status(self):
        """draw version with update status inline."""
        version = self._install_status.mbo_version if self._install_status else "?"

        checking = self.upgrade_manager.check_status == CheckStatus.CHECKING
        done = self.upgrade_manager.check_status == CheckStatus.DONE
        has_update = done and self.upgrade_manager.upgrade_available
        is_dev = done and self.upgrade_manager.is_dev_build
        up_to_date = done and not has_update and not is_dev

        # version text - green if up to date, normal otherwise
        if up_to_date:
            imgui.text_colored(COL_OK, f"v{version}")
            if imgui.is_item_hovered():
                imgui.set_tooltip("up to date")
        elif is_dev:
            imgui.text_colored(COL_TEXT_DIM, f"v{version}")
            if imgui.is_item_hovered():
                imgui.set_tooltip("development build")
        elif checking:
            imgui.text_colored(COL_TEXT_DIM, f"v{version}")
            imgui.same_line()
            imgui.text_colored(COL_TEXT_DIM, "...")
        else:
            imgui.text_colored(COL_TEXT_DIM, f"v{version}")

        # update button if available
        if has_update:
            imgui.same_line()
            upgrading = self.upgrade_manager.upgrade_status == UpgradeStatus.RUNNING
            if upgrading:
                imgui.begin_disabled()

            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.5, 0.2, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.25, 0.6, 0.25, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.4, 0.15, 1.0))
            text_h = imgui.get_text_line_height()
            if imgui.button("update", imgui.ImVec2(0, text_h)):
                self.upgrade_manager.start_upgrade()
            imgui.pop_style_color(3)

            if imgui.is_item_hovered():
                imgui.set_tooltip(f"v{self.upgrade_manager.latest_version} available")

            if upgrading:
                imgui.end_disabled()

            # show restart message after upgrade
            if self.upgrade_manager.upgrade_status == UpgradeStatus.SUCCESS:
                imgui.same_line()
                imgui.text_colored(COL_OK, "restart")
            elif self.upgrade_manager.upgrade_status == UpgradeStatus.ERROR:
                imgui.same_line()
                imgui.text_colored(COL_ERR, "failed")

    def _draw_dependency_group(self, name: str, pipeline_feature: str, requires: list[tuple[str, str]]):
        """draw a single dependency group inline: Name - Requirement vX.X (GPU/CPU)."""
        pipeline = self._get_feature(pipeline_feature)

        # not installed
        if pipeline is None or pipeline.status == Status.MISSING:
            imgui.text_colored(COL_NA, f"{name} - not installed")
            return

        # build the line: "Suite2p - PyTorch v2.1.0 (GPU)"
        parts = [name]

        # add requirement info inline
        for req_name, feature_name in requires:
            feat = self._get_feature(feature_name)
            if feat is None or feat.status == Status.MISSING:
                parts.append(f"- {req_name}: missing")
            else:
                ver_str = f"v{feat.version}" if feat.version and feat.version != "installed" else ""
                gpu_str = ""
                if feat.gpu_ok is True:
                    gpu_str = "GPU"
                elif feat.gpu_ok is False:
                    gpu_str = "CPU"

                req_parts = [f"- {req_name}"]
                if ver_str:
                    req_parts.append(ver_str)
                if gpu_str:
                    req_parts.append(f"({gpu_str})")
                parts.append(" ".join(req_parts))

        line = " ".join(parts)

        # color based on status
        if pipeline.status == Status.OK:
            imgui.text_colored(COL_OK, line)
        elif pipeline.status == Status.WARN:
            imgui.text_colored(COL_WARN, line)
        else:
            imgui.text_colored(COL_ERR, line)

        # tooltip with detailed info
        if imgui.is_item_hovered():
            tooltip_parts = []
            if pipeline.version and pipeline.version != "installed":
                tooltip_parts.append(f"{name} v{pipeline.version}")
            for req_name, feature_name in requires:
                feat = self._get_feature(feature_name)
                if feat and feat.message:
                    tooltip_parts.append(f"{req_name}: {feat.message}")
            if tooltip_parts:
                imgui.set_tooltip("\n".join(tooltip_parts))

    def _draw_formats_card_content(self):
        if self._install_status is None:
            # Vertical spacing
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + hello_imgui.em_size(1.0))

            # Progress text from callback (or default)
            msg = getattr(self, "_check_message", "Checking system configuration...")
            text_w = imgui.calc_text_size(msg).x
            avail_w = imgui.get_content_region_avail().x
            
            imgui.set_cursor_pos_x(max(0.0, (avail_w - text_w) * 0.5))
            imgui.text_colored(COL_TEXT_DIM, msg)

            # Vertical spacing
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + hello_imgui.em_size(0.5))

            # Progress bar from callback
            p = getattr(self, "_check_progress", 0.0)
            imgui.progress_bar(p)
            return

        # version line
        self._draw_version_status()

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

        # supported formats section
        imgui.text_colored(COL_ACCENT, "Supported Formats")
        imgui.same_line()
        push_button_style(primary=False)
        if imgui.small_button(f"{fa.ICON_FA_BOOK}  docs"):
            import webbrowser
            webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/file_formats.html")
        pop_button_style()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open documentation in browser")

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

        # calculate table width to fit content (not full available width)
        col1_width = hello_imgui.em_size(6)
        col2_width = hello_imgui.em_size(9)
        table_width = col1_width + col2_width

        table_flags = (
            imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.no_host_extend_x
        )
        if imgui.begin_table("##array_types", 2, table_flags, imgui.ImVec2(table_width, 0)):
            imgui.table_setup_column("Format", imgui.TableColumnFlags_.width_fixed, col1_width)
            imgui.table_setup_column("Extensions", imgui.TableColumnFlags_.width_fixed, col2_width)
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

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
        imgui.separator()
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

        # optional dependencies section
        imgui.text_colored(COL_ACCENT, "Optional Dependencies")
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

        # suite2p group
        self._draw_dependency_group(
            "Suite2p",
            "Suite2p",
            [("PyTorch", "PyTorch")]
        )

        # suite3d group
        self._draw_dependency_group(
            "Suite3D",
            "Suite3D",
            [("CuPy", "CuPy")]
        )

        # rastermap (no additional requirements)
        rastermap = self._get_feature("Rastermap")
        if rastermap is None or rastermap.status == Status.MISSING:
            imgui.text_colored(COL_NA, "Rastermap - not installed")
        else:
            ver = f" v{rastermap.version}" if rastermap.version and rastermap.version != "installed" else ""
            if rastermap.status == Status.OK:
                imgui.text_colored(COL_OK, f"Rastermap{ver}")
            else:
                imgui.text_colored(COL_WARN, f"Rastermap{ver}")

    def _center_text(self, text, color=None):
        """Draw centered text."""
        avail_w = imgui.get_content_region_avail().x
        text_sz = imgui.calc_text_size(text)
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_sz.x) * 0.5)
        if color:
            imgui.text_colored(color, text)
        else:
            imgui.text(text)

    def _center_widget(self, widget_width):
        """Set cursor to center a widget of given width."""
        avail_w = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - widget_width) * 0.5)

    def render(self):
        # global style
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

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0), window_flags=imgui.WindowFlags_.no_scrollbar):
            imgui.push_id("pfd")

            # header
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            self._center_text("Miller Brain Observatory", COL_ACCENT)
            self._center_text("Data Preview & Utilities", COL_TEXT_DIM)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # action buttons - use full available width minus padding
            avail_w = imgui.get_content_region_avail().x
            btn_w = min(avail_w - hello_imgui.em_size(2), hello_imgui.em_size(16))
            btn_h = hello_imgui.em_size(1.8)

            # Mode Selector
            self._center_widget(btn_w)
            imgui.set_next_item_width(btn_w)
            
            # Simple combo
            # ret, idx = imgui.combo("##mode", current_item, items)
            changed, self.selected_mode_index = imgui.combo(
                "##mode", 
                self.selected_mode_index, 
                self.gui_modes
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Select Application: {self.gui_modes[self.selected_mode_index]}")
            
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            self._center_widget(btn_w)
            if icon_button(
                fa.ICON_FA_FILE_IMAGE,
                "Open File(s)",
                imgui.ImVec2(btn_w, btn_h),
                "Select one or more image files"
            ):
                self._open_multi = pfd.open_file(
                    "Select files",
                    self._default_dir,
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin",
                     "All Files", "*"],
                    pfd.opt.multiselect
                )

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            self._center_widget(btn_w)
            if icon_button(
                fa.ICON_FA_FOLDER_OPEN,
                "Select Folder",
                imgui.ImVec2(btn_w, btn_h),
                "Select folder with image data"
            ):
                self._select_folder = pfd.select_folder("Select folder", self._default_dir)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

            # lazy load install status
            if self._install_status is None and self._check_thread is None:
                self._check_progress = 0.0
                self._check_message = "Starting..."

                def _progress_cb(p: float, msg: str):
                    self._check_progress = p
                    self._check_message = msg

                def _run_check():
                    self._install_status = check_installation(callback=_progress_cb)

                self._check_thread = threading.Thread(target=_run_check, daemon=True)
                self._check_thread.start()

            # card - use available width minus small margin
            avail_w = imgui.get_content_region_avail().x
            card_w = avail_w - hello_imgui.em_size(1)
            self._center_widget(card_w)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6.0)
            imgui.push_style_var(imgui.StyleVar_.cell_padding, hello_imgui.em_to_vec2(0.4, 0.2))

            # auto-resize height to content, no scrollbar
            child_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
            window_flags = imgui.WindowFlags_.no_scrollbar

            with imgui_ctx.begin_child("##formats", size=imgui.ImVec2(card_w, 0), child_flags=child_flags, window_flags=window_flags):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.6))

                self._draw_formats_card_content()

                imgui.unindent(hello_imgui.em_size(0.6))
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            imgui.pop_style_var(2)
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

            # quit button (centered)
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            qsz = imgui.ImVec2(hello_imgui.em_size(6), hello_imgui.em_size(1.5))
            self._center_widget(qsz.x)
            # Quit uses secondary style (gray bg, no border)
            push_button_style(primary=False)
            if imgui.button(f"{fa.ICON_FA_XMARK}  Quit", qsz) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


if __name__ == "__main__":
    pass
