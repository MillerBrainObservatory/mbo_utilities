import click
import numpy as np

from mbo_utilities.util import is_imgui_installed
from mbo_utilities.graphics.imgui import PreviewDataWidget
from mbo_utilities.file_io import (
    to_lazy_array,
    get_files,
    read_scan,
    _is_arraylike,
)

selected = []

if is_imgui_installed():
    import fastplotlib as fpl
    from imgui_bundle import portable_file_dialogs as pfd, immapp, imgui, imgui_md
    import OpenGL.GL as gl  # type: ignore
    import os

    if os.getenv("XDG_SESSION_TYPE") == "wayland" and not os.getenv("PYOPENGL_PLATFORM"):
        os.environ["PYOPENGL_PLATFORM"] = "x11"
    import glfw  # type: ignore
    from imgui_bundle import imgui, imgui_ctx
    from imgui_bundle import imgui_md
    import sys
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
    imgui.get_io().fonts.add_font_default()
    
@immapp.static(
    open_file_dialog=None,
    open_file_multiselect=None,
    save_file_dialog=None,
    select_folder_dialog=None,
    last_file_selection="",
    # Messages and Notifications
    icon_type=pfd.icon.info,
    message_dialog=None,
    message_choice_type=pfd.choice.ok,
)
def demo_portable_file_dialogs():
    # from imgui_bundle import portable_file_dialogs as pfd
    static = demo_portable_file_dialogs

    imgui.push_id("pfd")
    imgui_md.render_unindented(
        """
        # Portable File Dialogs
         [portable-file-dialogs](https://github.com/samhocevar/portable-file-dialogs) provides file dialogs
         as well as notifications and messages. They will use the native dialogs and notifications on each platform.
    """
    )

    def log_result_list(whats):
        static.last_file_selection = "\n".join(whats)
        selected.clear()
        selected.extend(whats)

    def log_result(what: str):
        static.last_file_selection = what
        selected.clear()
        selected.append(what)

    imgui.text("      ---   File dialogs   ---")
    if imgui.button("Open file"):
        static.open_file_dialog = pfd.open_file("Select file")
    if static.open_file_dialog is not None and static.open_file_dialog.ready():
        log_result_list(static.open_file_dialog.result())
        static.open_file_dialog = None

    imgui.same_line()

    if imgui.button("Open file (multiselect)"):
        static.open_file_multiselect = pfd.open_file(
            "Select file", options=pfd.opt.multiselect
        )
    if (
        static.open_file_multiselect is not None
        and static.open_file_multiselect.ready()
    ):
        log_result_list(static.open_file_multiselect.result())
        static.open_file_multiselect = None

    imgui.same_line()

    if imgui.button("Save file"):
        static.save_file_dialog = pfd.save_file("Save file")
    if static.save_file_dialog is not None and static.save_file_dialog.ready():
        log_result(static.save_file_dialog.result())
        static.save_file_dialog = None

    imgui.same_line()

    if imgui.button("Select folder"):
        static.select_folder_dialog = pfd.select_folder("Select folder")
    if static.select_folder_dialog is not None and static.select_folder_dialog.ready():
        log_result(static.select_folder_dialog.result())
        static.select_folder_dialog = None

    if len(static.last_file_selection) > 0:
        imgui.text(static.last_file_selection)

    imgui.text("      ---   Notifications and messages   ---")

    # icon type
    imgui.text("Icon type")
    imgui.same_line()
    for notification_icon in (pfd.icon.info, pfd.icon.warning, pfd.icon.error):
        if imgui.radio_button(notification_icon.name, static.icon_type == notification_icon):
            static.icon_type = notification_icon
        imgui.same_line()
    imgui.new_line()

    if imgui.button("Add Notif"):
        pfd.notify("Notification title", "This is an example notification", static.icon_type)

    # messages
    imgui.same_line()
    # 1. Display the message
    if imgui.button("Add message"):
        static.message_dialog = pfd.message("Message title", "This is an example message", static.message_choice_type, static.icon_type)
    # 2. Handle the message result
    if static.message_dialog is not None and static.message_dialog.ready():
        print("msg ready: " + str(static.message_dialog.result()))
        static.message_dialog = None
    # Optional: Select the message type
    imgui.same_line()
    for choice_type in (pfd.choice.ok, pfd.choice.yes_no, pfd.choice.yes_no_cancel, pfd.choice.retry_cancel, pfd.choice.abort_retry_ignore):
        if imgui.radio_button(choice_type.name, static.message_choice_type == choice_type):
            static.message_choice_type = choice_type
        imgui.same_line()
    imgui.new_line()

    imgui.pop_id()

def demo_gui():
    demo_portable_file_dialogs()

@click.command()
@click.option('--roi', '-r', type=click.IntRange(1, 10), default=None)
@click.option(
    '--gui/--no-gui',
    default=None,
    help="Enable or disable PreviewDataWidget. Default is auto."
)
@click.argument('data_in', required=False)
def run_gui(data_in=None, gui=None, roi=None, **kwargs):
    """Open a GUI to preview data of any supported type."""
    if data_in is None:
        immapp.run(demo_gui, with_markdown=True, window_size=(1000, 1000))  # type: ignore
        if not selected:
            print('not selected')
        else:
            fpath = selected
        files = get_files(fpath)
        data = read_scan(files)
    elif _is_arraylike(data_in):
        data = data_in
    else:
        data = to_lazy_array(data_in)

    if isinstance(data, list):
        sample = data[0]
    else:
        sample = data

    if sample.ndim < 2:
        raise ValueError(f"Invalid input shape: expected >=2D, got {sample.shape}")

    nx, ny = sample.shape[-2:]
    iw = fpl.ImageWidget(
        data=data,
        histogram_widget=False,
        figure_kwargs={"size": (nx, ny)},
        graphic_kwargs={"vmin": sample.min(), "vmax": sample.max()},
        window_funcs={"t": (np.mean, 0)},
    )

    if kwargs.get("gui"):
        gui = PreviewDataWidget(iw=iw)
        iw.figure.add_gui(gui)

    iw.show()
    fpl.loop.run()

def main():
    imgui.create_context()
    imgui.style_colors_dark()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    init_fonts_and_markdown()

if __name__ == "__main__":
    main()