from pathlib import Path
from typing import Optional, Union, Literal
from imgui_bundle import imgui, immapp, hello_imgui, portable_file_dialogs as pfd


class SimpleSelector:
    def __init__(
        self,
        mode: Literal["folder", "files"] = "files",
        title: str = "Select",
        filters: Optional[list[str]] = None,
        start_path: Optional[Union[str, Path]] = None,
    ):
        self.mode = mode
        self.title = title
        self.filters = filters or []
        self.start_path = str(start_path) if start_path else str(Path.home())
        self.selected_path = None
        self.selected_paths = []
        self.cancelled = False

    def draw(self):
        imgui.text(self.title)
        imgui.spacing()

        if self.mode == "folder":
            # native folder dialogs don't show files - this is an OS limitation
            if imgui.button("Select Folder"):
                result = pfd.select_folder(self.start_path)
                if result:
                    self.selected_path = Path(result.result())
                else:
                    self.cancelled = True

        elif self.mode == "files":
            if imgui.button("Select File(s)"):
                result = pfd.open_file(
                    self.title,
                    self.start_path,
                    self.filters,
                    pfd.opt.multiselect
                )
                if result and result.result():
                    self.selected_paths = [Path(p) for p in result.result()]
                else:
                    self.cancelled = True

        imgui.spacing()

        if self.selected_path:
            imgui.text(f"Selected: {self.selected_path}")
        elif self.selected_paths:
            imgui.text(f"Selected {len(self.selected_paths)} file(s):")
            for p in self.selected_paths[:5]:
                imgui.text(f"  {p.name}")
            if len(self.selected_paths) > 5:
                imgui.text(f"  ... and {len(self.selected_paths) - 5} more")

        imgui.spacing()

        if self.selected_path or self.selected_paths:
            if imgui.button("Continue"):
                hello_imgui.get_runner_params().app_shall_exit = True


def select_folder(
    title: str = "Select Folder",
    start_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Select a folder using native OS dialog.
    
    Note: Native folder dialogs don't show files - this is an OS limitation.
    """
    selector = SimpleSelector("folder", title, None, start_path)

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = title
    params.app_window_params.window_geometry.size = (500, 200)
    params.callbacks.show_gui = selector.draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = False
    addons.with_implot = False

    immapp.run(runner_params=params, add_ons_params=addons)

    return selector.selected_path


def select_files(
    title: str = "Select File(s)",
    filters: Optional[list[str]] = None,
    start_path: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Select one or more files using native OS dialog.
    
    Supports multi-select - user can select single or multiple files.
    """
    selector = SimpleSelector("files", title, filters, start_path)

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = title
    params.app_window_params.window_geometry.size = (500, 300)
    params.callbacks.show_gui = selector.draw

    addons = immapp.AddOnsParams()
    addons.with_markdown = False
    addons.with_implot = False

    immapp.run(runner_params=params, add_ons_params=addons)

    return selector.selected_paths


if __name__ == "__main__":
    selected = select_files(
        title="Choose TIFF file(s)",
        filters=None,
        start_path=Path.home()
    )

    if selected:
        print(f"You selected: {selected}")
    else:
        print("No files selected")
