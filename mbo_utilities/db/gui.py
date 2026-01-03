"""
imgui-based dataset browser for mbo_db.

provides a gui for browsing, filtering, and managing indexed datasets.
"""

from pathlib import Path

from mbo_utilities import log

logger = log.get("db.gui")


class DatasetBrowser:
    """imgui dataset browser widget."""

    def __init__(self):
        self.datasets = []
        self.selected_id = None
        self.filter_pipeline = ""
        self.filter_status = ""
        self.filter_search = ""
        self.show_details = False
        self.selected_dataset = None
        self._needs_refresh = True
        self._status_options = ["", "raw", "registered", "segmented", "complete", "error", "unknown"]
        self._pipeline_options = [""]  # populated on first refresh

    def refresh(self):
        """Reload datasets from database."""
        from mbo_utilities.db.database import get_datasets, get_stats
        from mbo_utilities.db.models import DatasetStatus

        # build filter kwargs
        kwargs = {"limit": 500}

        if self.filter_pipeline:
            kwargs["pipeline"] = self.filter_pipeline

        if self.filter_status:
            kwargs["status"] = DatasetStatus(self.filter_status)

        if self.filter_search:
            kwargs["search"] = self.filter_search

        self.datasets = get_datasets(**kwargs)

        # get unique pipelines for filter dropdown
        stats = get_stats()
        pipelines = list(stats.get("by_pipeline", {}).keys())
        self._pipeline_options = ["", *sorted(pipelines)]

        self._needs_refresh = False

    def render(self):
        """Render the browser ui."""
        from imgui_bundle import imgui

        if self._needs_refresh:
            self.refresh()

        # filter bar
        imgui.text("Filters:")
        imgui.same_line()

        # pipeline dropdown
        imgui.set_next_item_width(120)
        changed, idx = imgui.combo(
            "##pipeline",
            self._pipeline_options.index(self.filter_pipeline) if self.filter_pipeline in self._pipeline_options else 0,
            self._pipeline_options,
        )
        if changed:
            self.filter_pipeline = self._pipeline_options[idx]
            self._needs_refresh = True

        imgui.same_line()

        # status dropdown
        imgui.set_next_item_width(100)
        changed, idx = imgui.combo(
            "##status",
            self._status_options.index(self.filter_status) if self.filter_status in self._status_options else 0,
            self._status_options,
        )
        if changed:
            self.filter_status = self._status_options[idx]
            self._needs_refresh = True

        imgui.same_line()

        # search box
        imgui.set_next_item_width(200)
        changed, self.filter_search = imgui.input_text("##search", self.filter_search, 256)
        if changed:
            self._needs_refresh = True

        imgui.same_line()
        if imgui.button("Refresh"):
            self._needs_refresh = True

        imgui.separator()

        # dataset table
        avail = imgui.get_content_region_avail()
        table_height = avail.y - 30 if self.show_details else avail.y

        if imgui.begin_child("##table", imgui.ImVec2(0, table_height)):
            flags = (
                imgui.TableFlags_.borders_h
                | imgui.TableFlags_.row_bg
                | imgui.TableFlags_.resizable
                | imgui.TableFlags_.scroll_y
            )

            if imgui.begin_table("datasets", 5, flags):
                imgui.table_setup_column("ID", imgui.TableColumnFlags_.width_fixed, 40)
                imgui.table_setup_column("Pipeline", imgui.TableColumnFlags_.width_fixed, 80)
                imgui.table_setup_column("Status", imgui.TableColumnFlags_.width_fixed, 80)
                imgui.table_setup_column("Size", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_setup_column("Name")
                imgui.table_headers_row()

                for ds in self.datasets:
                    imgui.table_next_row()

                    # id column
                    imgui.table_next_column()
                    is_selected = self.selected_id == ds.id
                    if imgui.selectable(
                        str(ds.id),
                        is_selected,
                        imgui.SelectableFlags_.span_all_columns,
                    )[0]:
                        self.selected_id = ds.id
                        self.selected_dataset = ds
                        self.show_details = True

                    # pipeline column
                    imgui.table_next_column()
                    imgui.text(ds.pipeline or "-")

                    # status column
                    imgui.table_next_column()
                    status_colors = {
                        "raw": (0.7, 0.7, 0.3, 1.0),
                        "registered": (0.3, 0.7, 0.7, 1.0),
                        "segmented": (0.3, 0.7, 0.3, 1.0),
                        "complete": (0.2, 0.8, 0.2, 1.0),
                        "error": (0.8, 0.3, 0.3, 1.0),
                        "unknown": (0.5, 0.5, 0.5, 1.0),
                    }
                    color = status_colors.get(ds.status.value, (0.5, 0.5, 0.5, 1.0))
                    imgui.text_colored(imgui.ImVec4(*color), ds.status.value)

                    # size column
                    imgui.table_next_column()
                    imgui.text(ds.size_human)

                    # name column
                    imgui.table_next_column()
                    imgui.text(ds.name)

                imgui.end_table()

        imgui.end_child()

        # details panel (bottom)
        if self.show_details and self.selected_dataset:
            imgui.separator()
            ds = self.selected_dataset
            imgui.text(f"Path: {ds.path}")
            imgui.same_line()
            if imgui.small_button("Open"):
                self._open_dataset(ds)
            imgui.same_line()
            if imgui.small_button("Show Folder"):
                self._show_in_explorer(ds.path)

    def _open_dataset(self, dataset):
        """Open dataset in the mbo viewer."""
        try:
            from mbo_utilities.gui.run_gui import run_gui
            run_gui(dataset.path)
        except Exception as e:
            logger.exception(f"failed to open dataset: {e}")

    def _show_in_explorer(self, path: str):
        """Open containing folder in system file browser."""
        import subprocess
        import sys

        p = Path(path)
        folder = p.parent if p.is_file() else p

        if sys.platform == "win32":
            subprocess.run(["explorer", str(folder)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)


def launch_browser():
    """Launch the dataset browser gui."""
    from imgui_bundle import immapp, hello_imgui
    from mbo_utilities.gui._setup import get_default_ini_path
    from mbo_utilities.db.database import init_db

    # ensure database is initialized
    init_db()

    browser = DatasetBrowser()

    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "MBO Dataset Browser"
    params.app_window_params.window_geometry.size = (900, 600)
    params.ini_filename = get_default_ini_path("dataset_browser")
    params.callbacks.show_gui = browser.render

    addons = immapp.AddOnsParams()
    addons.with_markdown = True
    addons.with_implot = False
    addons.with_implot3d = False

    immapp.run(runner_params=params, add_ons_params=addons)
