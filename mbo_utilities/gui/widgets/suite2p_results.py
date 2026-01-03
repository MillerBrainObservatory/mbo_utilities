from pathlib import Path
import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd, implot

from mbo_utilities.preferences import get_last_dir, set_last_dir
import contextlib


class Suite2pResultsViewer:
    def __init__(self):
        self.ops_path = None
        self.stat = None
        self.F = None
        self.Fneu = None
        self.spks = None
        self.iscell = None
        self.ops = None
        self.masks = None
        self.image = None
        self.selected_cell = 0
        self.show_trace = False
        self.format = None  # "suite2p" or "cellpose"

    def load_ops_file(self, ops_path):
        """Load results from Suite2p or Cellpose format."""
        from lbm_suite2p_python.conversion import detect_format, masks_to_stat

        ops_path = Path(ops_path)
        self.ops_path = ops_path

        # Determine the directory to load from
        if ops_path.suffix == ".npy":
            load_dir = ops_path.parent
        else:
            load_dir = ops_path

        # Detect format and load appropriately
        self.format = detect_format(load_dir)

        if self.format in ("suite2p", "suite2p_minimal"):
            self._load_suite2p(load_dir)
        elif self.format == "cellpose":
            self._load_cellpose(load_dir)
        else:
            # Unknown format - try Suite2p first
            try:
                self._load_suite2p(load_dir)
            except Exception:
                self._load_cellpose(load_dir)

        # Set selected cell to 0 if we have data
        n_rois = len(self.stat) if self.stat is not None else 0
        if n_rois == 0 and self.masks is not None:
            n_rois = int(self.masks.max())
        if n_rois > 0:
            self.selected_cell = 0

    def _load_suite2p(self, load_dir):
        """Load Suite2p format results."""
        from lbm_suite2p_python.postprocessing import load_planar_results

        results = load_planar_results(load_dir)
        self.stat = results.get("stat")
        self.F = results.get("F")
        self.Fneu = results.get("Fneu")
        self.spks = results.get("spks")
        self.iscell = results.get("iscell")
        self.ops = results.get("ops")

        # Generate masks from stat if we have ops
        if self.stat is not None and self.ops is not None:
            from lbm_suite2p_python.conversion import stat_to_masks
            Ly, Lx = self.ops.get("Ly", 512), self.ops.get("Lx", 512)
            self.masks = stat_to_masks(self.stat, (Ly, Lx))
            self.image = self.ops.get("max_proj") or self.ops.get("meanImg")

    def _load_cellpose(self, load_dir):
        """Load Cellpose format results."""
        from lbm_suite2p_python.cellpose import load_seg_file, masks_to_stat

        load_dir = Path(load_dir)

        # Find _seg.npy file
        seg_files = list(load_dir.glob("*_seg.npy"))
        if seg_files:
            data = load_seg_file(seg_files[0])
            self.masks = data.get("masks")
            self.image = data.get("img")
            if self.masks is not None:
                self.stat = masks_to_stat(self.masks, self.image)
        elif (load_dir / "masks.npy").exists():
            self.masks = np.load(load_dir / "masks.npy")
            if self.masks is not None:
                self.stat = masks_to_stat(self.masks)

        # Load iscell if available
        iscell_file = load_dir / "iscell.npy"
        if iscell_file.exists():
            self.iscell = np.load(iscell_file)
        elif self.stat is not None:
            # Default: all cells accepted
            self.iscell = np.ones((len(self.stat), 2), dtype=np.float32)

        # Traces not available in Cellpose-only format
        self.F = None
        self.Fneu = None
        self.spks = None
        self.ops = None

    def draw(self):
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

        imgui.spacing()

        imgui.text("Segmentation Results Viewer")
        imgui.separator()
        imgui.spacing()

        if imgui.button("Select Results File"):
            default_dir = str(get_last_dir("suite2p_ops") or Path.home())
            result = pfd.open_file(
                "Select ops.npy or _seg.npy",
                default_dir,
                ["*.npy"],
            )
            if result and result.result():
                selected = result.result()[0]
                set_last_dir("suite2p_ops", selected)
                with contextlib.suppress(Exception):
                    self.load_ops_file(selected)

        imgui.same_line()
        if imgui.button("Open in Cellpose GUI"):
            if self.ops_path:
                with contextlib.suppress(Exception):
                    from lbm_suite2p_python.cellpose import open_in_gui
                    open_in_gui(self.ops_path.parent)

        imgui.spacing()

        if self.ops_path:
            imgui.text(f"Loaded: {self.ops_path.name}")
            if self.format:
                imgui.same_line()
                imgui.text_disabled(f"({self.format})")
            imgui.spacing()

            # Get number of cells from stat or masks
            num_cells = 0
            if self.stat is not None:
                num_cells = len(self.stat)
            elif self.masks is not None:
                num_cells = int(self.masks.max())

            if num_cells > 0:
                imgui.text(f"Total cells: {num_cells}")
                imgui.spacing()

                imgui.set_next_item_width(200)
                _changed, self.selected_cell = imgui.slider_int(
                    "Cell",
                    self.selected_cell,
                    0,
                    num_cells - 1
                )

                imgui.spacing()

                # Show cell probability if available
                if self.iscell is not None and self.selected_cell < len(self.iscell):
                    is_cell_prob = self.iscell[self.selected_cell]
                    if hasattr(is_cell_prob, "__len__"):
                        # Suite2p format: [is_cell, probability]
                        imgui.text(f"Cell probability: {is_cell_prob[1]:.3f}")
                    else:
                        imgui.text(f"Cell probability: {is_cell_prob:.3f}")

                # Show trace controls only if traces are available
                if self.F is not None and len(self.F) > 0:
                    imgui.spacing()
                    _, self.show_trace = imgui.checkbox("Show Trace", self.show_trace)

                    if self.show_trace and self.selected_cell < len(self.F):
                        imgui.spacing()
                        trace_data = self.F[self.selected_cell]
                        xs = np.arange(len(trace_data), dtype=np.float64)
                        ys = trace_data.astype(np.float64)

                        if implot.begin_plot(f"Cell {self.selected_cell} Trace", imgui.ImVec2(-1, 300)):
                            try:
                                implot.setup_axes("Frame", "Fluorescence")
                                implot.plot_line("F", xs, ys)
                            finally:
                                implot.end_plot()
                else:
                    imgui.spacing()
                    imgui.text_disabled("No traces available (Cellpose-only results)")
            else:
                imgui.text_disabled("No cells found in results")
        else:
            imgui.text("No file loaded")

        imgui.pop_style_var(2)


def draw_tab_suite2p_results(parent):
    if not hasattr(parent, "_suite2p_results_viewer"):
        parent._suite2p_results_viewer = Suite2pResultsViewer()

    try:
        parent._suite2p_results_viewer.draw()
    except Exception as e:
        imgui.text_colored(imgui.ImVec4(1.0, 0.0, 0.0, 1.0), f"Error: {e}")
        import traceback
        traceback.print_exc()
