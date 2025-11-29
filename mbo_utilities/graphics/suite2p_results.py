import sys
from pathlib import Path
import numpy as np
from imgui_bundle import imgui, portable_file_dialogs as pfd, implot


class Suite2pResultsViewer:
    def __init__(self):
        self.ops_path = None
        self.stat = None
        self.F = None
        self.Fneu = None
        self.spks = None
        self.iscell = None
        self.ops = None
        self.selected_cell = 0
        self.show_trace = False

    def load_ops_file(self, ops_path):
        from lbm_suite2p_python import load_planar_results

        ops_path = Path(ops_path)
        self.ops_path = ops_path

        results = load_planar_results(ops_path.parent)

        self.stat = results['stat']
        self.F = results['F']
        self.Fneu = results['Fneu']
        self.spks = results['spks']
        self.iscell = results['iscell']
        self.ops = results['ops']

        if self.F is not None and len(self.F) > 0:
            self.selected_cell = 0

    def draw(self):
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8, 8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(4, 3))

        imgui.spacing()

        imgui.text("Suite2p Results Viewer")
        imgui.separator()
        imgui.spacing()

        if imgui.button("Select ops.npy File"):
            result = pfd.open_file(
                "Select ops.npy",
                str(Path.home())
            )
            if result and result.result():
                selected = result.result()[0]
                try:
                    self.load_ops_file(selected)
                except Exception as e:
                    print(f"Error loading ops.npy: {e}")

        imgui.spacing()

        if self.ops_path:
            imgui.text(f"Loaded: {self.ops_path.name}")
            imgui.spacing()

            if self.F is not None and len(self.F) > 0:
                num_cells = len(self.F)
                imgui.text(f"Total cells: {num_cells}")
                imgui.spacing()

                imgui.set_next_item_width(200)
                changed, self.selected_cell = imgui.slider_int(
                    "Cell",
                    self.selected_cell,
                    0,
                    num_cells - 1
                )

                imgui.spacing()

                _, self.show_trace = imgui.checkbox("Show Trace", self.show_trace)

                imgui.spacing()

                if self.iscell is not None and self.selected_cell < len(self.iscell):
                    is_cell_prob = self.iscell[self.selected_cell]
                    imgui.text(f"Cell probability: {is_cell_prob:.3f}")

                imgui.spacing()

                if self.show_trace:
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
            imgui.text("No file loaded")

        imgui.pop_style_var(2)


def draw_tab_suite2p_results(parent):
    if not hasattr(parent, '_suite2p_results_viewer'):
        parent._suite2p_results_viewer = Suite2pResultsViewer()

    try:
        parent._suite2p_results_viewer.draw()
    except Exception as e:
        imgui.text_colored(imgui.ImVec4(1.0, 0.0, 0.0, 1.0), f"Error: {e}")
        import traceback
        print(f"Suite2p Results Viewer Error: {e}")
        traceback.print_exc()
