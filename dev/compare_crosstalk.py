"""
compare pre/post crosstalk-subtracted data using fastplotlib.

opens side-by-side views of raw vs demixed data for visual comparison.
"""

from pathlib import Path
import mbo_utilities as mbo
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui, portable_file_dialogs as pfd
import numpy as np


def compare_crosstalk(
    raw_path: str | Path | None = None,
    demixed_path: str | Path | None = None,
    plane: int = 0,
):
    """
    open fastplotlib viewer comparing raw vs crosstalk-subtracted data.

    parameters
    ----------
    raw_path : str or Path, optional
        path to raw data (before crosstalk subtraction). if None, opens dialog.
    demixed_path : str or Path, optional
        path to demixed data (after crosstalk subtraction). if None, opens dialog.
    plane : int
        initial plane/beam to display (default 0).
    """
    # get raw path
    if raw_path is None:
        dialog = pfd.select_folder("Select RAW data folder")
        while not dialog.ready():
            pass
        result = dialog.result()
        if not result:
            print("no raw folder selected")
            return
        raw_path = result

    # get demixed path
    if demixed_path is None:
        dialog = pfd.select_folder("Select DEMIXED data folder/file")
        while not dialog.ready():
            pass
        result = dialog.result()
        if not result:
            print("no demixed folder selected")
            return
        demixed_path = result

    raw_path = Path(raw_path)
    demixed_path = Path(demixed_path)

    print(f"loading raw from: {raw_path}")
    print(f"loading demixed from: {demixed_path}")

    # load data
    raw = mbo.imread(raw_path)
    demixed = mbo.imread(demixed_path)

    print(f"raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"demixed shape: {demixed.shape}, dtype: {demixed.dtype}")

    # create viewer
    iw = fpl.ImageWidget(
        data=[raw, demixed],
        names=["Raw", "Demixed"],
        figure_shape=(1, 2),
        figure_kwargs={"size": (1400, 700)},
        histogram_widget=True,
    )

    # add controls
    widget = CrosstalkCompareWidget(iw, raw, demixed, plane)
    iw.show()
    iw.figure.add_gui(widget)

    return iw


class CrosstalkCompareWidget(EdgeWindow):
    """imgui widget for crosstalk comparison controls."""

    def __init__(self, image_widget, raw_arr, demixed_arr, initial_plane: int):
        super().__init__(
            figure=image_widget.figure,
            size=280,
            location="right",
            title="Crosstalk Compare",
        )
        self.iw = image_widget
        self.raw = raw_arr
        self.demixed = demixed_arr
        self.current_plane = initial_plane

        # data dimensions
        self.num_frames = raw_arr.shape[0]
        self.num_planes = raw_arr.shape[1] if raw_arr.ndim >= 4 else 1

        self._status = ""

    def update(self):
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(8, 6))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(6, 4))

        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.4, 0.8, 1.0, 1.0), "Crosstalk Comparison")
        imgui.separator()
        imgui.spacing()

        # data info
        imgui.text("Raw:")
        imgui.indent()
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"{self.raw.shape}")
        imgui.unindent()

        imgui.text("Demixed:")
        imgui.indent()
        imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), f"{self.demixed.shape}")
        imgui.unindent()
        imgui.spacing()

        imgui.separator()
        imgui.spacing()

        # plane/beam selector
        if self.num_planes > 1:
            imgui.text("Plane / Beam:")
            changed, new_plane = imgui.slider_int(
                "##plane", self.current_plane, 0, self.num_planes - 1
            )
            if changed:
                self.current_plane = new_plane
                self._update_indices()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # difference stats (computed on current frame)
        try:
            t_idx = self.iw.current_index.get("t", 0)
            if self.num_planes > 1:
                raw_frame = np.asarray(self.raw[t_idx, self.current_plane])
                demix_frame = np.asarray(self.demixed[t_idx, self.current_plane])
            else:
                raw_frame = np.asarray(self.raw[t_idx])
                demix_frame = np.asarray(self.demixed[t_idx])

            diff = raw_frame.astype(np.float32) - demix_frame.astype(np.float32)
            removed_pct = 100 * np.clip(diff, 0, None).sum() / (raw_frame.sum() + 1e-12)

            imgui.text("Current Frame Stats:")
            imgui.indent()
            imgui.text(f"Raw mean: {raw_frame.mean():.1f}")
            imgui.text(f"Demixed mean: {demix_frame.mean():.1f}")
            imgui.text(f"Removed: {removed_pct:.1f}%")
            imgui.unindent()
        except Exception:
            pass

        # status
        if self._status:
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), self._status)

        imgui.pop_style_var(2)

    def _update_indices(self):
        """update z index in imagewidget."""
        if "z" in self.iw.current_index:
            self.iw.current_index["z"] = self.current_plane


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else None
    demixed = sys.argv[2] if len(sys.argv) > 2 else None

    iw = compare_crosstalk(raw, demixed)
    fpl.loop.run()
