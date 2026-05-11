"""Repro multi-ROI + PreviewDataWidget."""

import os

if "RENDERCANVAS_BACKEND" not in os.environ:
    try:
        import PyQt6  # noqa
        os.environ["RENDERCANVAS_BACKEND"] = "qt"
    except ImportError:
        pass

import numpy as np
import fastplotlib as fpl
from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget


def main():
    n_rois = 4
    figure_kwargs = {"size": (1200, 800)}
    ref_ranges = {"t": (0, 30, 1)}
    ndw = fpl.NDWidget(
        ref_ranges=ref_ranges,
        shape=(1, n_rois),
        controller_ids="sync",
        names=[[f"ROI {i+1}" for i in range(n_rois)]],
        **figure_kwargs,
    )
    for col in range(n_rois):
        data = np.random.rand(30, 64, 64).astype(np.float32)
        nd = ndw[0, col].add_nd_image(
            data=data,
            dims=("t", "y", "x"),
            spatial_dims=("y", "x"),
            compute_histogram=True,
        )
        nd.graphic.cmap = "gnuplot2"
    ndw.show()
    gui = PreviewDataWidget(iw=ndw, fpath=None, size=300)
    ndw.figure.add_gui(gui)
    fpl.loop.run()


if __name__ == "__main__":
    main()
