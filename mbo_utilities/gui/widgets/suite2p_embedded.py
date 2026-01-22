"""
Embedded Suite2p viewer widget.

Provides a dual-panel ROI viewer (cells/non-cells) with reactive
updates via SharedDataModel signals.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class Suite2pEmbedded(QWidget):
    """Embedded Suite2p-style dual-panel ROI viewer.

    Displays cells in left panel, non-cells in right panel.
    Supports click-to-select and right-click-to-toggle.

    Parameters
    ----------
    model : SharedDataModel
        Shared data model for synchronization.
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    cell_clicked : int, bool
        Emitted when a cell is clicked. Args: (roi_idx, is_right_click)
    """

    cell_clicked = pyqtSignal(int, bool)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model

        # visualization state (suite2p-style)
        self.Ly = 512
        self.Lx = 512
        self.rois = {}  # iROI, Lam, LamNorm, Sroi
        self.colors = {}  # RGB, cols

        self._setup_ui()
        self._connect_model()

    def _setup_ui(self):
        """Create the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # header with cell counts
        header = QHBoxLayout()
        self.cells_label = QLabel("Cells: 0")
        self.cells_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.cells_label.setStyleSheet("color: #00ff00;")

        self.noncells_label = QLabel("Non-cells: 0")
        self.noncells_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.noncells_label.setStyleSheet("color: #ff6666;")

        header.addWidget(self.cells_label)
        header.addStretch()
        header.addWidget(self.noncells_label)
        layout.addLayout(header)

        # graphics layout for mask panels
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground('k')

        # cells panel (left)
        self.p1 = CellViewBox(self, name="cells", panel_idx=0)
        self.graphics.addItem(self.p1, 0, 0)
        self.view1 = pg.ImageItem()
        self.color1 = pg.ImageItem()
        self.p1.addItem(self.view1)
        self.p1.addItem(self.color1)
        self.view1.setZValue(0)
        self.color1.setZValue(1)

        # non-cells panel (right)
        self.p2 = CellViewBox(self, name="noncells", panel_idx=1)
        self.graphics.addItem(self.p2, 0, 1)
        self.view2 = pg.ImageItem()
        self.color2 = pg.ImageItem()
        self.p2.addItem(self.view2)
        self.p2.addItem(self.color2)
        self.view2.setZValue(0)
        self.color2.setZValue(1)

        # link views for synchronized pan/zoom
        self.p2.setXLink(self.p1)
        self.p2.setYLink(self.p1)

        layout.addWidget(self.graphics, stretch=3)

        # trace plot (bottom)
        self.trace_widget = pg.PlotWidget()
        self.trace_widget.setBackground('k')
        self.trace_widget.setLabel('left', 'dF/F')
        self.trace_widget.setLabel('bottom', 'Frame')
        self.trace_plot = self.trace_widget.plot(pen='w')
        layout.addWidget(self.trace_widget, stretch=1)

    def _connect_model(self):
        """Connect to SharedDataModel signals."""
        print("[Suite2pEmbedded] connecting model signals")
        self.model.roi_selected.connect(self._on_roi_selected)
        self.model.iscell_changed.connect(self._on_iscell_changed)
        self.model.iscell_batch_changed.connect(self._on_iscell_batch_changed)
        self.model.data_loaded.connect(self._on_data_loaded)
        print("[Suite2pEmbedded] signals connected")

    def _on_data_loaded(self):
        """Handle new data load."""
        print("[Suite2pEmbedded] _on_data_loaded called")
        self._init_views()
        self._init_masks()
        self._update_labels()
        self._update_trace()
        print("[Suite2pEmbedded] _on_data_loaded done")

    def _on_roi_selected(self, roi_idx):
        """Handle ROI selection change."""
        self._update_trace()

    def _on_iscell_changed(self, roi_idx, is_cell):
        """Handle single cell toggle."""
        print(f"[_on_iscell_changed] roi_idx={roi_idx}, is_cell={is_cell}")
        # full redraw is simpler and more reliable than partial updates
        self._init_masks()
        self._update_labels()
        print("[_on_iscell_changed] done")

    def _on_iscell_batch_changed(self):
        """Handle batch iscell update (from slider filtering)."""
        self._init_masks()
        self._update_labels()

    def _init_views(self):
        """Initialize background views from ops."""
        if self.model.ops is None:
            return

        ops = self.model.ops
        self.Ly = ops.get("Ly", 512)
        self.Lx = ops.get("Lx", 512)

        # use meanImg as background
        if "meanImg" in ops:
            mimg = ops["meanImg"]
            mimg1 = np.percentile(mimg, 1)
            mimg99 = np.percentile(mimg, 99)
            mimg = (mimg - mimg1) / (mimg99 - mimg1 + 1e-6)
            mimg = np.clip(mimg, 0, 1) * 255
            mimg = mimg.astype(np.uint8)
        else:
            mimg = np.zeros((self.Ly, self.Lx), dtype=np.uint8)

        # convert to RGB
        view_rgb = np.stack([mimg, mimg, mimg], axis=-1)

        self.view1.setImage(view_rgb)
        self.view2.setImage(view_rgb)

        # set view ranges
        self.p1.setRange(xRange=[0, self.Lx], yRange=[0, self.Ly], padding=0)

    def _init_masks(self):
        """Initialize ROI masks using suite2p-style layered approach."""
        if self.model.stat is None:
            return

        stat = self.model.stat
        ncells = len(stat)

        # suite2p uses 3 layers for overlapping ROIs
        # iROI: pixel -> ROI index (-1 if none)
        # Lam: pixel weights
        # Sroi: binary mask of ROI presence
        self.rois["iROI"] = -np.ones((2, 3, self.Ly, self.Lx), dtype=np.int32)
        self.rois["Lam"] = np.zeros((2, 3, self.Ly, self.Lx), dtype=np.float32)
        self.rois["Sroi"] = np.zeros((2, self.Ly, self.Lx), dtype=bool)
        self.rois["LamNorm"] = np.zeros((2, self.Ly, self.Lx), dtype=np.float32)

        # generate random colors
        np.random.seed(42)
        self.colors["cols"] = np.random.randint(50, 255, size=(ncells, 3), dtype=np.uint8)

        # RGBA output
        self.colors["RGB"] = np.zeros((2, self.Ly, self.Lx, 4), dtype=np.uint8)

        # track total lam for normalization
        LamAll = np.zeros((self.Ly, self.Lx), dtype=np.float32)

        # process ROIs in reverse order (so later ROIs are on top)
        for n in range(ncells - 1, -1, -1):
            ypix = stat[n].get("ypix", np.array([]))
            xpix = stat[n].get("xpix", np.array([]))

            if len(ypix) == 0:
                continue

            # clip to bounds
            valid = (ypix >= 0) & (ypix < self.Ly) & (xpix >= 0) & (xpix < self.Lx)
            ypix = ypix[valid]
            xpix = xpix[valid]

            if len(ypix) == 0:
                continue

            # get lam weights
            lam = stat[n].get("lam", np.ones(len(ypix)))
            if len(lam) > len(ypix):
                lam = lam[valid]
            lam = lam / (lam.sum() + 1e-10)

            # panel: 0 = cells, 1 = non-cells
            is_cell = self.model.is_cell(n)
            i = 0 if is_cell else 1

            # push down existing layers
            self.rois["iROI"][i, 2, ypix, xpix] = self.rois["iROI"][i, 1, ypix, xpix]
            self.rois["iROI"][i, 1, ypix, xpix] = self.rois["iROI"][i, 0, ypix, xpix]
            self.rois["iROI"][i, 0, ypix, xpix] = n

            self.rois["Lam"][i, 2, ypix, xpix] = self.rois["Lam"][i, 1, ypix, xpix]
            self.rois["Lam"][i, 1, ypix, xpix] = self.rois["Lam"][i, 0, ypix, xpix]
            self.rois["Lam"][i, 0, ypix, xpix] = lam

            self.rois["Sroi"][i, ypix, xpix] = True
            LamAll[ypix, xpix] = lam

        # compute normalization
        lam_mean = LamAll[LamAll > 1e-10].mean() if (LamAll > 1e-10).any() else 1.0
        self.rois["LamMean"] = lam_mean
        self.rois["LamNorm"] = np.clip(0.75 * self.rois["Lam"][:, 0] / lam_mean, 0, 1)

        # draw RGB masks
        self._draw_all_masks()

    def _draw_all_masks(self):
        """Draw RGB masks for both panels."""
        for i in range(2):
            # get ROI indices for top layer
            roi_indices = self.rois["iROI"][i, 0]

            # create RGB from colors
            rgb = np.zeros((self.Ly, self.Lx, 3), dtype=np.uint8)
            valid_mask = roi_indices >= 0

            if valid_mask.any():
                rgb[valid_mask] = self.colors["cols"][roi_indices[valid_mask]]

            # alpha from LamNorm
            alpha = (self.rois["LamNorm"][i] * 200).astype(np.uint8)

            # combine
            self.colors["RGB"][i, :, :, :3] = rgb
            self.colors["RGB"][i, :, :, 3] = alpha

        # update images (use copy to force pyqtgraph refresh)
        self.color1.setImage(self.colors["RGB"][0].copy())
        self.color2.setImage(self.colors["RGB"][1].copy())

    def _flip_roi(self, roi_idx, to_cells):
        """Flip ROI between panels (like suite2p's flip_roi)."""
        if self.model.stat is None:
            return

        if "iROI" not in self.rois or "LamMean" not in self.rois:
            return

        stat = self.model.stat
        if roi_idx >= len(stat):
            return

        ypix = stat[roi_idx].get("ypix", np.array([]))
        xpix = stat[roi_idx].get("xpix", np.array([]))

        if len(ypix) == 0:
            return

        # clip to bounds
        valid = (ypix >= 0) & (ypix < self.Ly) & (xpix >= 0) & (xpix < self.Lx)
        ypix = ypix[valid]
        xpix = xpix[valid]

        if len(ypix) == 0:
            return

        lam = stat[roi_idx].get("lam", np.ones(len(ypix)))
        if len(lam) > len(ypix):
            lam = lam[valid]
        lam = lam / (lam.sum() + 1e-10)

        # source and destination panels
        i_dst = 0 if to_cells else 1  # destination
        i_src = 1 - i_dst  # source

        # remove from source panel
        self._remove_roi_from_panel(roi_idx, i_src, ypix, xpix)

        # add to destination panel
        self._add_roi_to_panel(roi_idx, i_dst, ypix, xpix, lam)

        # redraw affected pixels
        self._redraw_pixels(ypix, xpix)

    def _remove_roi_from_panel(self, roi_idx, panel, ypix, xpix):
        """Remove ROI from panel and push up layers."""
        # for each pixel, find which layer contains this ROI and shift up
        for idx in range(len(ypix)):
            yp, xp = ypix[idx], xpix[idx]

            # find which layer has this ROI
            for layer in range(3):
                if self.rois["iROI"][panel, layer, yp, xp] == roi_idx:
                    # shift all subsequent layers up
                    for l in range(layer, 2):
                        self.rois["iROI"][panel, l, yp, xp] = self.rois["iROI"][panel, l + 1, yp, xp]
                        self.rois["Lam"][panel, l, yp, xp] = self.rois["Lam"][panel, l + 1, yp, xp]
                    # clear bottom layer
                    self.rois["iROI"][panel, 2, yp, xp] = -1
                    self.rois["Lam"][panel, 2, yp, xp] = 0
                    break  # roi found and removed, done with this pixel

        # update Sroi
        self.rois["Sroi"][panel, ypix, xpix] = self.rois["iROI"][panel, 0, ypix, xpix] >= 0

        # update LamNorm
        self.rois["LamNorm"][panel, ypix, xpix] = np.clip(
            0.75 * self.rois["Lam"][panel, 0, ypix, xpix] / self.rois["LamMean"], 0, 1
        )

    def _add_roi_to_panel(self, roi_idx, panel, ypix, xpix, lam):
        """Add ROI to panel on top and push down layers."""
        # push down layers
        self.rois["iROI"][panel, 2, ypix, xpix] = self.rois["iROI"][panel, 1, ypix, xpix]
        self.rois["iROI"][panel, 1, ypix, xpix] = self.rois["iROI"][panel, 0, ypix, xpix]
        self.rois["iROI"][panel, 0, ypix, xpix] = roi_idx

        self.rois["Lam"][panel, 2, ypix, xpix] = self.rois["Lam"][panel, 1, ypix, xpix]
        self.rois["Lam"][panel, 1, ypix, xpix] = self.rois["Lam"][panel, 0, ypix, xpix]
        self.rois["Lam"][panel, 0, ypix, xpix] = lam

        # update Sroi
        self.rois["Sroi"][panel, ypix, xpix] = True

        # update LamNorm
        self.rois["LamNorm"][panel, ypix, xpix] = np.clip(
            0.75 * self.rois["Lam"][panel, 0, ypix, xpix] / self.rois["LamMean"], 0, 1
        )

    def _redraw_pixels(self, ypix, xpix):
        """Redraw RGB for affected pixels."""
        for i in range(2):
            roi_indices = self.rois["iROI"][i, 0, ypix, xpix]
            valid = roi_indices >= 0

            # clear pixels first
            self.colors["RGB"][i, ypix, xpix, :] = 0

            if valid.any():
                yv = ypix[valid]
                xv = xpix[valid]
                ri = roi_indices[valid]
                self.colors["RGB"][i, yv, xv, :3] = self.colors["cols"][ri]
                self.colors["RGB"][i, yv, xv, 3] = (self.rois["LamNorm"][i, yv, xv] * 200).astype(np.uint8)

        # force full image update (setImage with new array copy)
        self.color1.setImage(self.colors["RGB"][0].copy())
        self.color2.setImage(self.colors["RGB"][1].copy())

    def _update_trace(self):
        """Update the dF/F trace plot for selected ROI."""
        roi_idx = self.model.selected_roi
        dff = self.model.dff

        if dff is None or roi_idx >= len(dff):
            self.trace_plot.setData([], [])
            return

        trace = dff[roi_idx]
        xs = np.arange(len(trace))
        self.trace_plot.setData(xs, trace)

        # update title
        is_cell = self.model.is_cell(roi_idx)
        status = "Cell" if is_cell else "Non-cell"
        self.trace_widget.setTitle(f"ROI {roi_idx} ({status})")

    def _update_labels(self):
        """Update cell count labels."""
        n_cells = self.model.n_cells
        n_total = self.model.n_rois
        n_noncells = n_total - n_cells

        self.cells_label.setText(f"Cells: {n_cells}")
        self.noncells_label.setText(f"Non-cells: {n_noncells}")

    def handle_cell_click(self, panel_idx, x, y, is_right_click):
        """Handle click on a cell (called by ViewBox).

        Parameters
        ----------
        panel_idx : int
            Panel index (0=cells, 1=noncells).
        x : int
            X pixel coordinate.
        y : int
            Y pixel coordinate.
        is_right_click : bool
            Whether this was a right click.
        """
        print(f"[click] panel={panel_idx}, x={x}, y={y}, right={is_right_click}")

        if "iROI" not in self.rois:
            print("[click] no iROI")
            return

        # get ROI at this position (top layer)
        roi_idx = self.rois["iROI"][panel_idx, 0, y, x]
        print(f"[click] roi_idx={roi_idx}")

        if roi_idx < 0:
            return

        # select the ROI
        self.model.selected_roi = roi_idx

        if is_right_click:
            # toggle cell status
            print(f"[click] calling toggle_cell({roi_idx})")
            self.model.toggle_cell(roi_idx)
            print(f"[click] toggle_cell returned")

        self.cell_clicked.emit(roi_idx, is_right_click)


class CellViewBox(pg.ViewBox):
    """ViewBox that handles cell clicks and reports to parent."""

    def __init__(self, parent_widget, name="", panel_idx=0):
        super().__init__(invertY=True, lockAspect=True)
        self.parent_widget = parent_widget
        self.name = name
        self.panel_idx = panel_idx
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):
        """Handle mouse click events."""
        # map from scene to view coordinates
        pos = self.mapSceneToView(ev.scenePos())
        x = int(pos.x())
        y = int(pos.y())

        # clamp to valid range (ignore clicks outside image bounds)
        Lx = self.parent_widget.Lx
        Ly = self.parent_widget.Ly
        if x < 0 or x >= Lx or y < 0 or y >= Ly:
            ev.ignore()
            return

        is_right = ev.button() == Qt.MouseButton.RightButton

        self.parent_widget.handle_cell_click(self.panel_idx, x, y, is_right)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        """Handle double click to reset zoom."""
        Lx = self.parent_widget.Lx
        Ly = self.parent_widget.Ly
        self.setRange(xRange=[0, Lx], yRange=[0, Ly], padding=0)
        ev.accept()
