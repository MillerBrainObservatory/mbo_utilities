"""
Shared data model for Suite2p + Diagnostics integration.

Provides Qt signals for bidirectional synchronization between
Suite2p viewer and diagnostics widgets.
"""

import numpy as np
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal


class SharedDataModel(QObject):
    """Shared data model with Qt signals for reactive UI updates.

    This model holds Suite2p results and emits signals when data changes,
    enabling multiple widgets to stay synchronized.

    Signals
    -------
    roi_selected : int
        Emitted when a different ROI is selected.
    iscell_changed : int, bool
        Emitted when a single ROI's cell status is toggled.
        Args: (roi_index, is_cell)
    iscell_batch_changed : None
        Emitted when multiple ROIs are updated (e.g., from slider filtering).
    data_loaded : None
        Emitted when new data is loaded.
    data_saved : None
        Emitted when iscell.npy is saved to disk.
    """

    # signals
    roi_selected = pyqtSignal(int)
    iscell_changed = pyqtSignal(int, bool)
    iscell_batch_changed = pyqtSignal()
    data_loaded = pyqtSignal()
    data_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # data arrays
        self._stat = None
        self._iscell = None
        self._iscell_original = None
        self._F = None
        self._Fneu = None
        self._ops = None

        # paths
        self._plane_dir = None
        self._save_path = None

        # selection state
        self._selected_roi = 0
        self._merge_list = []  # for multi-select (shift+click)

        # computed metrics (cached)
        self._dff = None
        self._snr = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None

    # properties with change detection

    @property
    def stat(self):
        return self._stat

    @stat.setter
    def stat(self, value):
        self._stat = value

    @property
    def iscell(self):
        return self._iscell

    @iscell.setter
    def iscell(self, value):
        self._iscell = value

    @property
    def iscell_original(self):
        return self._iscell_original

    @iscell_original.setter
    def iscell_original(self, value):
        self._iscell_original = value

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def Fneu(self):
        return self._Fneu

    @Fneu.setter
    def Fneu(self, value):
        self._Fneu = value

    @property
    def ops(self):
        return self._ops

    @ops.setter
    def ops(self, value):
        self._ops = value

    @property
    def plane_dir(self):
        return self._plane_dir

    @property
    def save_path(self):
        return self._save_path

    @property
    def selected_roi(self):
        return self._selected_roi

    @selected_roi.setter
    def selected_roi(self, value):
        if self._stat is not None:
            value = max(0, min(value, len(self._stat) - 1))
        if value != self._selected_roi:
            self._selected_roi = value
            self.roi_selected.emit(value)

    @property
    def merge_list(self):
        return self._merge_list

    @property
    def n_rois(self):
        """Total number of ROIs."""
        return len(self._stat) if self._stat is not None else 0

    @property
    def n_cells(self):
        """Number of ROIs classified as cells."""
        if self._iscell is None:
            return 0
        if self._iscell.ndim == 2:
            return int(np.sum(self._iscell[:, 0] > 0.5))
        return int(np.sum(self._iscell > 0.5))

    @property
    def cell_indices(self):
        """Indices of ROIs classified as cells."""
        if self._iscell is None:
            return np.array([], dtype=int)
        if self._iscell.ndim == 2:
            return np.where(self._iscell[:, 0] > 0.5)[0]
        return np.where(self._iscell > 0.5)[0]

    @property
    def noncell_indices(self):
        """Indices of ROIs classified as non-cells."""
        if self._iscell is None:
            return np.array([], dtype=int)
        if self._iscell.ndim == 2:
            return np.where(self._iscell[:, 0] <= 0.5)[0]
        return np.where(self._iscell <= 0.5)[0]

    # computed properties

    @property
    def dff(self):
        """Delta F/F traces (computed on demand)."""
        if self._dff is None and self._F is not None:
            self._compute_dff()
        return self._dff

    @property
    def snr(self):
        """SNR values (computed on demand)."""
        if self._snr is None and self._F is not None:
            self._compute_metrics()
        return self._snr

    @property
    def skewness(self):
        """Skewness values."""
        if self._skewness is None and self._stat is not None:
            self._compute_metrics()
        return self._skewness

    @property
    def activity(self):
        """Activity values."""
        if self._activity is None and self._F is not None:
            self._compute_metrics()
        return self._activity

    @property
    def shot_noise(self):
        """Shot noise values."""
        if self._shot_noise is None and self._F is not None:
            self._compute_metrics()
        return self._shot_noise

    # methods

    def load_data(self, plane_dir):
        """Load Suite2p results from a plane directory.

        Parameters
        ----------
        plane_dir : str or Path
            Path to suite2p plane directory.
        """
        from mbo_utilities.util import load_npy

        plane_dir = Path(plane_dir)
        self._plane_dir = plane_dir

        # load ops first to get save_path
        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            ops_arr = load_npy(ops_path)
            self._ops = ops_arr.item() if ops_arr.ndim == 0 else ops_arr
            save_path = Path(self._ops.get("save_path", plane_dir))
            if not save_path.exists():
                save_path = plane_dir
        else:
            self._ops = {}
            save_path = plane_dir

        self._save_path = save_path

        # load arrays
        stat_path = save_path / "stat.npy"
        if stat_path.exists():
            self._stat = load_npy(stat_path)

        iscell_path = save_path / "iscell.npy"
        if iscell_path.exists():
            self._iscell = load_npy(iscell_path)
            self._iscell_original = self._iscell.copy()

        f_path = save_path / "F.npy"
        if f_path.exists():
            self._F = load_npy(f_path)

        fneu_path = save_path / "Fneu.npy"
        if fneu_path.exists():
            self._Fneu = load_npy(fneu_path)

        # clear cached metrics
        self._clear_cached_metrics()

        # reset selection
        self._selected_roi = 0
        self._merge_list = []

        # emit signal
        self.data_loaded.emit()

    def _clear_cached_metrics(self):
        """Clear cached computed metrics."""
        self._dff = None
        self._snr = None
        self._skewness = None
        self._activity = None
        self._shot_noise = None

    def _compute_dff(self, neuropil_coeff=0.7, baseline_percentile=20):
        """Compute delta F/F traces."""
        if self._F is None:
            return

        if self._Fneu is not None:
            F_corr = self._F - neuropil_coeff * self._Fneu
        else:
            F_corr = self._F

        baseline = np.percentile(F_corr, baseline_percentile, axis=1, keepdims=True)
        baseline = np.maximum(baseline, 1e-6)
        self._dff = (F_corr - baseline) / baseline

    def _compute_metrics(self):
        """Compute quality metrics."""
        if self._F is None or self._stat is None:
            return

        # ensure dff is computed
        if self._dff is None:
            self._compute_dff()

        n_rois = len(self._stat)

        # snr (MAD-based)
        signal = np.std(self._dff, axis=1)
        noise = np.median(np.abs(np.diff(self._dff, axis=1)), axis=1) / 0.6745
        self._snr = signal / (noise + 1e-6)

        # skewness from stat or compute
        self._skewness = np.array([s.get("skew", 0.0) for s in self._stat])

        # activity (fraction of frames > 0.5 dF/F)
        self._activity = np.array([
            np.sum(self._dff[i] > 0.5) / len(self._dff[i])
            for i in range(n_rois)
        ])

        # shot noise
        fs = self._ops.get("fs", 30.0) if self._ops else 30.0
        self._shot_noise = np.median(np.abs(np.diff(self._dff, axis=1)), axis=1) / np.sqrt(fs)

    def is_cell(self, roi_idx):
        """Check if ROI is classified as cell.

        Parameters
        ----------
        roi_idx : int
            ROI index.

        Returns
        -------
        bool
            True if classified as cell.
        """
        if self._iscell is None:
            return False
        if self._iscell.ndim == 2:
            return self._iscell[roi_idx, 0] > 0.5
        return self._iscell[roi_idx] > 0.5

    def get_iscell_prob(self, roi_idx):
        """Get iscell probability for an ROI.

        Parameters
        ----------
        roi_idx : int
            ROI index.

        Returns
        -------
        float
            Probability value (0-1).
        """
        if self._iscell is None:
            return 0.0
        if self._iscell.ndim == 2:
            return float(self._iscell[roi_idx, 0])
        return float(self._iscell[roi_idx])

    def toggle_cell(self, roi_idx):
        """Toggle cell/non-cell status for an ROI.

        Parameters
        ----------
        roi_idx : int
            ROI index to toggle.
        """
        if self._iscell is None:
            return

        was_cell = self.is_cell(roi_idx)
        new_val = 0.0 if was_cell else 1.0

        if self._iscell.ndim == 2:
            self._iscell[roi_idx, 0] = new_val
        else:
            self._iscell[roi_idx] = new_val

        print(f"[SharedDataModel] toggle_cell: roi={roi_idx}, was_cell={was_cell}, now_cell={not was_cell}")
        print(f"[SharedDataModel] emitting iscell_changed signal")
        self.iscell_changed.emit(roi_idx, not was_cell)

    def set_cell(self, roi_idx, is_cell):
        """Set cell status for an ROI.

        Parameters
        ----------
        roi_idx : int
            ROI index.
        is_cell : bool
            New cell status.
        """
        if self._iscell is None:
            return

        new_val = 1.0 if is_cell else 0.0
        changed = False

        if self._iscell.ndim == 2:
            if (self._iscell[roi_idx, 0] > 0.5) != is_cell:
                self._iscell[roi_idx, 0] = new_val
                changed = True
        else:
            if (self._iscell[roi_idx] > 0.5) != is_cell:
                self._iscell[roi_idx] = new_val
                changed = True

        if changed:
            self.iscell_changed.emit(roi_idx, is_cell)

    def set_iscell_batch(self, new_iscell):
        """Update entire iscell array (e.g., from slider filtering).

        Parameters
        ----------
        new_iscell : np.ndarray
            New iscell array.
        """
        self._iscell = new_iscell
        self.iscell_batch_changed.emit()

    def apply_filter(self, snr_min=None, shot_noise_max=None, skew_min=None, activity_min=None):
        """Apply filter thresholds to iscell.

        Only affects ROIs that were originally classified as cells.

        Parameters
        ----------
        snr_min : float, optional
            Minimum SNR threshold.
        shot_noise_max : float, optional
            Maximum shot noise threshold.
        skew_min : float, optional
            Minimum skewness threshold.
        activity_min : float, optional
            Minimum activity threshold.
        """
        if self._iscell is None or self._iscell_original is None:
            return

        # ensure metrics are computed
        _ = self.snr

        n_rois = self.n_rois
        changed = False

        for i in range(n_rois):
            # only filter originally-classified cells
            if self._iscell_original.ndim == 2:
                orig_is_cell = self._iscell_original[i, 0] > 0.5
            else:
                orig_is_cell = self._iscell_original[i] > 0.5

            if not orig_is_cell:
                continue

            passes = True
            if snr_min is not None and self._snr[i] < snr_min:
                passes = False
            if shot_noise_max is not None and self._shot_noise[i] > shot_noise_max:
                passes = False
            if skew_min is not None and self._skewness[i] < skew_min:
                passes = False
            if activity_min is not None and self._activity[i] < activity_min:
                passes = False

            new_val = 1.0 if passes else 0.0
            if self._iscell.ndim == 2:
                if self._iscell[i, 0] != new_val:
                    self._iscell[i, 0] = new_val
                    changed = True
            else:
                if self._iscell[i] != new_val:
                    self._iscell[i] = new_val
                    changed = True

        if changed:
            self.iscell_batch_changed.emit()

    def reset_filters(self):
        """Reset iscell to original values."""
        if self._iscell_original is not None:
            self._iscell = self._iscell_original.copy()
            self.iscell_batch_changed.emit()

    def save_iscell(self):
        """Save iscell.npy to disk."""
        if self._iscell is None or self._save_path is None:
            return False

        iscell_path = self._save_path / "iscell.npy"
        try:
            np.save(iscell_path, self._iscell)
            self.data_saved.emit()
            return True
        except Exception as e:
            print(f"Error saving iscell.npy: {e}")
            return False

    def add_to_merge(self, roi_idx):
        """Add ROI to merge selection list.

        Parameters
        ----------
        roi_idx : int
            ROI index to add.
        """
        if roi_idx not in self._merge_list:
            # only allow merging same type (all cells or all non-cells)
            if len(self._merge_list) == 0:
                self._merge_list.append(roi_idx)
            elif self.is_cell(roi_idx) == self.is_cell(self._merge_list[0]):
                self._merge_list.append(roi_idx)

    def remove_from_merge(self, roi_idx):
        """Remove ROI from merge selection list.

        Parameters
        ----------
        roi_idx : int
            ROI index to remove.
        """
        if roi_idx in self._merge_list:
            self._merge_list.remove(roi_idx)

    def clear_merge(self):
        """Clear merge selection list."""
        self._merge_list = []

    def set_merge(self, roi_indices):
        """Set merge selection list.

        Parameters
        ----------
        roi_indices : list
            List of ROI indices.
        """
        self._merge_list = list(roi_indices)
