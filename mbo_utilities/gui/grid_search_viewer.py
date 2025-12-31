"""Grid search results viewer for Suite2p parameter tuning.

Opens two suite2p GUI windows side-by-side for comparing parameter combinations.
"""

import numpy as np
from pathlib import Path
from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.util import load_npy


def _load_stats(plane_dir: Path) -> dict:
    """Load basic stats from a suite2p plane directory."""
    stats = {"n_cells": 0, "n_not_cells": 0, "mean_snr": 0.0}

    iscell_path = plane_dir / "iscell.npy"
    if iscell_path.exists():
        iscell = load_npy(iscell_path)
        if iscell.ndim == 2:
            stats["n_cells"] = int(np.sum(iscell[:, 0] > 0.5))
        else:
            stats["n_cells"] = int(np.sum(iscell > 0.5))
        stats["n_not_cells"] = len(iscell) - stats["n_cells"]

        # calculate SNR if F.npy exists
        f_path = plane_dir / "F.npy"
        if f_path.exists():
            F = load_npy(f_path)
            cell_mask = iscell[:, 0] > 0.5 if iscell.ndim == 2 else iscell > 0.5
            if np.any(cell_mask):
                F_cells = F[cell_mask]
                noise = np.std(F_cells, axis=1)
                signal = np.max(F_cells, axis=1) - np.min(F_cells, axis=1)
                snr = np.where(noise > 0, signal / noise, 0)
                stats["mean_snr"] = float(np.mean(snr))

    return stats


class GridSearchViewer:
    """Viewer for comparing grid search parameter combinations using suite2p GUIs."""

    def __init__(self):
        self.results_path = None
        self.param_combos = []
        self.loaded = False
        self._file_dialog = None

        # two suite2p windows for side-by-side comparison
        self._suite2p_left = None
        self._suite2p_right = None
        self._left_idx = 0
        self._right_idx = 1

        # stats cache
        self._stats_cache = {}

        # completeness cache (files required for suite2p GUI)
        self._complete_cache = {}

        # error message state (for showing in UI)
        self._error_message = None

    def load_results(self, path: Path):
        """Load grid search results from directory.

        Can handle two cases:
        1. Parent folder containing multiple parameter combination subfolders
        2. Single parameter combination folder (will use parent to find siblings)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Check if this is a single param combo folder (has suite2p results inside)
        # or the parent grid_search folder
        plane_dir = self._find_plane_dir(path)
        if plane_dir is not None:
            # User selected a single parameter combination folder
            # Use its parent as the results path to find sibling combinations
            self.results_path = path.parent
            self.param_combos = sorted([d for d in self.results_path.iterdir() if d.is_dir()])
            # Find the index of the selected folder
            try:
                selected_idx = next(i for i, c in enumerate(self.param_combos) if c.name == path.name)
            except StopIteration:
                selected_idx = 0
            self._left_idx = selected_idx
            self._right_idx = min(selected_idx + 1, len(self.param_combos) - 1) if len(self.param_combos) > 1 else 0
        else:
            # User selected the parent folder containing param combinations
            self.results_path = path
            self.param_combos = sorted([d for d in path.iterdir() if d.is_dir()])
            self._left_idx = 0
            self._right_idx = min(1, len(self.param_combos) - 1)

        if not self.param_combos:
            raise ValueError(f"No parameter combination folders found in {path}")

        self._stats_cache = {}
        self._complete_cache = {}
        self.loaded = True

    def _find_plane_dir(self, combo_dir: Path) -> Path | None:
        """Find the plane directory containing suite2p results."""
        candidates = [
            combo_dir / "suite2p" / "plane0",
            combo_dir / "plane0",
            combo_dir,
        ]
        for candidate in candidates:
            if (candidate / "stat.npy").exists():
                return candidate
        return None

    def _check_suite2p_complete(self, plane_dir: Path) -> tuple[bool, list[str]]:
        """Check if a plane directory has all files needed for suite2p GUI.

        Returns
        -------
        tuple[bool, list[str]]
            (is_complete, missing_files)
        """
        import os
        # All files required by suite2p GUI's load_files function
        required_files = ["stat.npy", "ops.npy", "F.npy", "Fneu.npy", "spks.npy"]
        # Use os.path.exists for consistent behavior with UNC paths
        missing = [f for f in required_files if not os.path.exists(os.path.join(str(plane_dir), f))]
        return len(missing) == 0, missing

    def _is_combo_complete(self, idx: int) -> tuple[bool, list[str]]:
        """Check if a parameter combination has all required files for suite2p GUI.

        Returns cached result to avoid repeated filesystem checks.
        """
        if idx not in self._complete_cache:
            combo_dir = self.param_combos[idx]
            plane_dir = self._find_plane_dir(combo_dir)
            if plane_dir is None:
                self._complete_cache[idx] = (False, ["no suite2p results found"])
            else:
                self._complete_cache[idx] = self._check_suite2p_complete(plane_dir)
        return self._complete_cache[idx]

    def _get_stats(self, idx: int) -> dict:
        """Get cached stats for a parameter combination."""
        if idx not in self._stats_cache:
            combo_dir = self.param_combos[idx]
            plane_dir = self._find_plane_dir(combo_dir)
            if plane_dir:
                self._stats_cache[idx] = _load_stats(plane_dir)
            else:
                self._stats_cache[idx] = {"n_cells": 0, "n_not_cells": 0, "mean_snr": 0.0}
        return self._stats_cache[idx]

    def _open_suite2p(self, idx: int, position: str = "left") -> str | None:
        """Open suite2p GUI for a parameter combination.

        Parameters
        ----------
        idx : int
            Index of the parameter combination
        position : str
            "left" or "right" - determines window positioning

        Returns
        -------
        str | None
            Error message if failed, None if successful
        """
        combo_dir = self.param_combos[idx]
        plane_dir = self._find_plane_dir(combo_dir)

        if plane_dir is None:
            return f"No suite2p results found in {combo_dir.name}"

        stat_path = plane_dir / "stat.npy"
        if not stat_path.exists():
            return f"No stat.npy found in {plane_dir}"

        # Check for required files BEFORE importing suite2p GUI
        import os

        # Use os.path for all checks - more reliable with various path types
        plane_dir_str = str(plane_dir)
        ops_check_path = os.path.join(plane_dir_str, "ops.npy")

        if not os.path.isfile(ops_check_path):
            return f"ops.npy not found at: {ops_check_path}\nSuite2p GUI requires ops.npy in the same folder as stat.npy.\nGrid search may have saved minimal results only."

        try:
            from suite2p.gui.gui2p import MainWindow as Suite2pMainWindow
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QRect

            # close existing window for this position
            if position == "left" and self._suite2p_left is not None:
                try:
                    self._suite2p_left.close()
                except (RuntimeError, AttributeError):
                    pass
            elif position == "right" and self._suite2p_right is not None:
                try:
                    self._suite2p_right.close()
                except (RuntimeError, AttributeError):
                    pass

            # Normalize paths for suite2p - it has issues with UNC paths
            import os
            normalized_stat_path = os.path.normpath(str(stat_path))
            window = Suite2pMainWindow(statfile=normalized_stat_path)
            window.setWindowTitle(f"Suite2p - {combo_dir.name}")

            # position windows side-by-side with proper margins for window decorations
            screen = QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                half_w = geom.width() // 2
                # Leave margin for title bar and taskbar
                margin_top = 30
                margin_bottom = 10
                win_height = geom.height() - margin_top - margin_bottom

                if position == "left":
                    window.setGeometry(QRect(geom.x(), geom.y() + margin_top, half_w, win_height))
                    self._suite2p_left = window
                    self._left_idx = idx
                else:
                    window.setGeometry(QRect(geom.x() + half_w, geom.y() + margin_top, half_w, win_height))
                    self._suite2p_right = window
                    self._right_idx = idx

            window.setMinimumSize(400, 300)
            window.show()
            # Ensure window has normal state (not maximized/fullscreen)
            window.showNormal()
            return None  # Success

        except ImportError as e:
            return f"Could not open suite2p GUI: {e}"
        except Exception as e:
            return f"Error opening suite2p GUI: {e}"

    def draw(self):
        """Draw the grid search viewer UI."""
        # check for pending file dialog
        if self._file_dialog is not None and self._file_dialog.ready():
            result = self._file_dialog.result()
            if result:
                try:
                    set_last_dir("grid_search", result)
                    self.load_results(Path(result))
                except Exception as e:
                    print(f"Error loading grid search results: {e}")
            self._file_dialog = None

        if not self.loaded:
            self._draw_load_ui()
            return

        self._draw_comparison_ui()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
        imgui.text("No grid search results loaded.")
        imgui.spacing()
        if imgui.button("Load Grid Search Results"):
            default_dir = str(get_last_dir("grid_search") or Path.home())
            self._file_dialog = pfd.select_folder(
                "Select grid search results folder", default_dir
            )

        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Select a folder containing grid search results.\n"
                "Each subfolder should be a parameter combination\n"
                "with suite2p/plane0/ containing the results."
            )

    def _draw_comparison_ui(self):
        """Draw the comparison UI with two columns."""
        n_combos = len(self.param_combos)
        imgui.text(f"Results: {self.results_path.name} ({n_combos} combinations)")

        # Show error message if any
        if self._error_message:
            imgui.spacing()
            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), self._error_message)
            imgui.same_line()
            if imgui.small_button("Dismiss"):
                self._error_message = None

        imgui.separator()
        imgui.spacing()

        # two-column layout
        avail = imgui.get_content_region_avail()
        col_width = (avail.x - 20) / 2

        # Check completeness for both selected combos
        left_complete, left_missing = self._is_combo_complete(self._left_idx)
        right_complete, right_missing = self._is_combo_complete(self._right_idx)

        # left column
        imgui.begin_group()
        imgui.text("Left Window")
        imgui.set_next_item_width(col_width - 100)
        changed_l, new_left = imgui.combo(
            "##left_combo",
            self._left_idx,
            [c.name for c in self.param_combos]
        )
        if changed_l:
            self._left_idx = new_left

        imgui.same_line()
        if not left_complete:
            imgui.begin_disabled()
        if imgui.button("Open##left"):
            err = self._open_suite2p(self._left_idx, "left")
            if err:
                self._error_message = err
        if not left_complete:
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(f"Missing: {', '.join(left_missing)}")

        # show stats for left
        stats_l = self._get_stats(self._left_idx)
        if not left_complete:
            imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.2, 1.0), f"Incomplete - missing: {', '.join(left_missing)}")
        elif stats_l['n_cells'] == 0:
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "No detected cells")
        else:
            imgui.text(f"Cells: {stats_l['n_cells']}  Non-cells: {stats_l['n_not_cells']}  SNR: {stats_l['mean_snr']:.2f}")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(imgui.ImVec2(20, 0))
        imgui.same_line()

        # right column
        imgui.begin_group()
        imgui.text("Right Window")
        imgui.set_next_item_width(col_width - 100)
        changed_r, new_right = imgui.combo(
            "##right_combo",
            self._right_idx,
            [c.name for c in self.param_combos]
        )
        if changed_r:
            self._right_idx = new_right

        imgui.same_line()
        if not right_complete:
            imgui.begin_disabled()
        if imgui.button("Open##right"):
            err = self._open_suite2p(self._right_idx, "right")
            if err:
                self._error_message = err
        if not right_complete:
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip(f"Missing: {', '.join(right_missing)}")

        # show stats for right
        stats_r = self._get_stats(self._right_idx)
        if not right_complete:
            imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.2, 1.0), f"Incomplete - missing: {', '.join(right_missing)}")
        elif stats_r['n_cells'] == 0:
            imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "No detected cells")
        else:
            imgui.text(f"Cells: {stats_r['n_cells']}  Non-cells: {stats_r['n_not_cells']}  SNR: {stats_r['mean_snr']:.2f}")
        imgui.end_group()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # open both button - disable if either is incomplete
        both_complete = left_complete and right_complete
        if not both_complete:
            imgui.begin_disabled()
        if imgui.button("Open Both Side-by-Side"):
            err_left = self._open_suite2p(self._left_idx, "left")
            err_right = self._open_suite2p(self._right_idx, "right")
            # Show first error encountered
            if err_left:
                self._error_message = err_left
            elif err_right:
                self._error_message = err_right
        if not both_complete:
            imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("One or both selections are missing required suite2p files")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Open both selected combinations in suite2p GUIs side-by-side")

        # quick stats comparison table
        imgui.spacing()
        imgui.separator()
        imgui.text("Quick Stats Comparison:")

        if imgui.begin_table("stats_table", 5, imgui.TableFlags_.borders | imgui.TableFlags_.row_bg):
            imgui.table_setup_column("Combination")
            imgui.table_setup_column("Status")
            imgui.table_setup_column("Cells")
            imgui.table_setup_column("Non-cells")
            imgui.table_setup_column("Mean SNR")
            imgui.table_headers_row()

            for i, combo in enumerate(self.param_combos):
                stats = self._get_stats(i)
                is_complete, missing = self._is_combo_complete(i)
                imgui.table_next_row()

                imgui.table_next_column()
                # highlight selected rows
                if i == self._left_idx or i == self._right_idx:
                    imgui.text_colored(imgui.ImVec4(0.3, 0.8, 0.3, 1.0), combo.name)
                else:
                    imgui.text(combo.name)

                imgui.table_next_column()
                if not is_complete:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.6, 0.2, 1.0), "Incomplete")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"Missing: {', '.join(missing)}")
                elif stats["n_cells"] == 0:
                    imgui.text_colored(imgui.ImVec4(0.7, 0.7, 0.7, 1.0), "No cells")
                else:
                    imgui.text_colored(imgui.ImVec4(0.3, 0.8, 0.3, 1.0), "Ready")

                imgui.table_next_column()
                imgui.text(str(stats["n_cells"]))

                imgui.table_next_column()
                imgui.text(str(stats["n_not_cells"]))

                imgui.table_next_column()
                imgui.text(f"{stats['mean_snr']:.2f}")

            imgui.end_table()

    def cleanup(self):
        """Clean up suite2p windows."""
        for window in [self._suite2p_left, self._suite2p_right]:
            if window is not None:
                try:
                    window.close()
                except (RuntimeError, AttributeError):
                    pass
        self._suite2p_left = None
        self._suite2p_right = None
