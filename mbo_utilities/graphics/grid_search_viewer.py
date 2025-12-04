"""Grid search results viewer for Suite2p parameter tuning."""

import numpy as np
from pathlib import Path
from imgui_bundle import imgui, portable_file_dialogs as pfd
from OpenGL import GL

from mbo_utilities.preferences import get_last_dir, set_last_dir


class GridSearchViewer:
    """Viewer for comparing grid search parameter combinations."""

    def __init__(self):
        self.results_path = None
        self.param_combos = []
        self.current_idx = 0
        self.images = {}
        self.textures = {}
        self.loaded = False
        self._file_dialog = None

    def load_results(self, path: Path):
        """Load grid search results from directory."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        self.results_path = path
        self.param_combos = sorted([d for d in path.iterdir() if d.is_dir()])

        if not self.param_combos:
            raise ValueError(f"No parameter combination folders found in {path}")

        self._cleanup_textures()
        self.images = {}
        self.textures = {}
        self.current_idx = 0
        self.loaded = True

        self._load_current_images()

    def _cleanup_textures(self):
        """Delete OpenGL textures."""
        for tex_id in self.textures.values():
            if tex_id:
                GL.glDeleteTextures(1, [tex_id])
        self.textures = {}

    def _load_current_images(self):
        """Load images for current parameter combination."""
        if not self.loaded or self.current_idx >= len(self.param_combos):
            return

        combo_dir = self.param_combos[self.current_idx]
        plane_dir = combo_dir / "suite2p" / "plane0"

        if not plane_dir.exists():
            self.images = {}
            return

        self.images = {}

        # load ops for meanImg
        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            ops = np.load(ops_path, allow_pickle=True).item()
            if "meanImg" in ops:
                self.images["mean"] = self._normalize_image(ops["meanImg"])

        # load stat for ROI visualization
        stat_path = plane_dir / "stat.npy"
        iscell_path = plane_dir / "iscell.npy"

        if stat_path.exists():
            stat = np.load(stat_path, allow_pickle=True)
            iscell = None
            if iscell_path.exists():
                iscell = np.load(iscell_path)

            if "mean" in self.images:
                h, w = self.images["mean"].shape[:2]
                self.images["rois"] = self._create_roi_image(stat, iscell, h, w)

        self._create_textures()

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 uint8."""
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        return img.astype(np.uint8)

    def _create_roi_image(
        self, stat: np.ndarray, iscell: np.ndarray | None, h: int, w: int
    ) -> np.ndarray:
        """Create RGB image with ROIs overlaid."""
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i, s in enumerate(stat):
            is_cell = True
            if iscell is not None:
                if iscell.ndim == 2:
                    is_cell = iscell[i, 0] > 0.5
                else:
                    is_cell = iscell[i] > 0.5

            ypix = s.get("ypix", [])
            xpix = s.get("xpix", [])

            if len(ypix) == 0:
                continue

            # clip to bounds
            mask = (ypix >= 0) & (ypix < h) & (xpix >= 0) & (xpix < w)
            ypix = ypix[mask]
            xpix = xpix[mask]

            if is_cell:
                rgb[ypix, xpix, 1] = 200  # green for cells
            else:
                rgb[ypix, xpix, 0] = 200  # red for non-cells

        return rgb

    def _create_textures(self):
        """Create OpenGL textures from loaded images."""
        self._cleanup_textures()

        for name, img in self.images.items():
            if img is None:
                continue

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            if img.ndim == 2:
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RED,
                    img.shape[1], img.shape[0], 0,
                    GL.GL_RED, GL.GL_UNSIGNED_BYTE, img
                )
            else:
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    img.shape[1], img.shape[0], 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img
                )

            self.textures[name] = tex_id

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

        self._draw_navigation()
        imgui.separator()
        self._draw_images()

    def _draw_load_ui(self):
        """Draw UI for loading results."""
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

    def _draw_navigation(self):
        """Draw navigation controls."""
        n_combos = len(self.param_combos)
        combo_name = self.param_combos[self.current_idx].name if n_combos > 0 else "None"

        imgui.text(f"Results: {self.results_path.name if self.results_path else 'None'}")
        imgui.text(f"Combination {self.current_idx + 1}/{n_combos}: {combo_name}")

        if imgui.button("< Prev") and self.current_idx > 0:
            self.current_idx -= 1
            self._load_current_images()

        imgui.same_line()

        if imgui.button("Next >") and self.current_idx < n_combos - 1:
            self.current_idx += 1
            self._load_current_images()

        imgui.same_line()

        imgui.set_next_item_width(200)
        changed, new_val = imgui.slider_int(
            "##combo_slider", self.current_idx, 0, max(0, n_combos - 1)
        )
        if changed and new_val != self.current_idx:
            self.current_idx = new_val
            self._load_current_images()

        imgui.same_line()

        if imgui.button("Load Different"):
            default_dir = str(get_last_dir("grid_search") or Path.home())
            self._file_dialog = pfd.select_folder(
                "Select grid search results folder", default_dir
            )

    def _draw_images(self):
        """Draw the images side by side."""
        if not self.images:
            imgui.text("No images found for this parameter combination")
            return

        avail = imgui.get_content_region_avail()
        img_width = avail.x / 2 - 10

        # mean image
        if "mean" in self.textures:
            img = self.images["mean"]
            aspect = img.shape[0] / img.shape[1]
            display_h = img_width * aspect

            imgui.text("Mean Image")
            imgui.image(self.textures["mean"], imgui.ImVec2(img_width, display_h))

        imgui.same_line()

        # ROI overlay
        if "rois" in self.textures:
            img = self.images["rois"]
            aspect = img.shape[0] / img.shape[1]
            display_h = img_width * aspect

            imgui.text("ROIs (green=cell, red=non-cell)")
            imgui.image(self.textures["rois"], imgui.ImVec2(img_width, display_h))

    def cleanup(self):
        """Clean up OpenGL resources."""
        self._cleanup_textures()
