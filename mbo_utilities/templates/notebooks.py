"""
notebook template generation for mbo_utilities.

generates jupyter notebooks from templates for common analysis pipelines.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# template registry: name -> (description, cell generator function)
TEMPLATES: dict[str, tuple[str, callable]] = {}


def register_template(name: str, description: str):
    """decorator to register a notebook template."""
    def decorator(func):
        TEMPLATES[name] = (description, func)
        return func
    return decorator


def list_templates() -> list[tuple[str, str]]:
    """return list of (name, description) for all templates."""
    return [(name, desc) for name, (desc, _) in TEMPLATES.items()]


def get_template_path() -> Path:
    """return path to custom templates directory."""
    config_dir = Path.home() / ".mbo" / "templates"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _make_cell(source: str | list[str], cell_type: str = "code") -> dict[str, Any]:
    """create a notebook cell."""
    if isinstance(source, str):
        source = source.split("\n")
    # ensure each line ends with newline except last
    lines = []
    for i, line in enumerate(source):
        if i < len(source) - 1 and not line.endswith("\n"):
            lines.append(line + "\n")
        else:
            lines.append(line)
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _make_notebook(cells: list[dict]) -> dict[str, Any]:
    """create a notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def create_notebook(
    template: str,
    output_path: Path | str | None = None,
    name: str | None = None,
    **kwargs
) -> Path:
    """
    create a notebook from a template.

    parameters
    ----------
    template : str
        template name (e.g., "lsp", "basic")
    output_path : Path | str, optional
        output directory. defaults to current directory.
    name : str, optional
        custom notebook name. defaults to yyyy-mm-dd_<template>.ipynb
    **kwargs
        additional arguments passed to the template generator

    returns
    -------
    Path
        path to the created notebook
    """
    if template not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template}'. Available: {available}")

    _, generator = TEMPLATES[template]
    cells = generator(**kwargs)
    notebook = _make_notebook(cells)

    # determine output path
    if output_path is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_path)

    if output_path.is_dir():
        # generate filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        if name:
            filename = f"{date_str}_{name}.ipynb"
        else:
            filename = f"{date_str}_{template}.ipynb"
        output_file = output_path / filename
    else:
        output_file = output_path

    # write notebook
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    return output_file


# full function signatures for templates
IMREAD_SIGNATURE = '''arr = mbo.imread(
    inputs=raw_data,  # str | Path | ndarray | Sequence[str | Path]
)'''

IMWRITE_SIGNATURE = '''mbo.imwrite(
    lazy_array=arr,
    outpath=output_dir,
    ext=".tiff",  # ".tiff" | ".bin" | ".zarr" | ".h5"
    planes=None,  # list | tuple | int | None - z-planes to export (1-based)
    frames=None,  # list | tuple | int | None - timepoints to export (1-based)
    num_frames=None,  # int | None - number of frames to export
    register_z=False,  # bool - perform z-plane registration via Suite3D
    roi_mode="concat_y",  # "concat_y" | "separate" - multi-ROI handling
    roi=None,  # int | Sequence[int] | None - specific ROI(s) when separate
    metadata=None,  # dict | None - additional metadata to merge
    overwrite=False,  # bool - overwrite existing files
    order=None,  # list | tuple | None - reorder planes
    target_chunk_mb=100,  # int - chunk size for streaming writes
    progress_callback=None,  # Callable | None - callback(progress, current_plane)
    debug=False,  # bool - verbose logging
    show_progress=True,  # bool - show tqdm progress bar
    shift_vectors=None,  # ndarray | None - pre-computed z-shift vectors
    output_name=None,  # str | None - filename for binary output
    output_suffix=None,  # str | None - custom suffix for output filenames
)'''

@register_template("lsp", "LBM-Suite2p-Python full pipeline")
def _template_lsp(data_path: str = "/path/to/data", **kwargs) -> list[dict]:
    """generate lbm-suite2p-python pipeline notebook."""
    cells = [
        # imports
        _make_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import mbo_utilities as mbo\n"
            "import lbm_suite2p_python as lsp"
        ),
        # run pipeline header
        _make_cell("## Run Pipeline", cell_type="markdown"),
        # paths
        _make_cell(
            f'input_data = r"{data_path}"\n'
            'save_path = r"D:/demo/segmentation"'
        ),
        # help info
        _make_cell(
            "To get help on any function or module:\n\n"
            "```python\n"
            "help(lsp)\n"
            "help(lsp.pipeline)\n"
            "```",
            cell_type="markdown"
        ),
        # pipeline call
        _make_cell(
            "ops = {\n"
            '    "diameter": 3,\n'
            '    "anatomical_only": 4,      # max projection\n'
            '    "accept_all_cells": True,  # skip acc/rej cells by s2p\n'
            '    "spatial_hp_cp": 3,\n'
            '    "denoise": 1,\n'
            '    "two_step_registration": 1,\n'
            "}\n"
            "\n"
            "results = lsp.pipeline(\n"
            "    input_data=input_data,      # path to .zarr, .tiff, or .bin file\n"
            "    save_path=save_path,        # default: save next to input file\n"
            "    ops=ops,                    # default: use MBO-optimized parameters\n"
            "    planes=np.arange(1, 4),     # None for all planes, or list of zplanes (1-based)\n"
            "    num_timepoints=500,         # None for all timepoints, or int\n"
            "    roi=None,                   # default: stitch multi-ROI data\n"
            "    keep_reg=True,              # default: keep data.bin (registered binary)\n"
            "    keep_raw=False,             # default: delete data_raw.bin after processing\n"
            "    force_reg=False,            # default: skip if already registered\n"
            "    force_detect=False,         # default: skip if stat.npy exists\n"
            "    dff_window_size=None,       # default: auto-calculate from tau and framerate\n"
            "    dff_percentile=20,          # default: 20th percentile for baseline\n"
            "    dff_smooth_window=None,     # default: auto-calculate from tau and framerate\n"
            "    accept_all_cells=False,     # default: use suite2p classification\n"
            "    cell_filters=[],            # default: set to [] to disable\n"
            "    reader_kwargs={},           # default: args passed to mbo.imread(input_path)\n"
            "    writer_kwargs={},           # default: args passed to mbo.imwrite(lazy_array, save_path)\n"
            ")"
        ),
        # load results header
        _make_cell("## Load Results", cell_type="markdown"),
        # load results
        _make_cell(
            "# get output folders\n"
            "folders = sorted(Path(save_path).glob(\"zplane*\"))\n"
            "print(f\"Found {len(folders)} plane folders\")\n"
            "\n"
            "# load results from first plane\n"
            "results = lsp.load_planar_results(folders[0])\n"
            "F = results[\"F\"]\n"
            "stat = results[\"stat\"]\n"
            "Fneu = results[\"Fneu\"]\n"
            "iscell = results[\"iscell\"]\n"
            "\n"
            "print(f\"Loaded {F.shape[0]} ROIs, {F.shape[1]} frames\")"
        ),
        # quality scoring
        _make_cell(
            "# plot top N neurons by quality score\n"
            "top_n = 20\n"
            "trace_quality = lsp.postprocessing.compute_trace_quality_score(F,)\n"
            "sort_idx = trace_quality[\"sort_idx\"]"
        ),
        # plot
        _make_cell(
            "top_indices = sort_idx[:top_n]\n"
            "\n"
            "# create boolean mask for plot_masks\n"
            "mask_idx = np.zeros(len(stat), dtype=bool)\n"
            "mask_idx[top_indices] = True\n"
            "\n"
            "# load ops to get the background image\n"
            "ops = lsp.load_ops(folders[0] / \"ops.npy\")\n"
            "img = ops.get(\"meanImgE\", ops.get(\"meanImg\"))\n"
            "\n"
            "# plot masks for top neurons\n"
            "lsp.plot_regional_zoom(\n"
            "    plane_dir=folders[0],\n"
            ")\n"
            "\n"
            "# plot traces for top neurons\n"
            "lsp.plot_traces(\n"
            "    f=F[top_indices],\n"
            "    fps=ops.get(\"fs\", 30.0),\n"
            "    num_neurons=top_n,\n"
            "    title=f\"Top {top_n} Neuron Traces\"\n"
            ")"
        ),
        # planar outputs table
        _make_cell(
            "### Planar Outputs\n\n"
            "Each z-plane directory contains:\n\n"
            "#### Data Files\n\n"
            "| File | Shape | Description |\n"
            "|------|-------|-------------|\n"
            "| `ops.npy` | dict | Processing parameters and metadata |\n"
            "| `stat.npy` | (n_rois,) | ROI definitions (pixel coordinates, weights, shape stats) |\n"
            "| `F.npy` | (n_rois, n_frames) | Raw fluorescence traces |\n"
            "| `Fneu.npy` | (n_rois, n_frames) | Neuropil fluorescence traces |\n"
            "| `spks.npy` | (n_rois, n_frames) | Deconvolved spike estimates |\n"
            "| `iscell.npy` | (n_rois, 2) | Cell classification: `[:, 0]` = is_cell (0/1), `[:, 1]` = probability |\n"
            "| `data.bin` | (n_frames, Ly, Lx) | Registered movie (if `keep_reg=True`) |",
            cell_type="markdown"
        ),
        # dff header
        _make_cell("### Calculate dF/F", cell_type="markdown"),
        # dff calculation
        _make_cell(
            "# calculate dF/F with rolling percentile baseline\n"
            "dff = lsp.dff_rolling_percentile(\n"
            "    results['F'],\n"
            "    window_size=300,    # frames (~10x tau x fs)\n"
            "    percentile=20       # baseline percentile\n"
            ")\n"
            "\n"
            "# filter for accepted cells only\n"
            "iscell_mask = results['iscell'][:, 0].astype(bool)\n"
            "dff_cells = dff[iscell_mask]\n"
            "print(f\"dF/F shape (accepted cells): {dff_cells.shape}\")"
        ),
        # gui header
        _make_cell(
            "## Open Suite2p GUI\n\n"
            "The Suite2p GUI provides interactive visualization and manual curation:\n\n"
            "- **View registered movie**: Suite2p -> Registration -> View Registration Binary\n"
            "- **Compare raw vs registered**: Check \"View raw binary\" (requires `keep_raw=True`)\n"
            "- **Registration quality**: Suite2p -> Registration -> View Registration Metrics (>1500 frames)",
            cell_type="markdown"
        ),
        # gui cell
        _make_cell(
            "# open GUI for manual curation\n"
            "run_gui = False\n"
            "if folders and run_gui:\n"
            "    stat_file = folders[0] / \"stat.npy\"\n"
            "    if stat_file.exists():\n"
            "        from suite2p import gui\n"
            "        gui.run(statfile=str(stat_file))"
        ),
    ]
    return cells


@register_template("basic", "Basic mbo_utilities data exploration")
def _template_basic(data_path: str = "/path/to/data", **kwargs) -> list[dict]:
    """generate basic data exploration notebook."""
    cells = [
        _make_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "import mbo_utilities as mbo"
        ),
        _make_cell(
            f'data_path = Path(r"{data_path}")'
        ),
        _make_cell(IMREAD_SIGNATURE.replace("raw_data", "data_path")),
        _make_cell(
            "print(f\"Shape: {arr.shape}\")\n"
            "print(f\"Type:  {type(arr).__name__}\")\n"
            "print(f\"Dtype: {arr.dtype}\")"
        ),
        _make_cell(
            "if hasattr(arr, 'metadata') and arr.metadata:\n"
            "    for k, v in list(arr.metadata.items())[:20]:\n"
            "        print(f\"{k}: {v}\")"
        ),
        _make_cell(
            "if arr.ndim == 4:  # TZYX\n"
            "    frame = arr[0, 0]\n"
            "elif arr.ndim == 3:  # TYX\n"
            "    frame = arr[0]\n"
            "else:\n"
            "    frame = arr\n"
            "\n"
            "plt.figure(figsize=(8, 8))\n"
            "plt.imshow(frame, cmap='gray')\n"
            "plt.colorbar(label='Intensity')\n"
            "plt.title('Single frame')\n"
            "plt.show()"
        ),
        _make_cell(
            "output_dir = data_path.parent / f\"{data_path.stem}_converted\"\n"
            "# " + IMWRITE_SIGNATURE.replace("\n", "\n# ")
        ),
    ]
    return cells


@register_template("dff", "Delta F/F analysis")
def _template_dff(data_path: str = "/path/to/data", **kwargs) -> list[dict]:
    """generate dff analysis notebook."""
    cells = [
        _make_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "import mbo_utilities as mbo\n"
            "from mbo_utilities.analysis import compute_dff"
        ),
        _make_cell(
            f'data_path = Path(r"{data_path}")'
        ),
        _make_cell(IMREAD_SIGNATURE.replace("raw_data", "data_path")),
        _make_cell(
            "print(f\"Shape: {arr.shape}\")"
        ),
        _make_cell(
            "if arr.ndim == 4:\n"
            "    plane_data = arr[:, 0]  # first z-plane\n"
            "else:\n"
            "    plane_data = arr\n"
            "\n"
            "dff = compute_dff(\n"
            "    plane_data,\n"
            "    method='percentile',  # 'percentile' | 'sliding' | 'first_n'\n"
            "    percentile=10,  # Nth percentile for baseline (when method='percentile')\n"
            ")\n"
            "\n"
            "print(f\"DFF shape: {dff.shape}\")\n"
            "print(f\"DFF range: [{dff.min():.2f}, {dff.max():.2f}]\")"
        ),
        _make_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "axes[0].imshow(plane_data[0], cmap='gray')\n"
            "axes[0].set_title('Raw (frame 0)')\n"
            "axes[1].imshow(dff.mean(axis=0), cmap='RdBu_r', vmin=-0.5, vmax=0.5)\n"
            "axes[1].set_title('Mean ΔF/F')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _make_cell(
            "activity = dff.std(axis=0)\n"
            "plt.figure(figsize=(8, 8))\n"
            "plt.imshow(activity, cmap='hot')\n"
            "plt.colorbar(label='Std(ΔF/F)')\n"
            "plt.title('Activity map')\n"
            "plt.show()"
        ),
    ]
    return cells
