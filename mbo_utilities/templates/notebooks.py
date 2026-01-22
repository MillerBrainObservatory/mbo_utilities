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

RUN_LSP_SIGNATURE = '''run_lsp(
    raw_data,
    output_dir,
    ops=ops,
    planes=None,  # list | None - specific planes to process, or None for all
)'''


# ============================================================================
# built-in templates
# ============================================================================

@register_template("lsp", "LBM-Suite2p-Python full pipeline")
def _template_lsp(data_path: str = "/path/to/data", **kwargs) -> list[dict]:
    """generate lbm-suite2p-python pipeline notebook."""
    cells = [
        _make_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "import mbo_utilities as mbo\n"
            "from lbm_suite2p_python import run_lsp\n"
            "from lbm_suite2p_python.utils import get_default_ops"
        ),
        _make_cell(
            f'raw_data = Path(r"{data_path}")\n'
            'output_dir = raw_data.parent / f"{raw_data.stem}_suite2p"'
        ),
        _make_cell(IMREAD_SIGNATURE),
        _make_cell(
            "print(f\"Shape: {arr.shape}\")\n"
            "print(f\"Type:  {type(arr).__name__}\")\n"
            "print(f\"Dtype: {arr.dtype}\")\n"
            "if hasattr(arr, 'metadata'):\n"
            "    md = arr.metadata\n"
            "    print(f\"Num planes: {md.get('num_planes', 'N/A')}\")\n"
            "    print(f\"Frame rate: {md.get('frame_rate', 'N/A')} Hz\")"
        ),
        _make_cell(
            "ops = get_default_ops()\n"
            "ops['tau'] = 1.3  # gcamp indicator decay time\n"
            "ops['fs'] = arr.metadata.get('frame_rate', 10.0)\n"
            "ops['nplanes'] = arr.metadata.get('num_planes', 1)\n"
            "ops['do_registration'] = True\n"
            "ops['nonrigid'] = True\n"
            "ops['sparse_mode'] = True\n"
            "ops['diameter'] = 12  # approximate neuron diameter in pixels"
        ),
        _make_cell(
            "# " + RUN_LSP_SIGNATURE.replace("\n", "\n# ")
        ),
        _make_cell(
            "plane_dir = output_dir / \"plane00\"\n"
            "if plane_dir.exists():\n"
            "    from mbo_utilities.arrays import Suite2pArray\n"
            "    s2p = Suite2pArray(plane_dir)\n"
            "    print(f\"Loaded {s2p.shape[0]} frames, {len(s2p.stat)} ROIs\")"
        ),
        _make_cell(
            "if 's2p' in dir():\n"
            "    fig, ax = plt.subplots(1, 1, figsize=(10, 10))\n"
            "    ax.imshow(s2p.mean_img, cmap='gray')\n"
            "    ax.set_title(f'Mean image with {len(s2p.stat)} ROIs')\n"
            "    for i, roi in enumerate(s2p.stat[:50]):\n"
            "        ypix, xpix = roi['ypix'], roi['xpix']\n"
            "        ax.scatter(xpix, ypix, s=1, alpha=0.5)\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
        _make_cell(
            "if 's2p' in dir():\n"
            "    F = s2p.F  # raw fluorescence\n"
            "    Fneu = s2p.Fneu  # neuropil\n"
            "    spks = s2p.spks  # deconvolved spikes\n"
            "    print(f\"Traces shape: {F.shape}\")\n"
            "    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)\n"
            "    t = np.arange(F.shape[1]) / ops['fs']\n"
            "    for i, ax in enumerate(axes):\n"
            "        ax.plot(t, F[i] - 0.7 * Fneu[i], 'k', lw=0.5)\n"
            "        ax.set_ylabel(f'ROI {i}')\n"
            "    axes[-1].set_xlabel('Time (s)')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
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
