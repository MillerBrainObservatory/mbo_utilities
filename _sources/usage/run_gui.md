---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: lsp
  language: python
  name: python3
---
# `run_gui`

{func}`mbo_utilities.run_gui` is a convenience function for viewing common data formats used in processing MBO datasets, usable from the terminal, jupyter-lab or a callable from a python script.

- raw scanimage tiffs
- any '[t(z)xy]' tiff, with or without the z or t dimensions
- any numpy array
- suite2p binary file (WIP, less tested)

Please ask a project maintainer to add a filetype or datatype to this list.

This takes advantage of [fastplotlib](https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib) to open a viewer in qt or jupyter, depending on the environment.

## Jupyter / IPython

Returns a {class}`fastplotlib.ImageWidget`:

```{code} python
from mbo_utilities.graphics import run_gui

# load directly from a ScanImage folder
iw = run_gui("path/to/scanimage/folder")

# or from preprocessed ScanMultiROIReordered object
from mbo_utilities.file_io import ScanMultiROIReordered
data = ScanMultiROIReordered(...)
iw = run_gui(data)
```

If no input is given and Qt is available, the widget will attempt to open a file dialog.
This is helpful when called from the command line:

``` bash
(venv) User ~/projects/AABBCC/
$ run_gui
No data provided.
Rendering qt widget
```

With a a native file dialog, now you can open a folder or filetype described above.

## Python script (non-Jupyter)

```{code} python
from mbo_utilities.graphics import run_gui

# Will launch native Qt viewer window
run_gui("path/to/data")
```
If run from a script without data_in, and Qt is installed, a file dialog will prompt for input.
