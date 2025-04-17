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
(run_gui)=
# Function Usage

Examples of some common function use cases.

{func}`mbo_utilities.run_gui` opens an interactive viewer for imaging data using [fastplotlib](https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib).
It supports execution in both **Jupyter** and **Qt-native** environments.

---

## Usage

### Jupyter / IPython

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

### Python script (non-Jupyter)

```{code} python
from mbo_utilities.graphics import run_gui

# Will launch native Qt viewer window
run_gui("path/to/data")
```
If run from a script without data_in, and Qt is installed, a file dialog will prompt for input.
