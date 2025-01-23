# MBO utilities
General python/shell utilities and environment/config files.

## Installation
```bash
conda create -n mbo_util python=3.10 imgui-bundle
conda activate mbo_util
pip install git+https://github.com/MillerBrainObservatory/mbo_utilities.git@master
```

## Usage

Open a terminal and run the following commands:

``` bash
# make sure the conda environment is activated
mbo --path ~/data/session/

# or open a dialog to select the folder
mbo --gui
```

or in Jupyter notebook:

```python
from mbo_utilities import run_gui

run_gui()
```
