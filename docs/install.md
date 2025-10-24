(installation)=
# Installation Guide


mbo_utilities has been developed to be a pure `pip` install.

This makes the choice of virtual-environment less relevant, you can use `venv`, `uv (recommended)`, `conda`, it does not matter.

## Stable from PyPi

::::{tab-set}

:::{tab-item} `uv`

```bash

uv venv --python 3.12.9 # or uv init
uv pip install mbo_utilities

```

:::

:::{tab-item} `conda`

```bash

conda create -n mbo -c conda-forge python=3.12.9
conda activate mbo
pip install mbo_utilities 

```

:::

:::{tab-item} `venv`

```bash

python -m venv
source .venv/bin/activate # omit "source" on powershell
pip install mbo_utilities

```

:::

::::

## Latest from Github

While this pipeline is early in development, its handing to keep a version of the codebase locally using `git`.

Code in the [master branch](https://github.com/MillerBrainObservatory/mbo_utilities/tree/master) is generally safe.
If it isn't please [report an issue!](https://github.com/MillerBrainObservatory/mbo_utilities/issues)

::::{tab-set}

:::{tab-item} `uv`

```bash

# clone anywhere
git clone https://github.com/MillerBrainObservatory/mbo_utilities.git

# from your project directory
uv venv --python 3.12.9 # if you don't already have a .venv created
uv pip install ../mbo_utilities

```

:::

:::{tab-item} `conda`

```bash

git clone https://github.com/MillerBrainObservatory/mbo_utilities.git
cd mbo_utilities

conda create -n mbo -c conda-forge python=3.12.9
conda activate mbo
pip install . 

```

:::

:::{tab-item} `venv`

```bash

git clone https://github.com/MillerBrainObservatory/mbo_utilities.git

cd my_project

python -m venv
source .venv/bin/activate # omit "source" on powershell

pip install ../mbo_utilities

```

:::

::::

## GUI Dependencies

::::{tab-set}

:::{tab-item} Linux / macOS

```bash

sudo apt install libxcursor-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev

```

:::

:::{tab-item} Windows
You will need [msvcc redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
:::

::::

## Troubleshooting

### Environment Issues

Many hard to diagnose installation/import bugs are due to environment issues.

The first thing you should do is check which python interpreter is being used. Generally this 
will point to your project like:

``` bash
C:/User/Username/repos/mbo_utilities/.venv//Scripts//python.exe
```

``` {figure} ./_images/env_jupyter.png
In jupyter, the terminal from which you ran `jupyter lab/notebook' will display the path to the python executable.
```

Once you know its location, open a python terminal.

For example, with the above `.venv`:

``` bash

$ python
Python 3.12.6 | packaged by conda-forge | (main, Sep 22 2024, 14:16:49) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mbo_utilities as mbo

$ uv run ipython
Python 3.11.12 (main, Apr 9 2025, 04:04:00) [Clang 20.1.0]
Type 'copyright', 'credits' or 'license' for more information.
IPython 9.2.0 -- An enhanced Interactive Python. Type '?' for help.
Tip: Use `object?` to see the help on `object`, `object??` to view its source

In [1]: import mbo_utilities as mbo
In [2]: mbo.__version__
Out[2]: '2.0.3'
```

### Git LFS Error: `smudge filter lfs failed`

If you see:

```
error: external filter 'git-lfs filter-process' failed
fatal: docs/source/_static/guide_hello_world.png: smudge filter lfs failed
```

Disable smudge during sync:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --active
```

In powershell:

```bash
$env:GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --active
```

