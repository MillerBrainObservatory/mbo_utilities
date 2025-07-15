# Virtual Environments

This guide covers managing python environments with [UV](https://docs.astral.sh/uv/) and [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

```{admonition} TLDR
:class: dropdown
If you know `conda`, use `conda`.

Otherwise, install and learn about `uv` and substitute `pip install <PACKAGE>` with `uv pip install <PACKAGE>` for a drop-in `pip` replacement.
```

```{list-table} UV vs Conda: Common Environment Commands
:header-rows: 1
:widths: 20 40 40

* - **Action**
  - **UV**
  - **Conda**
* - Create environment
  - `uv venv myenv --python=3.11`
  - `conda create -n myenv -c conda-forge python=3.11`
* - Activate
  - `source .venv/bin/activate`  
    `source .venv/Scripts/activate`
  - `conda activate myenv`
* - Install package
  - `uv pip install <package>`
  - `conda install <package>`
* - Add/remove package
  - `uv add <package>` / `uv remove <package>`
  - (edit `environment.yml` or reinstall)
* - Sync with pyproject.toml
  - `uv sync --all-extras`
  - ✖️ Not supported
* - Upgrade packages
  - `uv lock --upgrade`  
    `uv sync --upgrade-package <pkg>`
  - `conda update <package>`
```

(managing_uv)=
## Managing environments: `UV`

[uv](https://docs.astral.sh/uv/) is a complete drop-in replacement for `pip`.  
If you know `pip`, you already know `uv`.

You just prefix commands:

```bash
pip install .   # old
uv pip install . # new
```

Using `uv` has a huge upside: Packages are cached globally, so you can quickly recreate them without worrying about contaminating your base environment.

``` {tip}
With UV, re-creating your environment is much faster than with `conda` thanks to it's [dependency caching](https://docs.astral.sh/uv/concepts/cache/#dependency-caching).

This makes deleting your environment an appropriate solution if you run into environemnt conflicts.
```

### Create an environment

By default, `uv venv` will create a folder `.venv` which is used as your environemnt.

This folder will be placed in your current working directory:

```bash
USER@SERVER ~/repos/mypackage
$ uv venv
Using CPython 3.11.12
Creating virtual environment at: .venv
```

You can specify a name for the environment:

```bash
USER@SERVER ~/repos/mypackage
$ uv venv myenv
Using CPython 3.11.12
Creating virtual environment at: myenv
```

You can also indicate which python to use:

```bash
USER@SERVER ~/repos/mypackage
$ uv venv myenv --python=39  # or 310, 311, 312, 313
Using CPython 3.9.22
Creating virtual environment at: myenv
```

```{warning}
`uv venv` will create a folder in your current directory named `.venv`, which is a python standard and will be automatically chosen by many development environments e.g. `VSCode`.

To distinguish between environments, name the environment by substituting <VENV-NAME> with the name you wish to assign to that virtual environment.
```

### Activate the environment (optional, good practice)

If you named your environment, replace `.venv` below with the name of your environment.

::::{tab-set}
:::{tab-item} Linux/macOS

```bash
source .venv/bin/activate
```

:::

:::{tab-item} Windows

```bash
source .venv/Scripts/activate
```

:::
::::

If you don't activate the environment, but you are running code from a directory that has a `.venv` folder, it will still be used automatically in most cases.

### Install python packages

```bash
uv pip install <PACKAGE>
```

This works just like pip.

If you like to `git clone` repositories, you can:

```bash
uv pip install .
```

If you are in a directory with a `pyproject.toml` like many python projects will have:

```bash
cd repository
uv sync --all-extras
```

The `--all-extras` get's all extra dependencies, like [notebook, gui], if the repository has any.

(uv_cheatsheet)=
```{list-table} Most helpful UV Commands
:header-rows: 1
:name: uv-commands

* - **Command**
  - **Description**
* - `uv venv`, `uv venv .myvenv`
  - Create an env, optionally name it myenv (not recommended)
* - `uv sync` / `uv lock --upgrade`
  - Update your environment based on the current most up to date packages
* - `uv add <package>` / `uv remove <package>`
  - Add/remove packages (updates `pyproject.toml`, `uv.lock`, and your env)
* - `uv sync --upgrade-package <pkg>` / `uv lock --upgrade`
  - Upgrade single/all packages
```

---

## Managing environments: `conda`

### Create an environment

For more ways to create an environment (such as from a `environment.yml` file), see [Creating an Environment with Commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

``` bash
conda create -n myenv -c conda-forge python=3.xx
```

### Activate the environment


```{warning}
Don't forget to activate your environment!

If you do, it will be installed in the [base conda environment](https://www.anaconda.com/docs/tools/working-with-conda/environments#why-shouldn%E2%80%99t-i-work-in-the-base-environment%3F).
```

Unlike {ref}`uv <managing_uv>`,
which will look for a folder in your current directory,
`conda` will default to installing packages into the base environment which is shared across all user environments and often leads to package conflicts.

The only fix is to reinstall `conda`.

``` bash
conda activate myenv
```

### Install python packages

For details about installing python packages with conda, see conda docs on [Managing Packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#managing-packages).

``` bash
conda install <package1> <package2>
```

---

## Virtual environment options

You could learn the in's and out's for all of the virtual environments below.
Don't do this (use [UV](https://docs.astral.sh/uv/getting-started/), the community is finally settling on a standard).

There are generally three camps:

1. Tools that handle only Python packages (venv, pyenv)
2. Tools that handle python packages AND system dependencies like ffmpeg, opencv, cuda (mamba, conda)
3. Tools built on top of the above to make them easier to use, faster, etc. (UV, pixi, which uses UV under the hood).

::::{grid}
:::{grid-item-card} System + Python
 conda,
 miniconda,
 miniforge,
 anaconda,
 mamba,
 micromamba,
 pixi
:::

:::{grid-item-card} Python-only
 pip,
 pip-tools,
 pipx,
 venv / pyvenv,
 virtualenv,
 pipenv,
 poetry,
 twine,
 hatch,
 asdf,
 uv
:::
::::

---

## Why not `conda`?

- `pip` (and UV) can build most system packages you need from source.
- Most Python libraries now ship prebuilt wheels (binaries).
- The main selling point for `conda` (handling system binaries) is much less relevant today.
- The `conda` maintainers have moved toward venv-based setups (pixi and uv).

If you **must** use conda, **only** use `miniforge3`.
Many recommended setup steps for miniconda/anaconda are mimicking the defaults of `miniforge3`.

If you already know conda, that's fine — just **avoid mixing** `pip install` and `conda install` after your initial setup unless you have to.

---

## Mixing `conda` and `uv`

`uv` will automatically fallback to using a conda environment if no `.venv` folder is found:

```bash
USER at pop-os in ~/repos/work/mbo_utilities (master●)
$ uv pip list | grep imgui
Using Python 3.12.6 environment at: /home/USER/miniforge3
imgui-bundle              1.6.2
```

So even if you're using conda, `uv` will work all the same.

We recommend turning off automatic environment activation for `conda` in this case:

``` bash
conda config --set auto_activate_base false
```

You can still call `conda activate myenv`, but only if `conda` is called explicitly.

## Debugging environments

Most bugs and frustration in Python come from mismanaging virtual environments.

No matter what command you're running, check the terminal outputs for a filepath that points to python.

For jupyterlab:

```{figure} ./_images/venv_jlab.png
```

<!-- ### Two common examples -->
<!---->
<!-- 1. **Mixing `conda` and `pip`** -->
<!---->
<!--   `jupyter` is one of several libraries that will not work when mixing `conda` and `pip`. -->
<!---->
<!---->
<!-- **Wrong Python is being used** -->
<!---->
<!-- There are a tens of options for how to install Python on any of the operating systems. -->
<!---->
<!-- One of the most detrimental mistakes is adding Python to your system -->
<!---->
<!-- [PATH](https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them). -->
<!---->
<!-- `conda` warns you not to do this, [Windows Python Installer](https://docs.python.org/3/_images/win_installer.png) does too. -->
<!---->
<!-- Though for some reason users so often end up with system versions of python, often from the package managers. -->
<!---->
<!-- The problem here is that you will be using this system python and you won't know it. -->
<!---->
<!-- You will run a `jupyter lab` command, which will open properly but your environments are nowhere to be found. -->
<!---->
<!-- This can often be incredibly difficult and time-consuming to debug. -->
<!---->
<!-- This happens to experienced developers the same as anyone else. -->
<!---->
