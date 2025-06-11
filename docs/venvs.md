# Virtual Environments

```{admonition} TLDR
:class: dropdown
If you know conda, just use that.

Otherwise, install and learn a bit about [uv](https://docs.astral.sh/uv/) and substitute `pip install <PACKAGE>` with `uv pip install <PACKAGE>` until you learn more about its high-level features.
```

Most bugs and frustration in Python come from mismanaging virtual environments.

## Two common examples

**Mixing `conda` and `pip`**

You `conda install <PACKAGE>` as the installation instructions told you to do.
But you missed the [extra](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#extras) `[notebook]` dependency, so you `pip install jupyter`.
`jupyter` is one of several libraries that will not work when mixing `conda` and `pip`.

**Wrong Python is being used**

There are a tens of options for how to install Python on any of the operating systems.
One of the most detrimental mistakes is adding Python to your system
[PATH](https://superuser.com/questions/284342/what-are-path-and-other-environment-variables-and-how-can-i-set-or-use-them).
`conda` warns you not to do this, [Windows Python Installer](https://docs.python.org/3/_images/win_installer.png) does too.
Though for some reason users so often end up with system versions of python, often from the package managers.
The problem here is that you will be using this system python and you won't know it.
You will run a `jupyter lab` command, which will open properly but your environments are nowhere to be found.

This can often be incredibly difficult and time-consuming to debug. This happens to experienced developers the same as anyone else.

## How to prevent environment issues

No matter what command you're running, check the terminal outputs for a filepath that points to python.

For jupyterlab:

```{figure} ./_images/venv_jlab.png
```

## Virtual environment options

You could learn the in's and out's for all of the virtual environments below.
Don't do this (use [UV](https://docs.astral.sh/uv/getting-started/), the community is finally settling on a standard).

There are generally three camps:

1. Tools that handle only Python packages (venv, pyenv)
2. Tools that handle python packages AND system dependencies like ffmpeg, opencv, cuda (mamba, conda)
3. Tools built on top of the above to make them easier to use, faster, etc. (UV, pixi, which uses UV under the hood).

::::{grid}
:::{grid-item-card} System + Python
 conda
 miniconda
 miniforge
 anaconda
 mamba
 micromamba
 pixi
:::

:::{grid-item-card} Python-only
 pip
 pip-tools
 pipx
 venv / pyvenv
 virtualenv
 pipenv
 poetry
 twine
 hatch
 uv
 asdf
:::
::::

---

## Why not `conda`?

- `pip` (and UV) can build most system packages you need from source.
- Most Python libraries now ship prebuilt wheels (binaries).
- The main selling point for `conda` (handling system binaries) is much less relevant today.
- The `conda` maintainers have moved toward venv-based setups (pixi and uv).

If you **must** use conda, **only** use `miniforge3`.
Many recommended setup steps for miniconda/anaconda are basically mimicking the defaults of `miniforge3`.

If you already know conda, that's fine — just **avoid mixing** `pip install` and `conda install` after your initial setup unless you have to.

---

## Guide to using UV

[uv](https://docs.astral.sh/uv/) is a complete drop-in replacement for `pip`.  
If you know `pip`, you already know `uv`.

If you have a mess of environments, maybe you mixed `conda`, `venv`, and `pyenv`, installing and using `uv` will 
not conflict with your current environment.

You just prefix commands:

```bash
pip install .   # old
uv pip install . # new
```

Using `uv` has a huge upside:

- Packages are cached globally
- Install once, and you can recreate environments almost instantly

You want a directory to put your code. If you don't have any code (e.g. you just want to run `uv run mbo`), the directory can even be empty!
Each "project", "repository", "codebase", whatever you want to call it, get's its own folder.

You will have a different environment for each.

``` {tip}
UV makes it very cheap to delete your entire environment thanks to it's [dependency caching](https://docs.astral.sh/uv/concepts/cache/#dependency-caching).

So don't be afraid to delete the environment and make a new one (delete the .venv folder, or whatever you named it doing uv venv .myname). It should be nearly instant to recreate it.
```

1. In your terminal (MBO developers primarily use powershell or git bash on Windows) make an environment:

  ```bash
  uv venv
  ```

  ```{warning} 
  We stronly discourage naming the environment. `uv venv` will make a folder `.venv`, which is a python standard.
  Most IDE's like VS Code and Pycharm will recognize this folder as an environment automatically.
  There have been issues, especially with Pycharm, using named environments.
  ```

2. Activate the environment (optional but good practice)

  On Linux/Mac:

  ```bash
  source .venv/bin/activate
  ```

  On Windows:

  ```bash
  source .venv/Scripts/activate
  ```

If you don't activate the environment, but you are running code from a directory that has a `.venv` folder, it will be used as the environment.

Activating your environment is good practice. 

3) Install packages

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

---

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

## Mixing conda and pip

`uv` will automatically fallback to using a conda environment if no `.venv/` folder is found:

```bash
USER at pop-os in ~/repos/work/mbo_utilities (master●)
$ uv pip list | grep imgui
Using Python 3.12.6 environment at: /home/USER/miniforge3
imgui-bundle              1.6.2
```

So even if you're using conda, `uv` will work all the same.

We highly recommend turning off automatic environment activation for `conda` in this case:

``` bash
conda config --set auto_activate_base false
```

You can still call `conda activate myenv`, but only if `conda` is called explicitily.
