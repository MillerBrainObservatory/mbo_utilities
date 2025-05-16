# Virtual Environments

```{admonition} TLDR
:class: dropdown
If you know conda, just use that.

Otherwise, install and learn a bit about [uv](https://docs.astral.sh/uv/) and use it purely with `uv pip install <PACKAGE>` until you learn more about its high-level features.
```

Most bugs and frustration in Python come from mismanaging virtual environments.

Example: You install Python 3.10 on Windows or Linux. You must add this to your system PATH to use it, or it is added to your system path by default.
Now you want to install a package that requires Python 3.11. You're screwed because Python 3.10 being on you PATH means it will always be used unless
explicitly told not to (e.g. by activating a conda/venv virtual environment). The behavior that you will experience will vary drastically depending on if 
you use `conda`, `venv`, or `uv`. In most circumstances, you will be using Python 3.10 and you won't know it.

This happens to experienced developers the same as anyone else. You could learn the in's and out's for all of the virtual environments below.

You will only confuse yourself doing this.

There are three camps:

1. Tools that also handle system dependencies like ffmpeg, opencv, cuda (mamba, conda)
2. Tools that handle only Python packages (venv, pyenv)
3. Tools that wrap the above packages to make them easier to use (pixi, uv)

## Your Options

::::{grid}
:::{grid-item-card} System + Python
- conda
- miniconda
- miniforge
- anaconda
- mamba
- micromamba
- pixi
:::

:::{grid-item-card} Python-only
- pip
- pip-tools
- pipx
- venv / pyvenv
- virtualenv
- pipenv
- poetry
- twine
- hatch
- uv
- asdf
:::
::::

---

## Why not `conda`?

- `venv` can build most system packages you need from source.
- Most Python libraries now ship prebuilt wheels (binaries).
- The main selling point for `conda` (handling system binaries) is much less relevant today.
- Even the conda maintainers have moved toward venv-based setups (especially uv).

If you **must** use conda, **only** use `miniforge3`.  
Many recommended setup steps for miniconda/anaconda are basically mimicking the defaults of `miniforge3`.

If you already know conda, that's fine — just **avoid mixing** `pip install` and `conda install` after your initial setup.

---

## uv: Super pip

[uv](https://docs.astral.sh/uv/) is a complete drop-in replacement for `pip`.  
If you know `pip`, you already know `uv`.

You just prefix commands:

```bash
pip install .   # old
uv pip install . # new
```

Using `uv` has a huge upside:

- Packages are cached globally
- Install once, and you can recreate environments almost instantly

---

(uv_cheatsheet)=
```{list-table} UV CLI Cheatsheet (WIP)
:header-rows: 1
:name: uv-commands

* - **Command**
  - **Description**
* - `pyproject.toml`, `uv.lock`
  - Core files
* - `uv sync`, `uv run`
  - Create envs (auto-creates a venv on first use)
* - `uv sync` / `uv run`
  - Install & sync deps
* - `uv add <package>` / `uv remove <package>`
  - Add/remove packages (updates `pyproject.toml`, `uv.lock`, and your env)
* - `uv sync --upgrade-package <pkg>` / `uv lock --upgrade`
  - Upgrade single/all packages
```

---

## Mixing conda and pip

`uv` will automatically fallback to using a conda environment if no `.venv/` folder is found:

```bash
flynn at pop-os in ~/repos/work/mbo_utilities (master●)
$ uv pip list | grep imgui
Using Python 3.12.6 environment at: /home/flynn/miniforge3
imgui-bundle              1.6.2
```

So even if you're using conda, `uv` will work all the same.

We highly recommend turning off automatic environment activation for `conda` in this case:

``` bash
conda config --set auto_activate_base false
```

You can still call `conda activate myenv`, but only if `conda` is called explicitily.
