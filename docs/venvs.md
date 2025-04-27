# Virtual Environments

```{admonition} TLDR
:class: dropdown
If you know conda, just use that.

Otherwise, install and learn a bit about [uv](https://docs.astral.sh/uv/) and use it purely with `uv pip install <PACKAGE>` until you learn more about its high-level features.
```

Most bugs and frustration in Python come from mismanaging virtual environments.

It's tempting to want a firm grasp on all the options, their pros/cons, use cases. Just don't.

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

## uv Cheatsheet

::::{tab-set}

:::{tab-item} Core Files
- **Dependency file**: `pyproject.toml`
- **Lock file**: `uv.lock`
:::

:::{tab-item} Creating environments
- `uv sync`
- `uv run`

Uv will automatically create a virtualenv the first time you use it.
:::

:::{tab-item} Installing packages
- `uv sync`
- `uv run`

Installs *and* syncs all dependencies.
:::

:::{tab-item} Adding/Removing packages
- Add: `uv add package`
- Remove: `uv remove package`

Updates `pyproject.toml`, `uv.lock`, and your environment automatically.
:::

:::{tab-item} Upgrading
- Upgrade one package: `uv sync --upgrade-package package`
- Upgrade all packages: `uv lock --upgrade`
:::

::::

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

---

````{tab-set-code}

```{code-block} python
# Example: python file
def example():
    print("Hello World")
```

```{code-block} bash
# Example: bash command
uv pip install .
```
