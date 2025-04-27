# Virtual Environments

```{admonition} TLDR
:class: dropdown
If you know conda, just use that.

Otherwise, install and learn a bit about [uv](https://docs.astral.sh/uv/) and use it purely with `uv pip install <PACKAGE>` until you learn more about it's high-level features.
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

`venv` can build most system packages you need from source.

Most Python libraries now ship prebuilt wheels (binaries, built versions of code), the main selling point for `conda` was handling binaries your operating system needs which is less relevant today.

Even the conda maintainers have moved toward venv-based setups (uv in particular).

If you **must** use conda, **only** use `miniforge3`. Many recommended setup steps for miniconda and anaconda are those mimicing the defaults in `miniforge3`.

If you know conda workflows, stick with it, but avoid `conda install` unless absolutely necessary after your initial environment setup. `conda` works fine as an environment to pip install into, as long as you dont mix `pip install` and `conda install`.

---

## uv: Super pip

(https://docs.astral.sh/uv/) is a complete drop-in replacement for `pip`.If you know pip, you already know uv.

You just prefix commands:

```bash
pip install .   # old
uv pip install . # new
```

Letting `uv` manage your installs has a huge upside:

Packages are cached globally

Install once, then environments can be blown away and recreated almost instantly

---

## uv Cheatsheet

::::{tabs}

:::{tab-item} Core Files

**Dependency file**: `pyproject.toml`

**Lock file**: `uv.lock`
:::

:::{tab-item} Creating environments

`uv sync`

`uv run`

Uv will automatically create a virtualenv the first time you use it.
:::

:::{tab-item} Installing packages

`uv sync`

`uv run`

Installs *and* syncs all dependencies.
:::

:::{tab-item} Adding/Removing

Add: `uv add package`

Remove: `uv remove package`

Updates `pyproject.toml`, `uv.lock`, and your env.
:::

:::{tab-item} Upgrading

Single package: `uv sync --upgrade-package package`

All packages: `uv lock --upgrade`
:::

::::
