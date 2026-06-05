"""Build hook: copy the starter notebooks from demos/ into the package so
`mbo init` works for wheel / `uv tool install` installs. demos/ stays the
single source of truth; the package copy is a build artifact (see .gitignore)."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

NOTEBOOKS = ("mbo_user_guide.ipynb", "lsp_user_guide.ipynb")


class build_py_with_notebooks(build_py):
    def run(self):
        root = Path(__file__).parent
        src = root / "demos"
        dst = root / "mbo_utilities" / "assets" / "notebooks"
        dst.mkdir(parents=True, exist_ok=True)
        for name in NOTEBOOKS:
            f = src / name
            if f.exists():
                shutil.copy2(f, dst / name)
        super().run()


setup(cmdclass={"build_py": build_py_with_notebooks})
