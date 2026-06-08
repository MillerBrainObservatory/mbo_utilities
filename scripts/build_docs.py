#!/usr/bin/env python
"""
build and serve documentation locally.

usage:
    uv run python scripts/build_docs.py          # build and open (auto-installs docs deps)
    uv run python scripts/build_docs.py --clean  # clean build first
    uv run python scripts/build_docs.py --no-open  # build without opening
    uv run python scripts/build_docs.py --skip-install  # skip the dependency check
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

# non-base-sphinx extension modules conf.py needs; presence of these is the
# signal that the docs toolchain (the project's [docs] extra) is installed.
_DOCS_MODULES = ("myst_nb", "sphinx_book_theme", "sphinx_design", "sphinx_copybutton")


def ensure_docs_deps(root: Path, reinstall: bool = False) -> bool:
    """install the project's docs extra (.[docs]) if the toolchain is missing.

    augments the active environment with `uv pip install` (no pruning, so the
    rest of the dev env is preserved); falls back to pip.
    """
    if not reinstall and all(importlib.util.find_spec(m) for m in _DOCS_MODULES):
        return True

    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "-e", ".[docs]"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[docs]"]

    print(f"\ninstalling docs dependencies:\n  {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=root).returncode == 0


def build_docs(root: Path, clean: bool = False, install_deps: bool = True,
               reinstall_deps: bool = False):
    """build sphinx documentation."""
    docs_dir = root / "docs"
    build_dir = docs_dir / "_build"
    html_dir = build_dir / "html"

    # ensure the docs toolchain (myst_nb, theme, ...) is installed
    if install_deps and not ensure_docs_deps(root, reinstall=reinstall_deps):
        print("\n[ERROR] failed to install docs dependencies")
        return None

    # clean if requested
    if clean and build_dir.exists():
        print("cleaning build directory...")
        shutil.rmtree(build_dir)

    # build html (conf.py setup() copies the user_guide notebook into docs/)
    print("\nbuilding documentation...")
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(docs_dir), str(html_dir)],
        cwd=root,
    )

    if result.returncode != 0:
        print("\n[ERROR] documentation build failed")
        return None

    return html_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(description="build documentation")
    parser.add_argument("--clean", "-c", action="store_true", help="clean build directory first")
    parser.add_argument("--no-open", action="store_true", help="don't open browser")
    parser.add_argument("--skip-install", action="store_true", help="don't auto-install docs dependencies")
    parser.add_argument("--reinstall-deps", action="store_true", help="reinstall docs dependencies even if present")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    index_path = build_docs(root, clean=args.clean,
                            install_deps=not args.skip_install,
                            reinstall_deps=args.reinstall_deps)

    if index_path is None:
        sys.exit(1)

    print(f"\n[SUCCESS] documentation built: {index_path}")

    if not args.no_open:
        print("opening in browser...")
        webbrowser.open(index_path.as_uri())


if __name__ == "__main__":
    main()
