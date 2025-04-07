# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join("..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "mbo_utilities")))

project = "mbo_utilities"
copyright = "2024, Elizabeth R. Miller Brain Observatory | The Rockefeller University. All Rights Reserved"
release = "0.0.1"

# Copy example notebooks for rendering in the docs
print(f'Copying sphinx source files ...')
source_dir = Path(__file__).resolve().parent.parent / "demos"
dest_dir = Path(__file__).resolve().parent / "user_guide"

def copy_with_overwrite(src: Path, dst: Path):
    print(f'source: {src} being copied to destination: {dst}')
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

if source_dir.exists():
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.rglob("*"):
        relative_path = item.relative_to(source_dir)
        destination_path = dest_dir / relative_path
        copy_with_overwrite(item, destination_path)

exclude_patterns = ["Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.video",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_tippy",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

autodoc_mock_imports = ['scanreader']

nb_execution_mode = "off"

myst_admonition_enable = True
myst_amsmath_enable = True
myst_html_img_enable = True
myst_url_schemes = ("http", "https", "mailto")

images_config = {"cache_path": "./_images/"}

templates_path = ["_templates"]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "mbo_utilities"

# html_logo = "_static/lcp_logo.svg"
# html_favicon = "_static/icon_caiman_python.svg"
html_theme = "sphinx_book_theme"

html_short_title = "mbo utilities"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_copy_source = True
html_file_suffix = ".html"
# html_use_modindex = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "mbo": (
        "https://millerbrainobservatory.github.io/",
        None,
    ),
    "caiman": ("https://caiman.readthedocs.io/en/latest/", None),
    "mesmerize": ("https://mesmerize-core.readthedocs.io/en/latest", None),
    "suite2p": ("https://suite2p.readthedocs.io/en/latest/", None),
}

intersphinx_disabled_reftypes = ["*"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/MillerBrainObservatory/mbo_utilties/",
    "repository_branch": "master",
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "show_toc_level": 3,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "MBO User Hub",
            "url": "https://millerbrainobservatory.github.io/",
            "icon": "./_static/icon_mbo_home.png",
            "type": "local",
        },
        {
            "name": "MBO Github",
            "url": "https://github.com/MillerBrainObservatory/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Connect with MBO",
            "url": "https://mbo.rockefeller.edu/contact/",
            "icon": "fa-regular fa-address-card",
            "type": "fontawesome",
        },
    ],
}
