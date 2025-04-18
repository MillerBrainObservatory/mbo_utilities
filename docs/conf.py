# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..")))
sys.path.insert(0, os.path.abspath(os.path.join("..", "mbo_utilities")))

project = "mbo_utilities"
copyright = "2024, Elizabeth R. Miller Brain Observatory | The Rockefeller University. All Rights Reserved"
release = "0.0.1"

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
html_static_path = ["_static"]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_title = "mbo_utilities"

html_logo = "./_static/logo_utilities.png"
html_favicon = "./_static/utilities.png"
html_theme = "sphinx_book_theme"

html_short_title = "mbo utilities"
html_css_files = ["custom.css"]
html_copy_source = True
html_file_suffix = ".html"
html_use_modindex = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "fastplotlib": ("https://www.fastplotlib.org/", None),
    "numpydoc": ("https://numpydoc.readthedocs.io/en/latest", None),
    "mbo": ("https://millerbrainobservatory.github.io/", None),
    "lbm_suite2p_python": ("https://millerbrainobservatory.github.io/LBM-Suite2p-Python/", None),
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
            "icon": "_static/icon_mbo_home.png",
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
