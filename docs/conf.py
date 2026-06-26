"""Sphinx configuration for the colstore documentation.

Builds the HTML site from the Markdown guides in this directory plus an
autodoc/autosummary API reference pulled from the package's NumPy-style
docstrings. Importing ``colstore`` requires the compiled extension, so the
docs build installs the package first (see ``.github/workflows/docs.yml``).
"""

from __future__ import annotations

import colstore

# -- Project information ------------------------------------------------------

project = "colstore"
author = "Alkaid Cheng"
copyright = "2026, Alkaid Cheng"
release = colstore.__version__
version = release.split("+")[0].split(".dev")[0]

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Autodoc / autosummary ----------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# NumPy-style docstrings, not Google.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# -- Cross-project links ------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyarrow": ("https://arrow.apache.org/docs", None),
}

# -- MyST (Markdown) ----------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"colstore {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "source_repository": "https://github.com/AlkaidCheng/colstore/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#0f6e56",
        "color-brand-content": "#0f6e56",
    },
    "dark_css_variables": {
        "color-brand-primary": "#54d6a3",
        "color-brand-content": "#54d6a3",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/AlkaidCheng/colstore",
            "html": "",
            "class": "fa-brands fa-github fa-lg",
        },
    ],
}
