# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'TransformInvariantNMF'
copyright = '2021, Adrian Šošić, Mathias Winkel'
author = 'Adrian Šošić, Mathias Winkel'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#module-sphinx.ext.napoleon
    'm2r2',
]

autoapi_type = 'python'
autoapi_dirs = ['../tnmf', ]
autoapi_options = ['members', 'undoc-members', 'no-private-members', 'show-inheritance',
                   'show-module-summary', 'special-members', 'imported-members', ]

napoleon_attr_annotations = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False

m2r_parse_relative_links = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', ]
html_logo = "logos/tnmf_logo.svg"
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}

html_css_files = [
    'css/emdgroup.css',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    "torch": ("https://pytorch.org/docs/master/", None),
    }