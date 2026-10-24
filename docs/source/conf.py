# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# чтобы Sphinx находил ваш пакет (если не устанавливаете через pip)
sys.path.insert(0, os.path.abspath("../.."))  # путь до корня проекта

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Whydra'
copyright = '2026, Skoltech'
author = 'Skoltech'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",   # Google/NumPy style docstrings
    "sphinx.ext.viewcode",   # ссылки на исходники
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",   # обязательно
    'sphinx.ext.todo',      # добавлено из текстового описания
    'sphinx.ext.ifconfig',  # добавлено из текстового описания
]

nbsphinx_execute = "never"  # не запускать юпитер ноутбуки автоматически, в случае, если они когда то будут частью документации

templates_path = ['_templates']
exclude_patterns = []

language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
# html_theme = 'bizstyle'
# html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'titles_only': False,
}
autodoc_typehints = "description"