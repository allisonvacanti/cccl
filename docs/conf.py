# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import json

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

with open('../cccl-version.json', 'r') as f:
    cccl_version = json.load(f)['full']

project = 'CCCL'
copyright = f"{datetime.datetime.today().year}, NVIDIA Corporation"
author = 'NVIDIA'
release = cccl_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['breathe']

breathe_projects = {
    'cub': '../build/docs/cub/doxygen/xml',
    'cudax': '../build/docs/cudax/doxygen/xml',
    'thrust': '../build/docs/thrust/doxygen/xml',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'nvidia_sphinx_theme'
