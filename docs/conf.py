import os
import logging
import sys

mariner_path = os.path.abspath("../mariner")
fleet_path = os.path.abspath("../fleet")
api_path = os.path.abspath("../api")

logging.basicConfig(level=logging.INFO)

logging.info("Adding packages to sys.path")
logging.info(f"Mariner path: {mariner_path}")
logging.info(f"Fleet path: {fleet_path}")
logging.info(f"Api path: {api_path}")


sys.path.insert(0, mariner_path)
sys.path.insert(0, fleet_path)
sys.path.insert(0, api_path)

# Checks if importing added paths works:
try:
    import mariner
    import fleet
    import api
except ImportError:
    logging.error("Failed to import packages")
    sys.exit(1)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Mariner"
copyright = "2023, Tyler Shimko"
author = "Tyler Shimko"
release = "0.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx_toolbox.confval",
    "sphinx_copybutton",
]

copybutton_exclude = ".linenos, .gp, .go"

templates_path = ["templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
