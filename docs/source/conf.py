# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = 'Archetype Computing Adaptive System'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autodoc_mock_imports = [
    "aeon",
    "aeon.datasets",
    "avalanche",
    "avalanche.benchmarks",
    "avalanche.benchmarks.classic",
    "avalanche.evaluation",
    "avalanche.evaluation.metrics",
    "avalanche.logging",
    "avalanche.models",
    "avalanche.training",
    "avalanche.training.plugins",
    "avalanche.training.supervised",
    "deap",
    "deap.algorithms",
    "deap.tools",
    "graphviz",
    "gymnasium",
    "imageio",
    "matplotlib",
    "matplotlib.pyplot",
    "mpl_toolkits",
    "mpl_toolkits.mplot3d",
    "neat",
    "neat.config",
    "squid_inference",
    "sklearn",
    "sklearn.decomposition",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.preprocessing",
    "train_squid_center_cycle",
    "tqdm",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = "https://eu-emerge.github.io/archetype-computing-adaptive-system/"

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_copyright = False
html_show_sphinx = False
html_title = project

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
