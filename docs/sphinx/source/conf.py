# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NESO-Particles'
copyright = '2022, UKAEA'
author = 'UKAEA'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_sidebars = {
    "*": ["sidebar-nav-bs"]
}

with open("./docs_version") as fh:
    version = fh.read().strip()

html_theme_options = {
    "check_switcher": False,
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/ExCALIBUR-NEPTUNE/NESO-Particles/dev-docs/docs/switcher.json",
        "version_match": version,
    },
    "navbar_start": ["navbar-logo", "version-switcher"]
}

breathe_projects = {"NESO-Particles": "../../build/doxygen/xml"}
breathe_default_project = "NESO-Particles"

# Enable referencing figures etc by number.
numfig = True

