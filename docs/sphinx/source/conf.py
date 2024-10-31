# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NESO-Particles'
copyright = '2022, UKAEA'
author = 'UKAEA'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe"]

#templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.png"

html_sidebars = {
    "**": ["globaltoc.html"]
}

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_align": "left",
    "primary_sidebar_end": [],
    "navigation_depth": 0,
    "show_nav_level": 3,
    "logo": {
        "text": "",
        "image_light": "neso_particles_logo_light_small.png",
        "image_dark": "neso_particles_logo_dark_small.png",
    },
}

import os
docs_version = "./docs_version"
if os.path.exists(docs_version):

    with open(docs_version) as fh:
        version = fh.read().strip()

    html_theme_options.update({
        "check_switcher": False,
        "switcher": {
            "json_url": "https://excalibur-neptune.github.io/NESO-Particles/switcher.json",
            "version_match": version,
        },
        "navbar_start": ["navbar-logo", "version-switcher"]
    })

breathe_projects = {"NESO-Particles": "../../build/doxygen/xml"}
breathe_default_project = "NESO-Particles"

# Enable referencing figures etc by number.
numfig = True

