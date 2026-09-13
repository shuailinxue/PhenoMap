"""Sphinx configuration for the PhenoMap documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "step1"))
sys.path.insert(0, str(ROOT / "step2"))
sys.path.insert(0, str(ROOT / "step3"))

project = "PhenoMap"
author = "PhenoMap developers"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "PhenoMap documentation"
html_show_sourcelink = True

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"
nb_merge_streams = True
