# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 command-line entry points. Every module in here is a runnable step
(`python -m scripts.<name>` from wp2/); the analysis itself lives in the
b12wp2 package under wp2/src/.

The b12wp2 package is not pip-installed -- it is imported straight from the
working tree -- so wp2/src/ is put on sys.path here, when the first
`scripts.<name>` module is imported. That keeps `python -m scripts.<name>`
working from wp2/ with no install step, the same way it did when the
library modules still lived in this directory. pytest gets the same two
paths from pyproject.toml's `pythonpath`, and the OpenVSP scripts add them
themselves when run as a plain file (no package context).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
