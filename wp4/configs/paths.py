# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""Directories that every WP4 sub-package uses.

The paths come from the location of this file, not from the current working
directory. A script thus gives the same result from any start folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

WP4_DIR: Final[Path] = Path(__file__).resolve().parents[1]  # .../wp4/configs/paths.py
REPO_DIR: Final[Path] = WP4_DIR.parent
OUT_DIR: Final[Path] = WP4_DIR / "out"

# The airfoil coordinates stay in wp2 (one copy in the repository).
AIRFOILS_DIR: Final[Path] = REPO_DIR / "wp2" / "airfoils"
