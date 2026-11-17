# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4 directories, from the location of this file (not the working directory)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

WP4_DIR: Final[Path] = Path(__file__).resolve().parents[1]
REPO_DIR: Final[Path] = WP4_DIR.parent
OUT_DIR: Final[Path] = WP4_DIR / "out"

AIRFOILS_DIR: Final[Path] = REPO_DIR / "wp2" / "airfoils"
