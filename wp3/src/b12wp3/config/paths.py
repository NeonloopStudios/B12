# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Where WP3 reads its input files and writes its results.

Paths are derived from this file's own location, never from the current
working directory, so a script gives the same answer from wp3/ or from the
repo root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# .../wp3/src/b12wp3/config/paths.py -> wp3/
WP3_DIR: Final[Path] = Path(__file__).resolve().parents[3]
RESULTS_DIR: Final[Path] = WP3_DIR / "results"

# --- results, one directory per analysis stage ---

DRAG_DIR: Final[Path] = RESULTS_DIR / "drag"  # class II drag (WP3.3a)

DRAG_BUILDUP_CSV: Final[Path] = DRAG_DIR / "cd0_buildup.csv"
DRAG_BREAKDOWN_PNG: Final[Path] = DRAG_DIR / "cd0_breakdown.png"

# --- inputs ---

CONFIGS_DIR: Final[Path] = WP3_DIR / "configs"
GEOMETRY_YAML: Final[Path] = CONFIGS_DIR / "geometry.yaml"  # every geometric input
