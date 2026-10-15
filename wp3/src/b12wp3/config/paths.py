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

<<<<<<< HEAD
# --- inputs ---

CONFIGS_DIR: Final[Path] = WP3_DIR / "configs"
GEOMETRY_YAML: Final[Path] = CONFIGS_DIR / "geometry.yaml"  # every geometric input
=======
# drag due to lift at cruise (WP3.3a)
INDUCED_DRAG_CSV: Final[Path] = DRAG_DIR / "induced_drag.csv"
INDUCED_DRAG_PNG: Final[Path] = DRAG_DIR / "induced_drag.png"
DRAG_POLAR_CSV: Final[Path] = DRAG_DIR / "drag_polar.csv"
DRAG_POLAR_PNG: Final[Path] = DRAG_DIR / "drag_polar.png"

ITERATION_DIR: Final[Path] = RESULTS_DIR / "iteration"  # WP3.4 inputs and results

# Oswald factor and K of the clean, take-off and landing configurations,
# for the wing-loading diagram of WP3.4
OSWALD_CONFIGURATIONS_CSV: Final[Path] = ITERATION_DIR / "oswald_configurations.csv"
>>>>>>> e2a97674cc76c9266c85c4a6c536a737d5c4b0db
