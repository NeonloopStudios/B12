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

# class II CD0 build-up (scripts/class2_drag.py): one table and one breakdown
# chart per configuration, plus the CD0 of every configuration side by side
CD0_SUMMARY_CSV: Final[Path] = DRAG_DIR / "cd0_summary.csv"
CD0_COMPARISON_PNG: Final[Path] = DRAG_DIR / "cd0_comparison.png"
CD0_REPORT_TXT: Final[Path] = DRAG_DIR / "cd0_report.txt"  # every printed output


def drag_buildup_csv(configuration: str) -> Path:
    """CD0 build-up table of one configuration ("cruise", "takeoff", "landing")."""
    return DRAG_DIR / f"cd0_buildup_{configuration}.csv"


def drag_breakdown_png(configuration: str) -> Path:
    """CD0 breakdown chart of one configuration."""
    return DRAG_DIR / f"cd0_breakdown_{configuration}.png"


# --- inputs ---

CONFIGS_DIR: Final[Path] = WP3_DIR / "configs"
GEOMETRY_YAML: Final[Path] = CONFIGS_DIR / "geometry.yaml"  # every geometric input
PROPULSION_YAML: Final[Path] = CONFIGS_DIR / "propulsion.yaml"  # engine inputs

# drag due to lift at cruise (WP3.3a)
INDUCED_DRAG_CSV: Final[Path] = DRAG_DIR / "induced_drag.csv"
INDUCED_DRAG_PNG: Final[Path] = DRAG_DIR / "induced_drag.png"
DRAG_POLAR_CSV: Final[Path] = DRAG_DIR / "drag_polar.csv"
DRAG_POLAR_PNG: Final[Path] = DRAG_DIR / "drag_polar.png"

ITERATION_DIR: Final[Path] = RESULTS_DIR / "iteration"  # WP3.4 inputs and results

# Oswald factor and K of the clean, take-off and landing configurations,
# for the wing-loading diagram of WP3.4
OSWALD_CONFIGURATIONS_CSV: Final[Path] = ITERATION_DIR / "oswald_configurations.csv"

PROPULSION_DIR: Final[Path] = RESULTS_DIR / "propulsion"  # WP3.1 nacelle sizing
NACELLE_CSV: Final[Path] = PROPULSION_DIR / "nacelle.csv"
NACELLE_PNG: Final[Path] = PROPULSION_DIR / "nacelle.png"
NACELLE_REPORT_TXT: Final[Path] = PROPULSION_DIR / "nacelle_report.txt"
