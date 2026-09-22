# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Where WP2 reads and writes: the directories, and the candidate-airfoil
discovery every stage starts from.

Paths are derived from this file's own location, never from the current
working directory, so a script gives the same answer whether it was started
from wp2/ or from the repo root.
"""
from __future__ import annotations

from pathlib import Path

WP2_DIR = Path(__file__).resolve().parents[3]  # .../wp2/src/b12wp2/config/paths.py -> wp2/
AIRFOILS_DIR = WP2_DIR / "airfoils"
RESULTS_DIR = WP2_DIR / "results"


def discover_airfoils() -> list[Path]:
    """All candidate airfoil .dat files, sorted for deterministic ordering.

    Deliberately a directory scan, not a hardcoded list -- the candidate set
    has already changed once (EPPLER_395 -> NACA_64212); every later stage
    must pick up whatever is in wp2/airfoils/ without code changes.
    """
    return sorted(AIRFOILS_DIR.glob("*.dat"))
