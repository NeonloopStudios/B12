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

# --- results, one directory per analysis stage ---
#
#   section/   2D XFoil work: polars, Mach-critical sweeps and their figures
#   wing/      3D VSPAERO work, one subdirectory per airfoil
#   hld/       high-lift device comparison
#   scorecard/ the weighted selection tables

SECTION_DIR = RESULTS_DIR / "section"
DATA_DIR = SECTION_DIR / "data"  # XFoil section polars, Mach-critical sweeps
PLOTS_DIR = SECTION_DIR / "plots"  # per-airfoil section figures + the comparison

WING_DIR = RESULTS_DIR / "wing"  # 3D wing analysis, one subdirectory per airfoil
HLD_DIR = RESULTS_DIR / "hld"  # high-lift device comparison
SCORECARD_DIR = RESULTS_DIR / "scorecard"

SCORECARD_CSV = SCORECARD_DIR / "scorecard.csv"
SCORECARD_DETAIL_CSV = SCORECARD_DIR / "scorecard_detail.csv"


# --- the generated files, named in one place ---
#
# Every stage writes files another stage reads back by name, so the names
# are built here rather than f-strung at both ends of each hand-off.


def polar_csv(airfoil_stem: str, condition: str) -> Path:
    """XFoil polar of one airfoil at one flight condition ("cruise"/"landing")."""
    return DATA_DIR / f"{airfoil_stem}_{condition}_polar.csv"


def mcrit_csv(airfoil_stem: str) -> Path:
    """Karman-Tsien Mach sweep: cp_min_corrected and cp_crit vs Mach."""
    return DATA_DIR / f"{airfoil_stem}_mcrit.csv"


def baseline_cp_csv(airfoil_stem: str) -> Path:
    """Cp(x) at the subsonic baseline solve the Mach sweep starts from."""
    return DATA_DIR / f"{airfoil_stem}_baseline_cp.csv"


MCRIT_SUMMARY_CSV = DATA_DIR / "mcrit_summary.csv"


def section_plots_dir(airfoil_stem: str) -> Path:
    """Where one airfoil's 2D section figures go."""
    return PLOTS_DIR / airfoil_stem


COMPARISON_PLOTS_DIR = PLOTS_DIR / "comparison"


def wing_out_dir(airfoil_stem: str) -> Path:
    """Where the 3D wing analysis of one airfoil writes its results."""
    return WING_DIR / airfoil_stem


# VSPAERO's own run files (the .vsp3 model, its solver input/output). Not
# results: they are regenerated on every run and are gitignored.
RUN_SUBDIR = "vspaero_run"


def wing_run_dir(airfoil_stem: str) -> Path:
    return wing_out_dir(airfoil_stem) / RUN_SUBDIR


HLD_RUN_DIR = HLD_DIR / RUN_SUBDIR


def discover_airfoils() -> list[Path]:
    """All candidate airfoil .dat files, sorted for deterministic ordering.

    Deliberately a directory scan, not a hardcoded list -- the candidate set
    has already changed once (EPPLER_395 -> NACA_64212); every later stage
    must pick up whatever is in wp2/airfoils/ without code changes.
    """
    return sorted(AIRFOILS_DIR.glob("*.dat"))
