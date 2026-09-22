# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""Everything WP2 treats as an input, split by what it configures:

    paths       where the airfoils and the results live
    atmosphere  ISA relations the flight conditions are built on
    mission     flight conditions, sweep, chord -- the sweep-theory reduction
    wing        planform design variables and the VLM mesh
    hld         high-lift device table, span/chord fractions, requirements
    solvers     alpha/Mach sweep ranges and solver settings, per stage
    scoring     scorecard criteria, weights and per-airfoil Korn kappa_A

The mission-level names (flight conditions, paths, ISA) are re-exported here,
so `from b12wp2 import config` then `config.CRUISE` / `config.RESULTS_DIR`
works as it always has. The stage-specific modules are not flattened into
this namespace -- three of them define an ETA_IN or an ALPHA_START of their
own -- so they are imported explicitly:

    from b12wp2.config import wing as wing_cfg
"""
from __future__ import annotations

from b12wp2.config.atmosphere import isa_atmosphere, sutherland_viscosity
from b12wp2.config.mission import (
    CHORD_M,
    CRUISE,
    FT_TO_M,
    LANDING,
    LANDING_SPEED_MS,
    SWEEP_RAD,
    FlightCondition,
    level_flight_cl,
)
from b12wp2.config.paths import AIRFOILS_DIR, RESULTS_DIR, WP2_DIR, discover_airfoils
from b12wp2.config.scoring import (
    KAPPA_A_CONVENTIONAL,
    KAPPA_A_SUPERCRITICAL,
    SCORING_WEIGHTS,
    kappa_a,
)

__all__ = [
    "AIRFOILS_DIR",
    "CHORD_M",
    "CRUISE",
    "FT_TO_M",
    "FlightCondition",
    "KAPPA_A_CONVENTIONAL",
    "KAPPA_A_SUPERCRITICAL",
    "LANDING",
    "LANDING_SPEED_MS",
    "RESULTS_DIR",
    "SCORING_WEIGHTS",
    "SWEEP_RAD",
    "WP2_DIR",
    "discover_airfoils",
    "isa_atmosphere",
    "kappa_a",
    "level_flight_cl",
    "sutherland_viscosity",
]
