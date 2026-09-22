# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""The WP2 selection scorecard: the criteria, their weights, how each one is
scored, and the per-airfoil Korn kappa_A that the transonic criterion needs.

All of it is decision data, not derived from the airfoil coordinates, so it
lives here and is read by b12wp2.scoring.scorecard rather than written into
that module.
"""
from __future__ import annotations

# --- Korn equation kappa_A per airfoil (used by scripts/mcrit_sweep.py) ---
#
# M_dd + t/c + Cl/10 = kappa_A. kappa_A ~0.87 for a conventional section,
# ~0.95 for a supercritical one -- a real design-category judgment call,
# not something derivable from the .dat coordinates, so it's recorded here
# as an explicit per-airfoil decision rather than defaulted. NASA_SC(2)-0712
# is a NASA supercritical section (SC(2) series) by design; the other three
# candidates are conventional. If the candidate set changes again, this
# mapping must be updated too -- kappa_a() raises rather than silently
# assuming "conventional" for an unmapped airfoil.
KAPPA_A_CONVENTIONAL = 0.87
KAPPA_A_SUPERCRITICAL = 0.95

_KAPPA_A_BY_AIRFOIL: dict[str, float] = {
    "NACA_25112": KAPPA_A_CONVENTIONAL,
    "NACA_64212": KAPPA_A_CONVENTIONAL,
    "lockheed_c5a_bl758": KAPPA_A_CONVENTIONAL,
    "NASA_SC(2)-0712": KAPPA_A_SUPERCRITICAL,
}


def kappa_a(airfoil_stem: str) -> float:
    """Korn equation kappa_A for a given airfoil (by its .dat filename stem)."""
    try:
        return _KAPPA_A_BY_AIRFOIL[airfoil_stem]
    except KeyError:
        raise ValueError(
            f"kappa_a: no conventional/supercritical classification recorded "
            f"for airfoil {airfoil_stem!r} -- add it to _KAPPA_A_BY_AIRFOIL "
            "in config/scoring.py before running scripts/mcrit_sweep.py for this airfoil"
        ) from None


# --- WP2 scoring matrix (selection criteria and their weights) ---

SCORING_WEIGHTS: dict[str, float] = {
    "mach_critical": 0.25,
    "cl_cd_cruise": 0.30,
    "cl_max_landing": 0.15,
    "stall_margin": 0.15,  # stall angle minus cruise angle; raised from 0.125
    "cl_zero_angle": 0.05,
    "pitching_moment": 0.10,  # lowered from 0.125
}

if abs(sum(SCORING_WEIGHTS.values()) - 1.0) > 1e-9:
    raise AssertionError(
        f"SCORING_WEIGHTS must sum to 1.0, got {sum(SCORING_WEIGHTS.values())}"
    )

# Direction each raw criterion is scored in. "max" = higher raw value is
# better (normalize ascending). "min_abs" = smaller |value| is better
# (normalize the absolute value, then invert): pitching_moment is
# smaller-|Cm|-is-better (standard trim-drag rationale), cl_zero_angle is
# higher-is-better. The other four criteria (mach_critical, cl_cd_cruise,
# cl_max_landing, stall_margin) are unambiguously higher-is-better on
# their own terms.
CRITERION_DIRECTION: dict[str, str] = {
    "mach_critical": "max",
    "cl_cd_cruise": "max",
    "cl_max_landing": "max",
    "stall_margin": "max",
    "cl_zero_angle": "max",
    "pitching_moment": "min_abs",
}

CRITERION_LABELS: dict[str, str] = {
    "mach_critical": "Mach Critical",
    "cl_cd_cruise": "Cl/Cd cruise",
    "cl_max_landing": "Cl max landing",
    "stall_margin": "Stall Angle - Angle in Cruise",
    "cl_zero_angle": "Zero Angle Cl",
    "pitching_moment": "Pitching Moment",
}
