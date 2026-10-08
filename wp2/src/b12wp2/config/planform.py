# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-08
"""The wing planform both stages are built on: the four design variables and
what follows from them for a straight-tapered wing.

The 2D sweep-theory reduction (mission.py) needs the MAC and the half-chord
sweep, the 3D model (config/wing.py, wing/geometry.py) needs the same planform.
Both read it from here, so they cannot drift apart. Plain math only -- no
OpenVSP or XFoil import -- so either Python environment can use it.
"""
from __future__ import annotations

import math

# --- design variables (WP3 redesign) ---

S_REF = 66.7  # wing area, both halves [m^2]
AR = 9.50  # aspect ratio [-]
TAPER = 0.40  # c_tip / c_root [-]
SWEEP_C4_DEG = 30.0  # quarter-chord sweep [deg]
SWEEP_LOC = 0.25  # chord fraction SWEEP_C4_DEG is measured at

# Chord line whose sweep the 2D reduction (M_n, Cl_n, Re_n, normal section)
# and the 3D Korn wave-drag estimate both use: the half chord, as in ADSEE.
SECTION_SWEEP_LOC = 0.5

# --- derived (straight-tapered wing) ---

B = math.sqrt(S_REF * AR)  # span [m]
B_HALF = B / 2
C_ROOT = 2 * S_REF / (B * (1 + TAPER))
C_TIP = TAPER * C_ROOT
MAC = 2 / 3 * C_ROOT * (1 + TAPER + TAPER**2) / (1 + TAPER)
Y_MAC = B / 6 * (1 + 2 * TAPER) / (1 + TAPER)

TAN_SWEEP_LE = math.tan(math.radians(SWEEP_C4_DEG)) + SWEEP_LOC * (C_ROOT - C_TIP) / B_HALF


def sweep_at(chord_fraction: float) -> float:
    """Sweep angle [rad] of the line at the given chord fraction."""
    return math.atan(TAN_SWEEP_LE - chord_fraction * (C_ROOT - C_TIP) / B_HALF)


SECTION_SWEEP_RAD = sweep_at(SECTION_SWEEP_LOC)
