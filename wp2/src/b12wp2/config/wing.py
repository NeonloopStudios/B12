# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""Wing planform design variables and the VLM mesh they are meshed with.

The planform numbers are the WP1 sizing used in the stand-alone VSP_analysis
study (S, AR, taper, dihedral, incidence). The quarter-chord sweep is not
repeated here: it is mission.SWEEP_RAD, so the 3D model and the XFoil
sweep-theory reduction cannot drift apart.

Everything in this module is an input. The quantities derived from it (span,
chords, MAC, sweep of any chord line) are computed in b12wp2.wing.geometry.
"""
from __future__ import annotations

import math

from b12wp2.config.mission import SWEEP_RAD
from b12wp2.config.paths import AIRFOILS_DIR

# --- planform ---

S_REF = 66.7  # wing area, both halves [m^2]
AR = 9.50  # aspect ratio [-]
TAPER = 0.40  # c_tip / c_root [-]

SWEEP_LOC = 0.25  # chord fraction the sweep is measured at (quarter chord)
SWEEP_DEG = math.degrees(SWEEP_RAD)  # 24.02 deg

DIHEDRAL_DEG = 2.6
TWIST_TIP_DEG = 0.0  # tip twist relative to the root (negative = wash-out)
TWIST_LOC = 0.25  # twist axis as a chord fraction
INCIDENCE_DEG = 2.0  # wing incidence, added on top of the VSPAERO angle of attack

X_LE_ROOT = 0.0  # [m]
Z_ROOT = 0.3  # [m]

# --- sections ---

AIRFOIL_ROOT = AIRFOILS_DIR / "NACA_25112.dat"
AIRFOIL_TIP = AIRFOIL_ROOT

# --- VLM mesh ---
# The stand-alone study used OpenVSP's default of 5 spanwise strips per
# half-wing, too coarse to resolve the spanwise cl distribution the strip
# viscous correction is evaluated on. OUT_CLUSTER < 1 clusters strips
# towards the tip, where the load falls off fastest.
SPAN_TESS = 25  # spanwise section tessellation -> SPAN_TESS - 1 strips per half-wing
CHORD_TESS = 33  # chordwise tessellation (Tess_W)
OUT_CLUSTER = 0.5

WING_NAME = "Wing"  # the geom's name inside the OpenVSP model
