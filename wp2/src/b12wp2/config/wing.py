# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""Wing planform design variables and the VLM mesh they are meshed with.

S, AR, taper and the quarter-chord sweep are not repeated here: they live in
config/planform.py, which the XFoil sweep-theory reduction (mission.py) also
reads, so the 3D model and the 2D sections cannot drift apart. They are
re-exported below under their usual names.

Everything else in this module is an input of the 3D model only. The
quantities derived from the planform (span, chords, MAC, sweep of any chord
line) are computed in config/planform.py.
"""
from __future__ import annotations

from b12wp2.config.planform import AR as AR  # re-exported
from b12wp2.config.planform import S_REF as S_REF
from b12wp2.config.planform import SWEEP_C4_DEG
from b12wp2.config.planform import SWEEP_LOC as SWEEP_LOC
from b12wp2.config.planform import TAPER as TAPER
from b12wp2.config.paths import AIRFOILS_DIR

# --- planform ---

SWEEP_DEG = SWEEP_C4_DEG  # quarter-chord sweep, from config/planform.py

DIHEDRAL_DEG = 2.6
TWIST_TIP_DEG = -2.0  # tip twist relative to the root (negative = wash-out)
TWIST_LOC = 0.25  # twist axis as a chord fraction
INCIDENCE_DEG = 2.0  # wing incidence, added on top of the VSPAERO angle of attack

X_LE_ROOT = 0.0  # [m]
Z_ROOT = 0.3  # [m]

# --- sections ---

AIRFOIL_ROOT = AIRFOILS_DIR / "NASA_SC(2)-0712.dat"
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
