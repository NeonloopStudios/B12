# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Solver run settings: which angles, Machs and mesh each stage sweeps.

Numerical choices, not physics -- changing anything here changes how much is
computed and how finely, not what is being modelled. They are collected in
one place so a sweep can be widened without going through the scripts, and
because several of them (the XFoil alpha ranges, the VSPAERO alpha range)
are reported in the write-up as part of the method.

Names are prefixed by the stage that uses them, since three different stages
sweep an angle of attack over three different ranges for three different
reasons.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

# --- XFoil section polars (scripts/run_cruise_polars.py, run_landing_polars.py) ---

CRUISE_ALPHA_LOW_DEG = -6.0
CRUISE_ALPHA_HIGH_DEG = 20.0

# Low-speed, clean-configuration stall angles run noticeably higher than the
# cruise polar's (compressibility at M_n~0.70 cut cruise stall to ~3-5 deg;
# at landing's M~0.2 that effect is negligible), so this sweep goes wider.
LANDING_ALPHA_LOW_DEG = -8.0
LANDING_ALPHA_HIGH_DEG = 25.0

# --- Mach-critical sweep (scripts/mcrit_sweep.py) ---

# Baseline solve Mach: safely subsonic, so the Cp(x) it returns is the
# (approximately) incompressible distribution the Karman-Tsien correction
# expects as its input.
MCRIT_BASELINE_MACH = 0.2
MCRIT_MACH_GRID: npt.NDArray[np.float64] = np.arange(0.30, 0.951, 0.005)

# --- VSPAERO cruise sweep (scripts/vspaero_analysis.py) ---

VSPAERO_ALPHA_START = -4.0  # [deg], VSPAERO angle of attack (wing incidence comes on top)
VSPAERO_ALPHA_END = 10.0
VSPAERO_ALPHA_NPTS = 15

VSPAERO_WAKE_ITER = 5
VSPAERO_N_CPU = 4

KORN_SWEEP_LOC = 0.5  # chord fraction of the sweep line used in the Korn equation
SWEEP_DRAG_MODE = "friction"  # baseline, see b12wp2.wing.viscous_correction.sweep_drag_factor
SWEEP_DRAG_MODE_SENSITIVITY = "cos3"

# --- VSPAERO landing sweep for the clean wing (scripts/hld_analysis.py) ---

HLD_ALPHA_START = -4.0
HLD_ALPHA_END = 20.0
HLD_ALPHA_NPTS = 25
HLD_LINEAR_RANGE = (-4.0, 6.0)  # alpha range for the CL_alpha fit [deg]

# --- Lift-curve slope fit (scripts/lift_slope.py) ---

# Default OLS window. Deliberately narrow: Cl(alpha) is only straight near
# the middle of the polar, and the slope is defined on the linear part.
LIFT_SLOPE_ALPHA_LO_DEG = -3.0
LIFT_SLOPE_ALPHA_HI_DEG = 3.0
