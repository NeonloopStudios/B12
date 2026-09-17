# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Airfoil shape properties derived directly from parsed coordinates.

Split out from xfoil_runtime.py because this is pure geometry (no XFoil
calls involved) -- needed by mcrit_sweep.py's Korn equation (M_dd needs
t/c per airfoil).
"""
from __future__ import annotations

import numpy as np
from xfoil.model import Airfoil


def max_thickness_to_chord(airfoil: Airfoil, n_stations: int = 400) -> float:
    """Maximum thickness (upper y minus lower y, at matching x/c stations)
    as a fraction of chord -- the standard vertical thickness distribution
    definition (not perpendicular-to-camber-line).

    The coordinate loop (see xfoil_runtime.load_airfoil_dat) is
    TE -> upper surface -> LE -> lower surface -> TE, counterclockwise.
    Split at the leading edge (minimum-x point) into upper/lower branches,
    interpolate each onto a common x grid, and take the max of
    upper(x) - lower(x).

    Validated directly against XFoil's own internally-computed "Max
    thickness" (printed at airfoil-load time, not otherwise exposed by the
    Python bindings) for all four current candidates -- agrees to within
    0.1% in every case (see test_geometry.py), so this is not a rough
    stand-in for XFoil's number, it reproduces it.
    """
    x, y = airfoil.x, airfoil.y
    le_idx = int(np.argmin(x))

    upper_x, upper_y = x[: le_idx + 1][::-1], y[: le_idx + 1][::-1]
    lower_x, lower_y = x[le_idx:], y[le_idx:]
    if len(upper_x) < 2 or len(lower_x) < 2:
        raise ValueError("max_thickness_to_chord: not enough points on one surface")

    x_lo = max(upper_x.min(), lower_x.min())
    x_hi = min(upper_x.max(), lower_x.max())
    x_common = np.linspace(x_lo, x_hi, n_stations)

    upper_interp = np.interp(x_common, upper_x, upper_y)
    lower_interp = np.interp(x_common, lower_x, lower_y)
    return float((upper_interp - lower_interp).max())
