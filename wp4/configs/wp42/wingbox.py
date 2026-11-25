# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: wingbox geometry definition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class WingboxGeometry:
    """
    _c refers to the value being as a fraction of the chord
    (x,z) points describe geometry; x-coord is as a fraction of chord, z-coord is in dat units
    points must be given sequentially in CCW order starting from negative z-dir on the front spar
    """
    points: [tuple]
    airfoil_tmax_dat: float #max thickness of the airfoil in dat file units
    airfoil_tmax_c: float
    spar_thickness: float
    skin_thickness: float
    stringers: (list, list) #first list is nr of stringer; second is percent of half-span from 0 to 100
    stringer_area: float

WINGBOX_GEOM: Final[WingboxGeometry] = WingboxGeometry(
    points = [(0.2, -5.4), (0.2, 5.46), (0.7, 2.8), (0.7, -4.51)],
    airfoil_tmax_dat = 11.97, #dat
    airfoil_tmax_c = 0.12,
    spar_thickness = 10, #mm
    skin_thickness = 8, #mm
    stringers = (
        [7, 5, 3],
        [0, 50, 70]
    ),
    stringer_area = 400 #mm^2
)