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
    #_c refers to the value being as a fraction of the chord
    front_spar_c: float
    front_spar_point_up: float #wingbox corner on the front spar, in the negative z-dir (negative value!)
    front_spar_point_down: float
    rear_spar_c: float
    rear_spar_point_up: float
    rear_spar_point_down: float
    airfoil_tmax_dat: float #max thickness of the airfoil in dat file units
    airfoil_tmax_c: float

WINGBOX_GEOM: Final[WingboxGeometry] = WingboxGeometry(
    front_spar_c = 0.2,
    front_spar_point_up = -5.4,
    front_spar_point_down = 5.46,
    rear_spar_c = 0.7,
    rear_spar_point_up = -4.51,
    rear_spar_point_down = 2.8,
    airfoil_tmax_dat = 11.97,
    airfoil_tmax_c = 0.12
)
