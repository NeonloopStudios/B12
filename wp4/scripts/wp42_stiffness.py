# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-19
"""WP4.2: bending and torsional stiffness calculations from a given wing geometry."""

import math as m
import numpy as np
import scipy as sp
from pathlib import Path
import sys

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.wp42.wingbox import WINGBOX_GEOM
from configs.aircraft import WING
from src.wp42 import centroid
from src.wp42 import bending_stiffness
from src.wp42 import chord

span = m.sqrt(WING.aspect_ratio * WING.area_m2)

ctr = centroid.centroid_wingbox(
    fspar_loc = WINGBOX_GEOM.front_spar_c,
    fspar_up = WINGBOX_GEOM.front_spar_point_up,
    fspar_dn = WINGBOX_GEOM.front_spar_point_down,
    rspar_loc = WINGBOX_GEOM.rear_spar_c,
    rspar_up = WINGBOX_GEOM.rear_spar_point_up,
    rspar_dn = WINGBOX_GEOM.rear_spar_point_down,
    tmax_ul = WINGBOX_GEOM.airfoil_tmax_dat,
    tmax_c = WINGBOX_GEOM.airfoil_tmax_c
)
centroid_xz = ctr[0] #z as frac of chord
winbox_area_c = ctr[1] #as frac of chord

I_xx = bending_stiffness.second_area_moment(
    fspar_l = abs(WINGBOX_GEOM.front_spar_point_up - WINGBOX_GEOM.front_spar_point_down),
    fspar_ctr = (WINGBOX_GEOM.front_spar_point_up + WINGBOX_GEOM.front_spar_point_down)/2,
    rspar_l = abs(WINGBOX_GEOM.rear_spar_point_up - WINGBOX_GEOM.rear_spar_point_down),
    rspar_ctr = (WINGBOX_GEOM.rear_spar_point_up + WINGBOX_GEOM.rear_spar_point_down)/2,
    horizontal_l = WINGBOX_GEOM.rear_spar_c - WINGBOX_GEOM.front_spar_c
) #per unit chord

print(I_xx * chord.chord(span/2))