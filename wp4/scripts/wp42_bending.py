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
from src.wp42 import chord
from src.wp42 import centroid
from src.wp42 import bending_stiffness
from src.wp42 import bending_displacement

span = m.sqrt(WING.aspect_ratio * WING.area_m2)

ctr_out = centroid.centroid_quad(
    points = WINGBOX_GEOM.points,
    tmax_ul = WINGBOX_GEOM.airfoil_tmax_dat,
    tmax_c = WINGBOX_GEOM.airfoil_tmax_c
)
print(ctr_out)

def I_xx(y):
    return bending_stiffness.I_xx(
        points = WINGBOX_GEOM.points,
        y = y,
        t_sp = WINGBOX_GEOM.spar_thickness,
        t_sk = WINGBOX_GEOM.skin_thickness,
        centroid = ctr_out[0],
        dat_to_c = WINGBOX_GEOM.airfoil_tmax_c / WINGBOX_GEOM.airfoil_tmax_dat,
        stringers = WINGBOX_GEOM.stringers,
        stringer_area = WINGBOX_GEOM.stringer_area,
        halfspan = span/2
    )
print(I_xx(0))

E = 1
dvdy, dvdy_error = sp.integrate.quad(lambda y: bending_displacement.M_x(y, np.array([0, 5, 10]), np.array([100, 50, 0])) / (E * I_xx(y)), 0, span/4)
#print(dvdy)