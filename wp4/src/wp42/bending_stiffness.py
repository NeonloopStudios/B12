# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: calculation of the I_xx bending stiffness of the wingbox."""

import math as m
import numpy as np
import scipy as sp
from src.wp42 import chord

def I_xx(points: [tuple], y: float, t_sp: float, t_sk: float, centroid: tuple, dat_to_c: float, stringers: (list, list), stringer_area: float, span: float):
    """
    points must be given sequentially in CW or CCW order for any non-intersecting quadrilateral
    returns second area moment (in m^4) against bending about xx as a function of:
        geometry, y (in half-span direction), thickness, stringer distribution
    t_sp is spar thickness, uniform; t_sk is skin thickness, uniform
    centroid tuple has form (x,z)
    *dat_to_c turns a dat-unit to chord fraction

    assumption: skin stringers are distributed uniformly, thus, their Steiner terms are taken at mid_z
    """
    I_xx = 0
    s = sp.interpolate.interp1d(stringers[1], stringers[0], kind="previous", fill_value="extrapolate")

    for i in range(4):
        ctr = i+1
        if ctr>3: ctr=0

        c = chord.chord(y)
        n = dat_to_c * c #turn dat (z-coord) to meters
        delta_z = abs(points[i][1] - points[ctr][1])*n
        mid_z = min(points[i][1], points[ctr][1])*n + delta_z/2
        length = m.sqrt((points[i][0]*c - points[ctr][0]*c)**2 + (points[i][1]*n - points[ctr][1]*n)**2)

        t = t_sk / 1000
        if i%2==0: t = t_sp / 1000

        local_term = t * length * delta_z**2 / 12
        parallel_term = t * length * (mid_z - centroid[1]*c)**2
        I_xx += local_term + parallel_term

        #for skin segments
        if i%2!=0:
            stringer_term = (stringer_area / (1000**2)) * (mid_z - centroid[1]*c)**2
            I_xx += stringer_term * s(100*y/span)

    return I_xx