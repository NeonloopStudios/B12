# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-19
"""WP4.2: centroid calculations from wingbox edge coordinates."""

import math as m
import numpy as np 

def centroid_quad(points: [tuple], tmax_ul: float, tmax_c: float):
    """
    points must be given sequentially in CW or CCW order for any non-intersecting quadrilateral
    returns in this order: centroid x, centroid z, area; last two as fractions of chord
    tmax_ul is the max unitless (dat-format) thickness
    tmax_c is the max thickness as a frac of the chord
    """
    area = 0
    cx = 0
    cz = 0

    for i in range(4):
        ctr = i+1
        if ctr>3: ctr=0
        cross = (points[i][0] * points[ctr][1]) - (points[ctr][0] * points[i][1])
        area += cross
        cx += (points[i][0] + points[ctr][0]) * cross
        cz += (points[i][1] + points[ctr][1]) * cross

    area /= 2
    if area == 0:
        raise ValueError("Points are collinear; shape has zero area.")
    cx /= (6*area)
    cz /= (6*area)

    return ((cx, cz * tmax_c / tmax_ul), abs(area))