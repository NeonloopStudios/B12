# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: the airfoil chord length as a function of the half-span."""

import math as m
from configs.aircraft import WING

#y is in meters, maximum is half-span
def chord(y: float):
    area = WING.area_m2,
    aspect = WING.aspect_ratio,
    taper = WING.taper,
    root = WING.root_chord
    span = m.sqrt(aspect * area)
    return root - (root - taper*root)*y/(span/2)