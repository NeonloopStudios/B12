# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: the airfoil chord length as a function of the half-span."""

import math as m

#y is in meters
def chord(y: float, area: float, aspect: float, taper: float, root: float):
    span = m.sqrt(aspect * area)
    return root - (root - taper*root)*y/(span/2)