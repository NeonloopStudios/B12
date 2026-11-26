# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: calculation of spanwise displacement of the wing due to bending, given a wingbox stiffness per unit chord and a moment distribution."""

import math as m
import numpy as np
import scipy as sp
#from src.wp42 import chord

def M_x(y: float, b: np.array(), m: np.array()):
    s = sp.interpolate.interp1d(b, m, kind="quadratic", fill_value="extrapolate")
    return s(y)

#print(M_x(6.234, np.array([0, 5, 10]), np.array([100, 50, 0])))