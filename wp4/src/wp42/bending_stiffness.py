# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: calculation of the I_xx bending stiffness of the wingbox."""

import math as m
import numpy as np
import scipy as sp
from src.wp42 import chord

#horizontal_l is as a fraction of the chord
#function returns I_xx per unit chord
def second_area_moment(fspar_l: float, fspar_ctr: float, rspar_l: float, rspar_ctr: float, horizontal_l: float):
    a = fspar_l
    b = rspar_l
    h = horizontal_l
    dy = abs(fspar_ctr - rspar_ctr) #vertical offset between the midpoints of the spars

    I_xx = h*(a+b)*(a**2+b**2)/48 + h*(a**2 + b**2 + 4*a*b)*(dy**2)/(36 * (a+b))

    """
    #for testing
    print(f"fl {a}     rl {b}")
    print(f"fmid {fspar_ctr}     rmid {rspar_ctr}")
    print(f"offset {dy}     horiz {h}")
    """
    
    return I_xx