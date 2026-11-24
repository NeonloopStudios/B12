# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-24
"""WP4.2: calculation of the I_xx bending stiffness of the wingbox."""

import math as m
import numpy as np
import scipy as sp

def second_area_moment(y: float, fspar_l: float, rspar_l: float, horizontal_l: float, vert_offset: float):
    return 1