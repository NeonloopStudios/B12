# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-19
"""WP4.2: bending and torsional stiffness calculations from a given wing geometry."""

import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path
import sys

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.wp42.wingbox import WINGBOX_GEOM
from configs.aircraft import WING
from configs.mat_props import AL2024_T81
from src.wp42 import chord
from src.wp42 import centroid
from src.wp42 import bending_stiffness
from src.wp42 import bending_displacement
from src.wp42 import bending_figures
from configs.paths import OUT_DIR

span = m.sqrt(WING.aspect_ratio * WING.area_m2)
E = AL2024_T81.youngs_modulus_pa
moment_path = Path(__file__).resolve().parents[1] / "out" / "wp41" / "internal_moment_distribution.txt"  

ctr_out = centroid.centroid_quad(
    points = WINGBOX_GEOM.points,
    tmax_ul = WINGBOX_GEOM.airfoil_tmax_dat,
    tmax_c = WINGBOX_GEOM.airfoil_tmax_c
)

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

def load_moment(path):
    data = np.loadtxt(path, comments="#")
    return (data[:,0], data[:,1])

b_station = load_moment(moment_path)[0] #m, half-span position
m_station = load_moment(moment_path)[1] #internal moment, half-span direction
v = [float] * b_station.size #vertical wing displacement at a given half-span position
v[0] = 0
theta = [float] * b_station.size #slope of the wing at a given half-span position
theta[0] = 0

for i in range(b_station.size - 1):
    y0, y1 = b_station[i], b_station[i + 1]

    #slope change
    dtheta, _ = sp.integrate.quad(
        lambda y: bending_displacement.M_x(y, b_station, m_station * WINGBOX_GEOM.safety_factor) / (E * I_xx(y)),
        y0,
        y1,
    )
    theta[i+1] = theta[i] + dtheta

    #beam rise
    dv, _ = sp.integrate.quad(
        lambda y: theta[i] + (theta[i+1] - theta[i]) * ((y - y0) / (y1 - y0)),
        y0,
        y1,
    )

    v[i+1] = v[i] - dv

print(f"v_tip {v[b_station.size-1]}")
print(f"theta_tip {theta[b_station.size-1]}")

fig = bending_figures.bending_plot(b_station, m_station, I_xx, v)
path = OUT_DIR / "wp42" / "bending_plot.pdf"
fig.savefig(path)
plt.show()