# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yajvin Singhal and Rodrigo Rodrigues
# Created: 2026-11-26
"""WP4.2: torsional stiffness calculations from a given wing geometry."""

from pathlib import Path
import sys
from scipy import interpolate, integrate

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

import numpy as np
from configs.mat_props import MATERIAL
from configs.wp42.wingbox import WINGBOX_GEOM
from src.wp42.chord import chord
from src.wp42.torsional_displacement import compute_J_distribution
from src.wp42.torsional_displacement import dtheta_dy

torsion_data = compute_J_distribution(
    y_stations=y_stations,
    chord_func=chord,
    x_spars_norm=WINGBOX_GEOM.vertical_stiffeners_chordwise_pos,
    t_spars=[0.006, 0.004, 0.005],
    t_top=0.004,
    t_bot=0.0035,
    z_front_top_norm=-0.0544,
    z_front_bot_norm=0.0546,
    z_rear_top_norm=-0.0451,
    z_rear_bot_norm=0.0280,
    G=MATERIAL.shear_modulus_pa,
)
J_func = torsion_data["J_func"]  # Exact callable J(y) [N*m^2] for scipy.integrate.quad

#Spanwise locations and torque
y_stations = np.asarray(y_stations, dtype=float)
T_stations = np.asarray(T_stations, dtype=float)
G = MATERIAL.shear_modulus_pa

#Interpolate T(y) as continuous functions of y (Appendix D.1.4)
T_func = interpolate.interp1d(
    y_stations, T_stations, kind="cubic", fill_value="extrapolate"
)

#Integrate dtheta/dy = T(y) / (G * J(y)) from y = 0 to y using scipy.integrate.quad (Appendix D.2)
n_stations = len(y_stations)
theta_tot = np.zeros(n_stations)

if y_stations[0] > 0.0:
    theta_tot[0], _ = integrate.quad(
        dtheta_dy, 0.0, y_stations[0], args=(T_func, G)
    )

for k in range(1, n_stations):
    d_theta, _ = integrate.quad(
        dtheta_dy, y_stations[k - 1], y_stations[k], args=(T_func, G), limit=100
    )
    theta_tot[k] = theta_tot[k - 1] + d_theta

#Convert to degrees (should be within +/- 10 deg to pass)
theta_deg = np.degrees(theta_tot)