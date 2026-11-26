# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Yajvin Singhal and Rodrigo
# Created: 2026-11-19
"""WP4.2: torsional stiffness calculations from a given wing geometry."""

from pathlib import Path
import sys

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

import numpy as np
from configs.mat_props import MATERIAL
from src.wp42.chord import chord
from src.wp42.torsional_displacement import compute_J_distribution

y_stations = np.linspace(0.0, 24.37 / 2.0, 120)

torsion_data = compute_J_distribution(
    y_stations=y_stations,
    chord_func=chord,
    x_spars_norm=[0.20, 0.45, 0.70],
    t_spars=[0.006, 0.004, 0.005],
    t_top=0.004,
    t_bot=0.0035,
    z_front_top_norm=-0.0544,
    z_front_bot_norm=0.0546,
    z_rear_top_norm=-0.0451,
    z_rear_bot_norm=0.0280,
    G=MATERIAL.shear_modulus_pa,
)

J_y = torsion_data["J"]            # 120-element NumPy array J[y] [m^4]
GJ_func = torsion_data["GJ_func"]  # Exact callable GJ(y) [N*m^2] for scipy.integrate.quad

print(J_y)