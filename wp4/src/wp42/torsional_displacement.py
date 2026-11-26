# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rodrigo and Yajvin
# Created: 2026-11-19
"""WP4.2: Torsional displacement calculations."""

from scipy import interpolate, integrate
import numpy as np
from src.wp42 import chord

shear_modulus = 2.8 * 10**10
root_chord = 3.66 
tip_chord = 1.47
span = 24.37
half_span = span / 2.0
taper_ratio = tip_chord/root_chord

X_FRONT = 20 / 100
Z_FRONT_TOP = 5.44 / 100
Z_FRONT_BOT = -5.46 / 100

X_REAR = 70 / 100
Z_REAR_TOP = 4.51 / 100
Z_REAR_BOT = -2.80 / 100

def Am_per_chord (Am, y_pos):
    Am_y = Am*chord.chord(y = y_pos)**2
    return Am_y

def perimeter_per_chord (y_pos):
    #Chodwise coordinates for the perimeter
    x1 = 20/100
    y1=5.44/100
    y2=-5.46/100
    x2=70/100
    y3=4.51/100
    y4=-2.8/100

    #Normalised value for the lengths of the perimeter
    chordwise_length_top = np.sqrt((x2-x1)**2+(y3-y1)**2)
    chordwise_length_right = np.sqrt((y4-y3)**2)
    chordwise_length_bottom = np.sqrt((x2-x1)**2+(y4-y2)**2)
    chordwise_length_left = np.sqrt((y2-y1)**2)

    #Total value based on chord length
    length_bottom = chordwise_length_bottom * chord.chord(y = y_pos)
    length_left = chordwise_length_left * chord.chord(y = y_pos)
    length_right = chordwise_length_right * chord.chord(y = y_pos)
    length_top = chordwise_length_top * chord.chord(y = y_pos)
    
    return length_bottom, length_left, length_right, length_top

def get_multicell_geometry(y_pos, x_spars_norm=(X_FRONT, X_REAR)):
    """
    Computes physical spar heights, horizontal cell widths, inclined top/bottom
    skin lengths, and enclosed cell areas at spanwise location y_pos for an
    irregular asymmetric trapezoid with vertical parallel spars.

    Parameters
    ----------
    y_pos : float
        Spanwise position from wing root [m].
    x_spars_norm : sequence of float
        Normalized chordwise positions (x/c) of all spars from front to rear
        (length N + 1 for N cells). Default (0.20, 0.70) gives a single-cell box.
    """
    c = chord.chord(y=y_pos)
    x_spars_norm = np.asarray(x_spars_norm, dtype=float)

    # Linear interpolation fraction between front and rear spars
    xi = (x_spars_norm - X_FRONT) / (X_REAR - X_FRONT)

    # Top and bottom skin z/c positions at each spar
    z_top_norm = Z_FRONT_TOP + xi * (Z_REAR_TOP - Z_FRONT_TOP)
    z_bot_norm = Z_FRONT_BOT + xi * (Z_REAR_BOT - Z_FRONT_BOT)

    # Physical spar heights [m] (length N + 1)
    spar_heights = (z_top_norm - z_bot_norm) * c

    # Physical horizontal cell widths [m] (length N)
    dx_norm = np.diff(x_spars_norm)
    cell_widths = dx_norm * c

    # Physical inclined skin lengths [m] for each cell (length N)
    s_top = np.hypot(dx_norm, np.diff(z_top_norm)) * c
    s_bot = np.hypot(dx_norm, np.diff(z_bot_norm)) * c

    # Physical enclosed trapezoidal area [m^2] of each cell: A_i = 0.5*(h_l + h_r)*w_i
    cell_areas = 0.5 * (spar_heights[:-1] + spar_heights[1:]) * cell_widths

    return spar_heights, cell_widths, s_top, s_bot, cell_areas


def solve_multicell_trapezoid_torsion(
    T,
    spar_heights,
    cell_widths,
    s_top,
    s_bot,
    t_spars,
    t_top,
    t_bot,
    G=shear_modulus,
):
    """
    Solves the multi-cell torsion system (Appendix D.1.3) for an idealized
    N-cell irregular trapezoidal wing box using numpy.linalg.solve.
    Works for both single-cell (N = 1) and multi-cell (N >= 2) sections.
    """
    spar_heights = np.asarray(spar_heights, dtype=float)
    cell_widths = np.asarray(cell_widths, dtype=float)
    t_spars = np.asarray(t_spars, dtype=float)
    n_cells = len(cell_widths)

    if len(spar_heights) != n_cells + 1 or len(t_spars) != n_cells + 1:
        raise ValueError(f"Expected {n_cells + 1} spars for {n_cells} cells.")

    s_top = np.full(n_cells, s_top, dtype=float) if np.isscalar(s_top) else np.asarray(s_top, dtype=float)
    s_bot = np.full(n_cells, s_bot, dtype=float) if np.isscalar(s_bot) else np.asarray(s_bot, dtype=float)
    t_top = np.full(n_cells, t_top, dtype=float) if np.isscalar(t_top) else np.asarray(t_top, dtype=float)
    t_bot = np.full(n_cells, t_bot, dtype=float) if np.isscalar(t_bot) else np.asarray(t_bot, dtype=float)

    # Enclosed area of each irregular trapezoidal cell
    cell_areas = 0.5 * (spar_heights[:-1] + spar_heights[1:]) * cell_widths

    # Assemble linear system M * x = rhs for x = [q_1, ..., q_N, G * dtheta/dy]
    M = np.zeros((n_cells + 1, n_cells + 1), dtype=float)
    rhs = np.zeros(n_cells + 1, dtype=float)

    for i in range(n_cells):
        coeff = 1.0 / (2.0 * cell_areas[i])

        left_web = spar_heights[i] / t_spars[i]
        right_web = spar_heights[i + 1] / t_spars[i + 1]
        top_skin = s_top[i] / t_top[i]
        bot_skin = s_bot[i] / t_bot[i]

        M[i, i] = coeff * (left_web + right_web + top_skin + bot_skin)
        if i > 0:
            M[i, i - 1] = -coeff * left_web
        if i < n_cells - 1:
            M[i, i + 1] = -coeff * right_web

        M[i, -1] = -1.0
        M[-1, i] = 2.0 * cell_areas[i]

    rhs[-1] = T

    solution = np.linalg.solve(M, rhs)

    q_cells = solution[:-1]
    G_dtheta_dy = solution[-1]
    dtheta_dy = G_dtheta_dy / G
    J = T / G_dtheta_dy if T != 0 else np.nan
    GJ = G * J

    q_spars = np.zeros(n_cells + 1)
    q_spars[0] = q_cells[0]
    q_spars[-1] = q_cells[-1]
    for k in range(1, n_cells):
        q_spars[k] = q_cells[k - 1] - q_cells[k]

    return {
        "spar_heights": spar_heights,
        "cell_areas": cell_areas,
        "total_area": np.sum(cell_areas),
        "q_cells": q_cells,
        "q_spars": q_spars,
        "tau_spars": q_spars / t_spars,
        "tau_top": q_cells / t_top,
        "tau_bot": q_cells / t_bot,
        "dtheta_dy": dtheta_dy,
        "J": J,
        "GJ": GJ,
    }

def dtheta_dy(y, T_func, G):
    return float(T_func(y)) / (G * float(J(y)))

def theta(
    y_stations,
    T_stations,
    x_spars_list,
    t_spars_list,
    t_top_arr,
    t_bot_arr,
    G=shear_modulus,
    J_interp_kind="previous",
):
    """
    Computes spanwise torsional constant J(y) at all stations (e.g., 120 points)
    using unit torque (T = 1 Nm) and integrates dtheta/dy = T(y) / (G * J(y))
    from y = 0 to y using scipy.integrate.quad (Appendix D.2).

    Returns
    -------
    theta_tot : np.ndarray
        Cumulative twist angle [rad] at every station in y_stations.
    theta_deg : np.ndarray
        Cumulative twist angle [deg] at every station (must stay within +/- 10 deg).
    J_stations : np.ndarray
        Torsional constant J [m^4] at every station.
    q_cells_actual : list[np.ndarray]
        Actual torsional shear flows [N/m] in each cell at each station.
    """
    y_stations = np.asarray(y_stations, dtype=float)
    T_stations = np.asarray(T_stations, dtype=float)
    n_stations = len(y_stations)

    J_stations = np.zeros(n_stations)
    q_cells_actual = []

    # 1. Solve for geometric J(y_k) at each station with unit torque T = 1.0 Nm
    for k in range(n_stations):
        spar_h, cell_w, s_t, s_b, _ = get_multicell_geometry(
            y_pos=y_stations[k],
            x_spars_norm=x_spars_list[k],
        )
        res_unit = solve_multicell_trapezoid_torsion(
            T=1.0,
            spar_heights=spar_h,
            cell_widths=cell_w,
            s_top=s_t,
            s_bot=s_b,
            t_spars=t_spars_list[k],
            t_top=t_top_arr[k],
            t_bot=t_bot_arr[k],
            G=G,
        )
        J_stations[k] = res_unit["J"]
        q_cells_actual.append(res_unit["q_cells"] * T_stations[k])

    # 2. Interpolate T(y) and J(y) as continuous functions of y (Appendix D.1.4)
    T_func = interpolate.interp1d(
        y_stations, T_stations, kind="cubic", fill_value="extrapolate"
    )
    J_func = interpolate.interp1d(
        y_stations, J_stations, kind=J_interp_kind, fill_value="extrapolate"
    )
    dtheta_dy_func = lambda y: float(T_func(y)) / (G * float(J_func(y)))

    # 3. Integrate from root (theta(0) = 0) to each station (Appendix D.2)
    theta_tot = np.zeros(n_stations)
    if y_stations[0] > 0.0:
        theta_tot[0], _ = integrate.quad(dtheta_dy_func, 0.0, y_stations[0])

    for k in range(1, n_stations):
        d_theta, _ = integrate.quad(
            dtheta_dy_func, y_stations[k - 1], y_stations[k], limit=100
        )
        theta_tot[k] = theta_tot[k - 1] + d_theta

    theta_deg = np.degrees(theta_tot)
    return theta_tot, theta_deg, J_stations, q_cells_actual