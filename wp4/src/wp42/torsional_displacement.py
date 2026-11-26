# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rodrigo and Yajvin
# Created: 2026-11-19
"""WP4.2: Torsional displacement calculations."""

from scipy import interpolate, integrate
import numpy as np
from src.wp42 import chord
from typing import Callable, Sequence, Union


def Am_per_chord (Am, y_pos):
    Am_y = Am*chord.chord(y = y_pos)**2
    return Am_y

def perimeter_per_chord (y_pos, x1, y1, y2, x2, y3, y4):
    
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

def multicell_geometry(
    x_spars_norm: Sequence[float],
    z_front_top_norm: float,
    z_front_bot_norm: float,
    z_rear_top_norm: float,
    z_rear_bot_norm: float,
    t_spars: Sequence[float],
    t_top: Union[float, Sequence[float]],
    t_bot: Union[float, Sequence[float]],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Solves the Appendix D.1.3 torsion system at unit chord (c = 1.0 m) and unit
    torque (T = 1.0 N*m) for an irregular asymmetric trapezoidal wing box with
    vertical parallel spars and N >= 1 cells.

    Coordinate convention:
        +x : chordwise aft from leading edge (x/c in [0, 1])
        +y : spanwise from wing root (y = 0) to tip (y = b/2)
        +z : vertical downwards (top skin z < 0, bottom skin z > 0)

    Returns
    -------
    J_norm : float
        Chord-normalized torsional constant J / c^3 [m] (multiply by c(y)^3 for J(y)).
    q_unit_norm : np.ndarray
        Unit-chord, unit-torque cell shear flows [q_1, ..., q_N] (length N).
    cell_areas_norm : np.ndarray
        Chord-normalized enclosed cell areas A_i / c^2 [-] (length N).
    """
    # Convert input sequences into 1D floating-point NumPy arrays
    x_spars = np.asarray(x_spars_norm, dtype=float)
    t_spars_arr = np.asarray(t_spars, dtype=float)

    # Validate that at least two spars (front and rear) are given to form >= 1 cell
    if x_spars.ndim != 1 or len(x_spars) < 2:
        raise ValueError("x_spars_norm must be a 1D sequence with at least 2 spars.")

    # Ensure spars are strictly ordered from front (leading edge) to rear (trailing edge)
    if np.any(np.diff(x_spars) <= 0.0):
        raise ValueError("x_spars_norm must be strictly increasing from front to rear.")

    # Number of closed cells N equals (number of spars - 1)
    n_cells = len(x_spars) - 1

    # Verify that every spar has a corresponding thickness value
    if t_spars_arr.shape != (n_cells + 1,):
        raise ValueError(f"Expected {n_cells + 1} spar thicknesses for {n_cells} cells.")

    # Broadcast scalar skin thicknesses to length N if uniform across all cells,
    # or convert per-cell thickness sequences into 1D float arrays
    t_top_arr = np.full(n_cells, t_top, dtype=float) if np.isscalar(t_top) else np.asarray(t_top, dtype=float)
    t_bot_arr = np.full(n_cells, t_bot, dtype=float) if np.isscalar(t_bot) else np.asarray(t_bot, dtype=float)

    if t_top_arr.shape != (n_cells,) or t_bot_arr.shape != (n_cells,):
        raise ValueError(f"Skin thicknesses must be scalars or 1D sequences of length {n_cells}.")

    # Compute normalized position xi in [0, 1] of every spar between front and rear spars
    xi = (x_spars - x_spars[0]) / (x_spars[-1] - x_spars[0])

    # Linearly interpolate upper and lower skin z/c coordinates at each spar
    # (straight skin segments connecting front and rear spars)
    z_top_norm = z_front_top_norm + xi * (z_rear_top_norm - z_front_top_norm)
    z_bot_norm = z_front_bot_norm + xi * (z_rear_bot_norm - z_front_bot_norm)

    # Normalized vertical spar heights h_k / c (z_bot > z_top since +z points downwards)
    h_spars_norm = z_bot_norm - z_top_norm
    if np.any(h_spars_norm <= 0.0):
        raise ValueError("Computed non-positive spar height; verify +z downwards sign convention.")

    # Normalized horizontal cell widths w_i / c (length N)
    w_cells_norm = np.diff(x_spars)

    # Normalized inclined top and bottom skin panel lengths s_i / c (length N)
    s_top_norm = np.hypot(w_cells_norm, np.diff(z_top_norm))
    s_bot_norm = np.hypot(w_cells_norm, np.diff(z_bot_norm))

    # Normalized enclosed trapezoidal area A_i / c^2 of each cell i:
    # A_i = 0.5 * (h_left + h_right) * width_i
    cell_areas_norm = 0.5 * (h_spars_norm[:-1] + h_spars_norm[1:]) * w_cells_norm

    # Assemble the (N + 1) x (N + 1) linear system M * x = rhs (Appendix D.1.3).
    # Unknown vector x = [q_1, q_2, ..., q_N, G * (dtheta/dy)_norm]:
    # - Rows 0 ... N-1 : twist-rate compatibility (1 / 2A_i) * integral(q ds / t) - G*(dtheta/dy) = 0
    # - Row N (index -1): global torque equilibrium sum(2 * A_i * q_i) = T_unit
    # For N = 1 (single cell), this 2x2 system reduces identically to J = 4*A^2 / integral(ds/t).
    M = np.zeros((n_cells + 1, n_cells + 1), dtype=float)
    rhs = np.zeros(n_cells + 1, dtype=float)

    for i in range(n_cells):
        # Bredt-Batho prefactor 1 / (2 * A_i)
        coeff = 1.0 / (2.0 * cell_areas_norm[i])

        # Perimeter integral contributions (ds / t) for the 4 walls bounding cell i
        web_left = h_spars_norm[i] / t_spars_arr[i]
        web_right = h_spars_norm[i + 1] / t_spars_arr[i + 1]
        skin_top = s_top_norm[i] / t_top_arr[i]
        skin_bot = s_bot_norm[i] / t_bot_arr[i]

        # Self shear flow q_i acts positively around all 4 walls of cell i
        M[i, i] = coeff * (web_left + web_right + skin_top + skin_bot)

        # Adjacent cell shear flows oppose q_i along shared internal spars (q_i - q_{i+/-1})
        if i > 0:
            M[i, i - 1] = -coeff * web_left
        if i < n_cells - 1:
            M[i, i + 1] = -coeff * web_right

        # Compatibility condition: subtract G * (dtheta/dy)_norm
        M[i, -1] = -1.0

        # Global torque equilibrium contribution of cell i: 2 * A_i * q_i
        M[-1, i] = 2.0 * cell_areas_norm[i]

    # Apply unit torque T = 1.0 N*m
    T_unit = 1.0
    rhs[-1] = T_unit

    # Solve M * x = rhs using NumPy's LU linear solver (Appendix D.1.3)
    solution = np.linalg.solve(M, rhs)
    q_unit_norm = solution[:-1]
    G_dtheta_dy_norm = solution[-1]

    # Normalized torsional constant J_norm = T_unit / (G * dtheta/dy)_norm
    J_norm = float(T_unit / G_dtheta_dy_norm)

    return J_norm, q_unit_norm, cell_areas_norm


def compute_J_distribution(
    y_stations: Sequence[float],
    chord_func: Callable[[float], float],
    x_spars_norm: Sequence[float],
    t_spars: Sequence[float],
    t_top: Union[float, Sequence[float]],
    t_bot: Union[float, Sequence[float]],
    z_front_top_norm: float,
    z_front_bot_norm: float,
    z_rear_top_norm: float,
    z_rear_bot_norm: float,
    G: float,
) -> dict:
    """
    Computes the torsional constant J[y] across all spanwise stations y_stations
    using your existing chord(y) function, and returns exact callables J(y) and
    GJ(y) for scipy.integrate.quad.

    Parameters
    ----------
    y_stations : Sequence[float]
        1D array of spanwise coordinates [m] from wing root (0) to tip (b/2).
    chord_func : Callable[[float], float]
        Your existing chord(y) function returning local chord length [m] at y [m].
    x_spars_norm : Sequence[float]
        Normalized chordwise positions (x/c) of all spars ordered front-to-rear
        (length 2 for 1 cell, length 3 for 2 cells, ..., length N+1 for N cells).
    t_spars : Sequence[float]
        Constant thicknesses [m] of each spar from front to rear (length N + 1).
    t_top, t_bot : float or Sequence[float]
        Constant top and bottom skin thicknesses [m] (single float if uniform across
        all cells, or sequence of length N if varying cell-to-cell).
    z_front_top_norm, z_front_bot_norm, z_rear_top_norm, z_rear_bot_norm : float
        Normalized corner z/c coordinates of the trapezoidal wing box (+z down).
    G : float
        Material shear modulus [Pa] (passed in by caller from configs/material.py).
    """
    # Convert spanwise stations to a 1D float NumPy array
    y_arr = np.asarray(y_stations, dtype=float)

    # Solve the unit-chord (c = 1 m), unit-torque (T = 1 Nm) multi-cell system once
    J_norm, q_unit_norm, _ = multicell_geometry(
        x_spars_norm=x_spars_norm,
        z_front_top_norm=z_front_top_norm,
        z_front_bot_norm=z_front_bot_norm,
        z_rear_top_norm=z_rear_top_norm,
        z_rear_bot_norm=z_rear_bot_norm,
        t_spars=t_spars,
        t_top=t_top,
        t_bot=t_bot,
    )

    # Evaluate your existing chord(y) function at all 120 spanwise stations [m]
    c_arr = np.array([float(chord_func(float(y))) for y in y_arr], dtype=float)

    # Scale unit-chord J_norm by local chord cubed: J(y) = J_norm * c(y)^3 [m^4]
    J_arr = J_norm * (c_arr ** 3)

    # Scale unit-chord, unit-torque cell shear flows by 1 / c(y)^2 (shape: n_stations x n_cells).
    # Multiply by physical torque T(y) later to get actual cell shear flows q_i(y) [N/m].
    q_unit_arr = q_unit_norm[np.newaxis, :] / (c_arr[:, np.newaxis] ** 2)

    # Exact analytical callables using your chord_func directly (no interpolation needed)
    J_func = lambda y: J_norm * (float(chord_func(float(y))) ** 3)
    GJ_func = lambda y: float(G) * J_norm * (float(chord_func(float(y))) ** 3)

    return {
        "y": y_arr,
        "J": J_arr,
        "GJ": float(G) * J_arr,
        "J_func": J_func,
        "GJ_func": GJ_func,
        "q_unit": q_unit_arr,
    }

def theta(
    y_stations,
    T_stations,
    x_spars_list,
    t_spars_list,
    t_top_arr,
    t_bot_arr,
    G,
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