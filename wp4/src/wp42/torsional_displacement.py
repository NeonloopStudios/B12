# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rodrigo and Yajvin
# Created: 2026-11-19
"""WP4.2: Torsional displacement calculations."""

import scipy
import numpy as np
from src.wp42 import chord

shear_modulus = 2.8 * 10**10
root_chord = 3.66 
tip_chord = 1.47
span = 24.37
taper_ratio = tip_chord/root_chord

def Am_per_chord (Am, y_pos):
    Am_y = Am*chord.chord(y = y_pos)
    return Am_y

def perimeter_per_chord (y_pos):
    #Coordinates for the perimeter
    x1 = 20/100
    y1=5.44/100
    y2=-5.46/100
    x2=70/100
    y3=4.51/100
    y4=-2.8/100

    #Normalised value for the lengths of the perimeter
    chordwise_length_top = np.sqrt((x2-x1)^2+(y3-y1)^2)
    chordwise_length_right = np.sqrt((y4-y3)^2)
    chordwise_length_bottom = np.sqrt((x2-x1)^2+(y4-y2)^2)
    chordwise_length_left = np.sqrt((y2-y1)^2)

    #Total value based on chord length
    length_bottom = chordwise_length_bottom * chord.chord(y = y_pos)
    length_left = chordwise_length_left * chord.chord(y = y_pos)
    length_right = chordwise_length_right * chord.chord(y = y_pos)
    length_top = chordwise_length_top * chord.chord(y = y_pos)
    
    return length_bottom, length_left, length_right, length_top

# def theta ():


#     return theta_tot

def solve_multicell_trapezoid_torsion(
    T,
    h_front,
    h_rear,
    cell_widths,
    t_spars,
    t_top,  
    t_bot,
    G=28.0e9,
    trapezoid_shape="flat_bottom",
):
    """
    Solves the multi-cell torsion system (Appendix D.1.3) for an idealized
    N-cell trapezoidal wing box using numpy.linalg.solve.

    Parameters
    ----------
    T : float
        Applied torque [N*m] (positive counterclockwise).
    h_front : float
        Front spar height [m].
    h_rear : float
        Rear spar height [m].
    cell_widths : list[float] or np.ndarray
        Horizontal widths of each cell from front to rear [m] (length N).
    t_spars : list[float] or np.ndarray
        Thicknesses of all spars from front to rear [m] (length N + 1).
    t_top : float or list[float]
        Top skin thickness(es) [m] (scalar or length N).
    t_bot : float or list[float]
        Bottom skin thickness(es) [m] (scalar or length N).
    G : float, optional
        Shear modulus [Pa] (default: 28.0 GPa for AL2024-T81).
    trapezoid_shape : str, optional
        Geometry of the trapezoidal skins:
        - "flat_bottom" : Horizontal bottom skin, sloped top skin (e.g. Fig. 20.8)
        - "flat_top"    : Horizontal top skin, sloped bottom skin (e.g. Reader Fig. 1)
        - "symmetric"   : Both top and bottom skins sloped equally about the chordline

    Returns
    -------
    dict
        Enclosed areas, cell shear flows, internal spar net shear flows,
        rate of twist dtheta/dy, torsional constant J, and torsional stiffness GJ.
    """
    cell_widths = np.asarray(cell_widths, dtype=float)
    t_spars = np.asarray(t_spars, dtype=float)
    n_cells = len(cell_widths)

    if len(t_spars) != n_cells + 1:
        raise ValueError(f"Expected {n_cells + 1} spar thicknesses, got {len(t_spars)}.")

    t_top = np.full(n_cells, t_top, dtype=float) if np.isscalar(t_top) else np.asarray(t_top, dtype=float)
    t_bot = np.full(n_cells, t_bot, dtype=float) if np.isscalar(t_bot) else np.asarray(t_bot, dtype=float)

    # 1. Compute spar heights along the trapezoid by linear interpolation
    total_width = np.sum(cell_widths)
    x_spars = np.concatenate(([0.0], np.cumsum(cell_widths)))
    spar_heights = h_front + (h_rear - h_front) * (x_spars / total_width)

    # 2. Compute enclosed trapezoidal areas and skin panel lengths for each cell
    cell_areas = np.zeros(n_cells)
    s_top = np.zeros(n_cells)
    s_bot = np.zeros(n_cells)

    for i in range(n_cells):
        h_left = spar_heights[i]
        h_right = spar_heights[i + 1]
        w = cell_widths[i]
        dh = h_left - h_right

        # Enclosed area of trapezoidal cell i: A_i = 0.5 * (h_left + h_right) * w
        cell_areas[i] = 0.5 * (h_left + h_right) * w

        if trapezoid_shape == "flat_bottom":
            s_top[i] = np.hypot(w, dh)
            s_bot[i] = w
        elif trapezoid_shape == "flat_top":
            s_top[i] = w
            s_bot[i] = np.hypot(w, dh)
        elif trapezoid_shape == "symmetric":
            s_top[i] = np.hypot(w, dh / 2.0)
            s_bot[i] = np.hypot(w, dh / 2.0)
        else:
            raise ValueError("trapezoid_shape must be 'flat_bottom', 'flat_top', or 'symmetric'.")

    # 3. Assemble linear system M * x = rhs (Appendix D.1.3)
    # To keep the matrix well-conditioned numerically, solve for x = [q_1, ..., q_N, G * dtheta/dy]
    # and divide the final entry by G afterward.
    M = np.zeros((n_cells + 1, n_cells + 1), dtype=float)
    rhs = np.zeros(n_cells + 1, dtype=float)

    for i in range(n_cells):
        A_i = cell_areas[i]
        coeff = 1.0 / (2.0 * A_i)

        # ds / t terms for the four walls bounding cell i
        left_web = spar_heights[i] / t_spars[i]
        right_web = spar_heights[i + 1] / t_spars[i + 1]
        top_skin = s_top[i] / t_top[i]
        bot_skin = s_bot[i] / t_bot[i]

        # Coefficient of q_i (positive around full cell perimeter)
        M[i, i] = coeff * (left_web + right_web + top_skin + bot_skin)

        # Adjacent cell shear flows oppose q_i along shared internal spars
        if i > 0:
            M[i, i - 1] = -coeff * left_web
        if i < n_cells - 1:
            M[i, i + 1] = -coeff * right_web

        # Compatibility equation: (1 / 2A_i) * integral(q ds / t) - G*(dtheta/dy) = 0
        M[i, -1] = -1.0

        # Torque equilibrium row: sum(2 * A_i * q_i) = T
        M[-1, i] = 2.0 * A_i

    rhs[-1] = T

    # 4. Solve system using numpy.linalg.solve (Appendix D.1.3)
    solution = np.linalg.solve(M, rhs)

    q_cells = solution[:-1]              # Cell shear flows q_1 ... q_N [N/m]
    G_dtheta_dy = solution[-1]           # G * (dtheta/dy) [N/m^3]
    dtheta_dy = G_dtheta_dy / G          # Rate of twist [rad/m]

    # Torsional constant J = T / (G * dtheta/dy) [m^4]
    J = T / G_dtheta_dy if T != 0 else np.nan
    GJ = G * J

    # Net shear flows in all spars (positive upward on right side of cell i)
    q_spars = np.zeros(n_cells + 1)
    q_spars[0] = q_cells[0]                         # Front spar (downward for CCW q_1)
    q_spars[-1] = q_cells[-1]                       # Rear spar (upward for CCW q_N)
    for k in range(1, n_cells):
        q_spars[k] = q_cells[k - 1] - q_cells[k]    # Internal spar k: q_left - q_right

    # Shear stresses tau = q / t [Pa]
    tau_spars = q_spars / t_spars
    tau_top = q_cells / t_top
    tau_bot = q_cells / t_bot

    return {
        "spar_heights": spar_heights,
        "cell_areas": cell_areas,
        "total_area": np.sum(cell_areas),
        "q_cells": q_cells,
        "q_spars": q_spars,
        "tau_spars": tau_spars,
        "tau_top": tau_top,
        "tau_bot": tau_bot,
        "dtheta_dy": dtheta_dy,
        "J": J,
        "GJ": GJ,
    }


# ==============================================================================
# EXAMPLE: 3-cell trapezoidal section (dimensions from Figure 20.8)
# ==============================================================================
if __name__ == "__main__":
    results = solve_multicell_trapezoid_torsion(
        T=50_000.0,                        # Applied torque [N*m]
        h_front=0.240,                     # Front spar height: 240 mm
        h_rear=0.180,                      # Rear spar height: 180 mm
        cell_widths=[0.400, 0.400, 0.400], # Three 400 mm cells
        t_spars=[0.004, 0.002, 0.002, 0.003],  # Example spar thicknesses [m]
        t_top=0.0025,                      # Example top skin thickness [m]
        t_bot=0.0025,                      # Example bottom skin thickness [m]
        G=28.0e9,                          # AL2024-T81 shear modulus [Pa]
        trapezoid_shape="flat_bottom",
    )

    print("Spar heights [m]:      ", np.round(results["spar_heights"], 4))
    print("Cell areas A_i [m^2]:  ", np.round(results["cell_areas"], 5))
    print(f"Total enclosed area:    {results['total_area']:.5f} m^2")
    print("Cell shear flows [N/m]:", np.round(results["q_cells"], 2))
    print("Spar net q [N/m]:      ", np.round(results["q_spars"], 2))
    print("Spar shear stress [MPa]:", np.round(results["tau_spars"] / 1e6, 2))
    print(f"Rate of twist dθ/dy:    {results['dtheta_dy']:.6e} rad/m")
    print(f"Torsional constant J:   {results['J']:.6e} m^4")
    print(f"Torsional stiffness GJ: {results['GJ']:.4e} N*m^2")
