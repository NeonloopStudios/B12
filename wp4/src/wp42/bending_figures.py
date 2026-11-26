# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luca Paraschiv
# Created: 2026-11-26
"""WP4.2: plots of internal moment, bending stiffness, and vertical displacement."""

from typing import Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def bending_plot(
    b_station: [float],
    m_station: [float],
    I_xx: Callable[y],
    v: [float],
    num_smooth_points: int = 300,
    figsize: tuple[float, float] = (8.0, 9.0),
) -> plt.Figure:
    
    b = np.asarray(b_station, dtype=float)
    m = np.asarray(m_station, dtype=float)
    try:
        i_vals = np.asarray(I_xx(b), dtype=float)
        if i_vals.shape != b.shape:
            i_vals = np.asarray([I_xx(float(bi)) for bi in b], dtype=float)
    except TypeError:
        i_vals = np.asarray([I_xx(float(bi)) for bi in b], dtype=float)
    v_vals = np.asarray(v, dtype=float)

    if not (len(b) == len(m) == len(i_vals) == len(v_vals)):
        raise ValueError(
            "b_station, m_station, I_xx(b_station), and v must all have the same length."
        )
    if len(b) < 2:
        raise ValueError("At least two station points are required.")

    order = np.argsort(b)
    b = b[order]
    m = m[order]
    i_vals = i_vals[order]
    v_vals = v_vals[order]

    b_smooth = np.linspace(b.min(), b.max(), num_smooth_points)

    def smooth_curve(y_vals: np.ndarray) -> np.ndarray:
        k = min(3, len(b) - 1)
        spline = make_interp_spline(b, y_vals, k=k)
        return spline(b_smooth)

    m_smooth = smooth_curve(m)
    i_smooth = smooth_curve(i_vals)
    v_smooth = smooth_curve(v_vals)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    series_config = [
        (axes[0], m, m_smooth, "M [Nm]", "Internal Moment", False),
        (axes[1], i_vals, i_smooth, "I_xx [m^4]", "Bending Stiffness", False),
        (axes[2], v_vals, v_smooth, "v [m]", "Vertical Displacement", True),
    ]

    x_min = min(0.0, float(b.min()))
    x_max = float(b.max())

    for ax, raw_y, smooth_y, ylabel, title, is_negative_axis in series_config:
        ax.plot(b_smooth, smooth_y, linewidth=2.0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("data", x_min))

        ax.set_xlim(x_min, x_max)

        if is_negative_axis:
            y_origin = max(0.0, float(np.max(smooth_y)))
            y_most_negative = min(0.0, float(np.min(smooth_y)))
            if np.isclose(y_origin, y_most_negative):
                y_most_negative = -1.0
            ax.spines["bottom"].set_position(("data", y_origin))
            ax.set_ylim(y_origin, y_most_negative)
        else:
            y_bottom = min(0.0, float(np.min(smooth_y)))
            y_top = max(0.0, float(np.max(smooth_y)))
            if np.isclose(y_top, y_bottom):
                y_top = 1.0
            ax.spines["bottom"].set_position(("data", y_bottom))
            ax.set_ylim(y_bottom, y_top)

        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("half-span position [m]")
    fig.tight_layout()
    return fig