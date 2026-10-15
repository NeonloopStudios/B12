# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Drag-due-to-lift figures: induced drag with and without the winglet, and
the aircraft drag polar."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from b12wp3.drag.induced import InducedDrag  # noqa: E402
from b12wp3.drag.polar import DragPolar  # noqa: E402

FIGURE_DPI: Final[int] = 150
CL_MAX_PLOTTED: Final[float] = 1.0


def plot_induced_drag(
    without_winglet: InducedDrag,
    with_winglet: InducedDrag,
    cruise_cl: float,
    out_path: Path,
) -> Path:
    """C_Di against C_L for the wing without and with the winglet.

    Args:
        without_winglet: Clean wing, h = 0.
        with_winglet: Clean wing with the winglet.
        cruise_cl: Cruise lift coefficient, marked on both curves.
        out_path: PNG file to write. The parent directory is created.

    Returns:
        out_path.
    """
    cl = np.linspace(0.0, CL_MAX_PLOTTED, 101)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    try:
        for case, style in ((without_winglet, "--"), (with_winglet, "-")):
            label = (
                f"{case.name}: AR_eff = {case.effective_aspect_ratio:.2f}, "
                f"e = {case.oswald:.3f}"
            )
            ax.plot(cl, case.k * cl**2, style, label=label)
            ax.plot(cruise_cl, case.cdi(cruise_cl), "o", color="black", ms=4)
        ax.axvline(cruise_cl, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("$C_L$ [-]")
        ax.set_ylabel("$C_{D_i}$ [-]")
        ax.set_title("Drag due to lift, clean wing (Brandt, $AR_{eff}$)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path


def plot_drag_polar(polar: DragPolar, cruise_cl: float, out_path: Path) -> Path:
    """Drag polar C_L(C_D) and L/D(C_L), with the cruise and (L/D)max points.

    Args:
        polar: The aircraft drag polar.
        cruise_cl: Cruise lift coefficient.
        out_path: PNG file to write. The parent directory is created.

    Returns:
        out_path.
    """
    cl = np.linspace(0.0, CL_MAX_PLOTTED, 101)
    cd = polar.cd0 + polar.k * cl**2
    cl_opt = polar.cl_at_max_lift_to_drag
    fig, (ax_polar, ax_ld) = plt.subplots(1, 2, figsize=(10.0, 4.5))
    try:
        ax_polar.plot(cd, cl, "-")
        ax_polar.plot(polar.cd(cruise_cl), cruise_cl, "o", label="cruise")
        ax_polar.plot(polar.cd(cl_opt), cl_opt, "s", label="$(L/D)_{max}$")
        ax_polar.set_xlabel("$C_D$ [-]")
        ax_polar.set_ylabel("$C_L$ [-]")
        ax_polar.set_title(f"$C_D = {polar.cd0:.5f} + {polar.k:.5f}\\,C_L^2$")
        ax_polar.grid(alpha=0.3)
        ax_polar.legend()

        ax_ld.plot(cl, cl / cd, "-")
        ax_ld.plot(cruise_cl, polar.lift_to_drag(cruise_cl), "o", label="cruise")
        ax_ld.plot(cl_opt, polar.max_lift_to_drag, "s", label="$(L/D)_{max}$")
        ax_ld.set_xlabel("$C_L$ [-]")
        ax_ld.set_ylabel("$L/D$ [-]")
        ax_ld.grid(alpha=0.3)
        ax_ld.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path
