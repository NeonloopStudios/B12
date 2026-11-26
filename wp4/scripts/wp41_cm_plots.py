# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-26
"""WP4.1: plots of the VSPAERO pitching moment, one line per Mach number.

Reads the VSPAERO database (run wp41_aero_loads.py first). Writes two PNG
files to out/wp41/. All moments are nose-up positive.

- cm_vs_cl.png: wing CM about the MAC quarter chord against CL. Each line
  joins the two angles of attack of the database, so it is straight by
  construction (the database is linear in alpha).
- cm_span.png: spanwise cm about the local quarter chord (top row) and
  cm c^2 / MAC^2 = m' / (q MAC^2) (bottom row), one column per angle of
  attack.

    python wp4/scripts/wp41_cm_plots.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import WING  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from wp41 import aero_loads  # noqa: E402
from wp41 import lift_distribution as ld  # noqa: E402

logger = logging.getLogger("wp41_cm_plots")

# Single-hue sequential ramp, light (low Mach) to dark (high Mach)
COLORMAP: Final[str] = "Blues"
COLOR_LOW: Final[float] = 0.35
COLOR_HIGH: Final[float] = 1.0
LABEL_INK: Final[str] = "#444444"
GRID_INK: Final[str] = "#e6e6e6"
ZERO_INK: Final[str] = "#999999"
SPAN_LABEL_ETA: Final[float] = 0.55  # span fraction of the Mach labels


def _mach_colors(mach: NDArray[np.float64]) -> NDArray[np.float64]:
    """RGBA colour of each Mach number, light to dark, shape (n_mach, 4)."""
    return np.asarray(
        plt.get_cmap(COLORMAP)(
            np.linspace(COLOR_LOW, COLOR_HIGH, mach.size, dtype=np.float64)
        ),
        dtype=np.float64,
    )


def _mach_colorbar(
    mach: NDArray[np.float64], colors: NDArray[np.float64]
) -> ScalarMappable:
    """Stepped colour scale with one band per Mach number of the grid."""
    half_step = 0.5 * float(np.min(np.diff(mach)))
    edges = np.concatenate(([mach[0] - half_step], mach + half_step))
    return ScalarMappable(
        cmap=ListedColormap(colors), norm=BoundaryNorm(edges, mach.size)
    )


def _style_axes(ax: Axes, zero_line: bool) -> None:
    """Recessive grid, no top and right spines, optional zero line."""
    ax.grid(True, color=GRID_INK, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if zero_line:
        ax.axhline(0.0, color=ZERO_INK, linewidth=0.8)


def plot_cm_cl(database: ld.AeroDatabase, path: Path) -> None:
    """Plot CM against CL for every Mach of the database and save the PNG.

    Args:
        database: VSPAERO database with the wing totals CL and CM.
        path: PNG file.
    """
    mach = database.mach
    colors = _mach_colors(mach)
    labelled = {0, mach.size // 2, mach.size - 1}

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for i in range(mach.size):
        cl = database.cl_wing[i]
        cm = database.cm_wing[i]
        ax.plot(cl, cm, color=colors[i], linewidth=2.0, solid_capstyle="round")
        ax.plot(cl, cm, "o", markersize=5, color=colors[i], mec="white", mew=1.2)
        if i in labelled:
            ax.annotate(
                f"M {mach[i]:.2f}", (cl[1], cm[1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8, color=LABEL_INK,
            )
    alpha_lo, alpha_hi = database.alpha_deg
    ax.annotate(
        f"α = {alpha_lo:g}°", (database.cl_wing[-1, 0], database.cm_wing[-1, 0]),
        xytext=(0, -12), textcoords="offset points", ha="center", fontsize=8,
        color=LABEL_INK,
    )
    ax.annotate(
        f"α = {alpha_hi:g}°", (database.cl_wing[0, 1], database.cm_wing[0, 1]),
        xytext=(-4, 10), textcoords="offset points", ha="right", fontsize=8,
        color=LABEL_INK,
    )

    ax.set_xlabel("wing lift coefficient CL [-]")
    ax.set_ylabel("pitching moment coefficient CM about MAC c/4 [-]")
    ax.set_title(
        "Wing CM against CL per Mach (VSPAERO VLM, nose-up positive)",
        fontsize=10, loc="left",
    )
    _style_axes(ax, zero_line=False)
    ax.margins(x=0.08)

    bar = fig.colorbar(
        _mach_colorbar(mach, colors), ax=ax, ticks=mach[::2], pad=0.02
    )
    bar.set_label("Mach [-]")
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_cm_span(database: ld.AeroDatabase, mac_m: float, path: Path) -> None:
    """Plot the spanwise pitching moment per Mach and angle of attack.

    Top row: cm about the local quarter chord. Bottom row: cm c^2 / MAC^2,
    the moment per unit span m' divided by q MAC^2. One column per angle of
    attack of the database.

    Args:
        database: VSPAERO database with the strip cm_c4 and chords.
        mac_m: Mean aerodynamic chord, the scale of the bottom row [m].
        path: PNG file.
    """
    mach = database.mach
    colors = _mach_colors(mach)
    eta = database.y / database.half_span_m
    chord_ratio_sq = (database.chord / mac_m) ** 2
    k_label = int(np.searchsorted(eta, SPAN_LABEL_ETA))
    n_alpha = database.alpha_deg.size

    fig, axes = plt.subplots(
        2, n_alpha, figsize=(11.0, 7.4), sharex=True, sharey="row", squeeze=False
    )
    for j, alpha in enumerate(database.alpha_deg):
        ax_cm, ax_m = axes[0, j], axes[1, j]
        for i in range(mach.size):
            cm = database.cm_c4[:, i, j]
            ax_cm.plot(eta, cm, color=colors[i], linewidth=1.6)
            ax_m.plot(eta, cm * chord_ratio_sq, color=colors[i], linewidth=1.6)
        for i, offset in ((0, 6), (mach.size - 1, -9)):
            ax_cm.annotate(
                f"M {mach[i]:.2f}", (eta[k_label], database.cm_c4[k_label, i, j]),
                xytext=(0, offset), textcoords="offset points", ha="center",
                fontsize=8, color=LABEL_INK,
            )
        ax_cm.set_title(f"α = {alpha:g}°", fontsize=10, loc="left")
        ax_m.set_xlabel("span fraction η = 2y/b [-]")
        _style_axes(ax_cm, zero_line=True)
        _style_axes(ax_m, zero_line=True)
    axes[0, 0].set_ylabel("local cm about local c/4 [-]")
    axes[1, 0].set_ylabel("cm · c² / MAC²  (= m' / (q MAC²)) [-]")
    fig.suptitle(
        "Spanwise pitching moment, right half-wing (VSPAERO VLM, nose-up positive)",
        fontsize=11, x=0.01, ha="left",
    )

    bar = fig.colorbar(
        _mach_colorbar(mach, colors), ax=axes, ticks=mach[::2], pad=0.015,
        fraction=0.03,
    )
    bar.set_label("Mach [-]")
    bar.outline.set_visible(False)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def main() -> None:
    """Read the VSPAERO database and write the two pitching moment plots."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    database = ld.read_aero_database(
        settings.OUT_DIR / settings.LIFT_FILE, settings.OUT_DIR / settings.MOMENT_FILE
    )
    plot_cm_cl(database, settings.OUT_DIR / settings.CM_CL_PLOT_FILE)
    plot_cm_span(
        database,
        aero_loads.planform(WING).mac_m,
        settings.OUT_DIR / settings.CM_SPAN_PLOT_FILE,
    )


if __name__ == "__main__":
    main()
