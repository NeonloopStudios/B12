# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Nacelle figure: side-view outline of the Type B nacelle with its dimensions."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402

from b12wp3.propulsion.nacelle import NacelleGeometry  # noqa: E402

FIGURE_DPI: Final[int] = 150
OUTLINE_COLOR: Final[str] = "#4C72B0"


def plot_nacelle_outline(nacelle: NacelleGeometry, out_path: Path) -> Path:
    """Side view of the nacelle outline (highlight, max diameter, exit).

    Args:
        nacelle: Sized nacelle.
        out_path: PNG file to write. The parent directory is created.

    Returns:
        out_path.
    """
    x_max = nacelle.max_diameter_position * nacelle.length_m
    xs = [0.0, x_max, nacelle.length_m]
    radii = [
        nacelle.highlight_diameter_m / 2.0,
        nacelle.max_diameter_m / 2.0,
        nacelle.exit_diameter_m / 2.0,
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    try:
        ax.plot(xs, radii, color=OUTLINE_COLOR, marker="o")
        ax.plot(xs, [-r for r in radii], color=OUTLINE_COLOR, marker="o")
        ax.axhline(0.0, color="0.6", linestyle="-.", linewidth=0.8)
        labels = (
            f"D_h = {nacelle.highlight_diameter_m:.3f} m",
            f"D_n = {nacelle.max_diameter_m:.3f} m",
            f"D_ef = {nacelle.exit_diameter_m:.3f} m",
        )
        for x, r, label in zip(xs, radii, labels, strict=True):
            ax.annotate(
                label, (x, r), textcoords="offset points", xytext=(0, 8), ha="center"
            )
        ax.set_xlabel(
            f"x [m]   (l_n = {nacelle.length_m:.3f} m, "
            f"D_n at beta l_n = {x_max:.3f} m)"
        )
        ax.set_ylabel("r [m]")
        ax.set_title("Type B nacelle outline (Torenbeek)")
        ax.set_aspect("equal")
        ax.set_ylim(-0.85 * nacelle.max_diameter_m, 0.85 * nacelle.max_diameter_m)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path
