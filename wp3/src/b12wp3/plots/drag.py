# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Class II drag figure: the CD0 contribution of each component."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402

from b12wp3.drag.buildup import DragBuildup  # noqa: E402

DRAG_COUNT: Final[float] = 1e-4  # one drag count, in CD
FIGURE_DPI: Final[int] = 150


def plot_cd0_breakdown(buildup: DragBuildup, out_path: Path) -> Path:
    """Horizontal bar chart of the CD0 contribution of each term.

    Components are drawn first, then the miscellaneous terms.

    Args:
        buildup: The drag build-up.
        out_path: PNG file to write. The parent directory is created.

    Returns:
        out_path.
    """
    labels = [c.name for c in buildup.components] + [m.name for m in buildup.misc]
    counts = [buildup.contribution(c) / DRAG_COUNT for c in buildup.components]
    counts.extend(m.delta_cd / DRAG_COUNT for m in buildup.misc)
    colors = ["#4C72B0"] * len(buildup.components) + ["#DD8452"] * len(buildup.misc)

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(labels) + 1.5))
    try:
        bars = ax.barh(labels, counts, color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.invert_yaxis()
        ax.set_xlabel("CD0 contribution [drag counts]")
        ax.set_title(f"Class II drag build-up: CD0 = {buildup.cd0:.5f}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path
