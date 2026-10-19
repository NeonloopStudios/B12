# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Class II drag figures.

    plot_cd0_breakdown   CD0 contribution of each term, one configuration
    plot_cd0_comparison  total CD0 of every configuration, stacked by group

Colors are the categorical slots of the reference palette, in fixed order:
a group keeps its color in every configuration and every figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: only saves PNGs

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from b12wp3.drag.buildup import DragBuildup  # noqa: E402

DRAG_COUNT: Final[float] = 1e-4  # one drag count, in CD
FIGURE_DPI: Final[int] = 150
SEGMENT_GAP_COLOR: Final[str] = "#ffffff"  # surface gap between stacked fills
TEXT_COLOR: Final[str] = "#2b2b2a"
COMPONENT_COLOR: Final[str] = "#2a78d6"  # slot 1, blue
MISC_COLOR: Final[str] = "#eb6834"  # slot 2, orange

# Term name -> group of the comparison chart.
TERM_GROUPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "wing": "wing",
        "horizontal_tail": "tails",
        "vertical_tail": "tails",
        "fuselage": "fuselage",
        "fuselage_upsweep": "fuselage",
        "fuselage_base": "fuselage",
        "nacelle": "nacelles",
        "te_flap": "flaps",
        "le_flap": "flaps",
        "main_gear": "landing gear",
        "nose_gear": "landing gear",
        "wave_drag": "wave + excrescence",
        "excrescence_leakage": "wave + excrescence",
    }
)

# Group -> categorical slot (fixed order, never cycled).
GROUP_COLORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "wing": "#2a78d6",
        "tails": "#eb6834",
        "fuselage": "#1baf7a",
        "nacelles": "#eda100",
        "flaps": "#e87ba4",
        "landing gear": "#008300",
        "wave + excrescence": "#6250d6",
    }
)


def grouped_contributions(buildup: DragBuildup) -> dict[str, float]:
    """CD0 of each group of the comparison chart, in GROUP_COLORS order.

    Raises:
        KeyError: If a term has no group in TERM_GROUPS.
    """
    totals = dict.fromkeys(GROUP_COLORS, 0.0)
    for component in buildup.components:
        totals[TERM_GROUPS[component.name]] += buildup.contribution(component)
    for term in buildup.misc:
        totals[TERM_GROUPS[term.name]] += term.delta_cd
    return totals


def plot_cd0_breakdown(
    buildup: DragBuildup, out_path: Path, configuration: str = ""
) -> Path:
    """Horizontal bar chart of the CD0 contribution of each term.

    Components (blue) are drawn first, then the miscellaneous terms (orange).

    Args:
        buildup: The drag build-up.
        out_path: PNG file to write. The parent directory is created.
        configuration: Configuration name for the title.

    Returns:
        out_path.
    """
    labels = [c.name for c in buildup.components] + [m.name for m in buildup.misc]
    counts = [buildup.contribution(c) / DRAG_COUNT for c in buildup.components]
    counts.extend(m.delta_cd / DRAG_COUNT for m in buildup.misc)
    colors = [COMPONENT_COLOR] * len(buildup.components)
    colors += [MISC_COLOR] * len(buildup.misc)
    prefix = f"{configuration}: " if configuration else ""

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.6))
    try:
        bars = ax.barh(labels, counts, color=colors, height=0.6)
        ax.bar_label(bars, fmt="%.1f", padding=3, color=TEXT_COLOR, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("CD0 contribution [drag counts, 1 count = 0.0001]")
        ax.set_title(
            f"{prefix}class II CD0 = {buildup.cd0:.5f} "
            f"({buildup.cd0 / DRAG_COUNT:.1f} counts)"
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.12)
        handles = [
            Rectangle((0, 0), 1, 1, color=COMPONENT_COLOR),
            Rectangle((0, 0), 1, 1, color=MISC_COLOR),
        ]
        ax.legend(
            handles,
            ["component (Cf FF IF S_wet / S_ref)", "miscellaneous"],
            loc="lower right",
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path


def plot_cd0_comparison(
    buildups: Sequence[tuple[str, DragBuildup]], out_path: Path
) -> Path:
    """Stacked bars of the total CD0 of every configuration, by group.

    Args:
        buildups: (configuration name, build-up) pairs, in plot order.
        out_path: PNG file to write. The parent directory is created.

    Returns:
        out_path.

    Raises:
        ValueError: If no build-up is given.
    """
    if not buildups:
        raise ValueError("plot_cd0_comparison needs at least one build-up")
    names = [name for name, _ in buildups]
    groups = [grouped_contributions(buildup) for _, buildup in buildups]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    try:
        bottoms = [0.0] * len(buildups)
        for group, color in GROUP_COLORS.items():
            heights = [g[group] / DRAG_COUNT for g in groups]
            ax.bar(
                names,
                heights,
                bottom=bottoms,
                width=0.55,
                color=color,
                edgecolor=SEGMENT_GAP_COLOR,
                linewidth=2,
                label=group,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
        for index, (_, buildup) in enumerate(buildups):
            ax.annotate(
                f"CD0 = {buildup.cd0:.4f}",
                (index, buildup.cd0 / DRAG_COUNT),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                color=TEXT_COLOR,
                fontsize=9,
            )
        ax.set_ylabel("CD0 [drag counts, 1 count = 0.0001]")
        ax.set_title("Class II CD0 per configuration")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.1)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1],
            labels[::-1],
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI)
    finally:
        plt.close(fig)
    return out_path
