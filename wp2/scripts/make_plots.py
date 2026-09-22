# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 matplotlib figures for every candidate airfoil.

Reads only the CSVs the polar/scorecard scripts already wrote (no XFoil
calls here -- plotting is a pure post-processing step over static data).
Writes, per airfoil, into wp2/results/plots/<airfoil>/:

- cl_vs_alpha_cruise.png    -- full cruise polar, stall + cruise-alpha marked,
                                non-converged tail shown distinctly
- cd_and_drag_polar.png     -- Cd vs alpha and the Cl-Cd drag polar, two
                                side-by-side panels (never a dual y-axis)
- cl_cd_vs_alpha.png        -- Cl/Cd vs alpha, cruise value annotated
- cm_vs_alpha.png           -- Cm vs alpha, cruise value annotated
- cp_distribution.png       -- Cp(x) at the M=0.2/Cl_n baseline solve
- mcrit_sweep.png           -- Cp_min(M) vs Cp_crit(M), M_crit crossing marked
- landing_cl_vs_alpha.png   -- landing polar, Cl_max_landing marked

and, into wp2/results/plots/comparison/:

- criteria_breakdown.png    -- grouped bar chart of the six weighted
                                criteria per airfoil (mirrors the original
                                screenshot table)
- total_score.png           -- total weighted score, ranked

Color: a fixed categorical assignment (one hue per airfoil, in
config.discover_airfoils() order -- not re-cycled per plot) drawn from the
Okabe-Ito colorblind-safe palette, used consistently across every figure so
the same airfoil reads as the same color throughout the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves PNGs

from b12wp2 import config
from b12wp2.config import solvers
from b12wp2.xfoil import polar_analysis

# Okabe-Ito colorblind-safe categorical palette (Okabe & Ito, 2008),
# assigned in a fixed order by discover_airfoils()'s sort order -- never
# reassigned per-plot, so a given airfoil is the same color everywhere.
_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
_NONCONVERGED_COLOR = "#999999"

DATA_DIR = config.RESULTS_DIR / "data"
PLOTS_DIR = config.RESULTS_DIR / "plots"


def airfoil_colors() -> dict[str, str]:
    stems = [p.stem for p in config.discover_airfoils()]
    if len(stems) > len(_PALETTE):
        raise ValueError(
            f"airfoil_colors: {len(stems)} airfoils but only {len(_PALETTE)} "
            "palette colors defined -- add more Okabe-Ito colors before "
            "plotting a larger candidate set"
        )
    return dict(zip(stems, _PALETTE))


def _style_axes(
    ax: "plt.Axes", *, zero_lines: bool = True, horizontal_zero: bool = True, vertical_zero: bool = True,
) -> None:
    """Recessive gridlines behind the data, no chart-junk spines, and an
    emphasized reference line at x=0/y=0 wherever that value falls within
    the plotted range -- a plain gridline at zero reads identically to
    every other gridline, easy to miss when the sign of a value (Cl, Cm,
    Cp, alpha) is exactly the thing that matters. axhline/axvline use a
    blended transform (data y or x, axes-fraction for the other axis) so
    they track the final autoscaled view regardless of being added here,
    before any real data is plotted.

    zero_lines=False for the two comparison bar charts: their x-axis is
    categorical (one position per criterion/airfoil, not a physical
    quantity), so an "x=0" reference line is meaningless there -- verified
    directly, it rendered as a stray vertical line poking through the
    first bar group.

    horizontal_zero/vertical_zero independently drop just the y=0 or x=0
    line: a quantity that never crosses zero (e.g. Cd) shouldn't pull that
    value into the autoscaled view just to draw a reference line for it --
    axhline/axvline count toward autoscale, so drawing one at 0 when the
    data never gets near 0 pads the axis with dead space.
    """
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero_lines and horizontal_zero:
        ax.axhline(0, color="#888888", linewidth=1.1, zorder=1)
    if zero_lines and vertical_zero:
        ax.axvline(0, color="#888888", linewidth=1.1, zorder=1)


def _read_polar(stem: str, condition: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{stem}_{condition}_polar.csv")


def _cruise_operating_point(stem: str) -> dict[str, float]:
    cruise = _read_polar(stem, "cruise")
    return polar_analysis.interpolate_at_cl(cruise, config.CRUISE.cl_normal)


def _plot_nonconverged_markers(
    ax: "plt.Axes", polar: pd.DataFrame, converged: pd.DataFrame
) -> None:
    """Mark every non-converged alpha with an 'x', at a y-position that
    honestly reflects where it sits relative to the converged curve.

    Not a single fixed height: the two-leg sweep can leave non-converged
    points both in genuine tail-breakdown regions (past either end of the
    converged range) AND as isolated single-point misses scattered inside
    an otherwise-converged region (a numerical hiccup, not a real aero
    break -- verified directly: NACA_64212's cruise polar has isolated
    non-converged points at alpha=-3.25, -2.50, -1.00, -0.75, 0.50, mixed
    in with converged neighbors on both sides). np.interp handles both
    cases in one call: it linearly interpolates a point's y from its
    converged neighbors when it falls inside their alpha range, and
    clamps to the nearest endpoint's y when it falls outside it (the
    tail case) -- so an isolated miss lands naturally on the curve's own
    trajectory instead of being yanked to a distant tail height, and a
    real tail lands at that end's Cl instead of being extrapolated.
    """
    nonconverged = polar[~polar["converged"]]
    if nonconverged.empty or converged.empty:
        return
    y = np.interp(nonconverged["alpha"], converged["alpha"], converged["cl"])
    ax.scatter(
        nonconverged["alpha"], y,
        marker="x", s=40, color=_NONCONVERGED_COLOR,
        label="did not converge (breakdown)", zorder=4,
    )


# Per-stem label placement, tuned by eye against each airfoil's own curve
# shape (stall Cl, non-converged marker clusters, etc.). Keyed as
# stem -> (cruise-label data xy, stall-label absolute data xy). A stem not
# listed here falls back to _CRUISE_ALPHA_LABEL_DEFAULT_XYTEXT for the
# cruise label and (stall_alpha, cl_max + 0.1) for the stall label.
_CRUISE_ALPHA_LABEL_OVERRIDES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "NACA_25112": ((3.1, 0.25), (6.70, 1.626)),
    "NACA_64212": ((3, 0.25), (4.8, 1.175)),
    "NASA_SC(2)-0712": ((-2, 1), (2.25, 1.35)),
}
_CRUISE_ALPHA_LABEL_DEFAULT_XYTEXT: tuple[float, float] = (3, 0.25)


def plot_cl_vs_alpha_cruise(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    cl_max, stall_alpha = polar_analysis.find_cl_max(cruise)
    op = _cruise_operating_point(stem)
    cruise_xytext, stall_xy = _CRUISE_ALPHA_LABEL_OVERRIDES.get(
        stem, (_CRUISE_ALPHA_LABEL_DEFAULT_XYTEXT, (stall_alpha, cl_max + 0.1))
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    converged = cruise[cruise["converged"]].sort_values("alpha")
    ax.plot(
        converged["alpha"], converged["cl"], color=color, linewidth=2,
        marker="o", markersize=4, label="converged", zorder=3,
    )
    _plot_nonconverged_markers(ax, cruise, converged)

    ax.axvline(op["alpha"], color="#555555", linestyle="--", linewidth=1, zorder=2)
    # Fixed data-coordinate placement, centered: an open region of the axes
    # clear of both the rising curve and the dashed cruise-alpha line.
    ax.annotate(
        f"cruise α={op['alpha']:.2f}°\nCl={config.CRUISE.cl_normal:.3f}",
        xy=(op["alpha"], config.CRUISE.cl_normal), xytext=cruise_xytext,
        textcoords="data", fontsize=9, ha="center", va="center",
    )
    ax.plot([stall_alpha], [cl_max], marker="*", markersize=14, color="#D55E00", zorder=5)
    # always label below the marker: it sits at the curve's Cl peak, which
    # is often near the top of the axes, so a label placed above it
    # collides with the title
    ax.annotate(
        f"stall α={stall_alpha:.2f}°\nCl_max={cl_max:.3f}",
        xy=stall_xy, xytext=(-90, -30), textcoords="offset points", fontsize=9,
    )

    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl")
    ax.set_title(f"{stem}: Cl vs α (cruise, M_n={config.CRUISE.mach_normal:.3f})")
    ax.margins(y=0.15)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cl_vs_alpha_cruise.png", dpi=150)
    plt.close(fig)


def plot_cd_and_drag_polar(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    op = _cruise_operating_point(stem)
    converged = cruise[cruise["converged"]].sort_values("alpha")

    # two side-by-side panels, never a shared dual y-axis
    fig, (ax_cd, ax_polar) = plt.subplots(1, 2, figsize=(12, 5.5))
    # Cd is always positive on both panels -- forcing its zero line into
    # view just pads the axis with dead space, so each panel only draws
    # the zero line for the axis that actually crosses zero (alpha on the
    # left, Cl on the right).
    _style_axes(ax_cd, horizontal_zero=False)
    _style_axes(ax_polar, vertical_zero=False)

    ax_cd.plot(converged["alpha"], converged["cd"], color=color, linewidth=2, marker="o", markersize=4)
    ax_cd.plot([op["alpha"]], [op["cd"]], marker="*", markersize=14, color="#D55E00", zorder=5)
    # Centered above with real vertical clearance, not a small diagonal
    # offset -- the cruise point usually sits near the flat bottom of the
    # drag bucket, and a small rightward offset can run into the curve
    # turning back up (verified directly on NASA_SC(2)-0712).
    ax_cd.annotate(
        f"cruise: Cd={op['cd']:.4f}", xy=(op["alpha"], op["cd"]),
        xytext=(0, 14), textcoords="offset points", fontsize=9, ha="center",
    )
    ax_cd.set_xlabel("Angle of attack α (deg)")
    ax_cd.set_ylabel("Cd")
    ax_cd.set_title("Cd vs α")

    ax_polar.plot(converged["cd"], converged["cl"], color=color, linewidth=2, marker="o", markersize=4)
    ax_polar.plot([op["cd"]], [op["cl"]], marker="*", markersize=14, color="#D55E00", zorder=5)
    ax_polar.annotate(
        f"cruise: Cl={op['cl']:.3f}, Cd={op['cd']:.4f}", xy=(op["cd"], op["cl"]),
        xytext=(8, -14), textcoords="offset points", fontsize=9,
    )
    ax_polar.set_xlabel("Cd")
    ax_polar.set_ylabel("Cl")
    ax_polar.set_title("Drag polar (Cl vs Cd)")

    fig.suptitle(f"{stem}: cruise drag characteristics (M_n={config.CRUISE.mach_normal:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "cd_and_drag_polar.png", dpi=150)
    plt.close(fig)


def plot_cl_cd_vs_alpha(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    op = _cruise_operating_point(stem)
    converged = cruise[cruise["converged"]].sort_values("alpha")
    cl_over_cd = converged["cl"] / converged["cd"]
    op_cl_cd = op["cl"] / op["cd"]

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(converged["alpha"], cl_over_cd, color=color, linewidth=2, marker="o", markersize=4)
    ax.plot([op["alpha"]], [op_cl_cd], marker="*", markersize=14, color="#D55E00", zorder=5)
    # Up and to the left, right-aligned: Cl/Cd vs alpha is rising through
    # the cruise point for every current candidate, so a center-anchored
    # label's right half can still reach the curve as it climbs back up
    # further right (verified directly on lockheed_c5a_bl758). Right-
    # aligning the text at a leftward offset keeps the whole label to the
    # left of the point, where the curve is lower.
    ax.annotate(
        f"cruise: Cl/Cd={op_cl_cd:.1f}", xy=(op["alpha"] - 0.25, op_cl_cd - 15),
        xytext=(2, 20), textcoords="offset points", fontsize=9, ha="right",
    )
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl / Cd")
    ax.set_title(f"{stem}: Cl/Cd vs α (cruise)")
    fig.tight_layout()
    fig.savefig(out_dir / "cl_cd_vs_alpha.png", dpi=150)
    plt.close(fig)


# Per-stem horizontal nudge for the cruise-label anchor (alpha units),
# tuned by eye against each airfoil's own Cm curve shape.
_CM_LABEL_ALPHA_DELTA: dict[str, float] = {
    "NACA_25112": -0.7,
}
_CM_LABEL_ALPHA_DELTA_DEFAULT = -0.2


def plot_cm_vs_alpha(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    op = _cruise_operating_point(stem)
    converged = cruise[cruise["converged"]].sort_values("alpha")
    alpha_delta = _CM_LABEL_ALPHA_DELTA.get(stem, _CM_LABEL_ALPHA_DELTA_DEFAULT)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(converged["alpha"], converged["cm"], color=color, linewidth=2, marker="o", markersize=4)
    ax.plot([op["alpha"]], [op["cm"]], marker="*", markersize=14, color="#D55E00", zorder=5)
    # Centered above with real vertical clearance -- Cm vs alpha is rising
    # through the cruise point for every current candidate, same rationale
    # as the Cl/Cd and Cd plots above.
    ax.annotate(
        f"cruise: Cm={op['cm']:.4f}", xy=(op["alpha"] + alpha_delta, op["cm"]),
        xytext=(0, 16), textcoords="offset points", fontsize=9, ha="center",
    )
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cm")
    ax.set_title(f"{stem}: pitching moment vs α (cruise)")
    fig.tight_layout()
    fig.savefig(out_dir / "cm_vs_alpha.png", dpi=150)
    plt.close(fig)


def plot_cp_distribution(stem: str, color: str, out_dir: Path) -> None:
    cp = pd.read_csv(DATA_DIR / f"{stem}_baseline_cp.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(cp["x"], cp["cp"], color=color, linewidth=1.5)
    ax.invert_yaxis()  # convention: suction (negative Cp) plotted upward
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp")
    ax.set_title(
        f"{stem}: Cp distribution at M={solvers.MCRIT_BASELINE_MACH:.2f}, "
        f"Cl_n={config.CRUISE.cl_normal:.3f} (cruise operating Cl)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cp_distribution.png", dpi=150)
    plt.close(fig)


def plot_mcrit_sweep(stem: str, color: str, out_dir: Path) -> None:
    sweep = pd.read_csv(DATA_DIR / f"{stem}_mcrit.csv")
    summary = pd.read_csv(DATA_DIR / "mcrit_summary.csv").set_index("airfoil")
    m_crit = summary.loc[stem, "m_crit"]

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(sweep["mach"], sweep["cp_min_corrected"], color=color, linewidth=2, label="Cp_min (Karman-Tsien corrected)")
    ax.plot(sweep["mach"], sweep["cp_crit"], color="#555555", linewidth=2, linestyle="--", label="Cp_crit (sonic)")
    # Karman-Tsien blows up approaching its own sqrt(1-M^2) singularity
    # near M=1 -- physically meaningless well past M_crit, where local
    # sonic flow (and the correction's validity) has already broken down.
    # Left unclipped, that blow-up dominates the y-scale and makes the
    # actual crossing unreadable (verified directly: Cp_min reaches
    # <-3000 by M=0.93 on this data). Clip to the well-behaved Cp_crit
    # curve's own range instead, which is smooth and bounded -- the
    # crossing region stays legible, the blow-up simply runs off-axis.
    y_lo = sweep["cp_crit"].min()
    ax.set_ylim(y_lo * 1.15, 0.5)
    if pd.notna(m_crit):
        ax.axvline(m_crit, color="#D55E00", linewidth=1.5, zorder=2)
        ax.annotate(f"M_crit={m_crit:.3f}", xy=(m_crit, 0.5), xytext=(6, -14),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mach")
    ax.set_ylabel("Cp")
    ax.set_title(f"{stem}: Mach-critical determination")
    # Upper left, not lower left: both curves start well below Cp=0 at the
    # low-Mach end (Cp_crit around -7, Cp_min around -1.5), so that corner
    # is consistently empty -- lower left sits right where the descending
    # Cp_crit dashed curve passes through, verified directly.
    legend_x = ax.get_xlim()[0] + 0.05
    legend_y = ax.get_ylim()[1] - 0.5
    ax.legend(
        loc="upper left", bbox_to_anchor=(legend_x, legend_y), bbox_transform=ax.transData,
        frameon=False, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "mcrit_sweep.png", dpi=150)
    plt.close(fig)


def plot_landing_cl_vs_alpha(stem: str, color: str, out_dir: Path) -> None:
    landing = _read_polar(stem, "landing")
    cl_max, stall_alpha = polar_analysis.find_cl_max(landing)
    converged = landing[landing["converged"]].sort_values("alpha")

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(
        converged["alpha"], converged["cl"], color=color, linewidth=2,
        marker="o", markersize=3, label="converged", zorder=3,
    )
    _plot_nonconverged_markers(ax, landing, converged)
    ax.plot([stall_alpha], [cl_max], marker="*", markersize=14, color="#D55E00", zorder=5)
    # The star marks the curve's global Cl peak, so placing the label
    # above it is always clear of the data -- nothing on the curve is
    # ever higher. A downward/sideways offset risked colliding with the
    # curve itself or with an isolated non-converged 'x' sitting on it
    # (verified directly: NACA_64212's landing polar has one at alpha~14,
    # right where a left-and-down label used to land).
    ax.annotate(
        f"Cl_max_landing={cl_max:.3f}\nα={stall_alpha:.2f}°",
        xy=(stall_alpha, cl_max), xytext=(-60, 14), textcoords="offset points", fontsize=9,
    )
    ax.margins(y=0.15)
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl")
    ax.set_title(f"{stem}: landing polar (M_n={config.LANDING.mach_normal:.3f})")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "landing_cl_vs_alpha.png", dpi=150)
    plt.close(fig)


# --- Cross-airfoil comparison plots ---

_CRITERION_LABELS = {
    "mach_critical": "Mach\nCritical",
    "cl_cd_cruise": "Cl/Cd\ncruise",
    "cl_max_landing": "Cl max\nlanding",
    "stall_margin": "Stall\nmargin",
    "cl_zero_angle": "Zero angle\nCl",
    "pitching_moment": "Pitching\nmoment",
}


def plot_criteria_breakdown(colors: dict[str, str], out_dir: Path) -> None:
    detail = pd.read_csv(config.RESULTS_DIR / "scorecard_detail.csv", index_col=0)
    criteria = list(config.SCORING_WEIGHTS)
    airfoils = list(detail.index)  # already ranked by total_score

    n_airfoils = len(airfoils)
    group_width = 0.8
    bar_width = group_width / n_airfoils
    x = range(len(criteria))

    fig, ax = plt.subplots(figsize=(11, 6))
    _style_axes(ax, zero_lines=False)
    for i, airfoil in enumerate(airfoils):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        values = [detail.loc[airfoil, f"{c}_weighted"] for c in criteria]
        ax.bar(offsets, values, width=bar_width, color=colors[airfoil], label=airfoil, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([_CRITERION_LABELS[c] for c in criteria])
    ax.set_ylabel("Weighted contribution")
    ax.set_title("WP2 scorecard: weighted contribution per criterion")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "criteria_breakdown.png", dpi=150)
    plt.close(fig)


def plot_total_score(colors: dict[str, str], out_dir: Path) -> None:
    detail = pd.read_csv(config.RESULTS_DIR / "scorecard_detail.csv", index_col=0)
    detail = detail.sort_values("total_score", ascending=False)
    airfoils = list(detail.index)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax, zero_lines=False)
    bars = ax.bar(
        airfoils, detail["total_score"], color=[colors[a] for a in airfoils], zorder=3,
    )
    for bar, score in zip(bars, detail["total_score"]):
        ax.annotate(
            f"{score:.3f}", xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10,
        )
    ax.set_ylabel("Total weighted score")
    ax.set_title("WP2 scorecard: total score, ranked")
    ax.set_ylim(0, max(detail["total_score"]) * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "total_score.png", dpi=150)
    plt.close(fig)


PER_AIRFOIL_PLOTS = [
    plot_cl_vs_alpha_cruise,
    plot_cd_and_drag_polar,
    plot_cl_cd_vs_alpha,
    plot_cm_vs_alpha,
    plot_cp_distribution,
    plot_mcrit_sweep,
    plot_landing_cl_vs_alpha,
]


def main() -> None:
    colors = airfoil_colors()
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    for airfoil_path in config.discover_airfoils():
        stem = airfoil_path.stem
        out_dir = plots_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{stem}:")
        for plot_fn in PER_AIRFOIL_PLOTS:
            plot_fn(stem, colors[stem], out_dir)
            print(f"    -> {out_dir / (plot_fn.__name__.replace('plot_', '') + '.png')}")

    comparison_dir = plots_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_criteria_breakdown(colors, comparison_dir)
    plot_total_score(colors, comparison_dir)
    print("comparison:")
    print(f"    -> {comparison_dir / 'criteria_breakdown.png'}")
    print(f"    -> {comparison_dir / 'total_score.png'}")


if __name__ == "__main__":
    main()
