# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 Stage 8: matplotlib figures for every candidate airfoil.

Reads only the CSVs Stages 3/4/5/7 already wrote (no XFoil calls here --
plotting is a pure post-processing step over static data). Writes, per
airfoil, into wp2/results/plots/<airfoil>/:

1. cl_vs_alpha_cruise.png    -- full cruise polar, stall + cruise-alpha marked,
                                 non-converged tail shown distinctly
2. cd_and_drag_polar.png     -- Cd vs alpha and the Cl-Cd drag polar, two
                                 side-by-side panels (never a dual y-axis)
3. cl_cd_vs_alpha.png        -- Cl/Cd vs alpha, cruise value annotated
4. cm_vs_alpha.png           -- Cm vs alpha, cruise value annotated
5. cp_distribution.png       -- Cp(x) at the M=0.2/Cl_n baseline solve
6. mcrit_sweep.png           -- Cp_min(M) vs Cp_crit(M), M_crit crossing marked
7. landing_cl_vs_alpha.png   -- landing polar, Cl_max_landing marked

and, into wp2/results/plots/comparison/:

8. criteria_breakdown.png    -- grouped bar chart of the six weighted
                                 criteria per airfoil (mirrors the original
                                 screenshot table)
9. total_score.png           -- total weighted score, ranked

Color: a fixed categorical assignment (one hue per airfoil, in
config.discover_airfoils() order -- not re-cycled per plot) drawn from the
Okabe-Ito colorblind-safe palette, used consistently across every figure so
the same airfoil reads as the same color throughout the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # headless: this script only ever saves PNGs

from scripts import config, mcrit_sweep, polar_analysis

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


def _style_axes(ax: "plt.Axes") -> None:
    """Recessive gridlines behind the data, no chart-junk spines."""
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _read_polar(stem: str, condition: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{stem}_{condition}_polar.csv")


def _cruise_operating_point(stem: str) -> dict[str, float]:
    cruise = _read_polar(stem, "cruise")
    return polar_analysis.interpolate_at_cl(cruise, config.CRUISE.cl_normal)


def plot_cl_vs_alpha_cruise(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    cl_max, stall_alpha = polar_analysis.find_cl_max(cruise)
    op = _cruise_operating_point(stem)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    converged = cruise[cruise["converged"]].sort_values("alpha")
    ax.plot(
        converged["alpha"], converged["cl"], color=color, linewidth=2,
        marker="o", markersize=4, label="converged", zorder=3,
    )
    # Non-converged tail: two separate breakdown regions (the two-leg sweep
    # stops independently at the low-alpha end and the high-alpha end), so
    # each tail is plotted at the Cl of the curve's own nearest end, not a
    # single shared height -- a flat height for both sides would place the
    # low-alpha breakdown points at the high-alpha Cl, misrepresenting
    # where the curve actually broke down.
    nonconverged = cruise[~cruise["converged"]]
    if not nonconverged.empty and not converged.empty:
        alpha_min, alpha_max = converged["alpha"].min(), converged["alpha"].max()
        cl_at_min, cl_at_max = converged["cl"].iloc[0], converged["cl"].iloc[-1]
        tail_y = nonconverged["alpha"].apply(
            lambda a: cl_at_min if a < alpha_min else cl_at_max
        )
        ax.scatter(
            nonconverged["alpha"], tail_y,
            marker="x", s=40, color=_NONCONVERGED_COLOR,
            label="did not converge (breakdown)", zorder=4,
        )

    ax.axvline(op["alpha"], color="#555555", linestyle="--", linewidth=1, zorder=2)
    ax.annotate(
        f"cruise α={op['alpha']:.2f}°\nCl={config.CRUISE.cl_normal:.3f}",
        xy=(op["alpha"], config.CRUISE.cl_normal), xytext=(8, -18),
        textcoords="offset points", fontsize=9,
    )
    ax.plot([stall_alpha], [cl_max], marker="*", markersize=14, color="#D55E00", zorder=5)
    # always label below the marker: it sits at the curve's Cl peak, which
    # is often near the top of the axes, so a label placed above it
    # collides with the title
    ax.annotate(
        f"stall α={stall_alpha:.2f}°\nCl_max={cl_max:.3f}",
        xy=(stall_alpha, cl_max), xytext=(-90, -30), textcoords="offset points", fontsize=9,
    )

    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl")
    ax.set_title(f"{stem} — Cl vs α (cruise, M_n={config.CRUISE.mach_normal:.3f})")
    ax.margins(y=0.15)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "cl_vs_alpha_cruise.png", dpi=150)
    plt.close(fig)


def plot_cd_and_drag_polar(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    op = _cruise_operating_point(stem)
    converged = cruise[cruise["converged"]].sort_values("alpha")

    # two side-by-side panels, never a shared dual y-axis
    fig, (ax_cd, ax_polar) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax in (ax_cd, ax_polar):
        _style_axes(ax)

    ax_cd.plot(converged["alpha"], converged["cd"], color=color, linewidth=2, marker="o", markersize=4)
    ax_cd.plot([op["alpha"]], [op["cd"]], marker="*", markersize=14, color="#D55E00", zorder=5)
    ax_cd.annotate(
        f"cruise: Cd={op['cd']:.4f}", xy=(op["alpha"], op["cd"]),
        xytext=(8, 8), textcoords="offset points", fontsize=9,
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

    fig.suptitle(f"{stem} — cruise drag characteristics (M_n={config.CRUISE.mach_normal:.3f})")
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
    ax.annotate(
        f"cruise: Cl/Cd={op_cl_cd:.1f}", xy=(op["alpha"], op_cl_cd),
        xytext=(8, 8), textcoords="offset points", fontsize=9,
    )
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl / Cd")
    ax.set_title(f"{stem} — Cl/Cd vs α (cruise)")
    fig.tight_layout()
    fig.savefig(out_dir / "cl_cd_vs_alpha.png", dpi=150)
    plt.close(fig)


def plot_cm_vs_alpha(stem: str, color: str, out_dir: Path) -> None:
    cruise = _read_polar(stem, "cruise")
    op = _cruise_operating_point(stem)
    converged = cruise[cruise["converged"]].sort_values("alpha")

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(converged["alpha"], converged["cm"], color=color, linewidth=2, marker="o", markersize=4)
    ax.plot([op["alpha"]], [op["cm"]], marker="*", markersize=14, color="#D55E00", zorder=5)
    ax.annotate(
        f"cruise: Cm={op['cm']:.4f}", xy=(op["alpha"], op["cm"]),
        xytext=(8, 8), textcoords="offset points", fontsize=9,
    )
    ax.axhline(0.0, color="#999999", linewidth=1, linestyle=":", zorder=1)
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cm")
    ax.set_title(f"{stem} — pitching moment vs α (cruise)")
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
        f"{stem} — Cp distribution at M={mcrit_sweep.BASELINE_MACH:.2f}, "
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
    ax.set_title(f"{stem} — Mach-critical determination")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "mcrit_sweep.png", dpi=150)
    plt.close(fig)


def plot_landing_cl_vs_alpha(stem: str, color: str, out_dir: Path) -> None:
    landing = _read_polar(stem, "landing")
    cl_max, stall_alpha = polar_analysis.find_cl_max(landing)
    converged = landing[landing["converged"]].sort_values("alpha")

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    ax.plot(converged["alpha"], converged["cl"], color=color, linewidth=2, marker="o", markersize=3)
    ax.plot([stall_alpha], [cl_max], marker="*", markersize=14, color="#D55E00", zorder=5)
    ax.annotate(
        f"Cl_max_landing={cl_max:.3f}\nα={stall_alpha:.2f}°",
        xy=(stall_alpha, cl_max), xytext=(-110, -35), textcoords="offset points", fontsize=9,
    )
    ax.margins(y=0.15)
    ax.set_xlabel("Angle of attack α (deg)")
    ax.set_ylabel("Cl")
    ax.set_title(f"{stem} — landing polar (M_n={config.LANDING.mach_normal:.3f})")
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
    _style_axes(ax)
    for i, airfoil in enumerate(airfoils):
        offsets = [xi - group_width / 2 + bar_width * (i + 0.5) for xi in x]
        values = [detail.loc[airfoil, f"{c}_weighted"] for c in criteria]
        ax.bar(offsets, values, width=bar_width, color=colors[airfoil], label=airfoil, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels([_CRITERION_LABELS[c] for c in criteria])
    ax.set_ylabel("Weighted contribution")
    ax.set_title("WP2 scorecard — weighted contribution per criterion")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "criteria_breakdown.png", dpi=150)
    plt.close(fig)


def plot_total_score(colors: dict[str, str], out_dir: Path) -> None:
    detail = pd.read_csv(config.RESULTS_DIR / "scorecard_detail.csv", index_col=0)
    detail = detail.sort_values("total_score", ascending=False)
    airfoils = list(detail.index)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style_axes(ax)
    bars = ax.bar(
        airfoils, detail["total_score"], color=[colors[a] for a in airfoils], zorder=3,
    )
    for bar, score in zip(bars, detail["total_score"]):
        ax.annotate(
            f"{score:.3f}", xy=(bar.get_x() + bar.get_width() / 2, score),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10,
        )
    ax.set_ylabel("Total weighted score")
    ax.set_title("WP2 scorecard — total score, ranked")
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
