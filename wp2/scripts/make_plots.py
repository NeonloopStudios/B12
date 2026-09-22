# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2: draw every 2D section figure for every candidate airfoil.

Runs the figures of b12wp2.plots.section over the current candidate set and
writes them to results/section/plots/<airfoil>/ plus the two cross-candidate
figures to results/section/plots/comparison/. Needs the polar, mcrit and scorecard CSVs to
exist already -- it reads them, it never re-runs XFoil.
"""
from __future__ import annotations

from b12wp2 import config
from b12wp2.config import paths
from b12wp2.plots.section import PER_AIRFOIL_PLOTS, airfoil_colors, plot_criteria_breakdown, plot_total_score


def main() -> None:
    colors = airfoil_colors()
    paths.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for airfoil_path in config.discover_airfoils():
        stem = airfoil_path.stem
        out_dir = paths.section_plots_dir(stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{stem}:")
        for plot_fn in PER_AIRFOIL_PLOTS:
            plot_fn(stem, colors[stem], out_dir)
            print(f"    -> {out_dir / (plot_fn.__name__.replace('plot_', '') + '.png')}")

    comparison_dir = paths.COMPARISON_PLOTS_DIR
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_criteria_breakdown(colors, comparison_dir)
    plot_total_score(colors, comparison_dir)
    print("comparison:")
    print(f"    -> {comparison_dir / 'criteria_breakdown.png'}")
    print(f"    -> {comparison_dir / 'total_score.png'}")


if __name__ == "__main__":
    main()
