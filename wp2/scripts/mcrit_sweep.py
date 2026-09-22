# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 Mach-critical determination for every candidate airfoil.

Runs b12wp2.xfoil.mcrit.run_mcrit_analysis over every airfoil in wp2/airfoils/
and writes, per airfoil, the Mach sweep (mach, cp_min_corrected, cp_crit) and
the baseline Cp(x) distribution the sweep was built from -- make_plots reads
that back instead of re-invoking XFoil -- plus one summary row per airfoil
(m_crit, m_dd, margin, thickness_to_chord, cl_cruise, kappa_a) in
mcrit_summary.csv. File locations: config.paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from b12wp2 import config
from b12wp2.config import paths, solvers
from b12wp2.xfoil.mcrit import run_mcrit_analysis


def main() -> None:
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Baseline Mach={solvers.MCRIT_BASELINE_MACH}, Cl_n={config.CRUISE.cl_normal:.4f}, "
        f"cruise M_n={config.CRUISE.mach_normal:.4f} (Korn margin reference)"
    )

    summary_rows: list[dict[str, float | str]] = []
    for airfoil_path in config.discover_airfoils():
        stem = airfoil_path.stem
        m_crit, sweep, baseline_cp, summary = run_mcrit_analysis(airfoil_path)

        sweep_path = paths.mcrit_csv(stem)
        sweep.to_csv(sweep_path, index=False)
        cp_path = paths.baseline_cp_csv(stem)
        baseline_cp.to_csv(cp_path, index=False)

        if m_crit is None:
            print(f"  [WARN] {stem}: baseline Cl={config.CRUISE.cl_normal:.4f} solve at "
                  f"M={solvers.MCRIT_BASELINE_MACH} did not converge -- no M_crit, M_dd/margin still computed")
        elif np.isnan(summary["m_crit"]) and not sweep.empty:
            print(f"  [WARN] {stem}: no Mach in [{solvers.MCRIT_MACH_GRID[0]:.2f}, {solvers.MCRIT_MACH_GRID[-1]:.2f}] "
                  f"reached critical -- M_crit above this grid's range")
        else:
            print(
                f"  {stem}: M_crit={m_crit:.3f}, M_dd={summary['m_dd']:.3f} "
                f"(kappa_A={summary['kappa_a']}, t/c={summary['thickness_to_chord']:.4f}), "
                f"margin(M_dd-M_n)={summary['margin_m_dd_minus_m_n']:+.3f}"
            )
        print(f"    -> {sweep_path}")
        print(f"    -> {cp_path}")

        summary_rows.append({"airfoil": stem, **summary})

    summary_path = paths.MCRIT_SUMMARY_CSV
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"  -> {summary_path}")


if __name__ == "__main__":
    main()
