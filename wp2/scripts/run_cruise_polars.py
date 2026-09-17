# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 Stage 3: cruise polars for every candidate airfoil.

For each airfoil in wp2/airfoils/, runs an angle-of-attack sweep at the
CRUISE flight condition (config.CRUISE: M_n, Re_n, Ncrit) and writes the
full polar to wp2/results/data/<airfoil>_cruise_polar.csv.

This stage only generates and saves the raw polar. Extracting the cruise
operating point (matching Cl(alpha) = Cl_n), stall angle, Cl_max, and
zero-angle Cl from it is Stage 7's job (build_scorecard.py) -- this module
prints a lightweight sanity check (whether Cl_n was actually reached before
the sweep broke down) so a bad run is caught immediately rather than three
stages later.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import config, xfoil_runtime

ALPHA_LOW_DEG = -6.0
ALPHA_HIGH_DEG = 20.0


def run_cruise_polar(airfoil_path: Path) -> pd.DataFrame:
    """Run the cruise alpha sweep for one airfoil and return its polar.

    Uses xfoil_runtime.run_two_leg_polar (warm-started from alpha=0 in
    both directions) rather than a single cold sweep from ALPHA_LOW_DEG --
    see that function's docstring for why a cold start reliably fails at
    this Mach.
    """
    airfoil = xfoil_runtime.load_airfoil_dat(airfoil_path)
    return xfoil_runtime.run_two_leg_polar(
        airfoil,
        mach=config.CRUISE.mach_normal,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
        alpha_low_deg=ALPHA_LOW_DEG,
        alpha_high_deg=ALPHA_HIGH_DEG,
    )


def _sanity_check(airfoil_stem: str, polar: pd.DataFrame) -> None:
    converged = polar[polar["converged"]]
    n_converged = len(converged)
    n_total = len(polar)
    if n_converged == 0:
        print(f"  [WARN] {airfoil_stem}: no converged points at all")
        return

    cl_n = config.CRUISE.cl_normal
    cl_min, cl_max = converged["cl"].min(), converged["cl"].max()
    reachable = cl_min <= cl_n <= cl_max
    flag = "" if reachable else "  [WARN] Cl_n not bracketed by converged range"
    print(
        f"  {airfoil_stem}: {n_converged}/{n_total} converged, "
        f"alpha in [{converged['alpha'].min():.2f}, {converged['alpha'].max():.2f}] deg, "
        f"cl in [{cl_min:.3f}, {cl_max:.3f}] (Cl_n={cl_n:.3f}){flag}"
    )


def main() -> None:
    out_dir = config.RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Cruise condition: M_n={config.CRUISE.mach_normal:.4f}, "
        f"Re_n={config.CRUISE.reynolds_normal:,.0f}, Ncrit={config.CRUISE.ncrit}, "
        f"Cl_n={config.CRUISE.cl_normal:.4f}"
    )
    for airfoil_path in config.discover_airfoils():
        polar = run_cruise_polar(airfoil_path)
        out_path = out_dir / f"{airfoil_path.stem}_cruise_polar.csv"
        polar.to_csv(out_path, index=False)
        _sanity_check(airfoil_path.stem, polar)
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
