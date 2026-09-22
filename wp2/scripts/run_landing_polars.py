# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""WP2 landing polars for every candidate airfoil.

For each airfoil in wp2/airfoils/, runs an angle-of-attack sweep at the
LANDING flight condition (config.LANDING: M_n, Re_n, Ncrit) and writes the
full polar to wp2/results/data/<airfoil>_landing_polar.csv.

Unlike cruise, landing needs no required-Cl (config.LANDING.cl_wing is
intentionally unset -- see config.py): "Cl max landing" in the scorecard is
simply the polar's Cl_max at the landing Re/M, a property of the airfoil at
that condition, not a trim point to locate. This module's sanity check
reports that Cl_max directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from b12wp2 import config
from b12wp2.xfoil import polar_analysis, runtime as xfoil_runtime

# Low-speed, clean-configuration stall angles run noticeably higher than the
# cruise polar's (compressibility at M_n~0.70 cut cruise stall to ~3-5 deg;
# at landing's M~0.2 that effect is negligible), so this sweep goes wider.
ALPHA_LOW_DEG = -8.0
ALPHA_HIGH_DEG = 25.0


def run_landing_polar(airfoil_path: Path) -> pd.DataFrame:
    """Run the landing alpha sweep for one airfoil and return its polar."""
    airfoil = xfoil_runtime.load_airfoil_dat(airfoil_path)
    return xfoil_runtime.run_two_leg_polar(
        airfoil,
        mach=config.LANDING.mach_normal,
        reynolds=config.LANDING.reynolds_normal,
        ncrit=config.LANDING.ncrit,
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

    # NOT converged['cl'].max() -- XFoil can keep numerically converging
    # well past real stall (verified: NACA 25112 here converges cleanly to
    # alpha=40 deg with Cl settling on a non-physical ~0.75 plateau below
    # its real ~1.80 peak). find_cl_max detects the real stall break.
    cl_max, alpha_at_cl_max = polar_analysis.find_cl_max(polar)
    print(
        f"  {airfoil_stem}: {n_converged}/{n_total} converged, "
        f"alpha in [{converged['alpha'].min():.2f}, {converged['alpha'].max():.2f}] deg, "
        f"Cl_max_landing={cl_max:.3f} at alpha={alpha_at_cl_max:.2f} deg"
    )
    if converged["alpha"].max() >= ALPHA_HIGH_DEG - 0.5:
        print(
            f"  [NOTE] {airfoil_stem}: still converging at the top of the swept "
            f"range ({ALPHA_HIGH_DEG} deg) -- expected (see polar_analysis.find_cl_max), "
            "Cl_max_landing above already excludes that non-physical tail"
        )


def main() -> None:
    out_dir = config.RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Landing condition (sea level, V={config.LANDING_SPEED_MS} m/s): "
        f"M_n={config.LANDING.mach_normal:.4f}, "
        f"Re_n={config.LANDING.reynolds_normal:,.0f}, Ncrit={config.LANDING.ncrit}"
    )
    for airfoil_path in config.discover_airfoils():
        polar = run_landing_polar(airfoil_path)
        out_path = out_dir / f"{airfoil_path.stem}_landing_polar.csv"
        polar.to_csv(out_path, index=False)
        _sanity_check(airfoil_path.stem, polar)
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
