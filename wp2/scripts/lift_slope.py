# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""Print the lift-curve slope of one airfoil's polar.

Fits Cl(alpha) over an explicit alpha window by closed-form OLS
(b12wp2.xfoil.lift_slope) and prints the slope, the zero-lift angle and the
goodness-of-fit numbers. --scan adds a window-sensitivity table, which is
the honest uncertainty on Cl_alpha here: XFoil output is deterministic, so
the formal standard error only measures how straight the chosen window is,
while the spread between windows measures how much the answer depends on
where the "linear part" is taken to end.

    python -m scripts.lift_slope NACA_25112 --condition cruise --scan
"""
from __future__ import annotations

import argparse
from pathlib import Path

from b12wp2.config import paths, solvers
from b12wp2.xfoil.lift_slope import OLSFit, fit_window


def report(polar_csv: Path, alpha_lo: float, alpha_hi: float) -> OLSFit:
    fit = fit_window(polar_csv, alpha_lo, alpha_hi)
    print(f"file            : {polar_csv}")
    print(f"window requested: [{alpha_lo:+.2f}, {alpha_hi:+.2f}] deg")
    print(f"points used     : n = {fit.n}, alpha in [{fit.alpha_min:+.2f}, {fit.alpha_max:+.2f}] deg")
    print(f"Cl_alpha        : {fit.slope_per_deg:.6f} 1/deg  = {fit.slope_per_rad:.4f} 1/rad")
    print(f"se(Cl_alpha)    : {fit.se_slope_per_deg:.2e} 1/deg = {fit.se_slope_per_rad:.2e} 1/rad")
    print(f"Cl at alpha=0   : {fit.intercept:.6f}")
    print(f"alpha_L0        : {fit.alpha_L0:+.4f} deg")
    print(f"R^2             : {1 - (1 - fit.r2):.8f}   (1 - R^2 = {1 - fit.r2:.2e})")
    print(f"max |residual|  : {fit.max_abs_residual:.2e} in Cl")
    return fit


def sensitivity_scan(polar_csv: Path, windows: list[tuple[float, float]]) -> None:
    print(f"{'window [deg]':>18}  {'n':>3}  {'1/deg':>9}  {'1/rad':>7}  {'1-R^2':>9}  {'max|res|':>9}")
    for lo, hi in windows:
        try:
            f = fit_window(polar_csv, lo, hi)
        except ValueError as exc:
            print(f"{f'[{lo:+.2f},{hi:+.2f}]':>18}  -- {exc}")
            continue
        print(
            f"{f'[{lo:+.2f},{hi:+.2f}]':>18}  {f.n:>3}  {f.slope_per_deg:9.6f}  "
            f"{f.slope_per_rad:7.4f}  {1 - f.r2:9.2e}  {f.max_abs_residual:9.2e}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("airfoil", help="airfoil tag, e.g. NACA_25112")
    p.add_argument("--condition", default="cruise", choices=["cruise", "landing"])
    p.add_argument("--alpha-lo", type=float, default=solvers.LIFT_SLOPE_ALPHA_LO_DEG)
    p.add_argument("--alpha-hi", type=float, default=solvers.LIFT_SLOPE_ALPHA_HI_DEG)
    p.add_argument("--scan", action="store_true", help="also print a window-sensitivity table")
    args = p.parse_args()

    csv = paths.polar_csv(args.airfoil, args.condition)
    report(csv, args.alpha_lo, args.alpha_hi)
    if args.scan:
        print()
        sensitivity_scan(
            csv,
            [
                (-1.0, 1.0),
                (-2.0, 2.0),
                (-3.0, 3.0),
                (-3.25, 4.0),
                (-3.25, 5.0),
                (-3.25, 6.5),
                (0.0, 4.0),
                (0.0, 6.5),
            ],
        )


if __name__ == "__main__":
    main()
