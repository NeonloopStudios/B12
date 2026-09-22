# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-22
"""Lift-curve slope Cl_alpha from an XFoil polar, by closed-form OLS.

The fit is the textbook simple-linear-regression normal-equation solution,
written out explicitly rather than handed to numpy.polyfit/lstsq, so every
number in the WP2 report traces back to an equation on this page:

    model            Cl(alpha) = a + b * alpha          (alpha in degrees)
    Sxx  = sum (x_i - xbar)^2
    Sxy  = sum (x_i - xbar)(y_i - ybar)
    b    = Sxy / Sxx                                     (slope, 1/deg)
    a    = ybar - b * xbar                               (intercept, Cl at alpha=0)
    alpha_L0 = -a / b                                    (zero-lift angle, deg)

Uncertainty (only meaningful as a *linearity* measure here -- XFoil output is
deterministic, not noisy, so these are goodness-of-fit numbers, not experimental
error bars):

    s^2    = SSE / (n - 2),  SSE = sum (y_i - a - b x_i)^2
    se(b)  = sqrt(s^2 / Sxx)
    R^2    = 1 - SSE / Syy

Only rows with converged=True and a finite Cl are used; the window is an
explicit alpha range, because the "lift-curve slope" is only defined on the
linear part of the curve and XFoil's polar runs well into the nonlinear
pre-stall region.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    from .config import RESULTS_DIR
except ImportError:  # running as a plain script, not a package module
    from config import RESULTS_DIR


class OLSFit:
    """Result of a closed-form straight-line least-squares fit."""

    def __init__(self, x: list[float], y: list[float]) -> None:
        n = len(x)
        if n < 3:
            raise ValueError(f"need at least 3 points for an OLS fit, got {n}")
        xbar = sum(x) / n
        ybar = sum(y) / n
        sxx = sum((xi - xbar) ** 2 for xi in x)
        sxy = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
        syy = sum((yi - ybar) ** 2 for yi in y)
        if sxx == 0.0:
            raise ValueError("all alpha values identical; slope undefined")

        self.n = n
        self.slope_per_deg = sxy / sxx
        self.intercept = ybar - self.slope_per_deg * xbar
        residuals = [yi - (self.intercept + self.slope_per_deg * xi) for xi, yi in zip(x, y)]
        sse = sum(r * r for r in residuals)
        self.sse = sse
        self.r2 = 1.0 - sse / syy if syy > 0 else float("nan")
        self.s2 = sse / (n - 2)
        self.se_slope_per_deg = math.sqrt(self.s2 / sxx)
        self.max_abs_residual = max(abs(r) for r in residuals)
        self.alpha_min = min(x)
        self.alpha_max = max(x)

    @property
    def slope_per_rad(self) -> float:
        return self.slope_per_deg * 180.0 / math.pi

    @property
    def se_slope_per_rad(self) -> float:
        return self.se_slope_per_deg * 180.0 / math.pi

    @property
    def alpha_L0(self) -> float:
        """Zero-lift angle of attack [deg], from the same fit."""
        return -self.intercept / self.slope_per_deg


def load_linear_points(
    polar_csv: Path, alpha_lo: float, alpha_hi: float
) -> tuple[list[float], list[float]]:
    df = pd.read_csv(polar_csv)
    mask = (
        df["converged"].astype(str).str.lower().eq("true")
        & df["cl"].notna()
        & df["alpha"].between(alpha_lo, alpha_hi)
    )
    sub = df.loc[mask, ["alpha", "cl"]]
    return sub["alpha"].tolist(), sub["cl"].tolist()


def fit_window(polar_csv: Path, alpha_lo: float, alpha_hi: float) -> OLSFit:
    x, y = load_linear_points(polar_csv, alpha_lo, alpha_hi)
    return OLSFit(x, y)


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
    p.add_argument("--alpha-lo", type=float, default=-3.0)
    p.add_argument("--alpha-hi", type=float, default=3.0)
    p.add_argument("--scan", action="store_true", help="also print a window-sensitivity table")
    args = p.parse_args()

    csv = RESULTS_DIR / "data" / f"{args.airfoil}_{args.condition}_polar.csv"
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
