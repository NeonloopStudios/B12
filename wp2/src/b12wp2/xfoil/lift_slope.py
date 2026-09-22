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

import math
from pathlib import Path

import pandas as pd


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
