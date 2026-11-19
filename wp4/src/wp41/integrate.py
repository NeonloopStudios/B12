# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tip-to-station integral of a load that is piecewise linear in y.

The WP4.1 diagrams are integrals from the tip to the station y (reader,
Appendix C). The load is known at fixed stations. It is linear between them
and constant beyond the first and the last station. The trapezoidal rule is
exact for such a load.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy import integrate

FloatArray: TypeAlias = npt.NDArray[np.float64]


def integral_from_tip(
    y: FloatArray, values: FloatArray, half_span_m: float, knots: FloatArray
) -> FloatArray:
    """Integral of the load from each knot to the tip.

    Args:
        y: Stations of the load [m], strictly increasing.
        values: Load at the stations [load unit].
        half_span_m: Station of the tip, b/2 [m].
        knots: Stations where the integral is wanted [m]. Must lie in
            [0, half_span_m].

    Returns:
        Integral from each knot to half_span_m [load unit * m], same shape as knots.

    Raises:
        ValueError: If the inputs have the wrong shape or order, or a knot lies
            outside [0, half_span_m].
    """
    if y.ndim != 1 or y.shape != values.shape or y.size < 2:
        raise ValueError(
            f"y {y.shape} and values {values.shape} need the same 1-D shape"
        )
    if not (np.all(np.diff(y) > 0.0)):
        raise ValueError("y must be strictly increasing")
    if not (y[0] >= 0.0 and y[-1] <= half_span_m):
        raise ValueError(f"stations [{y[0]}, {y[-1]}] must lie in [0, {half_span_m}]")
    if not (np.all(knots >= 0.0) and np.all(knots <= half_span_m)):
        raise ValueError(f"knots must lie in [0, {half_span_m}]")

    # Knots of the exact trapezoid: the stations, both ends, and the wanted knots.
    # Inserting a point into a piecewise linear function does not change it.
    grid = np.unique(np.concatenate(([0.0], y, knots, [half_span_m])))
    load = np.interp(grid, y, values)
    cumulative = integrate.cumulative_trapezoid(load, grid, initial=0.0)
    tail = cumulative[-1] - cumulative
    return np.asarray(np.interp(knots, grid, tail), dtype=np.float64)
