# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-24
"""WP4.1 wing weights on the global stations: fuel and engine.

Model: wp4/README_41.md, section 6. All values are for n = 1, right half-wing.
The wing structure weight is neglected (assumption A-7), so its column in
weight.csv is zero. The fuel formula is in wp41.weight_distrubution (c(y)^2
from the fuselage side to the tip). This module evaluates it on the global
stations and scales it so the station integral is exactly W_fuel / 2.
"""

from __future__ import annotations

import numpy as np
from numpy import trapezoid

from wp41.contracts import FloatArray, PointLoads, WeightTable
from wp41.weight_distrubution import weight_distribution


def fuel_per_span(y: FloatArray, w_fuel_n: float, tolerance: float) -> FloatArray:
    """Fuel weight per unit span on the stations, scaled to W_fuel / 2.

    Args:
        y: Global stations [m].
        w_fuel_n: Fuel weight of both halves [N].
        tolerance: Largest allowed |scale - 1|, a fraction.

    Returns:
        w'_fuel at the stations [N/m], with 2 * integral(w'_fuel dy) = w_fuel_n.

    Raises:
        ValueError: If the weight is negative, or the scale factor differs from 1
            by more than the tolerance.
    """
    if not (w_fuel_n >= 0.0 and np.isfinite(w_fuel_n)):
        raise ValueError(f"w_fuel_n must be >= 0 and finite, got {w_fuel_n}")
    if not w_fuel_n > 0.0:
        return np.zeros(y.shape, dtype=np.float64)
    points, weight = weight_distribution(w_fuel_n)
    on_stations = np.interp(y, points, weight)
    scale = 0.5 * w_fuel_n / float(trapezoid(on_stations, y))
    if not (abs(scale - 1.0) <= tolerance):
        raise ValueError(
            f"fuel table integral differs from W_fuel / 2 by a factor {scale}, "
            f"more than the tolerance {tolerance}"
        )
    return np.asarray(on_stations * scale, dtype=np.float64)


def engine_ahead_of_centroid(
    f_sc: float,
    chord_at_engine_m: float,
    inlet_ahead_of_le_m: float,
    nacelle_length_m: float,
    cg_fraction: float,
) -> float:
    """Distance of the engine centre of gravity ahead of the shear centre [m].

    d = f_sc c + (inlet ahead of the leading edge) - cg_fraction * nacelle length.
    The shear centre is f_sc c behind the leading edge. The centre of gravity is
    cg_fraction of the nacelle length behind the inlet.

    Args:
        f_sc: Chord fraction of the shear centre [-].
        chord_at_engine_m: Chord at the engine station [m].
        inlet_ahead_of_le_m: Inlet plane ahead of the leading edge [m].
        nacelle_length_m: Nacelle length [m].
        cg_fraction: Centre of gravity behind the inlet, fraction of the length [-].

    Returns:
        Forward-positive distance [m].
    """
    return (
        f_sc * chord_at_engine_m + inlet_ahead_of_le_m - cg_fraction * nacelle_length_m
    )


def weight_tables(
    case: str,
    y: FloatArray,
    chord: FloatArray,
    w_fuel_n: float,
    f_fuel: float,
    engine_y_m: float,
    engine_weight_n: float,
    engine_ahead_m: float,
    tolerance: float,
) -> tuple[WeightTable, PointLoads]:
    """Build weight.csv and point_loads.csv of one load case.

    The structure column is zero. Its chord fraction is set equal to the fuel
    one, so it has no effect.

    Args:
        case: Load case name.
        y: Global stations [m].
        chord: Chord at the stations [m].
        w_fuel_n: Fuel weight of the mass case, both halves [N].
        f_fuel: Chord fraction of the fuel centre of gravity [-].
        engine_y_m: Span station of the engine [m].
        engine_weight_n: Weight of the engines of one half-wing [N].
        engine_ahead_m: Engine centre of gravity ahead of the shear centre [m].
        tolerance: Largest allowed fuel scale error, a fraction.

    Returns:
        The weight table and the point loads.
    """
    ones = np.ones(y.shape, dtype=np.float64)
    weight = WeightTable(
        case=case,
        y_m=y,
        chord_m=chord,
        w_str_npm=np.zeros(y.shape, dtype=np.float64),
        w_fuel_npm=fuel_per_span(y, w_fuel_n, tolerance),
        f_str=f_fuel * ones,
        f_fuel=f_fuel * ones,
    )
    points = PointLoads(
        case=case,
        names=("engine",),
        y_m=np.array([engine_y_m], dtype=np.float64),
        w_n=np.array([engine_weight_n], dtype=np.float64),
        d_ahead_of_sc_m=np.array([engine_ahead_m], dtype=np.float64),
    )
    return weight, points
