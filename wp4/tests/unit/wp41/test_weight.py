# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-24
"""Tests of wp41.weight: weight integrals, fuel shape, engine offset, tables.

The stations and the chord are those of the real wing (WING). The weights are
test data. Expected values follow from the model: 2 * integral = W, a c^2
shape, zero fuel inside the fuselage.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import trapezoid

from configs.aircraft import WING
from wp41 import aero_loads
from wp41.stations import station_y
from wp41.contracts import PointLoads, WeightTable
from wp41.weight import (
    engine_ahead_of_centroid,
    fuel_per_span,
    structure_per_span,
    weight_tables,
)

PF = aero_loads.planform(WING)
Y = station_y(PF.half_span_m)
CHORD = PF.chord_at(Y)
Y_FUSELAGE = WING.fuselage_diameter_m / 2.0
TOLERANCE = 0.01


def test_structure_per_span_integral_returns_half_the_weight() -> None:
    w = structure_per_span(Y, CHORD, 30_417.3)

    assert w.shape == Y.shape
    assert w.dtype == np.float64
    assert 2.0 * trapezoid(w, Y) == pytest.approx(30_417.3, rel=1e-12)


def test_structure_per_span_is_proportional_to_chord_squared() -> None:
    w = structure_per_span(Y, CHORD, 30_417.3)

    i, j = 23, 87
    assert w[i] / w[j] == pytest.approx((CHORD[i] / CHORD[j]) ** 2, rel=1e-12)


def test_structure_per_span_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="w_structure_n"):
        structure_per_span(Y, CHORD, -1.0)


def test_fuel_per_span_integral_returns_half_the_weight() -> None:
    w = fuel_per_span(Y, 55_803.7, TOLERANCE)

    assert w.dtype == np.float64
    assert 2.0 * trapezoid(w, Y) == pytest.approx(55_803.7, rel=1e-12)


def test_fuel_per_span_is_zero_inside_the_fuselage_and_positive_outside() -> None:
    w = fuel_per_span(Y, 55_803.7, TOLERANCE)

    inside = Y < Y_FUSELAGE - 0.2
    outside = Y > Y_FUSELAGE + 0.2
    assert inside.any() and outside.any()
    assert np.all(w[inside] == 0.0)
    assert np.all(w[outside] > 0.0)


def test_fuel_per_span_outside_the_fuselage_follows_chord_squared() -> None:
    w = fuel_per_span(Y, 55_803.7, TOLERANCE)

    i, j = 40, 101  # both stations well outside the fuselage step
    assert Y[i] > Y_FUSELAGE + 0.2
    assert w[i] / w[j] == pytest.approx((CHORD[i] / CHORD[j]) ** 2, rel=1e-3)


def test_fuel_per_span_no_fuel_returns_zeros() -> None:
    w = fuel_per_span(Y, 0.0, TOLERANCE)

    assert w.shape == Y.shape
    assert not w.any()


def test_fuel_per_span_scale_error_above_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="differs from W_fuel / 2"):
        fuel_per_span(Y, 55_803.7, 1.0e-12)


def test_fuel_per_span_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="w_fuel_n"):
        fuel_per_span(Y, -5.0, TOLERANCE)


def test_engine_ahead_of_centroid_returns_hand_value() -> None:
    # 0.41 * 3.1 + 2.5 - 0.35 * 4.0 = 1.271 + 2.5 - 1.4
    d = engine_ahead_of_centroid(0.41, 3.1, 2.5, 4.0, 0.35)

    assert d == pytest.approx(2.371, rel=1e-12)


def _tables() -> tuple[WeightTable, PointLoads]:
    return weight_tables(
        case="LC-T", y=Y, chord=CHORD, w_structure_n=30_417.3, w_fuel_n=55_803.7,
        f_str=0.43, f_fuel=0.47, engine_y_m=4.4, engine_weight_n=18_960.1,
        engine_ahead_m=2.37, tolerance=TOLERANCE,
    )


def test_weight_tables_returns_consistent_tables() -> None:
    weight, points = _tables()

    assert weight.case == points.case == "LC-T"
    assert weight.y_m.tobytes() == Y.tobytes()
    assert np.all(weight.f_str == 0.43) and np.all(weight.f_fuel == 0.47)
    assert points.names == ("engine",)
    assert points.w_n[0] == 18_960.1 and points.d_ahead_of_sc_m[0] == 2.37
    assert points.y_m[0] == 4.4


def test_weight_tables_repeated_call_returns_bitwise_equal_arrays() -> None:
    first, _ = _tables()
    second, _ = _tables()

    assert first.w_fuel_npm.tobytes() == second.w_fuel_npm.tobytes()
    assert first.w_str_npm.tobytes() == second.w_str_npm.tobytes()
