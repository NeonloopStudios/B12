# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of wp41.torsion against closed-form torques.

Every case uses constant or linear loads on the 11.7 m span, so the torque
follows from one integral by hand (README_41, section 7). Axes: x forward,
y right, z down; torque M_y nose-up positive.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from wp41.contracts import (
    ContractError,
    LiftTable,
    PointLoads,
    TorqueTable,
    WeightTable,
)
from wp41.torsion import point_torques, torque_diagram, torque_per_span

L = 11.7
C = 3.1
SWEEP = 30.0
COMMON = {"sweep_c4_deg": SWEEP, "thrust_n": 0.0, "h_thrust_m": 0.8, "half_span_m": L}

Lift = Callable[..., LiftTable]
Weight = Callable[..., WeightTable]
Points = Callable[..., PointLoads]


def _diagram(
    lift: LiftTable, weight: WeightTable, points: PointLoads, f_sc: float, **kw: float
) -> TorqueTable:
    args = {**COMMON, **kw}
    return torque_diagram(lift, weight, points, f_sc=f_sc, **args)


def _tau_at(table: TorqueTable, y: float, row: int = 0) -> float:
    """Torque at station y; row picks the first or the second row of a step."""
    return float(table.torque_nm[np.flatnonzero(table.y_m == y)[row]])


# ============================================================
#  DISTRIBUTED LOADS
# ============================================================


def test_torque_diagram_constant_lift_aft_of_quarter_chord_returns_closed_form(
    make_lift: Lift, make_weight: Weight, make_points: Points, stations: np.ndarray
) -> None:
    n_prime, f_sc = 412.7, 0.43
    table = _diagram(make_lift(n_prime=n_prime), make_weight(), make_points(), f_sc)

    expected = n_prime * (f_sc - 0.25) * C * (L - table.y_m)
    np.testing.assert_allclose(table.torque_nm, expected, rtol=1e-12, atol=1e-9)
    assert table.torque_nm.dtype == np.float64
    assert table.y_m.size == stations.size


def test_torque_diagram_shear_centre_ahead_of_quarter_chord_returns_negative_torque(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    table = _diagram(make_lift(n_prime=412.7), make_weight(), make_points(), 0.19)

    assert np.all(table.torque_nm[:-1] < 0.0)
    assert table.torque_nm[-1] == pytest.approx(0.0, abs=1e-12)


def test_torque_diagram_linear_lift_returns_closed_form(
    make_lift: Lift, make_weight: Weight, make_points: Points, stations: np.ndarray
) -> None:
    a, b, f_sc = 310.0, -17.3, 0.41
    lift = make_lift(n_prime=a + b * stations)
    table = _diagram(lift, make_weight(), make_points(), f_sc)

    y = table.y_m
    expected = (f_sc - 0.25) * C * (a * (L - y) + b * (L**2 - y**2) / 2.0)
    np.testing.assert_allclose(table.torque_nm, expected, rtol=1e-12, atol=1e-9)


def test_torque_diagram_pitching_moment_only_returns_moment_times_length(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    m_prime = -55.5
    table = _diagram(make_lift(m_prime=m_prime), make_weight(), make_points(), 0.43)

    expected = m_prime * (L - table.y_m)
    np.testing.assert_allclose(table.torque_nm, expected, rtol=1e-12, atol=1e-9)


def test_torque_diagram_structure_weight_scales_with_load_factor(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    w_str, f_str, f_sc, n = 120.4, 0.47, 0.43, 2.3
    weight = make_weight(w_str=w_str, f_str=f_str)
    table = _diagram(make_lift(n=n), weight, make_points(), f_sc)

    # Structure weight acts down aft of the centroid (f_str > f_sc): nose-up torque.
    expected = -n * w_str * (f_sc - f_str) * C * (L - table.y_m)
    np.testing.assert_allclose(table.torque_nm, expected, rtol=1e-12, atol=1e-9)
    assert table.torque_nm[0] > 0.0


def test_torque_diagram_fuel_weight_with_negative_load_factor_flips_sign(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    w_fuel, f_fuel, f_sc, n = 88.9, 0.36, 0.43, -1.4
    weight = make_weight(w_fuel=w_fuel, f_fuel=f_fuel)
    table = _diagram(make_lift(n=n), weight, make_points(), f_sc)

    # Fuel weight acts ahead of the centroid (f_fuel < f_sc): nose-down at n > 0.
    expected = -n * w_fuel * (f_sc - f_fuel) * C * (L - table.y_m)
    np.testing.assert_allclose(table.torque_nm, expected, rtol=1e-12, atol=1e-9)
    assert table.torque_nm[0] > 0.0


def test_torque_per_span_sums_the_four_terms(
    make_lift: Lift, make_weight: Weight
) -> None:
    lift = make_lift(n_prime=412.7, m_prime=-55.5, n=2.3)
    weight = make_weight(w_str=120.4, w_fuel=88.9, f_str=0.47, f_fuel=0.36)
    f_sc = 0.43

    t = torque_per_span(lift, weight, f_sc)

    expected = (
        -55.5
        + 412.7 * (f_sc - 0.25) * C
        - 2.3 * 120.4 * (f_sc - 0.47) * C
        - 2.3 * 88.9 * (f_sc - 0.36) * C
    )
    np.testing.assert_allclose(t, expected, rtol=1e-12)


# ============================================================
#  POINT TORQUES AND THE STEP
# ============================================================


def test_point_torques_weight_and_thrust_return_hand_value(make_points: Points) -> None:
    points = make_points(((5.5, 7340.0, 0.62),))
    n, thrust, sweep, h = 2.3, 18_200.0, 30.0, 0.8

    result = point_torques(points, n, thrust, sweep, h)

    expected = -n * 7340.0 * 0.62 + thrust * math.cos(math.radians(sweep)) * h
    assert result.shape == (1,)
    assert result[0] == pytest.approx(expected, rel=1e-12)


def test_torque_diagram_engine_off_station_adds_a_step_of_the_point_torque(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    lift = make_lift(n_prime=412.7, n=2.3)
    points = make_points(((5.5, 7340.0, 0.62),))
    kw = {"thrust_n": 18_200.0}
    with_engine = _diagram(lift, make_weight(), points, 0.43, **kw)
    without = _diagram(lift, make_weight(), make_points(), 0.43)

    t_j = float(point_torques(points, 2.3, 18_200.0, SWEEP, 0.8)[0])
    inboard = _tau_at(with_engine, 5.5, 0)
    outboard = _tau_at(with_engine, 5.5, 1)
    # The step station appears twice, so the engine adds the knot and the step.
    assert np.count_nonzero(with_engine.y_m == 5.5) == 2
    assert inboard - outboard == pytest.approx(t_j, rel=1e-12)
    # Outboard of the engine nothing changes. Inboard of it, everything shifts by T_j.
    assert outboard == pytest.approx(412.7 * (0.43 - 0.25) * C * (L - 5.5), rel=1e-12)
    root_shift = _tau_at(with_engine, 0.0) - _tau_at(without, 0.0)
    assert root_shift == pytest.approx(t_j, rel=1e-12)
    assert _tau_at(with_engine, L) == pytest.approx(0.0, abs=1e-12)


def test_torque_diagram_engine_on_station_gives_one_step_pair(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    points = make_points(((4.1, 7340.0, 0.62),))

    table = _diagram(make_lift(n_prime=412.7), make_weight(), points, 0.43)

    assert np.count_nonzero(table.y_m == 4.1) == 2
    assert table.y_m.size == 9


def test_torque_diagram_engine_with_negative_thrust_raises(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    with pytest.raises(ValueError, match="thrust_n"):
        _diagram(
            make_lift(), make_weight(), make_points(((5.5, 7340.0, 0.62),)), 0.43,
            thrust_n=-1.0,
        )


def test_torque_diagram_engine_beyond_tip_raises(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    with pytest.raises(ValueError, match="beyond the tip"):
        _diagram(make_lift(), make_weight(), make_points(((12.4, 7340.0, 0.62),)), 0.43)


# ============================================================
#  INPUT CHECKS
# ============================================================


@pytest.mark.parametrize("f_sc", [-0.01, 1.01, float("nan")])
def test_torque_diagram_f_sc_outside_unit_interval_raises(
    make_lift: Lift, make_weight: Weight, make_points: Points, f_sc: float
) -> None:
    with pytest.raises(ValueError, match="f_sc"):
        _diagram(make_lift(), make_weight(), make_points(), f_sc)


def test_torque_diagram_different_stations_raises(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    weight = make_weight()
    shifted = WeightTable(
        case=weight.case, y_m=weight.y_m * 0.99, chord_m=weight.chord_m,
        w_str_npm=weight.w_str_npm, w_fuel_npm=weight.w_fuel_npm,
        f_str=weight.f_str, f_fuel=weight.f_fuel,
    )

    with pytest.raises(ContractError, match="differ in y_m or chord_m"):
        _diagram(make_lift(), shifted, make_points(), 0.43)


def test_torque_diagram_different_case_names_raises(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    with pytest.raises(ContractError, match="case names differ"):
        _diagram(make_lift(), make_weight(case="LC-X"), make_points(), 0.43)


def test_torque_diagram_repeated_call_returns_bitwise_equal_arrays(
    make_lift: Lift, make_weight: Weight, make_points: Points
) -> None:
    args = (
        make_lift(n_prime=412.7, m_prime=-55.5),
        make_weight(w_str=120.4),
        make_points(((5.5, 7340.0, 0.62),)),
    )

    first = _diagram(*args, 0.43, thrust_n=18_200.0)
    second = _diagram(*args, 0.43, thrust_n=18_200.0)

    assert first.torque_nm.tobytes() == second.torque_nm.tobytes()
    assert first.y_m.tobytes() == second.y_m.tobytes()
