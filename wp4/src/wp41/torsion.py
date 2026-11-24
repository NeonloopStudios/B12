# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""WP4.1 torque diagram of the wing about the shear centre axis.

Model: wp4/README_41.md, section 7. Reader: WP4-5, WP4.1 and Appendix C.5,
Appendix A.3 (equivalent unswept wing, thrust rotated by the sweep).

Axes: body frame, x forward, y right, z down. Torque is M_y about the shear
centre axis, nose-up positive. The shear centre is taken at the centroid of
the wing box (reader, WP4.1; assumption A-6). The diagram value at y is the
resultant torque of all loads outboard of y.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from wp41.contracts import (
    ContractError,
    FloatArray,
    LiftTable,
    PointLoads,
    TorqueTable,
    WeightTable,
)
from wp41.integrate import integral_from_tip

QUARTER_CHORD: Final[float] = 0.25
GRID_RTOL: Final[float] = 1.0e-6
GRID_ATOL_M: Final[float] = 1.0e-9
# The lift script takes the chord from a line fitted to the VSPAERO strips, the
# weight script from the planform formula. They differ by 0.10 % at the tip for
# the current wing (measured). The torque uses the chord of lift.csv.
CHORD_RTOL: Final[float] = 5.0e-3


def _check_same_grid(lift: LiftTable, weight: WeightTable) -> None:
    """Check that lift.csv and weight.csv share stations and chords.

    Raises:
        ContractError: If the stations or the chords differ.
    """
    same = (
        lift.y_m.shape == weight.y_m.shape
        and np.allclose(lift.y_m, weight.y_m, rtol=GRID_RTOL, atol=GRID_ATOL_M)
        and np.allclose(lift.chord_m, weight.chord_m, rtol=CHORD_RTOL, atol=GRID_ATOL_M)
    )
    if not same:
        raise ContractError(
            f"lift.csv ({lift.y_m.size} stations) and weight.csv "
            f"({weight.y_m.size} stations) differ in y_m or chord_m"
        )


def torque_per_span(lift: LiftTable, weight: WeightTable, f_sc: float) -> FloatArray:
    """Torque per unit span about the shear centre, nose-up positive.

    t = m'_c4 + N' (f_sc - 1/4) c - n w'_str (f_sc - f_str) c
        - n w'_fuel (f_sc - f_fuel) c

    Each force term is force times the chordwise arm from the shear centre.
    Lift acts at c/4. An upward force ahead of the shear centre gives a
    nose-up torque. A downward force ahead of it gives a nose-down torque.

    Args:
        lift: Lift table of the case (carries n).
        weight: Weight table of the case, n = 1 values.
        f_sc: Chord fraction of the shear centre [-], in [0, 1].

    Returns:
        t at the stations of the tables [N m/m].

    Raises:
        ContractError: If the two tables do not share their stations.
        ValueError: If f_sc is outside [0, 1].
    """
    if not (0.0 <= f_sc <= 1.0):
        raise ValueError(f"f_sc must be in [0, 1], got {f_sc}")
    _check_same_grid(lift, weight)
    chord = lift.chord_m
    lift_arm = lift.n_prime_npm * (f_sc - QUARTER_CHORD) * chord
    str_arm = lift.n * weight.w_str_npm * (f_sc - weight.f_str) * chord
    fuel_arm = lift.n * weight.w_fuel_npm * (f_sc - weight.f_fuel) * chord
    total = lift.m_prime_c4_nmpm + lift_arm - str_arm - fuel_arm
    return np.asarray(total, dtype=np.float64)


def point_torques(
    points: PointLoads,
    n: float,
    thrust_n: float,
    sweep_c4_deg: float,
    h_thrust_m: float,
) -> FloatArray:
    """Torque of each engine about the shear centre, nose-up positive.

    T_j = - n W d + T cos(sweep) h_thrust

    The weight W acts down at the distance d ahead of the shear centre. The
    thrust is rotated by the quarter-chord sweep (reader, Appendix A.3). Its
    chordwise part acts h_thrust below the shear centre, so it gives a
    nose-up torque (M_y = z F_x for z down, F_x forward).

    Args:
        points: Point loads of the case (one row per engine).
        n: Load factor [-].
        thrust_n: Thrust of one engine [N], >= 0.
        sweep_c4_deg: Quarter-chord sweep [deg].
        h_thrust_m: Height of the thrust line below the shear centre [m].

    Returns:
        One torque per row of points [N m].

    Raises:
        ValueError: If the thrust is negative or not finite.
    """
    if not (thrust_n >= 0.0 and math.isfinite(thrust_n)):
        raise ValueError(f"thrust_n must be >= 0 and finite, got {thrust_n}")
    thrust_part = thrust_n * math.cos(math.radians(sweep_c4_deg)) * h_thrust_m
    weight_part = -n * points.w_n * points.d_ahead_of_sc_m
    return np.asarray(weight_part + thrust_part, dtype=np.float64)


def torque_diagram(
    lift: LiftTable,
    weight: WeightTable,
    points: PointLoads,
    *,
    f_sc: float,
    sweep_c4_deg: float,
    thrust_n: float,
    h_thrust_m: float,
    half_span_m: float,
) -> TorqueTable:
    """Torque diagram of one load case, from the tip to the centreline.

    tau(y) = integral from y to b/2 of t(eta) d(eta) + sum of T_j for y_j >= y.
    Between the stations t is linear. Beyond the end stations it is constant.

    Args:
        lift: Lift table of the case.
        weight: Weight table of the case.
        points: Point loads of the case.
        f_sc: Chord fraction of the shear centre [-].
        sweep_c4_deg: Quarter-chord sweep [deg].
        thrust_n: Thrust of one engine [N].
        h_thrust_m: Height of the thrust line below the shear centre [m].
        half_span_m: b/2 [m].

    Returns:
        The torque table. Each engine station appears twice: first with its
        point torque, then without it.

    Raises:
        ContractError: If the tables do not match each other or the case.
        ValueError: If an input is outside its range.
    """
    if not (half_span_m > 0.0):
        raise ValueError(f"half_span_m must be > 0, got {half_span_m}")
    if not (lift.case == weight.case == points.case):
        raise ContractError(
            f"case names differ: lift {lift.case!r}, weight {weight.case!r}, "
            f"point loads {points.case!r}"
        )
    if not (np.all(points.y_m <= half_span_m)):
        raise ValueError(
            f"a point load lies beyond the tip {half_span_m}: {points.y_m}"
        )

    t_stations = torque_per_span(lift, weight, f_sc)
    t_point = point_torques(points, lift.n, thrust_n, sweep_c4_deg, h_thrust_m)

    knots = np.unique(np.concatenate(([0.0], lift.y_m, points.y_m, [half_span_m])))
    t_knots = np.interp(knots, lift.y_m, t_stations)
    distributed = integral_from_tip(lift.y_m, t_stations, half_span_m, knots)

    y_out: list[float] = []
    t_out: list[float] = []
    tau_out: list[float] = []
    for knot, t_value, base in zip(knots, t_knots, distributed, strict=True):
        with_point = base + float(t_point[points.y_m >= knot].sum())
        without_point = base + float(t_point[points.y_m > knot].sum())
        # Exact match: the knots hold copies of the values in points.y_m.
        at_step = bool(np.any(points.y_m == knot))
        for tau in [with_point, without_point] if at_step else [with_point]:
            y_out.append(float(knot))
            t_out.append(float(t_value))
            tau_out.append(tau)
    return TorqueTable(
        case=lift.case,
        n=lift.n,
        y_m=np.array(y_out, dtype=np.float64),
        t_prime_nmpm=np.array(t_out, dtype=np.float64),
        torque_nm=np.array(tau_out, dtype=np.float64),
    )
