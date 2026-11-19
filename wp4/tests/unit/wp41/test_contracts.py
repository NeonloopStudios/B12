# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of wp41.contracts: round trips and every rejection of a bad file."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from wp41.contracts import (
    ContractError,
    LiftTable,
    PointLoads,
    TorqueTable,
    WeightTable,
)

Lift = Callable[..., LiftTable]
Weight = Callable[..., WeightTable]
Points = Callable[..., PointLoads]


def _irregular(stations: np.ndarray, scale: float) -> np.ndarray:
    """Values that are different at every station and not round."""
    return scale * (1.0 + 0.137 * np.arange(stations.size)) / 3.0


# ============================================================
#  ROUND TRIPS
# ============================================================


def test_lift_table_round_trip_returns_bitwise_equal_arrays(
    make_lift: Lift, stations: np.ndarray, tmp_path: Path
) -> None:
    lift = make_lift(
        n_prime=_irregular(stations, 412.7), m_prime=_irregular(stations, -55.5)
    )
    path = tmp_path / "lift.csv"

    lift.to_csv(path)
    back = LiftTable.from_csv(path)

    assert back.n_prime_npm.tobytes() == lift.n_prime_npm.tobytes()
    assert back.m_prime_c4_nmpm.tobytes() == lift.m_prime_c4_nmpm.tobytes()
    assert back.y_m.tobytes() == lift.y_m.tobytes()
    assert (back.case, back.n) == (lift.case, lift.n)
    assert (back.w_n, back.q_pa) == (lift.w_n, lift.q_pa)
    assert (back.cl_total, back.alpha_d_deg) == (lift.cl_total, lift.alpha_d_deg)
    assert (back.mach_case, back.mach_run, back.extrapolated) == (0.71, 0.77, False)


def test_weight_table_round_trip_returns_bitwise_equal_arrays(
    make_weight: Weight, tmp_path: Path
) -> None:
    weight = make_weight(w_str=120.4, w_fuel=88.9, f_str=0.4713, f_fuel=0.3611)
    path = tmp_path / "weight.csv"

    weight.to_csv(path)
    back = WeightTable.from_csv(path)

    assert back.w_str_npm.tobytes() == weight.w_str_npm.tobytes()
    assert back.f_fuel.tobytes() == weight.f_fuel.tobytes()
    assert back.case == weight.case


def test_point_loads_round_trip_keeps_names_and_values(
    make_points: Points, tmp_path: Path
) -> None:
    points = make_points(((4.4, 7340.3, 0.62), (6.1, 912.7, -0.31)))
    path = tmp_path / "point_loads.csv"

    points.to_csv(path)
    back = PointLoads.from_csv(path)

    assert back.names == ("engine_0", "engine_1")
    assert back.y_m.tobytes() == points.y_m.tobytes()
    assert back.d_ahead_of_sc_m.tobytes() == points.d_ahead_of_sc_m.tobytes()


def test_point_loads_round_trip_with_no_row_returns_empty_arrays(
    make_points: Points, tmp_path: Path
) -> None:
    path = tmp_path / "point_loads.csv"
    make_points().to_csv(path)

    back = PointLoads.from_csv(path)

    assert back.names == ()
    assert back.y_m.shape == (0,)


def test_torque_table_round_trip_keeps_a_repeated_station(tmp_path: Path) -> None:
    table = TorqueTable(
        case="LC-T", n=-1.4,
        y_m=np.array([0.0, 4.4, 4.4, 11.7]),
        t_prime_nmpm=np.array([3.1, 2.7, 2.7, 1.3]),
        torque_nm=np.array([910.3, 400.1, 350.7, 0.0]),
    )
    path = tmp_path / "torque.csv"

    table.to_csv(path)
    back = TorqueTable.from_csv(path)

    assert back.y_m.tobytes() == table.y_m.tobytes()
    assert back.torque_nm.tobytes() == table.torque_nm.tobytes()
    assert back.n == -1.4


# ============================================================
#  REJECTIONS
# ============================================================


def test_from_csv_wrong_schema_raises(make_weight: Weight, tmp_path: Path) -> None:
    path = tmp_path / "weight.csv"
    make_weight().to_csv(path)

    with pytest.raises(ContractError, match="schema"):
        LiftTable.from_csv(path)


def test_from_csv_wrong_column_name_raises(make_weight: Weight, tmp_path: Path) -> None:
    path = tmp_path / "weight.csv"
    make_weight().to_csv(path)
    path.write_text(path.read_text().replace("w_fuel_npm", "w_fuel"), encoding="utf-8")

    with pytest.raises(ContractError, match="columns"):
        WeightTable.from_csv(path)


def test_from_csv_missing_comment_line_raises(make_lift: Lift, tmp_path: Path) -> None:
    path = tmp_path / "lift.csv"
    make_lift().to_csv(path)
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("# n:")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="missing comment line '# n:'"):
        LiftTable.from_csv(path)


def test_from_csv_bad_extrapolated_flag_raises(make_lift: Lift, tmp_path: Path) -> None:
    path = tmp_path / "lift.csv"
    make_lift().to_csv(path)
    path.write_text(path.read_text().replace("extrapolated: false", "extrapolated: 0"))

    with pytest.raises(ContractError, match="true or false"):
        LiftTable.from_csv(path)


def test_from_csv_short_row_raises(make_weight: Weight, tmp_path: Path) -> None:
    path = tmp_path / "weight.csv"
    make_weight().to_csv(path)
    path.write_text(path.read_text() + "1.0,2.0\n", encoding="utf-8")

    with pytest.raises(ContractError, match="fields"):
        WeightTable.from_csv(path)


def test_from_csv_text_in_number_column_raises(
    make_weight: Weight, tmp_path: Path
) -> None:
    path = tmp_path / "weight.csv"
    make_weight().to_csv(path)
    lines = path.read_text().splitlines()
    lines[-1] = "abc," + lines[-1].split(",", 1)[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ContractError, match="non-number"):
        WeightTable.from_csv(path)


def test_lift_table_unordered_stations_raises(make_lift: Lift) -> None:
    lift = make_lift()
    y = lift.y_m.copy()
    y[3], y[4] = y[4], y[3]

    with pytest.raises(ContractError, match="not increasing"):
        LiftTable(
            case="LC-T", n=2.3, w_n=1.0, q_pa=1.0, cl_total=0.7, alpha_d_deg=1.0,
            mach_case=0.7, mach_run=0.77, extrapolated=False, y_m=y,
            chord_m=lift.chord_m, cl_d=lift.cl_d, cm_c4_d=lift.cm_c4_d,
            n_prime_npm=lift.n_prime_npm, m_prime_c4_nmpm=lift.m_prime_c4_nmpm,
        )


def test_lift_table_nan_value_raises(make_lift: Lift) -> None:
    lift = make_lift()
    bad = lift.n_prime_npm.copy()
    bad[2] = np.nan

    with pytest.raises(ContractError, match="not finite"):
        LiftTable(
            case="LC-T", n=2.3, w_n=1.0, q_pa=1.0, cl_total=0.7, alpha_d_deg=1.0,
            mach_case=0.7, mach_run=0.77, extrapolated=False, y_m=lift.y_m,
            chord_m=lift.chord_m, cl_d=lift.cl_d, cm_c4_d=lift.cm_c4_d,
            n_prime_npm=bad, m_prime_c4_nmpm=lift.m_prime_c4_nmpm,
        )


def test_weight_table_chord_fraction_above_one_raises(make_weight: Weight) -> None:
    weight = make_weight()

    with pytest.raises(ContractError, match="f_str"):
        WeightTable(
            case="LC-T", y_m=weight.y_m, chord_m=weight.chord_m,
            w_str_npm=weight.w_str_npm, w_fuel_npm=weight.w_fuel_npm,
            f_str=weight.f_str + 0.6, f_fuel=weight.f_fuel,
        )


def test_point_loads_shape_mismatch_raises() -> None:
    with pytest.raises(ContractError, match="shape"):
        PointLoads(
            case="LC-T", names=("a", "b"), y_m=np.array([1.0, 2.0]),
            w_n=np.array([3.0]), d_ahead_of_sc_m=np.array([0.1, 0.2]),
        )
