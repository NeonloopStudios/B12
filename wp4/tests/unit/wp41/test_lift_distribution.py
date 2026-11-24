# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""Tests of wp41.lift_distribution on a synthetic VSPAERO database.

The database is written with wp41.aero_loads.write_distribution, so the
tests also check the file contract between the two modules. Its cl(y) is an
elliptic-type loading and its cm_c4(y) a linear one. Both are linear in
alpha and scale with the Prandtl-Glauert factor 1/sqrt(1 - M^2). Its CL is
the strip sum 2 sum(cl dA) / S, as in VSPAERO. Equilibrium checked:
lift = n W, 2 integral(N' dy) = n W.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import trapezoid

from configs.aircraft import MASSES, WING
from configs.wp41 import settings
from configs.wp41.load_cases import LoadCase
from wp41 import aero_loads as al
from wp41 import lift_distribution as ld
from wp41.atmosphere import G0, isa
from wp41.contracts import LiftTable

MACHS = (0.2, 0.4, 0.6, 0.8)
ALPHAS = (0.0, 10.0)
N_STRIP = 100
A0_SEA_LEVEL = isa(0.0)[1]
FULL_GRID = [(m, a) for m in MACHS for a in ALPHAS]
INNER = 0.95  # compare station values inboard of 95 % span (PCHIP near the tip)


def _cl(y: np.ndarray, mach: float, alpha: float, half_span: float) -> np.ndarray:
    """Synthetic strip cl: elliptic in y, linear in alpha, Prandtl-Glauert in Mach."""
    return 1.2 * (0.3 + 0.08 * alpha) / math.sqrt(1.0 - mach**2) * np.sqrt(
        1.0 - (y / half_span) ** 2
    )


def _cm(y: np.ndarray, mach: float, alpha: float, half_span: float) -> np.ndarray:
    """Synthetic strip cm_c4: linear in y and in alpha, Prandtl-Glauert in Mach."""
    return (-0.13 + 0.004 * alpha) * (1.0 - 0.3 * y / half_span) / math.sqrt(1.0 - mach**2)


def _write_database(
    folder: Path, cases: list[tuple[float, float]], moment_cases: list[tuple[float, float]]
) -> tuple[Path, Path]:
    """Write synthetic lift and moment distribution files for the design wing."""
    pf = al.planform(WING)
    dy = pf.half_span_m / N_STRIP
    y = (np.arange(N_STRIP, dtype=np.float64) + 0.5) * dy
    chord = pf.chord_at(y)
    paths = []
    for name, grid, func, key in (
        ("lift.txt", cases, _cl, settings.KEY_CL),
        ("moment.txt", moment_cases, _cm, settings.KEY_CMY),
    ):
        rows = []
        for mach, alpha in grid:
            value = func(y, mach, alpha, pf.half_span_m)
            for i in range(N_STRIP):
                rows.append(
                    [mach, alpha, y[i], chord[i], chord[i] * dy, value[i], 0.0, value[i], 0.0]
                )
        table = al.span_table(np.array(rows, dtype=np.float64))
        totals = 2.0 * (table.cl * table.area[:, None]).sum(axis=0) / WING.area_m2
        path = folder / name
        al.write_distribution(path, "synthetic", "c", table, table.cl, totals, key, WING, pf)
        paths.append(path)
    return paths[0], paths[1]


@pytest.fixture(scope="session")
def database(tmp_path_factory: pytest.TempPathFactory) -> ld.AeroDatabase:
    """Synthetic database on the full (Mach, alpha) grid."""
    lift_path, moment_path = _write_database(
        tmp_path_factory.mktemp("db"), FULL_GRID, FULL_GRID
    )
    return ld.read_aero_database(lift_path, moment_path)


def _weight_for_cl(database: ld.AeroDatabase, cl_wing: float, v_eas: float, n: float) -> float:
    """Aircraft weight that needs the wing CL cl_wing at v_eas and n."""
    return cl_wing * 0.5 * ld.RHO0 * v_eas**2 * database.s_ref_m2 / n


# ============================================================
#  DATABASE
# ============================================================


def test_read_aero_database_full_grid_returns_sorted_arrays(
    database: ld.AeroDatabase,
) -> None:
    pf = al.planform(WING)

    np.testing.assert_allclose(database.mach, MACHS)
    np.testing.assert_allclose(database.alpha_deg, ALPHAS)
    assert database.cl.shape == database.cm_c4.shape == (N_STRIP, len(MACHS), 2)
    assert database.cl_wing.shape == (len(MACHS), 2)
    assert database.cl.dtype == database.cm_c4.dtype == np.float64
    assert database.s_ref_m2 == pytest.approx(WING.area_m2)
    np.testing.assert_allclose(
        database.cm_c4[:, 2, 1], _cm(database.y, 0.6, 10.0, pf.half_span_m), atol=1e-6
    )


def test_read_aero_database_missing_alpha_case_raises(tmp_path: Path) -> None:
    lift_path, moment_path = _write_database(tmp_path, FULL_GRID[:-1], FULL_GRID)

    with pytest.raises(ValueError, match="2 angles of attack at every Mach"):
        ld.read_aero_database(lift_path, moment_path)


def test_read_aero_database_files_of_different_runs_raise(tmp_path: Path) -> None:
    other = [(m + 0.01, a) for m, a in FULL_GRID]
    lift_path, moment_path = _write_database(tmp_path, FULL_GRID, other)

    with pytest.raises(ValueError, match="same VSPAERO run"):
        ld.read_aero_database(lift_path, moment_path)


def test_read_aero_database_missing_header_key_raises(tmp_path: Path) -> None:
    lift_path, moment_path = _write_database(tmp_path, FULL_GRID, FULL_GRID)
    lines = [
        line for line in lift_path.read_text().splitlines()
        if not line.startswith(f"# {settings.KEY_S_REF}:")
    ]
    lift_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=settings.KEY_S_REF):
        ld.read_aero_database(lift_path, moment_path)


def test_read_aero_database_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ld.read_aero_database(tmp_path / "none.txt", tmp_path / "none2.txt")


# ============================================================
#  FLIGHT STATE AND WEIGHTS
# ============================================================


def test_flight_state_sea_level_tas_equals_eas() -> None:
    state = ld.flight_state(150.0, 0.0)

    assert state.v_tas == pytest.approx(150.0)
    assert state.mach == pytest.approx(150.0 / A0_SEA_LEVEL)
    assert state.dynamic_pressure == pytest.approx(0.5 * 1.225 * 150.0**2, rel=1e-4)


def test_flight_state_altitude_keeps_q_and_raises_tas() -> None:
    _, a, rho = isa(8_000.0)

    state = ld.flight_state(127.0, 8_000.0)

    assert state.v_tas == pytest.approx(127.0 * math.sqrt(ld.RHO0 / rho))
    assert state.mach == pytest.approx(state.v_tas / a)
    assert state.dynamic_pressure == pytest.approx(0.5 * rho * state.v_tas**2)


@pytest.mark.parametrize("v_eas", [0.0, -10.0, math.nan])
def test_flight_state_speed_not_positive_raises(v_eas: float) -> None:
    with pytest.raises(ValueError, match="v_eas"):
        ld.flight_state(v_eas, 0.0)


@pytest.mark.parametrize(
    ("mass_case", "expected_kg"),
    [
        ("oew", (15_873.0, 0.0, 0.0)),
        ("no_fuel", (15_873.0, 9_302.0, 0.0)),
        ("mtow", (15_873.0, 9_302.0, 30_865.0 - 15_873.0 - 9_302.0)),
    ],
)
def test_case_weights_mass_cases_return_oem_payload_fuel(
    mass_case: str, expected_kg: tuple[float, float, float]
) -> None:
    weights = ld.case_weights(mass_case, MASSES)  # type: ignore[arg-type]

    assert weights == pytest.approx(tuple(G0 * m for m in expected_kg))


def test_case_weights_unknown_case_raises() -> None:
    with pytest.raises(ValueError, match="unknown mass case"):
        ld.case_weights("full", MASSES)  # type: ignore[arg-type]


# ============================================================
#  LIFT DISTRIBUTION
# ============================================================


@pytest.mark.parametrize("j_alpha", [0, 1])
def test_lift_distribution_s_at_database_angle_returns_that_table(
    database: ld.AeroDatabase, j_alpha: int
) -> None:
    """s = 0 gives the alpha = 0 deg table, s = 1 the alpha = 10 deg table."""
    v_eas = 0.4 * A0_SEA_LEVEL  # M = 0.4 at sea level
    weight = _weight_for_cl(database, database.cl_wing[1, j_alpha], v_eas, 2.5)
    pf = al.planform(WING)

    lift = ld.lift_distribution(
        database, w_oem=0.6 * weight, w_payload=0.25 * weight, v_eas=v_eas,
        altitude_m=0.0, load_factor=2.5, w_fuel=0.15 * weight,
    )

    inner = lift.y <= INNER * pf.half_span_m
    alpha = ALPHAS[j_alpha]
    assert lift.s == pytest.approx(float(j_alpha), abs=1e-12)
    assert lift.alpha_deg == pytest.approx(alpha, abs=1e-9)
    assert not lift.extrapolated
    np.testing.assert_allclose(
        lift.cl[inner], _cl(lift.y[inner], 0.4, alpha, pf.half_span_m), rtol=2e-3
    )
    np.testing.assert_allclose(
        lift.cm_c4[inner], _cm(lift.y[inner], 0.4, alpha, pf.half_span_m), rtol=2e-3
    )


def test_lift_distribution_s_between_angles_follows_linear_closed_form(
    database: ld.AeroDatabase,
) -> None:
    v_eas = 0.6 * A0_SEA_LEVEL  # M = 0.6 at sea level
    s = 0.37
    cl_wing = database.cl_wing[2, 0] + s * (database.cl_wing[2, 1] - database.cl_wing[2, 0])
    weight = _weight_for_cl(database, cl_wing, v_eas, 1.8)
    pf = al.planform(WING)

    lift = ld.lift_distribution(
        database, w_oem=0.7 * weight, w_payload=0.3 * weight, v_eas=v_eas,
        altitude_m=0.0, load_factor=1.8,
    )

    inner = lift.y <= INNER * pf.half_span_m
    y = lift.y[inner]
    cl0, cl1 = _cl(y, 0.6, 0.0, pf.half_span_m), _cl(y, 0.6, 10.0, pf.half_span_m)
    cm0, cm1 = _cm(y, 0.6, 0.0, pf.half_span_m), _cm(y, 0.6, 10.0, pf.half_span_m)
    assert lift.s == pytest.approx(s)
    assert lift.alpha_deg == pytest.approx(10.0 * s)
    np.testing.assert_allclose(lift.cl[inner], cl0 + s * (cl1 - cl0), rtol=2e-3)
    np.testing.assert_allclose(lift.cm_c4[inner], cm0 + s * (cm1 - cm0), rtol=2e-3)
    np.testing.assert_allclose(
        lift.moment_per_span,
        lift.state.dynamic_pressure * lift.cm_c4 * lift.chord**2,
        rtol=1e-9, atol=1e-6,
    )


def test_lift_distribution_negative_load_factor_is_extrapolated(
    database: ld.AeroDatabase,
) -> None:
    lift = ld.lift_distribution(
        database, w_oem=1.76e5, w_payload=7.06e4, v_eas=150.0, altitude_m=0.0,
        load_factor=-1.4,
    )

    total = 2.0 * float(trapezoid(lift.lift_per_span, lift.y))
    assert lift.s < 0.0
    assert lift.extrapolated
    assert total == pytest.approx(-1.4 * lift.weight, rel=settings.LIFT_CHECK_TOLERANCE)


def test_lift_distribution_general_case_lift_equals_n_w(database: ld.AeroDatabase) -> None:
    lift = ld.lift_distribution(
        database, w_oem=1.76e5, w_payload=7.06e4, v_eas=140.0, altitude_m=3_000.0,
        load_factor=2.5, w_fuel=7.64e4,
    )

    total = 2.0 * float(trapezoid(lift.lift_per_span, lift.y))
    assert lift.weight == pytest.approx(1.76e5 + 7.06e4 + 7.64e4)
    assert total == pytest.approx(2.5 * lift.weight, rel=settings.LIFT_CHECK_TOLERANCE)
    assert lift.y.shape == lift.lift_per_span.shape == (settings.N_SPAN_STATIONS,)
    assert lift.lift_per_span.dtype == lift.moment_per_span.dtype == np.float64
    assert lift.lift_per_span[-1] == pytest.approx(0.0, abs=1e-9)
    assert lift.moment_per_span[-1] == pytest.approx(0.0, abs=1e-9)
    assert lift.y[-1] == pytest.approx(database.half_span_m)
    np.testing.assert_allclose(lift.inertial_per_span, 0.0)


@given(
    weight=st.floats(1.0e5, 4.0e5),
    v_eas=st.floats(70.0, 200.0),
    load_factor=st.floats(-1.0, 3.8).filter(lambda n: abs(n) > 0.05),
)
def test_lift_distribution_any_case_matches_equilibrium_and_grid(
    database: ld.AeroDatabase, weight: float, v_eas: float, load_factor: float
) -> None:
    """2 integral(N' dy) = n W, and alpha agrees with aoa_grid (M 0.23 to 0.66)."""
    lift = ld.lift_distribution(
        database, w_oem=0.7 * weight, w_payload=0.3 * weight, v_eas=v_eas,
        altitude_m=2_000.0, load_factor=load_factor,
    )
    grid = ld.aoa_grid(
        database, np.array([weight], dtype=np.float64), np.array([v_eas], dtype=np.float64),
        2_000.0, load_factor,
    )

    total = 2.0 * float(trapezoid(lift.lift_per_span, lift.y))
    assert abs(total - load_factor * weight) <= settings.LIFT_CHECK_TOLERANCE * weight
    assert lift.alpha_deg == pytest.approx(float(grid.alpha_deg[0, 0]), abs=1e-9)
    assert lift.cl_wing == pytest.approx(float(grid.cl_wing[0, 0]))


def test_lift_distribution_weight_function_gives_inertial_load(
    database: ld.AeroDatabase,
) -> None:
    def weight_per_span(y: np.ndarray) -> np.ndarray:
        return 1500.0 - 80.0 * y

    lift = ld.lift_distribution(
        database, w_oem=1.8e5, w_payload=7.0e4, v_eas=130.0, altitude_m=0.0,
        load_factor=3.2, weight_per_span=weight_per_span,
    )

    np.testing.assert_allclose(lift.inertial_per_span, 3.2 * (1500.0 - 80.0 * lift.y))
    assert lift.weight == pytest.approx(2.5e5)  # w(y) is not added to W


def test_lift_distribution_weight_function_wrong_shape_raises(
    database: ld.AeroDatabase,
) -> None:
    with pytest.raises(ValueError, match="weight_per_span"):
        ld.lift_distribution(
            database, w_oem=1.8e5, w_payload=7.0e4, v_eas=130.0, altitude_m=0.0,
            load_factor=1.0, weight_per_span=lambda y: np.ones(3, dtype=np.float64),
        )


def test_lift_distribution_mach_above_grid_raises(database: ld.AeroDatabase) -> None:
    with pytest.raises(ValueError, match="outside the database grid"):
        ld.lift_distribution(
            database, w_oem=1.8e5, w_payload=7.0e4, v_eas=300.0, altitude_m=0.0,
            load_factor=1.0,
        )


@pytest.mark.parametrize(
    "change",
    [{"w_oem": 0.0}, {"w_oem": math.nan}, {"w_payload": -1.0}, {"w_fuel": -1.0},
     {"load_factor": math.nan}, {"v_eas": math.nan}],
)
def test_lift_distribution_invalid_input_raises(
    database: ld.AeroDatabase, change: dict[str, float]
) -> None:
    inputs = {"w_oem": 1.8e5, "w_payload": 7.0e4, "v_eas": 130.0, "altitude_m": 0.0,
              "load_factor": 1.0, "w_fuel": 5.0e4}
    inputs.update(change)

    with pytest.raises(ValueError):
        ld.lift_distribution(database, **inputs)


# ============================================================
#  LOAD CASE AND lift.csv
# ============================================================

CASE = LoadCase(
    name="LC-T", n=2.7, v_eas=162.3, altitude_m=3_050.0, mass_case="mtow", thrust_n=18_200.0
)


def test_lift_for_case_uses_the_mass_case_weight(database: ld.AeroDatabase) -> None:
    lift = ld.lift_for_case(database, CASE, MASSES)

    assert lift.weight == pytest.approx(G0 * MASSES.mtom_kg)
    assert lift.load_factor == 2.7
    assert lift.state.v_eas == 162.3


def test_to_lift_table_round_trip_keeps_every_value(
    database: ld.AeroDatabase, tmp_path: Path
) -> None:
    lift = ld.lift_for_case(database, dataclasses.replace(CASE, n=-1.1), MASSES)
    path = tmp_path / "LC-T" / settings.LIFT_CSV

    ld.to_lift_table(lift, CASE.name).to_csv(path)
    table = LiftTable.from_csv(path)

    assert table.case == "LC-T"
    assert table.extrapolated
    assert table.mach_run == table.mach_case == pytest.approx(lift.state.mach)
    assert (table.n, table.w_n, table.alpha_d_deg) == (-1.1, lift.weight, lift.alpha_deg)
    np.testing.assert_array_equal(table.y_m, lift.y)
    np.testing.assert_array_equal(table.n_prime_npm, lift.lift_per_span)
    np.testing.assert_array_equal(table.m_prime_c4_nmpm, lift.moment_per_span)
    np.testing.assert_array_equal(table.cm_c4_d, lift.cm_c4)


# ============================================================
#  ANGLE-OF-ATTACK GRID
# ============================================================


def test_aoa_grid_weights_and_speeds_return_2d_arrays(database: ld.AeroDatabase) -> None:
    weights = np.array([1.76e5, 2.67e5, 3.23e5], dtype=np.float64)
    v_eas = np.linspace(80.0, 200.0, 5, dtype=np.float64)

    grid = ld.aoa_grid(database, weights, v_eas, 1_000.0, 1.0)

    assert grid.alpha_deg.shape == grid.cl_wing.shape == (3, 5)
    assert grid.alpha_deg.dtype == np.float64
    assert np.all(np.diff(grid.alpha_deg, axis=0) > 0.0)  # heavier -> higher alpha
    assert np.all(np.diff(grid.alpha_deg, axis=1) < 0.0)  # faster -> lower alpha


def test_aoa_grid_speed_above_grid_raises(database: ld.AeroDatabase) -> None:
    with pytest.raises(ValueError, match="outside the database grid"):
        ld.aoa_grid(
            database, np.array([2.0e5], dtype=np.float64),
            np.array([150.0, 300.0], dtype=np.float64), 0.0, 1.0,
        )


def test_aoa_grid_empty_weights_raises(database: ld.AeroDatabase) -> None:
    with pytest.raises(ValueError, match="weights"):
        ld.aoa_grid(
            database, np.array([], dtype=np.float64),
            np.array([150.0], dtype=np.float64), 0.0, 1.0,
        )
