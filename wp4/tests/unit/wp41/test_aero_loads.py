# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""Tests of wp41.aero_loads: pre-processing, moment transfer, file I/O.

Planform values: closed-form trapezoidal-wing relations (Raymer, Aircraft
Design, ch. 7). The OpenVSP test post-processes the real VSPAERO run in
out/wp41/vspaero_run/ when it exists.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from configs.aircraft import WING, WingGeometry
from configs.wp41 import settings
from wp41 import aero_loads as al

# ============================================================
#  PRE-PROCESSING
# ============================================================


def test_planform_rectangular_wing_returns_closed_form(rectangular_wing: WingGeometry) -> None:
    pf = al.planform(rectangular_wing)

    assert pf.span_m == pytest.approx(10.0)
    assert pf.root_chord_m == pytest.approx(1.0)
    assert pf.tip_chord_m == pytest.approx(1.0)
    assert pf.mac_m == pytest.approx(1.0)
    assert pf.y_mac_m == pytest.approx(2.5)
    assert pf.tan_sweep_le == pytest.approx(0.0)


def test_planform_design_wing_matches_hand_check() -> None:
    """Hand check in wp3/configs/geometry.yaml: b, c_r, c_t."""
    pf = al.planform(WING)

    assert pf.span_m == pytest.approx(24.4001, abs=1e-4)
    assert pf.root_chord_m == pytest.approx(3.6692, abs=1e-4)
    assert pf.tip_chord_m == pytest.approx(1.4677, abs=1e-4)


@given(
    area=st.floats(1.0, 500.0),
    aspect_ratio=st.floats(2.0, 15.0),
    taper=st.floats(0.1, 1.0),
    sweep=st.floats(-40.0, 40.0),
)
def test_planform_any_trapezoid_recovers_area_and_taper(
    area: float, aspect_ratio: float, taper: float, sweep: float
) -> None:
    wing = dataclasses.replace(
        WING, area_m2=area, aspect_ratio=aspect_ratio, taper=taper, sweep_c4_deg=sweep
    )

    pf = al.planform(wing)

    assert (pf.root_chord_m + pf.tip_chord_m) / 2.0 * pf.span_m == pytest.approx(area)
    assert pf.tip_chord_m / pf.root_chord_m == pytest.approx(taper)
    assert pf.tip_chord_m - 1e-12 <= pf.mac_m <= pf.root_chord_m + 1e-12


def test_quarter_chord_points_flat_wing_returns_root_le_plus_quarter_chord(
    rectangular_wing: WingGeometry,
) -> None:
    pf = al.planform(rectangular_wing)
    y = np.array([0.0, 2.0, 5.0], dtype=np.float64)

    x, z = al.quarter_chord_points(rectangular_wing, pf, y)

    np.testing.assert_allclose(x, 1.25)
    np.testing.assert_allclose(z, 0.5)


def test_quarter_chord_points_incidence_rotates_about_root_le(
    rectangular_wing: WingGeometry,
) -> None:
    wing = dataclasses.replace(rectangular_wing, incidence_deg=10.0)
    pf = al.planform(wing)
    inc = math.radians(10.0)

    x, z = al.quarter_chord_points(wing, pf, np.array([3.0], dtype=np.float64))

    assert x[0] == pytest.approx(1.0 + 0.25 * math.cos(inc))
    assert z[0] == pytest.approx(0.5 - 0.25 * math.sin(inc))


def test_quarter_chord_points_dihedral_puts_point_on_panel(
    rectangular_wing: WingGeometry,
) -> None:
    wing = dataclasses.replace(rectangular_wing, dihedral_deg=5.0)
    pf = al.planform(wing)
    y = np.array([4.0], dtype=np.float64)

    _, z = al.quarter_chord_points(wing, pf, y)

    assert z[0] == pytest.approx(0.5 + 4.0 * math.tan(math.radians(5.0)))


def test_moment_reference_rectangular_wing_returns_mac_quarter_chord(
    rectangular_wing: WingGeometry,
) -> None:
    x_ref, z_ref = al.moment_reference(rectangular_wing, al.planform(rectangular_wing))

    assert (x_ref, z_ref) == pytest.approx((1.25, 0.5))


def test_read_airfoil_dat_design_airfoil_has_unit_chord() -> None:
    upper, lower = al.read_airfoil_dat(WING.airfoil_root)

    np.testing.assert_allclose(upper[0], [0.0, 0.0], atol=1e-12)
    assert max(upper[:, 0].max(), lower[:, 0].max()) == pytest.approx(1.0)
    assert upper[:, 1].max() > 0.0 > lower[:, 1].min()


def test_read_airfoil_dat_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        al.read_airfoil_dat(tmp_path / "none.dat")


# ============================================================
#  CONFIG VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "change",
    [{"area_m2": 0.0}, {"taper": 0.0}, {"taper": 1.5}, {"sweep_c4_deg": 70.0},
     {"twist_axis_chord_fraction": 1.2}, {"fuselage_diameter_m": -1.0}],
)
def test_wing_geometry_invalid_value_raises(change: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(WING, **change)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "change",
    [{"mach_start": 0.0}, {"mach_end": 1.0}, {"mach_start": 0.9}, {"mach_npts": 1}],
)
def test_vspaero_settings_invalid_mach_grid_raises(change: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="mach"):
        dataclasses.replace(settings.VSPAERO, **change)  # type: ignore[arg-type]


@pytest.mark.parametrize("npts", [1, 3])
def test_vspaero_settings_alpha_count_not_two_raises(npts: int) -> None:
    with pytest.raises(ValueError, match="alpha_npts"):
        dataclasses.replace(settings.VSPAERO, alpha_npts=npts)


def test_vspaero_settings_single_wake_node_raises() -> None:
    with pytest.raises(ValueError, match="wake_nodes"):
        dataclasses.replace(settings.VSPAERO, wake_nodes=1)


# ============================================================
#  POST-PROCESSING
# ============================================================


def test_quarter_chord_moment_zero_offset_returns_cmy() -> None:
    cmy = np.array([-0.1, 0.2], dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    ones = np.ones(2, dtype=np.float64)

    cm = al.quarter_chord_moment(cmy, ones, ones, ones, zero, zero)

    np.testing.assert_allclose(cm, cmy)


@given(
    cx=st.floats(-1.0, 1.0), cz=st.floats(-2.0, 2.0), chord=st.floats(0.5, 5.0),
    dx=st.floats(-10.0, 10.0), dz=st.floats(-2.0, 2.0), cm_c4=st.floats(-0.3, 0.3),
)
def test_quarter_chord_moment_recovers_section_moment(
    cx: float, cz: float, chord: float, dx: float, dz: float, cm_c4: float
) -> None:
    """A force (cx, cz) at the quarter chord plus a moment cm_c4.

    About the reference point: M_ref = M_c4 + dz Fx - dx Fz (x aft, z up),
    so cmy = cm_c4 + (dz cx - dx cz) / c. The transfer must give cm_c4 back.
    """
    cmy = cm_c4 + (dz * cx - dx * cz) / chord
    arr = lambda v: np.array([v], dtype=np.float64)  # noqa: E731

    cm = al.quarter_chord_moment(arr(cmy), arr(cx), arr(cz), arr(chord), arr(dx), arr(dz))

    assert cm[0] == pytest.approx(cm_c4, abs=1e-9)




def _strip_rows(cases: list[tuple[float, float]], y: list[float]) -> np.ndarray:
    """Strip rows (STRIP_COLUMNS) with cl = 0.1 alpha + mach and cmy = -0.01 alpha."""
    rows = []
    for mach, alpha in cases:
        for yi in y:
            cl = 0.1 * alpha + mach
            rows.append([mach, alpha, yi, 1.0, 0.5, cl, 0.0, cl, -0.01 * alpha])
    return np.array(rows, dtype=np.float64)


def test_span_table_unsorted_rows_returns_cases_by_mach_then_alpha() -> None:
    rows = _strip_rows([(0.6, 10.0), (0.3, 10.0), (0.6, 0.0), (0.3, 0.0)], [3.0, 1.0, -1.0, 2.0])

    table = al.span_table(rows)

    np.testing.assert_allclose(table.mach, [0.3, 0.3, 0.6, 0.6])
    np.testing.assert_allclose(table.alpha, [0.0, 10.0, 0.0, 10.0])
    np.testing.assert_allclose(table.y, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(table.cl[:, 3], 1.6)
    assert table.cmy.shape == (3, 4)
    assert table.cl.dtype == np.float64


def test_span_table_different_strips_per_case_raises() -> None:
    rows = np.vstack([_strip_rows([(0.3, 0.0)], [1.0, 2.0]), _strip_rows([(0.3, 5.0)], [1.0, 2.5])])

    with pytest.raises(ValueError, match="differ"):
        al.span_table(rows)


def test_span_table_no_right_wing_strips_raises() -> None:
    with pytest.raises(ValueError, match="y > 0"):
        al.span_table(_strip_rows([(0.3, 0.0)], [-1.0, 0.0]))


def _totals(cases: list[tuple[float, float]], scale: float) -> np.ndarray:
    """Totals (TOTAL_COLUMNS) that match _strip_rows on 3 strips of 0.5 m^2, S = 10 m^2.

    2 * 3 strips * 0.5 m^2 / 10 m^2 = 0.3 times the strip value.
    """
    rows = [[m, a, scale * 0.3 * (0.1 * a + m), 0.3 * (-0.01 * a)] for m, a in cases]
    return al.sort_cases(np.array(rows, dtype=np.float64))


CASES = [(0.3, 0.0), (0.3, 10.0), (0.6, 0.0), (0.6, 10.0)]


def test_check_totals_consistent_data_passes(rectangular_wing: WingGeometry) -> None:
    pf = al.planform(rectangular_wing)
    table = al.span_table(_strip_rows(CASES, [1.0, 2.0, 3.0]))

    al.check_totals(table, _totals(CASES, 1.0), rectangular_wing, pf, tolerance=1e-9)


def test_check_totals_mismatch_raises(rectangular_wing: WingGeometry) -> None:
    pf = al.planform(rectangular_wing)
    table = al.span_table(_strip_rows(CASES, [1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="differs"):
        al.check_totals(table, _totals(CASES, 1.5), rectangular_wing, pf, tolerance=1e-3)


def test_check_totals_missing_case_raises(rectangular_wing: WingGeometry) -> None:
    pf = al.planform(rectangular_wing)
    table = al.span_table(_strip_rows(CASES, [1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Cases differ"):
        al.check_totals(table, _totals(CASES[:3], 1.0), rectangular_wing, pf, tolerance=1e-3)


def test_read_csv_wrong_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "totals.csv"
    path.write_text("# alpha,CL\n0,0.5\n")

    with pytest.raises(ValueError, match="header"):
        al.read_csv(path, al.TOTAL_COLUMNS)


def test_read_csv_header_only_raises(tmp_path: Path) -> None:
    path = tmp_path / "totals.csv"
    path.write_text("# mach,alpha,CL,CMy\n")

    with pytest.warns(UserWarning), pytest.raises(ValueError, match="shape"):
        al.read_csv(path, al.TOTAL_COLUMNS)


def test_write_distribution_file_has_keyed_header_and_case_columns(
    tmp_path: Path, rectangular_wing: WingGeometry
) -> None:
    pf = al.planform(rectangular_wing)
    table = al.span_table(_strip_rows(CASES, [1.0, 2.0, 3.0]))
    totals = _totals(CASES, 1.0)
    path = tmp_path / "lift.txt"

    al.write_distribution(
        path, "title", "cl", table, table.cl, totals[:, 2], settings.KEY_CL,
        rectangular_wing, pf,
    )

    data = np.loadtxt(path)
    text = path.read_text()
    assert data.shape == (3, 2 + len(CASES))
    np.testing.assert_allclose(data[:, 0], table.y)
    np.testing.assert_allclose(data[:, 5], table.cl[:, 3])
    assert f"# {settings.KEY_MACH}: 0.3000 0.3000 0.6000 0.6000" in text
    assert f"# {settings.KEY_S_REF}: 10.000000" in text


# ============================================================
#  REAL VSPAERO RUN (OpenVSP env)
# ============================================================


@pytest.mark.openvsp
def test_post_process_real_run_writes_full_grid(tmp_path: Path) -> None:
    """Post-process out/wp41/vspaero_run/ (skip if wp41_aero_loads.py has not run)."""
    if not (settings.RUN_DIR / settings.STRIPS_FILE).is_file():
        pytest.skip("no VSPAERO run in out/wp41/vspaero_run/")
    n_case = settings.VSPAERO.mach_npts * settings.VSPAERO.alpha_npts

    lift_path, moment_path = al.post_process(settings.RUN_DIR, tmp_path)

    lift = np.loadtxt(lift_path)
    moment = np.loadtxt(moment_path)
    assert lift.shape == moment.shape == (settings.VSPAERO.span_tess - 1, 2 + n_case)
    assert np.all(lift[:, 3::2] > lift[:, 2::2])  # cl rises with alpha at every Mach
    assert np.all(lift[:, 0] < al.planform(WING).half_span_m)
