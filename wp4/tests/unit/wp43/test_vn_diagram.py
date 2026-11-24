# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-24
"""Tests of wp43.vn_diagram. CL, SPEEDS and masses are test data, not design values."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from configs.wp43.settings import DesignSpeeds, LiftCoefficients
from wp43 import speeds as sp
from wp43.atmosphere import isa
from wp43.vn_diagram import (
    VnSpeeds,
    clean_envelope,
    flap_curve,
    manoeuvre_envelope,
    n_min,
    vn_speeds,
)

H_CRUISE_M = 35_000 * 0.3048
MTOW_KG = 26_000.0
AREA_M2 = 60.0

CL = LiftCoefficients(
    cl_max_clean=1.4, cl_max_takeoff=2.0, cl_max_app=2.3, cl_max_landing=2.6
)
SPEEDS = DesignSpeeds(
    v_c_eas_ms=120.0,
    m_c=0.70,
    k_vd=1.25,
    dm_d=0.07,
    k_vf_takeoff=1.7,
    k_vf_approach=1.9,
    k_vf_landing=2.0,
)

# A valid set: 0 < V_S0 < V_S1 < V_A < V_C < V_D, V_F > V_S0, n_max > 0.
VALID = {
    "v_s0": 50.0, "v_s1": 65.0, "v_a": 105.0, "v_c": 120.0,
    "v_d": 150.0, "v_f": 95.0, "n_max": 2.6,
}


# --- VnSpeeds ---


def test_vn_speeds_class_accepts_valid_order() -> None:
    vs = VnSpeeds(**VALID)

    assert (vs.v_s1, vs.v_d, vs.n_max) == (65.0, 150.0, 2.6)


def test_vn_speeds_class_is_frozen() -> None:
    vs = VnSpeeds(**VALID)

    with pytest.raises(dataclasses.FrozenInstanceError):
        vs.v_a = 110.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("v_s0", 0.0),  # not positive
        ("v_s0", 70.0),  # above V_S1
        ("v_s1", 110.0),  # above V_A
        ("v_a", 125.0),  # above V_C
        ("v_c", 160.0),  # above V_D
        ("v_d", 120.0),  # equal to V_C
        ("v_s1", math.nan),
    ],
)
def test_vn_speeds_class_wrong_order_raises(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match="V_S0 < V_S1 < V_A < V_C < V_D"):
        VnSpeeds(**{**VALID, field: bad})


@pytest.mark.parametrize("bad", [50.0, 40.0, math.nan])
def test_vn_speeds_class_v_f_not_above_v_s0_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="V_F"):
        VnSpeeds(**{**VALID, "v_f": bad})


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_vn_speeds_class_n_max_not_positive_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="n_max"):
        VnSpeeds(**{**VALID, "n_max": bad})


# --- vn_speeds() ---


def test_vn_speeds_fields_come_from_speeds_module() -> None:
    atm = isa(0.0)

    vs = vn_speeds(21_400.0, MTOW_KG, AREA_M2, CL, SPEEDS, atm)

    assert vs.v_s0 == sp.stall_speed_flaps_landing(21_400.0, AREA_M2, CL)
    assert vs.v_s1 == sp.stall_speed_no_flaps(21_400.0, AREA_M2, CL)
    assert vs.v_a == sp.manoeuvering_speed(21_400.0, AREA_M2, CL, MTOW_KG)
    assert vs.v_c == SPEEDS.v_c_eas_ms
    assert vs.v_d == sp.dive_speed(SPEEDS, atm)
    assert vs.v_f == sp.design_wing_flap_speed(SPEEDS, CL, MTOW_KG, AREA_M2)
    assert vs.n_max == sp.n_max(MTOW_KG)


def test_vn_speeds_hand_values_at_mtow_sea_level() -> None:
    # V_S1 = sqrt(2 * 26 000 * 9.80665 / (1.225 * 60 * 1.4)) = 70.40 m/s
    # n_max: W = 57 269 lb -> 2.457, raised to 2.5; V_A = 70.40 sqrt(2.5)
    vs = vn_speeds(MTOW_KG, MTOW_KG, AREA_M2, CL, SPEEDS, isa(0.0))

    assert vs.v_s1 == pytest.approx(70.40, abs=0.01)
    assert vs.v_s0 == pytest.approx(70.40 * math.sqrt(1.4 / 2.6), abs=0.01)
    assert vs.n_max == pytest.approx(2.5)
    assert vs.v_a == pytest.approx(70.40 * math.sqrt(2.5), abs=0.01)
    assert vs.v_d == pytest.approx(150.0)


def test_vn_speeds_only_v_d_depends_on_altitude() -> None:
    low = vn_speeds(21_400.0, MTOW_KG, AREA_M2, CL, SPEEDS, isa(0.0))
    high = vn_speeds(21_400.0, MTOW_KG, AREA_M2, CL, SPEEDS, isa(H_CRUISE_M))

    assert high.v_d < low.v_d
    assert dataclasses.replace(high, v_d=low.v_d) == low


def test_vn_speeds_mass_changes_stall_and_v_a_but_not_mtow_terms() -> None:
    heavy = vn_speeds(MTOW_KG, MTOW_KG, AREA_M2, CL, SPEEDS, isa(0.0))
    light = vn_speeds(16_900.0, MTOW_KG, AREA_M2, CL, SPEEDS, isa(0.0))

    ratio = math.sqrt(16_900.0 / MTOW_KG)
    assert light.v_s1 == pytest.approx(heavy.v_s1 * ratio, rel=1e-12)
    assert light.v_s0 == pytest.approx(heavy.v_s0 * ratio, rel=1e-12)
    assert light.v_a == pytest.approx(heavy.v_a * ratio, rel=1e-12)
    assert (light.n_max, light.v_f, light.v_c, light.v_d) == (
        heavy.n_max, heavy.v_f, heavy.v_c, heavy.v_d
    )


def test_vn_speeds_v_a_above_v_c_raises() -> None:
    # V_C = 100 m/s is below V_A = 111.3 m/s at MTOW.
    low_v_c = dataclasses.replace(SPEEDS, v_c_eas_ms=100.0)

    with pytest.raises(ValueError, match="V_A"):
        vn_speeds(MTOW_KG, MTOW_KG, AREA_M2, CL, low_v_c, isa(0.0))


# --- n_min ---


@pytest.mark.parametrize(
    ("v", "expected"),
    [
        (0.0, -1.0),
        (87.3, -1.0),
        (120.0, -1.0),  # at V_C
        (135.0, -0.5),  # halfway between V_C and V_D
        (142.5, -0.25),
        (150.0, 0.0),  # at V_D
    ],
)
def test_n_min_flat_then_linear_to_zero(v: float, expected: float) -> None:
    assert n_min(v, 120.0, 150.0) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("v", [-1.0, 150.1, math.nan])
def test_n_min_speed_out_of_range_raises(v: float) -> None:
    with pytest.raises(ValueError, match="v_eas_ms"):
        n_min(v, 120.0, 150.0)


def test_n_min_v_d_not_above_v_c_raises() -> None:
    with pytest.raises(ValueError, match="V_D"):
        n_min(100.0, 120.0, 120.0)


# --- clean envelope ---

# Consistent hand-made set: V_A = V_S1 sqrt(n_max) = 60 sqrt(2.5) = 94.868 m/s.
V_A_CONSISTENT = 60.0 * math.sqrt(2.5)
HAND = VnSpeeds(
    v_s0=45.0, v_s1=60.0, v_a=V_A_CONSISTENT, v_c=120.0, v_d=150.0,
    v_f=95.0, n_max=2.5,
)


def _has_point(v: np.ndarray, n: np.ndarray, v0: float, n0: float) -> bool:
    """True if (v0, n0) is one of the polygon points."""
    return bool(np.any(np.isclose(v, v0, rtol=1e-12) & np.isclose(n, n0, atol=1e-12)))


def test_clean_envelope_is_closed_float64_and_same_shape() -> None:
    v, n = clean_envelope(HAND, n_points=50)

    assert v.shape == n.shape == (50 + 4 + 49,)
    assert v.dtype == n.dtype == np.float64
    assert (v[0], n[0]) == (0.0, 0.0)
    assert (v[-1], n[-1]) == pytest.approx((0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize(
    ("v0", "n0"),
    [
        (V_A_CONSISTENT, 2.5),  # A
        (150.0, 2.5),  # D
        (150.0, 0.0),  # E
        (120.0, -1.0),  # F
        (60.0, -1.0),  # H
    ],
)
def test_clean_envelope_passes_through_corners(v0: float, n0: float) -> None:
    v, n = clean_envelope(HAND)

    assert _has_point(v, n, v0, n0)


def test_clean_envelope_follows_stall_parabolas() -> None:
    v, n = clean_envelope(HAND, n_points=200)

    upper = slice(0, 200)
    lower = slice(203, None)
    assert n[upper] == pytest.approx((v[upper] / 60.0) ** 2, abs=1e-12)
    assert n[lower] == pytest.approx(-((v[lower] / 60.0) ** 2), abs=1e-12)


def test_clean_envelope_v_a_above_curve_gives_flat_top_to_v_a() -> None:
    # CS 25.335(c): V_A may be above V_S1 sqrt(n_max); A is then on the flat top.
    vs = dataclasses.replace(HAND, v_a=105.0)

    v, n = clean_envelope(vs)

    assert _has_point(v, n, V_A_CONSISTENT, 2.5)
    assert n.max() == pytest.approx(2.5)


def test_clean_envelope_v_a_below_curve_raises() -> None:
    with pytest.raises(ValueError, match="V_A"):
        clean_envelope(dataclasses.replace(HAND, v_a=90.0))


# --- flaps-down curve ---


def test_flap_curve_v_f_below_clean_meeting_point_drops_to_clean_curve() -> None:
    # V_S1 sqrt(2) = 84.85 m/s; with V_F = 80 the n = 2 line ends at V_F and
    # drops to the clean curve, (80 / 60)^2 = 1.778.
    vs = dataclasses.replace(HAND, v_f=80.0)

    v, n = flap_curve(vs, n_points=100)

    assert v.shape == n.shape == (102,)
    assert n[:100] == pytest.approx((v[:100] / 45.0) ** 2, abs=1e-12)
    assert (v[99], n[99]) == pytest.approx((45.0 * math.sqrt(2.0), 2.0))
    assert (v[100], n[100]) == pytest.approx((80.0, 2.0))
    assert (v[101], n[101]) == pytest.approx((80.0, (80.0 / 60.0) ** 2))


def test_flap_curve_v_f_above_clean_meeting_point_ends_on_clean_curve() -> None:
    v, n = flap_curve(HAND, n_points=100)  # V_F = 95 > 84.85

    assert v.shape == (101,)
    assert (v[-1], n[-1]) == pytest.approx((60.0 * math.sqrt(2.0), 2.0))
    assert n.max() == pytest.approx(2.0)


def test_flap_curve_v_f_below_v_s0_sqrt2_raises() -> None:
    with pytest.raises(ValueError, match="V_F"):
        flap_curve(dataclasses.replace(HAND, v_f=60.0))  # 45 sqrt(2) = 63.6


# --- full envelope ---


def test_manoeuvre_envelope_corners_named_and_on_boundary() -> None:
    vs = vn_speeds(MTOW_KG, MTOW_KG, AREA_M2, CL, SPEEDS, isa(H_CRUISE_M))

    env = manoeuvre_envelope(vs)

    assert [c.name for c in env.corners] == ["A", "D", "E", "F", "H"]
    for corner in env.corners:
        assert _has_point(env.v_eas, env.n, corner.v_eas_ms, corner.n)
    assert env.speeds is vs
