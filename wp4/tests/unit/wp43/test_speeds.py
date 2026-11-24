# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-24
"""Tests of wp43.speeds. CL and DesignSpeeds below are test data, not design values."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from configs.wp43.settings import DesignSpeeds, LiftCoefficients
from wp43 import speeds as sp
from wp43.atmosphere import isa

H_CRUISE_M = 35_000 * 0.3048

# Distinct CL_max per configuration, so a wrong attribute gives a wrong speed.
CL = LiftCoefficients(
    cl_max_clean=1.4, cl_max_takeoff=2.0, cl_max_app=2.3, cl_max_landing=2.6
)
# Distinct k_vf factors, so a wrong factor gives a wrong V_F.
SPEEDS = DesignSpeeds(
    v_c_eas_ms=120.0,
    m_c=0.70,
    k_vd=1.25,
    dm_d=0.07,
    k_vf_takeoff=1.7,
    k_vf_approach=1.9,
    k_vf_landing=2.0,
)

StallFunction = Callable[[float, float, LiftCoefficients], float]


# --- conversions ---


def test_kg_to_lb_uses_guide_factor() -> None:
    assert sp.kg_to_lb(908.0) == pytest.approx(2_000.0)


def test_tas_from_eas_at_cruise_altitude() -> None:
    # sigma(35 000 ft) = 0.30988, sqrt(sigma) = 0.55667
    assert sp.tas_from_eas(100.0, isa(H_CRUISE_M)) == pytest.approx(179.64, abs=0.01)


def test_tas_equals_eas_at_sea_level() -> None:
    assert sp.tas_from_eas(83.4, isa(0.0)) == pytest.approx(83.4, rel=1e-5)


def test_eas_from_mach_at_cruise_altitude() -> None:
    assert sp.eas_from_mach(0.77, isa(H_CRUISE_M)) == pytest.approx(127.10, abs=0.01)


def test_eas_from_mach_round_trips_to_mach_times_a() -> None:
    atm = isa(7_300.0)

    v_tas = sp.tas_from_eas(sp.eas_from_mach(0.63, atm), atm)

    assert v_tas == pytest.approx(0.63 * atm.speed_of_sound_ms, rel=1e-12)


@pytest.mark.parametrize("mach", [-0.1, math.nan])
def test_eas_from_mach_invalid_raises(mach: float) -> None:
    with pytest.raises(ValueError, match="mach"):
        sp.eas_from_mach(mach, isa(0.0))


# --- stall speed ---


def test_stall_speed_eas_matches_hand_value() -> None:
    # sqrt(2 * 20 000 * 9.80665 / (1.225 * 50 * 1.5)) = 65.342 m/s
    assert sp.stall_speed_eas(20_000.0, 1.5, 50.0) == pytest.approx(65.342, abs=1e-3)


def test_stall_speed_eas_scales_with_sqrt_n() -> None:
    v_1g = sp.stall_speed_eas(18_500.0, 1.7, 47.0)

    v_n = sp.stall_speed_eas(18_500.0, 1.7, 47.0, n=2.7)

    assert v_n == pytest.approx(v_1g * math.sqrt(2.7), rel=1e-12)


@pytest.mark.parametrize("name", ["mass_kg", "cl_max", "area_m2", "n"])
@pytest.mark.parametrize("bad", [0.0, -3.0, math.nan])
def test_stall_speed_eas_invalid_input_raises(name: str, bad: float) -> None:
    kwargs = {"mass_kg": 20_000.0, "cl_max": 1.5, "area_m2": 50.0, "n": 1.0}
    kwargs[name] = bad

    with pytest.raises(ValueError, match=name):
        sp.stall_speed_eas(**kwargs)


@pytest.mark.parametrize(
    ("function", "cl_max"),
    [
        (sp.stall_speed_no_flaps, CL.cl_max_clean),
        (sp.stall_speed_flaps_takeoff, CL.cl_max_takeoff),
        (sp.stall_speed_flaps_approach, CL.cl_max_app),
        (sp.stall_speed_flaps_landing, CL.cl_max_landing),
    ],
)
def test_stall_speed_per_configuration_uses_its_cl_max(
    function: StallFunction, cl_max: float
) -> None:
    expected = sp.stall_speed_eas(24_300.0, cl_max, 55.0)

    assert function(24_300.0, 55.0, CL) == pytest.approx(expected, rel=1e-12)


def test_stall_speeds_decrease_as_flaps_extend() -> None:
    v_s1 = sp.stall_speed_no_flaps(24_300.0, 55.0, CL)
    v_to = sp.stall_speed_flaps_takeoff(24_300.0, 55.0, CL)
    v_app = sp.stall_speed_flaps_approach(24_300.0, 55.0, CL)
    v_s0 = sp.stall_speed_flaps_landing(24_300.0, 55.0, CL)

    assert v_s1 > v_to > v_app > v_s0


# --- n_max ---


@pytest.mark.parametrize(
    ("mtow_kg", "expected"),
    [
        (9_080.0, 2.9),  # W = 20 000 lb: 2.1 + 24 000 / 30 000, inside the limits
        (27_240.0, 2.5),  # W = 60 000 lb: formula gives 2.443, raised to 2.5
        (908.0, 3.8),  # W = 2 000 lb: formula gives 4.1, capped at 3.8
    ],
)
def test_n_max_formula_and_limits(mtow_kg: float, expected: float) -> None:
    assert sp.n_max(mtow_kg) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("mtow_kg", [0.0, -1.0, math.nan])
def test_n_max_invalid_mass_raises(mtow_kg: float) -> None:
    with pytest.raises(ValueError, match="mtow_kg"):
        sp.n_max(mtow_kg)


# --- V_A ---


def test_manoeuvering_speed_is_v_s1_times_sqrt_n_max() -> None:
    v_a = sp.manoeuvering_speed(9_080.0, 30.0, CL, mtow_kg=9_080.0)

    v_s1 = sp.stall_speed_no_flaps(9_080.0, 30.0, CL)
    assert v_a == pytest.approx(v_s1 * math.sqrt(2.9), rel=1e-12)


def test_manoeuvering_speed_takes_n_max_from_mtow_not_mass() -> None:
    # Diagram mass 15 000 kg, MTOM 27 240 kg: n_max = 2.5, from the MTOM.
    v_a = sp.manoeuvering_speed(15_000.0, 50.0, CL, mtow_kg=27_240.0)

    v_s1 = sp.stall_speed_no_flaps(15_000.0, 50.0, CL)
    assert v_a == pytest.approx(v_s1 * math.sqrt(2.5), rel=1e-12)


# --- V_D ---


def test_dive_speed_at_sea_level_is_v_c_over_0_8() -> None:
    assert sp.dive_speed(SPEEDS, isa(0.0)) == pytest.approx(150.0, rel=1e-12)


def test_dive_speed_at_cruise_altitude_uses_mach_margin() -> None:
    # M_D = 0.70 + 0.07 = 0.77 at 35 000 ft is 127.10 m/s EAS, below 150 m/s.
    v_d = sp.dive_speed(SPEEDS, isa(H_CRUISE_M))

    assert v_d == pytest.approx(127.10, abs=0.01)
    assert v_d == pytest.approx(sp.eas_from_mach(0.77, isa(H_CRUISE_M)), rel=1e-12)


def test_dive_speed_does_not_increase_with_altitude() -> None:
    v_d = [sp.dive_speed(SPEEDS, isa(h)) for h in (0.0, 4_000.0, 8_000.0, 12_000.0)]

    assert all(lo >= hi for lo, hi in zip(v_d, v_d[1:]))


# --- V_F ---


def test_design_wing_flap_speed_approach_governs() -> None:
    # 1.9 / sqrt(2.3) = 1.253 > 2.0 / sqrt(2.6) = 1.240 > 1.7 / sqrt(2.0) = 1.202
    v_f = sp.design_wing_flap_speed(SPEEDS, CL, 26_000.0, 60.0)

    expected = 1.9 * sp.stall_speed_flaps_approach(26_000.0, 60.0, CL)
    assert v_f == pytest.approx(expected, rel=1e-12)


def test_design_wing_flap_speed_takeoff_governs() -> None:
    # Low take-off CL_max: 1.7 / sqrt(1.5) = 1.388 is the highest.
    cl = LiftCoefficients(
        cl_max_clean=1.4, cl_max_takeoff=1.5, cl_max_app=2.3, cl_max_landing=2.6
    )

    v_f = sp.design_wing_flap_speed(SPEEDS, cl, 26_000.0, 60.0)

    expected = 1.7 * sp.stall_speed_flaps_takeoff(26_000.0, 60.0, cl)
    assert v_f == pytest.approx(expected, rel=1e-12)


def test_design_wing_flap_speed_is_at_least_each_minimum() -> None:
    v_f = sp.design_wing_flap_speed(SPEEDS, CL, 26_000.0, 60.0)

    assert v_f >= 1.7 * sp.stall_speed_flaps_takeoff(26_000.0, 60.0, CL)
    assert v_f >= 1.9 * sp.stall_speed_flaps_approach(26_000.0, 60.0, CL)
    assert v_f >= 2.0 * sp.stall_speed_flaps_landing(26_000.0, 60.0, CL)
