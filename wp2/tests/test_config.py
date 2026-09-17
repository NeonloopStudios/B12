"""Tests for wp2/scripts/config.py.

The ISA atmosphere and sweep-theory checks are cross-checked against the
hand-derived numbers in wp2/xfoil_plan.md Sec. 1 -- these are not just
"does it run" tests, they verify the physics matches the documented
derivation within reasonable rounding tolerance.
"""
from __future__ import annotations

import math

import pytest

from scripts import config


def test_isa_sea_level_matches_standard_values() -> None:
    t, a, rho = config.isa_atmosphere(0.0)
    assert t == pytest.approx(288.15, abs=1e-6)
    assert a == pytest.approx(340.29, abs=0.05)
    assert rho == pytest.approx(1.225, abs=0.001)


def test_isa_at_35000ft_matches_plan_derivation() -> None:
    # wp2/xfoil_plan.md Sec. 1: 35,000 ft -> T ~ 218.8 K, a ~ 296.5 m/s,
    # rho ~ 0.380 kg/m^3
    altitude_m = 35_000 * config.FT_TO_M
    t, a, rho = config.isa_atmosphere(altitude_m)
    assert t == pytest.approx(218.8, abs=0.05)
    assert a == pytest.approx(296.5, abs=0.1)
    assert rho == pytest.approx(0.380, abs=0.001)


def test_isa_rejects_out_of_range_altitude() -> None:
    with pytest.raises(ValueError):
        config.isa_atmosphere(-1.0)
    with pytest.raises(ValueError):
        config.isa_atmosphere(25_000.0)


def test_cruise_mach_normal_matches_plan_derivation() -> None:
    # wp2/xfoil_plan.md Sec. 1: M_n = M_inf * cos(Lambda) ~ 0.703
    assert config.CRUISE.mach_normal == pytest.approx(0.703, abs=0.001)


def test_cruise_v_freestream_matches_plan_derivation() -> None:
    # wp2/xfoil_plan.md Sec. 1: V_inf = M * a ~ 228.4 m/s
    assert config.CRUISE.v_freestream == pytest.approx(228.4, abs=0.1)


def test_sweep_angle_matches_plan_value() -> None:
    assert math.degrees(config.SWEEP_RAD) == pytest.approx(24.02, abs=0.01)


def test_cl_normal_raises_without_cl_wing() -> None:
    cond = config.FlightCondition(
        name="test", altitude_m=0.0, mach_freestream=0.5,
        sweep_rad=config.SWEEP_RAD, ncrit=9.0,
    )
    with pytest.raises(ValueError, match="cl_wing is not set"):
        _ = cond.cl_normal


def test_reynolds_normal_raises_without_chord() -> None:
    cond = config.FlightCondition(
        name="test", altitude_m=0.0, mach_freestream=0.5,
        sweep_rad=config.SWEEP_RAD, ncrit=9.0,
    )
    with pytest.raises(ValueError, match="chord_m is not set"):
        _ = cond.reynolds_normal


def test_cruise_and_landing_reynolds_normal_now_computable() -> None:
    # chord_m is set (WP1 sizing); Re_n should compute without raising and
    # land in a physically sane range for a ~2.6 m chord jet wing.
    assert 5e6 < config.CRUISE.reynolds_normal < 5e7
    assert config.LANDING.reynolds_normal > 0.0


def test_cruise_cl_normal_matches_sweep_correction() -> None:
    # Cl_n = Cl_wing / cos^2(sweep); cl_wing = 0.489433403 (WP1, eq. 8.13).
    cl_wing = 0.489433403
    expected = cl_wing / config.CRUISE.cos_sweep**2
    assert config.CRUISE.cl_wing == pytest.approx(cl_wing)
    assert config.CRUISE.cl_normal == pytest.approx(expected)
    assert 0.5 < config.CRUISE.cl_normal < 0.7


def test_level_flight_cl_matches_lift_equals_weight() -> None:
    # L = W => Cl = 2W / (rho V^2 S); sanity-check with round numbers where
    # dynamic pressure * S is trivially invertible.
    weight_n = 100_000.0
    rho = 1.0
    v = 100.0
    wing_area_m2 = 20.0
    q = 0.5 * rho * v**2
    expected = weight_n / (q * wing_area_m2)
    assert config.level_flight_cl(weight_n, rho, v, wing_area_m2) == pytest.approx(expected)


def test_cl_normal_and_reynolds_normal_when_set() -> None:
    cond = config.FlightCondition(
        name="test", altitude_m=0.0, mach_freestream=0.2,
        sweep_rad=0.0, ncrit=9.0, chord_m=1.0, cl_wing=0.5,
    )
    # zero sweep: normal quantities reduce to the freestream ones
    assert cond.cl_normal == pytest.approx(0.5)
    assert cond.mach_normal == pytest.approx(0.2)
    assert cond.reynolds_normal > 0.0


def test_scoring_weights_sum_to_one() -> None:
    assert sum(config.SCORING_WEIGHTS.values()) == pytest.approx(1.0)


def test_discover_airfoils_finds_final_candidate_set() -> None:
    found = {p.stem for p in config.discover_airfoils()}
    assert found == {
        "NACA_25112",
        "NACA_64212",
        "WORTMANN_FX_62-K-131",
        "lockheed_c5a_bl758",
    }
