# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Tests for wp2/src/b12wp2/config/.

The ISA atmosphere and sweep-theory checks are cross-checked against
hand-derived reference numbers -- these are not just "does it run" tests,
they verify the physics matches the expected derivation within reasonable
rounding tolerance.
"""
from __future__ import annotations

import math

import pytest

from b12wp2 import config
from b12wp2.config import planform  # noqa: F401  (makes config.planform resolvable)


def test_isa_sea_level_matches_standard_values() -> None:
    t, a, rho = config.isa_atmosphere(0.0)
    assert t == pytest.approx(288.15, abs=1e-6)
    assert a == pytest.approx(340.29, abs=0.05)
    assert rho == pytest.approx(1.225, abs=0.001)


def test_isa_at_35000ft_matches_hand_derivation() -> None:
    # 35,000 ft -> T ~ 218.8 K, a ~ 296.5 m/s, rho ~ 0.380 kg/m^3
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


def test_cruise_mach_normal_matches_hand_derivation() -> None:
    # M_n = M_inf * cos(Lambda_c/2) = 0.77 * cos(28.024 deg) ~ 0.680
    assert config.CRUISE.mach_normal == pytest.approx(0.6797, abs=0.0005)


def test_cruise_v_freestream_matches_hand_derivation() -> None:
    # V_inf = M * a ~ 228.4 m/s
    assert config.CRUISE.v_freestream == pytest.approx(228.4, abs=0.1)


def test_planform_matches_hand_derivation() -> None:
    # S = 66.7, AR = 9.5, taper 0.4: b = sqrt(S AR) = 25.17 m,
    # c_r = 2S / (b (1 + taper)) = 3.785 m, MAC = 2/3 c_r (1+t+t^2)/(1+t) = 2.812 m,
    # y_MAC = b/6 (1+2t)/(1+t) = 5.394 m.
    assert config.planform.B == pytest.approx(25.172, abs=0.001)
    assert config.planform.C_ROOT == pytest.approx(3.785, abs=0.001)
    assert config.planform.MAC == pytest.approx(2.812, abs=0.001)
    assert config.planform.Y_MAC == pytest.approx(5.394, abs=0.001)


def test_sweep_lines_match_hand_derivation() -> None:
    # tan(L_x) = tan(L_c/4) - (4/AR)(x - 1/4)(1 - t)/(1 + t), L_c/4 = 30 deg
    sweep_le = math.degrees(config.planform.sweep_at(0.0))
    sweep_c2 = math.degrees(config.planform.sweep_at(0.5))
    assert math.degrees(config.planform.sweep_at(0.25)) == pytest.approx(30.0, abs=1e-9)
    assert sweep_le == pytest.approx(31.901, abs=0.001)
    assert sweep_c2 == pytest.approx(28.024, abs=0.001)


def test_section_conditions_use_half_chord_sweep_and_mac() -> None:
    assert config.SECTION_SWEEP_RAD == pytest.approx(config.planform.sweep_at(0.5))
    for cond in (config.CRUISE, config.LANDING):
        assert cond.sweep_rad == pytest.approx(config.SECTION_SWEEP_RAD)
        assert cond.chord_m == pytest.approx(config.planform.MAC)


def test_cl_normal_raises_without_cl_wing() -> None:
    cond = config.FlightCondition(
        name="test", altitude_m=0.0, mach_freestream=0.5,
        sweep_rad=config.SECTION_SWEEP_RAD, ncrit=9.0,
    )
    with pytest.raises(ValueError, match="cl_wing is not set"):
        _ = cond.cl_normal


def test_reynolds_normal_raises_without_chord() -> None:
    cond = config.FlightCondition(
        name="test", altitude_m=0.0, mach_freestream=0.5,
        sweep_rad=config.SECTION_SWEEP_RAD, ncrit=9.0,
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


def test_landing_mach_derived_from_given_speed() -> None:
    # landing is 65 m/s at sea level
    sea_level_a = config.isa_atmosphere(0.0)[1]
    assert config.LANDING.mach_freestream == pytest.approx(65.0 / sea_level_a)
    assert config.LANDING.v_freestream == pytest.approx(65.0, abs=1e-6)


def test_kappa_a_matches_given_classification() -> None:
    assert config.kappa_a("NASA_SC(2)-0712") == pytest.approx(0.95)
    for conventional in ("NACA_25112", "NACA_64212", "lockheed_c5a_bl758"):
        assert config.kappa_a(conventional) == pytest.approx(0.87)


def test_kappa_a_rejects_unmapped_airfoil() -> None:
    with pytest.raises(ValueError, match="no conventional/supercritical classification"):
        config.kappa_a("some_new_airfoil")


def test_discover_airfoils_finds_final_candidate_set() -> None:
    # WORTMANN_FX_62-K-131 was cut (unfixable source-data defect, see
    # test_xfoil_runtime.py's history); NASA_SC(2)-0712 replaced it.
    found = {p.stem for p in config.discover_airfoils()}
    assert found == {
        "NACA_25112",
        "NACA_64212",
        "NASA_SC(2)-0712",
        "lockheed_c5a_bl758",
    }
