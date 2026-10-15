# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/config/.

ISA reference at 35,000 ft: T = 218.8 K, a = 296.5 m/s, rho = 0.380 kg/m^3
(same cross-check as wp2/tests/test_config.py). Laminar fractions and
roughness: ADSEE-II slide tables.
"""

from __future__ import annotations

import pytest

from b12wp3.config import drag, geometry, mission


def test_isa_cruise_altitude_returns_reference_values() -> None:
    condition = mission.CRUISE

    result = (
        condition.temperature_k,
        condition.speed_of_sound_m_s,
        condition.density_kg_m3,
    )

    assert result == pytest.approx((218.8, 296.5, 0.380), rel=2e-3)


def test_cruise_condition_is_transonic_at_mach_077() -> None:
    condition = mission.CRUISE

    assert (condition.mach, condition.regime) == (0.77, mission.FlowRegime.TRANSONIC)


def test_flight_condition_zero_mach_raises() -> None:
    with pytest.raises(ValueError):
        mission.FlightCondition("bad", 0.0, 0.0, mission.FlowRegime.SUBSONIC)


def test_selected_laminar_fraction_is_civil_jet_classic_metal() -> None:
    fraction = drag.LAMINAR_FRACTION

    assert (fraction.fuselage, fraction.wing_and_tail) == (0.05, 0.10)


def test_selected_roughness_is_smooth_paint() -> None:
    assert drag.ROUGHNESS_M == pytest.approx(0.634e-5)


@pytest.mark.parametrize(("fuselage", "wing_and_tail"), [(-0.1, 0.1), (0.1, 1.1)])
def test_laminar_fraction_out_of_range_raises(
    fuselage: float, wing_and_tail: float
) -> None:
    with pytest.raises(ValueError):
        drag.LaminarFraction(fuselage, wing_and_tail)


def test_aircraft_geometry_unset_input_raises_with_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry, "FUSELAGE_DIAMETER_M", None)
    monkeypatch.setattr(geometry, "ENGINE_COUNT", None)

    with pytest.raises(ValueError) as error:
        geometry.aircraft_geometry()

    assert "FUSELAGE_DIAMETER_M" in str(error.value)
    assert "ENGINE_COUNT" in str(error.value)


def test_aircraft_geometry_all_inputs_set_returns_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: dict[str, float] = {}
    for surface in ("WING", "HORIZONTAL_TAIL", "VERTICAL_TAIL", "PYLON"):
        values |= {
            f"{surface}_AREA_M2": 10.0,
            f"{surface}_ASPECT_RATIO": 4.0,
            f"{surface}_TAPER": 0.5,
            f"{surface}_SWEEP_C4_DEG": 20.0,
            f"{surface}_THICKNESS_TO_CHORD": 0.1,
            f"{surface}_X_C_MAX_THICKNESS": 0.35,
            f"{surface}_COVERED_SPAN_M": 0.0,
        }
    for body in ("FUSELAGE", "NACELLE"):
        values |= {
            f"{body}_NOSE_LENGTH_M": 1.0,
            f"{body}_CYLINDER_LENGTH_M": 5.0,
            f"{body}_TAIL_LENGTH_M": 2.0,
            f"{body}_DIAMETER_M": 1.5,
        }
    for name, value in values.items():
        monkeypatch.setattr(geometry, name, value)
    monkeypatch.setattr(geometry, "ENGINE_COUNT", 2)

    result = geometry.aircraft_geometry()

    assert result.engine_count == 2
    assert result.fuselage.length_m == pytest.approx(8.0)
    assert result.vertical_tail.panels == 1
