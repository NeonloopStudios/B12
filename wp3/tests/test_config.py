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

from b12wp3.config import drag, drag_tables, geometry, mission


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
        drag_tables.LaminarFraction(fuselage, wing_and_tail)


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
    values |= {"FUSELAGE_UPSWEEP_DEG": 5.0, "FUSELAGE_BASE_AREA_M2": 0.0}
    for gear in ("MAIN_GEAR", "NOSE_GEAR"):
        values |= {
            f"{gear}_FRONTAL_AREA_M2": 0.3,
            f"{gear}_TIRE_WIDTH_M": 0.4,
            f"{gear}_TIRE_DIAMETER_M": 1.0,
        }
    for name, value in values.items():
        monkeypatch.setattr(geometry, name, value)
    for name in ("ENGINE_COUNT", "MAIN_GEAR_COUNT", "NOSE_GEAR_COUNT"):
        monkeypatch.setattr(geometry, name, 2)

    result = geometry.aircraft_geometry()

    assert result.engine_count == 2
    assert result.fuselage.length_m == pytest.approx(8.0)
    assert result.vertical_tail.panels == 1


def test_interference_table_has_every_slide_row() -> None:
    table = drag_tables.INTERFERENCE_TABLE

    assert len(table) == 20
    assert table[drag_tables.InterferenceCase.CRUCIFORM_TAIL].low == 1.06
    assert table[drag_tables.InterferenceCase.UNFILLETED_LOW_WING] == (
        drag_tables.ValueRange(1.1, 1.4)
    )


def test_selected_interference_factors_match_choices() -> None:
    factors = drag.DRAG_SETTINGS.interference

    result = (
        factors.wing,
        factors.horizontal_tail,
        factors.vertical_tail,
        factors.fuselage,
        factors.nacelle,
        factors.pylon,
    )

    assert result == (1.4, 1.06, 1.06, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("case", "value"),
    [
        (drag_tables.InterferenceCase.UNFILLETED_LOW_WING, None),
        (drag_tables.InterferenceCase.UNFILLETED_LOW_WING, 1.5),
        (drag_tables.InterferenceCase.CRUCIFORM_TAIL, 1.04),
        (None, None),
        (None, 0.0),
    ],
)
def test_interference_choice_outside_table_raises(
    case: drag_tables.InterferenceCase | None, value: float | None
) -> None:
    with pytest.raises(ValueError):
        drag_tables.InterferenceChoice(case, value)


def test_selected_drag_settings_match_decisions() -> None:
    settings = drag.DRAG_SETTINGS

    result = (
        settings.kappa_a,
        settings.wing_mach_critical_normal,
        settings.te_flap_factor,
        settings.le_flap_factor,
        settings.wheel_well_coeff,
        settings.excrescence_fraction,
    )

    assert result == (0.95, 0.56, 0.0074, 0.0144, 0.05328, 0.035)


def test_excrescence_fraction_outside_jet_range_raises() -> None:
    jet = drag_tables.EXCRESCENCE_FRACTION_TABLE[
        drag_tables.ExcrescenceCategory.JET_TRANSPORT
    ]

    with pytest.raises(ValueError):
        jet.select(0.06)


def test_landing_configuration_unset_deflection_raises_with_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mission, "LANDING_TE_FLAP_DEFLECTION_DEG", None)

    with pytest.raises(ValueError) as error:
        mission.landing_configuration()

    assert "LANDING_TE_FLAP_DEFLECTION_DEG" in str(error.value)


def test_takeoff_configuration_all_inputs_set_returns_gear_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mission, "TAKEOFF_SPEED_M_S", 75.0)
    monkeypatch.setattr(mission, "TAKEOFF_TE_FLAP_DEFLECTION_DEG", 15.0)
    monkeypatch.setattr(mission, "TAKEOFF_LE_FLAP_DEFLECTION_DEG", 10.0)

    configuration = mission.takeoff_configuration()

    assert configuration.gear_down
    assert configuration.condition.velocity_m_s == pytest.approx(75.0)
