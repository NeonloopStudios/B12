# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/config/.

ISA reference at 35,000 ft: T = 218.8 K, a = 296.5 m/s, rho = 0.380 kg/m^3
(same cross-check as wp2/tests/test_config.py). Laminar fractions and
roughness: ADSEE-II slide tables.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from b12wp3.config import drag, drag_tables, geometry, mission, paths


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


def _filled_geometry_dict() -> dict[str, Any]:
    """The real geometry YAML with every null replaced by a valid test value."""
    raw: dict[str, Any] = yaml.safe_load(
        paths.GEOMETRY_YAML.read_text(encoding="utf-8")
    )
    fills = {
        "area_m2": 10.0,
        "aspect_ratio": 4.0,
        "taper": 0.5,
        "sweep_c4_deg": 20.0,
        "thickness_to_chord": 0.1,
        "x_c_max_thickness": 0.35,
        "covered_span_m": 0.0,
        "nose_length_m": 1.0,
        "cylinder_length_m": 5.0,
        "tail_length_m": 2.0,
        "diameter_m": 1.5,
        "upsweep_deg": 5.0,
        "base_area_m2": 0.0,
        "count": 2,
        "length_m": 4.0,
        "highlight_diameter_m": 1.07,
        "max_diameter_m": 1.34,
        "max_diameter_position": 0.35,
        "exit_diameter_m": 0.9,
        "frontal_area_m2": 0.3,
        "tire_width_m": 0.4,
        "tire_diameter_m": 1.0,
    }
    for section in raw.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if value is None:
                    section[key] = fills[key]
    raw["engine_count"] = 2
    return raw


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_load_geometry_null_inputs_raise_with_their_names(tmp_path: Path) -> None:
    raw = _filled_geometry_dict()
    raw["wing"]["covered_span_m"] = None
    raw["fuselage"]["diameter_m"] = None
    raw["engine_count"] = None
    path = _write(tmp_path, raw)

    with pytest.raises(geometry.GeometryConfigError) as error:
        geometry.load_geometry(path)

    message = str(error.value)
    assert "wing.covered_span_m" in message
    assert "fuselage.diameter_m" in message
    assert "engine_count" in message
    assert "wing.area_m2" not in message


def test_load_geometry_repo_file_has_valid_structure() -> None:
    # the repo file may still hold nulls, but never a structure or type error
    try:
        geometry.load_geometry()
    except geometry.GeometryConfigError as error:
        assert "not set (null)" in str(error)


def test_load_geometry_all_inputs_set_returns_geometry(tmp_path: Path) -> None:
    path = _write(tmp_path, _filled_geometry_dict())

    result = geometry.load_geometry(path)

    assert result.engine_count == 2
    assert result.wing.area_m2 == pytest.approx(66.7)
    assert result.fuselage.length_m == pytest.approx(28.68125)
    assert result.vertical_tail.panels == 1
    assert result.te_flap.chord_ratio == pytest.approx(0.30)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["wing"].update(span_m=30.0),
        lambda raw: raw["wing"].pop("taper"),
        lambda raw: raw.pop("nose_gear"),
        lambda raw: raw.update(extra_section={}),
        lambda raw: raw["wing"].update(taper="0.4"),
        lambda raw: raw["main_gear"].update(count=2.5),
        lambda raw: raw.update(engine_count=True),
    ],
)
def test_load_geometry_bad_structure_or_type_raises(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object]
) -> None:
    raw = _filled_geometry_dict()
    mutate(raw)
    path = _write(tmp_path, raw)

    with pytest.raises(geometry.GeometryConfigError):
        geometry.load_geometry(path)


def test_load_geometry_unphysical_value_raises(tmp_path: Path) -> None:
    raw = _filled_geometry_dict()
    raw["fuselage"]["diameter_m"] = -1.0
    path = _write(tmp_path, raw)

    with pytest.raises(ValueError):
        geometry.load_geometry(path)


def test_load_geometry_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(geometry.GeometryConfigError):
        geometry.load_geometry(tmp_path / "absent.yaml")


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
