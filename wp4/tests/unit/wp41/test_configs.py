# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of configs.wp41.load_cases and configs.wp41.torsion."""

from __future__ import annotations

import pytest

from configs.wp41.load_cases import LoadCase, get_case
from configs.wp41.torsion import TORSION_INPUTS, TorsionInputs
from configs.wp41.weight import WEIGHT_INPUTS, WeightInputs

CASE = LoadCase(
    name="LC-T", n=2.3, v_eas=187.3, altitude_m=3_050.0, mass_case="mtow",
    thrust_n=18_200.0,
)


def test_get_case_known_name_returns_the_case() -> None:
    assert get_case("LC-T", (CASE,)) is CASE


def test_get_case_unknown_name_raises_with_known_names() -> None:
    with pytest.raises(KeyError, match="known: \\['LC-T'\\]"):
        get_case("LC-9", (CASE,))


@pytest.mark.parametrize(
    ("field", "value"),
    [("n", 0.0), ("n", float("nan")), ("v_eas", 0.0), ("altitude_m", 25_000.0),
     ("thrust_n", -5.0), ("name", "a/b")],
)
def test_load_case_invalid_field_raises(field: str, value: object) -> None:
    fields: dict[str, object] = {
        "name": "LC-T", "n": 2.3, "v_eas": 187.3, "altitude_m": 3_050.0,
        "mass_case": "mtow", "thrust_n": 18_200.0,
    }
    fields[field] = value

    with pytest.raises(ValueError, match=field):
        LoadCase(**fields)  # type: ignore[arg-type]


def test_torsion_inputs_f_sc_unset_raises_naming_wp42() -> None:
    inputs = TorsionInputs(f_sc=None, h_thrust_m=0.83)

    with pytest.raises(ValueError, match="ask WP4.2"):
        inputs.require_f_sc()


def test_torsion_inputs_f_sc_set_returns_value() -> None:
    assert TorsionInputs(f_sc=0.413, h_thrust_m=0.83).require_f_sc() == 0.413


@pytest.mark.parametrize("f_sc", [-0.1, 1.2, float("nan")])
def test_torsion_inputs_f_sc_outside_unit_interval_raises(f_sc: float) -> None:
    with pytest.raises(ValueError, match="f_sc"):
        TorsionInputs(f_sc=f_sc, h_thrust_m=0.83)


def test_torsion_inputs_design_values_are_set_and_inside_the_chord() -> None:
    assert TORSION_INPUTS.require_f_sc() == pytest.approx(0.4337186)
    assert 0.0 < TORSION_INPUTS.h_thrust_m < 0.8


@pytest.mark.parametrize(
    ("field", "value"),
    [("f_str", 1.2), ("f_fuel", -0.1), ("engine_cg_fraction", float("nan")),
     ("engine_inlet_ahead_of_le_m", 0.0), ("nacelle_length_m", float("inf"))],
)
def test_weight_inputs_invalid_field_raises(field: str, value: float) -> None:
    fields = {
        "f_str": 0.43, "f_fuel": 0.47, "engine_inlet_ahead_of_le_m": 2.5,
        "nacelle_length_m": 4.0, "engine_cg_fraction": 0.35,
    }
    fields[field] = value

    with pytest.raises(ValueError, match=field):
        WeightInputs(**fields)


def test_weight_inputs_design_values_put_the_engine_cg_ahead_of_the_wing_box() -> None:
    inputs = WEIGHT_INPUTS

    assert inputs.f_str == inputs.f_fuel
    assert 0.0 < inputs.engine_cg_fraction < 1.0
