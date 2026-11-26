# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of configs.wp41.load_cases and configs.wp41.torsion."""

from __future__ import annotations

import pytest

from configs.wp41 import settings
from configs.wp41.load_cases import LOAD_CASES, LoadCase, get_case
from configs.wp41.torsion import TORSION_INPUTS, TorsionInputs
from configs.wp41.weight import WEIGHT_INPUTS, WeightInputs
from wp41.lift_distribution import flight_state

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
    [("f_fuel", -0.1), ("f_fuel", 1.3), ("engine_cg_fraction", float("nan")),
     ("engine_inlet_ahead_of_le_m", 0.0), ("nacelle_length_m", float("inf"))],
)
def test_weight_inputs_invalid_field_raises(field: str, value: float) -> None:
    fields = {
        "f_fuel": 0.47, "engine_inlet_ahead_of_le_m": 2.5,
        "nacelle_length_m": 4.0, "engine_cg_fraction": 0.35,
    }
    fields[field] = value

    with pytest.raises(ValueError, match=field):
        WeightInputs(**fields)


def test_weight_inputs_design_values_put_the_engine_cg_ahead_of_the_wing_box() -> None:
    inputs = WEIGHT_INPUTS

    assert inputs.f_fuel == pytest.approx(0.4337186)
    assert 0.0 < inputs.engine_cg_fraction < 1.0


def test_load_cases_hold_a_positive_and_a_negative_case_inside_the_mach_grid() -> None:
    signs = sorted(case.n > 0.0 for case in LOAD_CASES)
    machs = [flight_state(c.v_eas, c.altitude_m).mach for c in LOAD_CASES]

    assert signs == [False, True]
    low, high = settings.VSPAERO.mach_start, settings.VSPAERO.mach_end
    assert all(low <= m <= high for m in machs)
    assert machs == pytest.approx([0.4671, 0.7703], abs=5e-4)
