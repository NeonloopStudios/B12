# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Tests of configs.wp41.load_cases and configs.wp41.torsion."""

from __future__ import annotations

import pytest

from configs.wp41.load_cases import LoadCase, get_case
from configs.wp41.torsion import TorsionInputs

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
