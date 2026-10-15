# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/propulsion/nacelle.py and config/propulsion.py.

Spec: Vos, "Airplane Design and Analysis", section 8.3, equations 8.22-8.30
and Table 8.1 (after Torenbeek). Reference values are evaluated by hand from
those equations for the GE CF34-8C5B1 inputs (T = 61.3 kN per engine, B = 5,
T_t4 = 1830 K, eta_noz = 0.98, eta_tf = 0.75) at the textbook sea-level
values a0 = 340 m/s, rho0 = 1.225 kg/m^3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from b12wp3.config import paths
from b12wp3.config.propulsion import EngineInputs, load_propulsion
from b12wp3.config.yaml_io import ConfigError
from b12wp3.plots.nacelle import plot_nacelle_outline
from b12wp3.propulsion.nacelle import (
    fan_exit_diameter,
    gas_generator_function,
    inlet_diameter,
    mass_flow,
    max_diameter,
    nacelle_length,
    size_type_b_nacelle,
    spinner_ratio,
)

A0 = 340.0
RHO0 = 1.225
MASS_FLOW_HAND = 167.12895992127298


def _cf34() -> EngineInputs:
    return EngineInputs(61300.0, 5.0, 1830.0, 0.98, 0.75)


def test_gas_generator_1830_k_returns_1_8() -> None:
    assert gas_generator_function(1830.0) == pytest.approx(1.8)


def test_gas_generator_at_750_k_raises() -> None:
    with pytest.raises(ValueError):
        gas_generator_function(750.0)


def test_mass_flow_cf34_returns_hand_value() -> None:
    flow = mass_flow(61300.0, A0, 5.0, 0.98, 1.8, 0.75)

    assert flow == pytest.approx(MASS_FLOW_HAND)


def test_mass_flow_turbojet_reduces_to_bypass_free_form() -> None:
    # B = 0: m_dot = T / (a0 sqrt(5 eta_noz G))
    flow = mass_flow(50000.0, A0, 0.0, 0.98, 1.8, 0.75)

    assert flow == pytest.approx(50000.0 / (A0 * (5.0 * 0.98 * 1.8) ** 0.5))


def test_inlet_and_length_cf34_return_hand_values() -> None:
    spinner = spinner_ratio(MASS_FLOW_HAND, RHO0, A0, 5.0)

    d_i = inlet_diameter(MASS_FLOW_HAND, RHO0, A0, spinner)
    l_n = nacelle_length(MASS_FLOW_HAND, RHO0, A0, 5.0, 9.8, 0.05)

    assert spinner == pytest.approx(0.18746043774209434)
    assert d_i == pytest.approx(1.0706795187389075)
    assert l_n == pytest.approx(4.07413038894084)


def test_type_b_diameters_return_hand_values() -> None:
    d_n = max_diameter(1.0706795187389075, 1.0, 4.07413038894084)

    d_ef = fan_exit_diameter(d_n, 1.0)

    assert d_n == pytest.approx(1.345127342075358)
    assert d_ef == pytest.approx(2.0 / 3.0 * d_n)


def test_size_type_b_nacelle_isa_matches_hand_values() -> None:
    # ISA a0 = 340.29 m/s vs the textbook 340 m/s: agreement within 0.2 %
    nacelle = size_type_b_nacelle(_cf34())

    assert nacelle.mass_flow_kg_s == pytest.approx(MASS_FLOW_HAND, rel=2e-3)
    assert nacelle.inlet_diameter_m == pytest.approx(1.0707, rel=2e-3)
    assert nacelle.length_m == pytest.approx(4.0741, rel=2e-3)
    assert nacelle.highlight_diameter_m == nacelle.inlet_diameter_m
    assert nacelle.fan_cowl_length_m == nacelle.length_m
    assert nacelle.max_diameter_position == 0.35


@pytest.mark.parametrize(
    "kwargs",
    [
        {"takeoff_thrust_per_engine_n": 0.0},
        {"bypass_ratio": -1.0},
        {"turbine_inlet_temperature_k": 700.0},
        {"nozzle_efficiency": 1.2},
        {"turbine_fan_efficiency": 0.0},
    ],
)
def test_engine_inputs_invalid_raises(kwargs: dict[str, float]) -> None:
    values = {
        "takeoff_thrust_per_engine_n": 61300.0,
        "bypass_ratio": 5.0,
        "turbine_inlet_temperature_k": 1830.0,
        "nozzle_efficiency": 0.98,
        "turbine_fan_efficiency": 0.75,
    } | kwargs

    with pytest.raises(ValueError):
        EngineInputs(**values)


def test_load_propulsion_repo_file_returns_cf34() -> None:
    inputs = load_propulsion()

    assert inputs.engine == _cf34()
    assert inputs.reference.fan_diameter_m == pytest.approx(1.17)


def test_load_propulsion_null_input_raises_with_its_name(tmp_path: Path) -> None:
    raw = yaml.safe_load(paths.PROPULSION_YAML.read_text(encoding="utf-8"))
    raw["engine"]["bypass_ratio"] = None
    path = tmp_path / "propulsion.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError) as error:
        load_propulsion(path)

    assert "engine.bypass_ratio" in str(error.value)


def test_plot_nacelle_outline_writes_png(tmp_path: Path) -> None:
    nacelle = size_type_b_nacelle(_cf34())

    out = plot_nacelle_outline(nacelle, tmp_path / "nacelle.png")

    assert out.is_file()
    assert out.stat().st_size > 0
