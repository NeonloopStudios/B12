# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Tests for wp3/src/b12wp3/config/vsp_model.py: the VSP model YAML loader."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from b12wp3.config import vsp_model
from b12wp3.config.vsp_model import VspModelConfigError, VspModelInputs


def _write(path: Path, content: dict[str, Any]) -> Path:
    """Write a YAML file and return its path."""
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def _as_dict(inputs: VspModelInputs) -> dict[str, Any]:
    """The YAML content that loads to the given inputs."""
    return dataclasses.asdict(inputs)


def test_load_vsp_model_valid_file_returns_inputs(
    tmp_path: Path, sample_vsp_inputs: VspModelInputs
) -> None:
    path = _write(tmp_path / "vsp.yaml", _as_dict(sample_vsp_inputs))

    result = vsp_model.load_vsp_model(path)

    assert result == sample_vsp_inputs


def test_load_vsp_model_nulls_names_every_null_input(
    tmp_path: Path, sample_vsp_inputs: VspModelInputs
) -> None:
    content = _as_dict(sample_vsp_inputs)
    content["wing"]["root_le_x_m"] = None
    content["ground"]["fuselage_clearance_m"] = None
    path = _write(tmp_path / "vsp.yaml", content)

    with pytest.raises(VspModelConfigError) as info:
        vsp_model.load_vsp_model(path)

    assert "wing.root_le_x_m" in str(info.value)
    assert "ground.fuselage_clearance_m" in str(info.value)


def test_load_vsp_model_unknown_key_raises(
    tmp_path: Path, sample_vsp_inputs: VspModelInputs
) -> None:
    content = _as_dict(sample_vsp_inputs)
    content["pylon"]["colour"] = 3.0
    path = _write(tmp_path / "vsp.yaml", content)

    with pytest.raises(VspModelConfigError, match="unknown keys"):
        vsp_model.load_vsp_model(path)


def test_load_vsp_model_float_tire_count_raises(
    tmp_path: Path, sample_vsp_inputs: VspModelInputs
) -> None:
    content = _as_dict(sample_vsp_inputs)
    content["main_gear_position"]["tires_per_unit"] = 2.5
    path = _write(tmp_path / "vsp.yaml", content)

    with pytest.raises(VspModelConfigError, match="integer"):
        vsp_model.load_vsp_model(path)


def test_repo_vsp_model_yaml_only_positions_are_null() -> None:
    expected_nulls = {
        "wing.root_le_x_m",
        "wing.root_z_m",
        "centre_of_gravity.x_m",
        "nacelle_position.axis_y_m",
        "nacelle_position.inlet_ahead_of_le_m",
        "nacelle_position.axis_below_chord_m",
        "nose_gear_position.contact_x_m",
        "main_gear_position.contact_x_m",
        "main_gear_position.track_m",
        "ground.fuselage_clearance_m",
    }

    try:
        vsp_model.load_vsp_model()
    except VspModelConfigError as error:
        listed = set(str(error).split(": ", 1)[1].split(", "))
        assert listed <= expected_nulls
    else:
        pytest.skip("every position is set; nothing left to check")


@pytest.mark.parametrize(
    ("section", "values"),
    [
        ("centre_of_gravity", {"x_m": 0.0}),
        ("tails", {"horizontal_height_fraction": 1.5}),
        ("nacelle_detail", {"lip_length_fraction": 0.2, "fan_face_position": 0.1}),
        ("pylon", {"le_position": 0.5, "chord_fraction": 0.6}),
        ("main_gear_position", {"tires_per_unit": 0}),
        ("ground", {"fuselage_clearance_m": 0.0}),
        ("wing", {"incidence_deg": 20.0}),
    ],
)
def test_vsp_inputs_unphysical_value_raises(
    sample_vsp_inputs: VspModelInputs, section: str, values: dict[str, float]
) -> None:
    current = getattr(sample_vsp_inputs, section)

    with pytest.raises(ValueError):
        dataclasses.replace(current, **values)
