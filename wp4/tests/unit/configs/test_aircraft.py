# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""Tests of configs.aircraft.AircraftMasses: derived masses and validation."""

from __future__ import annotations

import dataclasses
import math

import pytest

from configs.aircraft import MASSES, AircraftMasses

SAMPLE = AircraftMasses(
    oem_kg=16_431.0, design_payload_kg=6_870.0, max_payload_kg=8_915.0,
    fuel_kg=7_213.0, engine_count=2, engine_dry_kg=1_017.0, engine_installed_kg=1_806.0,
)


def test_masses_sample_returns_mtom_and_mzfm() -> None:
    assert SAMPLE.mtom_kg == pytest.approx(16_431.0 + 6_870.0 + 7_213.0)
    assert SAMPLE.mzfm_kg == pytest.approx(16_431.0 + 8_915.0)


def test_masses_design_engines_are_a_part_of_oem() -> None:
    engines = MASSES.engine_count * MASSES.engine_installed_kg

    assert engines == pytest.approx(3_866.76)
    assert MASSES.engine_dry_kg < MASSES.engine_installed_kg
    assert engines < MASSES.oem_kg


@pytest.mark.parametrize(
    "change",
    [
        {"engine_dry_kg": 0.0},
        {"engine_dry_kg": math.nan},
        {"engine_installed_kg": 900.0},  # below the dry mass
        {"engine_installed_kg": 9_000.0},  # two engines above the OEM
        {"engine_count": 0},
        {"engine_count": 3},
        {"oem_kg": -1.0},
        {"max_payload_kg": 5_000.0},  # below the design payload
    ],
)
def test_masses_invalid_value_raises(change: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SAMPLE, **change)  # type: ignore[arg-type]
