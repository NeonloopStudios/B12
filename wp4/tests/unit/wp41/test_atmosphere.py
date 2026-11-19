# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""Tests of wp41.atmosphere against the ICAO Standard Atmosphere (Doc 7488) tables."""

from __future__ import annotations

import math

import pytest

from wp41.atmosphere import isa


@pytest.mark.parametrize(
    ("altitude", "temperature", "speed_of_sound", "density"),
    [
        (0.0, 288.15, 340.294, 1.22500),
        (5_000.0, 255.65, 320.529, 0.73612),
        (11_000.0, 216.65, 295.070, 0.36392),
        (35_000 * 0.3048, 218.81, 296.54, 0.37960),
        (15_000.0, 216.65, 295.070, 0.19367),  # geopotential altitude
    ],
)
def test_isa_table_points_match_icao(
    altitude: float, temperature: float, speed_of_sound: float, density: float
) -> None:
    t, a, rho = isa(altitude)

    assert t == pytest.approx(temperature, abs=0.01)
    assert a == pytest.approx(speed_of_sound, abs=0.01)
    assert rho == pytest.approx(density, abs=1e-4)


def test_isa_tropopause_is_continuous() -> None:
    below = isa(11_000.0 - 1e-6)
    above = isa(11_000.0 + 1e-6)

    assert below == pytest.approx(above, rel=1e-8)


@pytest.mark.parametrize("altitude", [-1.0, 20_001.0, math.nan])
def test_isa_altitude_out_of_range_raises(altitude: float) -> None:
    with pytest.raises(ValueError, match="altitude_m"):
        isa(altitude)
