# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-19
"""Tests of wp43.atmosphere. Reference values: ICAO Standard Atmosphere (Doc 7488)."""

from __future__ import annotations

import math

import pytest

from wp43.atmosphere import ISA_RHO0_KG_M3, isa


@pytest.mark.parametrize(
    ("altitude", "temperature", "pressure", "density", "speed_of_sound"),
    [
        (0.0, 288.15, 101_325.0, 1.22500, 340.294),
        (5_000.0, 255.65, 54_019.9, 0.73612, 320.529),
        (11_000.0, 216.65, 22_632.1, 0.36392, 295.070),
        (35_000 * 0.3048, 218.81, 23_842.3, 0.37960, 296.54),
        (15_000.0, 216.65, 12_044.6, 0.19367, 295.070),
    ],
)
def test_isa_table_points_match_icao(
    altitude: float,
    temperature: float,
    pressure: float,
    density: float,
    speed_of_sound: float,
) -> None:
    state = isa(altitude)

    assert state.altitude_m == altitude
    assert state.temperature_k == pytest.approx(temperature, abs=0.01)
    assert state.pressure_pa == pytest.approx(pressure, rel=1e-4)
    assert state.density_kg_m3 == pytest.approx(density, abs=1e-4)
    assert state.speed_of_sound_ms == pytest.approx(speed_of_sound, abs=0.01)


def test_isa_tropopause_is_continuous() -> None:
    below = isa(11_000.0 - 1e-6)
    above = isa(11_000.0 + 1e-6)

    assert below.temperature_k == pytest.approx(above.temperature_k, rel=1e-8)
    assert below.pressure_pa == pytest.approx(above.pressure_pa, rel=1e-8)
    assert below.density_kg_m3 == pytest.approx(above.density_kg_m3, rel=1e-8)


def test_isa_sea_level_density_matches_rho0() -> None:
    state = isa(0.0)

    assert state.density_kg_m3 == pytest.approx(ISA_RHO0_KG_M3, rel=1e-5)
    assert state.sigma == pytest.approx(1.0, rel=1e-5)


def test_isa_sigma_at_cruise_converts_eas_to_mach() -> None:
    # V_C = 127.157 m/s EAS at 35 000 ft is M 0.77 (WP4.3 cruise condition).
    state = isa(35_000 * 0.3048)

    mach = 127.157 / math.sqrt(state.sigma) / state.speed_of_sound_ms

    assert mach == pytest.approx(0.77, abs=1e-3)


def test_isa_density_decreases_with_altitude() -> None:
    densities = [isa(h).density_kg_m3 for h in (0.0, 3_000.0, 9_000.0, 13_000.0, 19_000.0)]

    assert all(lo > hi for lo, hi in zip(densities, densities[1:]))


@pytest.mark.parametrize("altitude", [-1.0, 20_001.0, math.nan])
def test_isa_altitude_out_of_range_raises(altitude: float) -> None:
    with pytest.raises(ValueError, match="altitude_m"):
        isa(altitude)
