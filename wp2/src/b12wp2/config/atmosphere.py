# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""ISA standard atmosphere and air properties.

Physics constants and relations, not project settings -- they would be the
same numbers in any other analysis. Kept inside the config package because
every flight condition in mission.py is built on them.
"""
from __future__ import annotations

import math

# Troposphere (h <= 11 km): linear lapse rate.
# Lower stratosphere (11 km < h <= 20 km): isothermal at 216.65 K.
# Source: ICAO Standard Atmosphere. Constants below are the standard ISA
# values, not fitted to this project.

_T0 = 288.15  # K, sea-level standard temperature
_P0 = 101_325.0  # Pa, sea-level standard pressure
_L = 0.0065  # K/m, tropospheric lapse rate
_R_AIR = 287.05287  # J/(kg K), specific gas constant for dry air
_GAMMA = 1.4  # ratio of specific heats for air
_G0 = 9.80665  # m/s^2, standard gravity
_H_TROPOPAUSE = 11_000.0  # m
_T_TROPOPAUSE = _T0 - _L * _H_TROPOPAUSE  # 216.65 K


def isa_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """Return (temperature K, speed of sound m/s, density kg/m^3) at a given
    geopotential altitude, ISA standard atmosphere, valid 0-20 km.

    Cross-checked in wp2/tests/test_config.py against the hand-derived
    values at 35,000 ft (T ~ 218.8 K, a ~ 296.5 m/s, rho ~ 0.380 kg/m^3).
    """
    if altitude_m < 0.0:
        raise ValueError(f"altitude_m must be >= 0, got {altitude_m}")
    if altitude_m <= _H_TROPOPAUSE:
        t = _T0 - _L * altitude_m
        p = _P0 * (t / _T0) ** (_G0 / (_L * _R_AIR))
    elif altitude_m <= 20_000.0:
        t = _T_TROPOPAUSE
        p_tropopause = _P0 * (_T_TROPOPAUSE / _T0) ** (_G0 / (_L * _R_AIR))
        p = p_tropopause * math.exp(-_G0 * (altitude_m - _H_TROPOPAUSE) / (_R_AIR * t))
    else:
        raise ValueError(
            f"altitude_m={altitude_m} is above the validated ISA range (0-20 km) "
            "for this implementation"
        )
    rho = p / (_R_AIR * t)
    a = math.sqrt(_GAMMA * _R_AIR * t)
    return t, a, rho


def sutherland_viscosity(temperature_k: float) -> float:
    """Dynamic viscosity of air (Pa s) via Sutherland's law, for Reynolds
    number. Constants are the standard values for air (mu0 at T0=273.15 K).
    """
    mu0, t0, s = 1.716e-5, 273.15, 110.4
    return mu0 * (temperature_k / t0) ** 1.5 * (t0 + s) / (temperature_k + s)
