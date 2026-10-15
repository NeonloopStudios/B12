# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""ISA standard atmosphere and air properties.

Copied unchanged from wp2/src/b12wp2/config/atmosphere.py, so that wp3 does
not import the wp2 package. Physics constants and relations, not project
settings.
"""

from __future__ import annotations

import math
from typing import Final

# Troposphere (h <= 11 km): linear lapse rate.
# Lower stratosphere (11 km < h <= 20 km): isothermal at 216.65 K.
# Source: ICAO Standard Atmosphere.

_T0: Final[float] = 288.15  # K, sea-level standard temperature
_P0: Final[float] = 101_325.0  # Pa, sea-level standard pressure
_L: Final[float] = 0.0065  # K/m, tropospheric lapse rate
_R_AIR: Final[float] = 287.05287  # J/(kg K), specific gas constant for dry air
_GAMMA: Final[float] = 1.4  # ratio of specific heats for air
_G0: Final[float] = 9.80665  # m/s^2, standard gravity
_H_TROPOPAUSE: Final[float] = 11_000.0  # m
_T_TROPOPAUSE: Final[float] = _T0 - _L * _H_TROPOPAUSE  # 216.65 K
_H_MAX: Final[float] = 20_000.0  # m, top of the implemented range


def isa_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """ISA temperature, speed of sound and density at a geopotential altitude.

    Args:
        altitude_m: Geopotential altitude [m], valid 0-20 km.

    Returns:
        (temperature [K], speed of sound [m/s], density [kg/m^3]).

    Raises:
        ValueError: If the altitude is outside 0-20 km.
    """
    if altitude_m < 0.0:
        raise ValueError(f"altitude_m must be >= 0, got {altitude_m}")
    if altitude_m <= _H_TROPOPAUSE:
        t = _T0 - _L * altitude_m
        p = _P0 * (t / _T0) ** (_G0 / (_L * _R_AIR))
    elif altitude_m <= _H_MAX:
        t = _T_TROPOPAUSE
        p_tropopause = _P0 * (_T_TROPOPAUSE / _T0) ** (_G0 / (_L * _R_AIR))
        p = p_tropopause * math.exp(-_G0 * (altitude_m - _H_TROPOPAUSE) / (_R_AIR * t))
    else:
        raise ValueError(
            f"altitude_m={altitude_m} is above the validated ISA range (0-20 km)"
        )
    rho = p / (_R_AIR * t)
    a = math.sqrt(_GAMMA * _R_AIR * t)
    return t, a, rho


def sutherland_viscosity(temperature_k: float) -> float:
    """Dynamic viscosity of air by Sutherland's law.

    Law: Sutherland, Phil. Mag. 5th ser. 36 (1893) 507-531, eq. (4).
    S = 110.4 K: ISO 2533:1975, Standard Atmosphere.

    Args:
        temperature_k: Static temperature [K].

    Returns:
        Dynamic viscosity [Pa s].
    """
    mu0, t0, s = 1.716e-5, 273.15, 110.4
    temperature_ratio: float = (temperature_k / t0) ** 1.5
    return mu0 * temperature_ratio * (t0 + s) / (temperature_k + s)
