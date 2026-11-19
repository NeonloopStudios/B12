# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-19
"""ISA atmosphere, 0 to 20 km (ICAO Standard Atmosphere, Doc 7488).

Troposphere (h <= 11 km): T = T0 - L h, p = p0 (T / T0)^(g0 / (L R)).
Lower stratosphere (11-20 km): T = T11, p = p11 exp(-g0 (h - 11 km) / (R T11)).
Density from the ideal gas law, rho = p / (R T); speed of sound a = sqrt(gamma R T).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

ISA_T0_K: Final[float] = 288.15
ISA_P0_PA: Final[float] = 101_325.0
ISA_RHO0_KG_M3: Final[float] = 1.225
ISA_LAPSE_K_M: Final[float] = 0.0065
ISA_R_AIR: Final[float] = 287.05287
ISA_GAMMA: Final[float] = 1.4
ISA_G0: Final[float] = 9.80665
ISA_H_TROPOPAUSE_M: Final[float] = 11_000.0
ISA_H_MAX_M: Final[float] = 20_000.0


@dataclass(frozen=True)
class AtmosphereState:
    """ISA properties at one altitude.

    Attributes:
        altitude_m: Geopotential altitude [m].
        temperature_k: Static temperature [K].
        pressure_pa: Static pressure [Pa].
        density_kg_m3: Density [kg/m^3].
        speed_of_sound_ms: Speed of sound [m/s].
    """

    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_ms: float

    @property
    def sigma(self) -> float:
        """Density ratio rho / rho0 [-], used to convert EAS and TAS."""
        return self.density_kg_m3 / ISA_RHO0_KG_M3


def isa(altitude_m: float) -> AtmosphereState:
    """ISA state at a geopotential altitude.

    Args:
        altitude_m: Geopotential altitude, 0 to 20 km [m].

    Returns:
        Temperature, pressure, density and speed of sound at that altitude.

    Raises:
        ValueError: If the altitude is outside 0 to 20 km or is NaN.
    """
    if not 0.0 <= altitude_m <= ISA_H_MAX_M:
        raise ValueError(f"altitude_m must be in [0, {ISA_H_MAX_M}], got {altitude_m}")
    exponent = ISA_G0 / (ISA_LAPSE_K_M * ISA_R_AIR)
    if altitude_m <= ISA_H_TROPOPAUSE_M:
        t = ISA_T0_K - ISA_LAPSE_K_M * altitude_m
        p = ISA_P0_PA * (t / ISA_T0_K) ** exponent
    else:
        t = ISA_T0_K - ISA_LAPSE_K_M * ISA_H_TROPOPAUSE_M
        p_tropopause = ISA_P0_PA * (t / ISA_T0_K) ** exponent
        dh = altitude_m - ISA_H_TROPOPAUSE_M
        p = p_tropopause * math.exp(-ISA_G0 * dh / (ISA_R_AIR * t))
    return AtmosphereState(
        altitude_m=altitude_m,
        temperature_k=t,
        pressure_pa=p,
        density_kg_m3=p / (ISA_R_AIR * t),
        speed_of_sound_ms=math.sqrt(ISA_GAMMA * ISA_R_AIR * t),
    )
