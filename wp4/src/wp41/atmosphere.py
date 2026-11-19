# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""ICAO standard atmosphere (ISA), 0 to 20 km, and standard gravity.

Same constants as wp2/src/b12wp2/config/atmosphere.py. Python 3.11-compatible
(vspaero_runner.py imports aero_loads, which imports this module).
"""

from __future__ import annotations

import math
from typing import Final

G0: Final[float] = 9.80665  # standard gravity [m/s^2]

ISA_T0_K: Final[float] = 288.15
ISA_P0_PA: Final[float] = 101_325.0
ISA_LAPSE_K_M: Final[float] = 0.0065
ISA_R_AIR: Final[float] = 287.05287
ISA_GAMMA: Final[float] = 1.4
ISA_H_TROPOPAUSE_M: Final[float] = 11_000.0
ISA_H_MAX_M: Final[float] = 20_000.0


def isa(altitude_m: float) -> tuple[float, float, float]:
    """ISA temperature [K], speed of sound [m/s] and density [kg/m^3].

    Troposphere: linear lapse rate. 11 to 20 km: isothermal at 216.65 K.

    Args:
        altitude_m: Geopotential altitude, 0 to 20 km [m].

    Returns:
        (temperature, speed of sound, density).

    Raises:
        ValueError: If the altitude is outside 0 to 20 km, or NaN.
    """
    if not 0.0 <= altitude_m <= ISA_H_MAX_M:
        raise ValueError(f"altitude_m must be in [0, {ISA_H_MAX_M}], got {altitude_m}")
    exponent = G0 / (ISA_LAPSE_K_M * ISA_R_AIR)
    t_tropopause = ISA_T0_K - ISA_LAPSE_K_M * ISA_H_TROPOPAUSE_M
    if altitude_m <= ISA_H_TROPOPAUSE_M:
        t = ISA_T0_K - ISA_LAPSE_K_M * altitude_m
        p = ISA_P0_PA * (t / ISA_T0_K) ** exponent
    else:
        t = t_tropopause
        p_tropopause = ISA_P0_PA * (t_tropopause / ISA_T0_K) ** exponent
        dh = altitude_m - ISA_H_TROPOPAUSE_M
        p = p_tropopause * math.exp(-G0 * dh / (ISA_R_AIR * t))
    rho = p / (ISA_R_AIR * t)
    a = math.sqrt(ISA_GAMMA * ISA_R_AIR * t)
    return t, a, rho
