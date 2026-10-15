# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""The WP3 flight conditions.

The cruise values are the same as wp2/src/b12wp2/config/mission.py
(35,000 ft, M = 0.77). Each condition also states its flow regime, which
selects the cutoff Reynolds number of the skin-friction method. The regime
is set here explicitly, never inferred from the Mach number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from b12wp3.config.atmosphere import isa_atmosphere, sutherland_viscosity

FT_TO_M: Final[float] = 0.3048


class FlowRegime(Enum):
    """Flow regime that selects the cutoff Reynolds number (ADSEE-II)."""

    SUBSONIC = "subsonic"
    TRANSONIC = "transonic"


@dataclass(frozen=True)
class FlightCondition:
    """One flight condition and the air properties at it.

    Attributes:
        name: Label used in result files.
        altitude_m: Geopotential altitude [m].
        mach: Free-stream Mach number [-].
        regime: Flow regime for the cutoff Reynolds number.
    """

    name: str
    altitude_m: float
    mach: float
    regime: FlowRegime

    def __post_init__(self) -> None:
        if not math.isfinite(self.mach) or self.mach <= 0.0:
            raise ValueError(
                f"FlightCondition({self.name!r}): mach must be > 0, got {self.mach}"
            )
        isa_atmosphere(self.altitude_m)  # raises for an altitude outside 0-20 km

    @property
    def temperature_k(self) -> float:
        """Static temperature [K]."""
        return isa_atmosphere(self.altitude_m)[0]

    @property
    def speed_of_sound_m_s(self) -> float:
        """Speed of sound [m/s]."""
        return isa_atmosphere(self.altitude_m)[1]

    @property
    def density_kg_m3(self) -> float:
        """Air density [kg/m^3]."""
        return isa_atmosphere(self.altitude_m)[2]

    @property
    def velocity_m_s(self) -> float:
        """True airspeed [m/s]."""
        return self.mach * self.speed_of_sound_m_s

    @property
    def viscosity_pa_s(self) -> float:
        """Dynamic viscosity [Pa s]."""
        return sutherland_viscosity(self.temperature_k)


# Cruise: M = 0.77 at 35,000 ft. The transonic cutoff Reynolds number is used,
# because M = 0.77 is in the transonic range of the swept wing.
CRUISE: Final[FlightCondition] = FlightCondition(
    name="cruise",
    altitude_m=35_000 * FT_TO_M,
    mach=0.77,
    regime=FlowRegime.TRANSONIC,
)
