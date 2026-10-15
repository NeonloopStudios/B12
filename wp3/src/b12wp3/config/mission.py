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

# Landing: sea level, 65 m/s approach speed, as in WP2 (config/mission.py).
LANDING_SPEED_M_S: Final[float] = 65.0
LANDING: Final[FlightCondition] = FlightCondition(
    name="landing",
    altitude_m=0.0,
    mach=LANDING_SPEED_M_S / isa_atmosphere(0.0)[1],
    regime=FlowRegime.SUBSONIC,
)


# ============================================================
#  DRAG CONFIGURATIONS
# ============================================================


@dataclass(frozen=True)
class DragConfiguration:
    """One aircraft configuration for the drag build-up.

    Attributes:
        name: Label used in result files.
        condition: Flight condition.
        lift_coefficient: Wing C_L for the drag-divergence Mach number [-].
            Only needed if M >= M_crit; None otherwise.
        te_flap_deflection_deg: Trailing-edge flap deflection [deg].
        le_flap_deflection_deg: Leading-edge flap deflection [deg].
        gear_down: True if the landing gear is extended.
    """

    name: str
    condition: FlightCondition
    lift_coefficient: float | None
    te_flap_deflection_deg: float
    le_flap_deflection_deg: float
    gear_down: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("te_flap_deflection_deg", self.te_flap_deflection_deg),
            ("le_flap_deflection_deg", self.le_flap_deflection_deg),
        ):
            if not 0.0 <= value < 90.0:
                raise ValueError(
                    f"DragConfiguration({self.name!r}): {label} must be in [0, 90), "
                    f"got {value}"
                )


# Cruise C_L: level-flight trim at the WP1 cruise weight and the WP2 wing area
# (wp2/results/aero_summary_3d.csv, CL_design).
CRUISE_LIFT_COEFFICIENT: Final[float] = 0.489433403

CRUISE_CONFIGURATION: Final[DragConfiguration] = DragConfiguration(
    name="cruise",
    condition=CRUISE,
    lift_coefficient=CRUISE_LIFT_COEFFICIENT,
    te_flap_deflection_deg=0.0,
    le_flap_deflection_deg=0.0,
    gear_down=False,
)

# --- take-off and landing inputs, not set yet ---
TAKEOFF_SPEED_M_S: Final[float | None] = None  # sea level
TAKEOFF_TE_FLAP_DEFLECTION_DEG: Final[float | None] = None
TAKEOFF_LE_FLAP_DEFLECTION_DEG: Final[float | None] = None
LANDING_TE_FLAP_DEFLECTION_DEG: Final[float | None] = None
LANDING_LE_FLAP_DEFLECTION_DEG: Final[float | None] = None


def _raise_if_unset(
    configuration: str, named: tuple[tuple[str, float | None], ...]
) -> None:
    """Raise one error that names every unset input of a configuration."""
    missing = [name for name, value in named if value is None]
    if missing:
        raise ValueError(
            f"{configuration} inputs not set in b12wp3/config/mission.py: "
            + ", ".join(missing)
        )


def takeoff_configuration() -> DragConfiguration:
    """Take-off configuration: sea level, take-off flaps, gear down.

    Raises:
        ValueError: If a take-off input is None. The message names all of them.
    """
    speed = TAKEOFF_SPEED_M_S
    te = TAKEOFF_TE_FLAP_DEFLECTION_DEG
    le = TAKEOFF_LE_FLAP_DEFLECTION_DEG
    _raise_if_unset(
        "take-off",
        (
            ("TAKEOFF_SPEED_M_S", speed),
            ("TAKEOFF_TE_FLAP_DEFLECTION_DEG", te),
            ("TAKEOFF_LE_FLAP_DEFLECTION_DEG", le),
        ),
    )
    if speed is None or te is None or le is None:
        raise AssertionError("unreachable: _raise_if_unset checked the inputs")
    condition = FlightCondition(
        name="takeoff",
        altitude_m=0.0,
        mach=speed / isa_atmosphere(0.0)[1],
        regime=FlowRegime.SUBSONIC,
    )
    return DragConfiguration("takeoff", condition, None, te, le, gear_down=True)


def landing_configuration() -> DragConfiguration:
    """Landing configuration: sea level, 65 m/s, landing flaps, gear down.

    Raises:
        ValueError: If a landing input is None. The message names all of them.
    """
    te = LANDING_TE_FLAP_DEFLECTION_DEG
    le = LANDING_LE_FLAP_DEFLECTION_DEG
    _raise_if_unset(
        "landing",
        (
            ("LANDING_TE_FLAP_DEFLECTION_DEG", te),
            ("LANDING_LE_FLAP_DEFLECTION_DEG", le),
        ),
    )
    if te is None or le is None:
        raise AssertionError("unreachable: _raise_if_unset checked the inputs")
    return DragConfiguration("landing", LANDING, None, te, le, gear_down=True)
