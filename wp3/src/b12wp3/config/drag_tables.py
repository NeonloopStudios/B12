# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Method tables of the ADSEE-II class II drag estimation.

Every table here is copied from the ADSEE-II slides. Nothing in this module
selects a row; the selected rows are in b12wp3.config.drag.

    LAMINAR_FRACTION_TABLE      "Flat plate skin friction coefficient"
    ROUGHNESS_TABLE_M           "Flat plate skin friction coefficient"
    INTERFERENCE_TABLE          "Component interference factor"
    FLAP_FACTOR_TABLE           "Miscellaneous drag - flap"
    WHEEL_WELL_COEFF_TABLE      "Miscellaneous drag - landing gear"
    EXCRESCENCE_FRACTION_TABLE  "Miscellaneous drag - Excrescence and leakage"
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True)
class ValueRange:
    """A table entry that is one value (low == high) or a range.

    Attributes:
        low: Lower bound.
        high: Upper bound.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"ValueRange: low {self.low} > high {self.high}")

    @property
    def is_single(self) -> bool:
        """True if the entry is one value."""
        return self.low == self.high

    def select(self, value: float | None) -> float:
        """Validated value from this entry.

        Args:
            value: The chosen value, or None for a single-value entry.

        Returns:
            The table value of a single-value entry, or the chosen value.

        Raises:
            ValueError: If a range entry has no chosen value, or the chosen
                value is outside the entry.
        """
        if value is None:
            if not self.is_single:
                raise ValueError(
                    f"table range [{self.low}, {self.high}] needs a chosen value"
                )
            return self.low
        if not self.low <= value <= self.high:
            raise ValueError(
                f"value {value} is outside the table range [{self.low}, {self.high}]"
            )
        return value


# ============================================================
#  SKIN FRICTION
# ============================================================


class AircraftCategory(Enum):
    """Rows of the laminar-flow table."""

    GA_CLASSIC_METAL = "General aviation - classic production metal"
    GA_SMOOTH_METAL = "General aviation - smooth metal (no rivets or cracks)"
    GA_SMOOTH_COMPOSITE = "General aviation - smooth molded composites"
    SAILPLANE_SMOOTH_COMPOSITE = "Sailplane - smooth molded composites"
    CIVIL_JET_CLASSIC_METAL = "Civil jet - classic production metal"
    MILITARY = "Military aircraft"


@dataclass(frozen=True)
class LaminarFraction:
    """Fraction of the wetted area with laminar flow, for one category.

    Attributes:
        fuselage: Laminar fraction of the fuselage [-], 0 to 1.
        wing_and_tail: Laminar fraction of the wing and tails [-], 0 to 1.
    """

    fuselage: float
    wing_and_tail: float

    def __post_init__(self) -> None:
        for name, value in (
            ("fuselage", self.fuselage),
            ("wing_and_tail", self.wing_and_tail),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"LaminarFraction.{name} must be in [0, 1], got {value}"
                )


LAMINAR_FRACTION_TABLE: Final[Mapping[AircraftCategory, LaminarFraction]] = (
    MappingProxyType(
        {
            AircraftCategory.GA_CLASSIC_METAL: LaminarFraction(0.00, 0.10),
            AircraftCategory.GA_SMOOTH_METAL: LaminarFraction(0.10, 0.35),
            AircraftCategory.GA_SMOOTH_COMPOSITE: LaminarFraction(0.25, 0.50),
            AircraftCategory.SAILPLANE_SMOOTH_COMPOSITE: LaminarFraction(0.35, 0.70),
            AircraftCategory.CIVIL_JET_CLASSIC_METAL: LaminarFraction(0.05, 0.10),
            AircraftCategory.MILITARY: LaminarFraction(0.00, 0.00),
        }
    )
)


class SurfaceFinish(Enum):
    """Rows of the surface roughness table."""

    CAMOUFLAGE_PAINT_ON_ALUMINUM = "Camouflage paint on aluminum"
    SMOOTH_PAINT = "Smooth paint"
    PRODUCTION_SHEET_METAL = "Production sheet metal"
    POLISHED_SHEET_METAL = "Polished sheet metal"
    SMOOTH_MOLDED_COMPOSITE = "Smooth molded composite"


ROUGHNESS_TABLE_M: Final[Mapping[SurfaceFinish, float]] = MappingProxyType(
    {
        SurfaceFinish.CAMOUFLAGE_PAINT_ON_ALUMINUM: 1.015e-5,
        SurfaceFinish.SMOOTH_PAINT: 0.634e-5,
        SurfaceFinish.PRODUCTION_SHEET_METAL: 0.405e-5,
        SurfaceFinish.POLISHED_SHEET_METAL: 0.152e-5,
        SurfaceFinish.SMOOTH_MOLDED_COMPOSITE: 0.052e-5,
    }
)


# ============================================================
#  INTERFERENCE FACTOR
# ============================================================


class InterferenceCase(Enum):
    """Rows of the component interference factor table."""

    NACELLE_DIRECTLY_UNDER_FUSELAGE = (
        "Nacelle or external store, directly under a fuselage"
    )
    NACELLE_UNDER_FUSELAGE_NEAR = (
        "Nacelle or external store, under a fuselage, less than about 1 diameter away"
    )
    NACELLE_UNDER_FUSELAGE_FAR = (
        "Nacelle or external store, under a fuselage, more than about 1 diameter away"
    )
    WINGTIP_OBJECT = "Object, such as a fuel tank, mounted to a wingtip"
    HIGH_OR_MID_WING_FAIRED = "High wing or mid wing with carefully designed fairing"
    UNFILLETED_LOW_WING = "Un-filleted low wing"
    WHITCOMB_WINGLET = "Whitcomb winglet"
    AIRBUS_WINGLET = '"Airbus" style winglet'
    BLENDED_WINGLET = "Modern blended winglet"
    LANDING_GEAR_STRUT = (
        "Aerodynamic or square leaf-spring main landing gear strut entering wing "
        "or fuselage"
    )
    WING_STRUT = (
        "Aerodynamic wing strut entering wing on one end and fuselage on the other"
    )
    BOUNDARY_LAYER_DIVERTER = "Boundary layer diverter"
    CONVENTIONAL_TAIL = "Conventional tail"
    CRUCIFORM_TAIL = "Cruciform tail"
    V_TAIL = "V-tail"
    H_TAIL_B25 = "H-tail (B-25 Mitchell or A-10 Warthog style)"
    H_TAIL_ELECTRA = "H-tail (Lockheed Model 10 Electra style)"
    H_TAIL_D18 = "H-tail (Beech D-18 style)"
    TRIPLE_TAIL = "Triple-tail (Lockheed Constellation style)"
    T_TAIL = "T-tail"


# "Boundary layer diverter ~1.0" is stored as 1.0.
INTERFERENCE_TABLE: Final[Mapping[InterferenceCase, ValueRange]] = MappingProxyType(
    {
        InterferenceCase.NACELLE_DIRECTLY_UNDER_FUSELAGE: ValueRange(1.5, 1.5),
        InterferenceCase.NACELLE_UNDER_FUSELAGE_NEAR: ValueRange(1.3, 1.3),
        InterferenceCase.NACELLE_UNDER_FUSELAGE_FAR: ValueRange(1.0, 1.0),
        InterferenceCase.WINGTIP_OBJECT: ValueRange(1.25, 1.25),
        InterferenceCase.HIGH_OR_MID_WING_FAIRED: ValueRange(1.0, 1.0),
        InterferenceCase.UNFILLETED_LOW_WING: ValueRange(1.1, 1.4),
        InterferenceCase.WHITCOMB_WINGLET: ValueRange(1.04, 1.04),
        InterferenceCase.AIRBUS_WINGLET: ValueRange(1.04, 1.04),
        InterferenceCase.BLENDED_WINGLET: ValueRange(1.00, 1.01),
        InterferenceCase.LANDING_GEAR_STRUT: ValueRange(1.10, 1.10),
        InterferenceCase.WING_STRUT: ValueRange(1.10, 1.10),
        InterferenceCase.BOUNDARY_LAYER_DIVERTER: ValueRange(1.0, 1.0),
        InterferenceCase.CONVENTIONAL_TAIL: ValueRange(1.04, 1.05),
        InterferenceCase.CRUCIFORM_TAIL: ValueRange(1.06, 1.06),
        InterferenceCase.V_TAIL: ValueRange(1.03, 1.03),
        InterferenceCase.H_TAIL_B25: ValueRange(1.08, 1.08),
        InterferenceCase.H_TAIL_ELECTRA: ValueRange(1.13, 1.13),
        InterferenceCase.H_TAIL_D18: ValueRange(1.06, 1.06),
        InterferenceCase.TRIPLE_TAIL: ValueRange(1.10, 1.10),
        InterferenceCase.T_TAIL: ValueRange(1.04, 1.04),
    }
)


@dataclass(frozen=True)
class InterferenceChoice:
    """The interference factor chosen for one component.

    Attributes:
        case: Table row, or None for a component with no row in the table.
        value: Chosen IF. Required for a range row and for case None.
            Optional for a single-value row (the table value is used).
    """

    case: InterferenceCase | None
    value: float | None = None

    def __post_init__(self) -> None:
        _ = self.factor  # validates the choice against the table

    @property
    def factor(self) -> float:
        """The interference factor IF [-].

        Raises:
            ValueError: If the choice does not match the table, or a choice
                without a table row has no positive value.
        """
        if self.case is None:
            if self.value is None or self.value <= 0.0:
                raise ValueError(
                    f"InterferenceChoice without a table row needs a value > 0, "
                    f"got {self.value}"
                )
            return self.value
        return INTERFERENCE_TABLE[self.case].select(self.value)


# ============================================================
#  MISCELLANEOUS DRAG
# ============================================================


class FlapType(Enum):
    """Flap types of the flap drag equation."""

    PLAIN = "plain"
    SLOTTED = "slotted"


FLAP_FACTOR_TABLE: Final[Mapping[FlapType, float]] = MappingProxyType(
    {
        FlapType.PLAIN: 0.0144,
        FlapType.SLOTTED: 0.0074,
    }
)


class WheelWells(Enum):
    """Wheel-well state of the retractable landing gear drag fit."""

    OPEN = "open"
    CLOSED = "closed"


# Delta C_Ds = coefficient * exp(5.615 * S_A / (d * w)), on S_s = d * w
WHEEL_WELL_COEFF_TABLE: Final[Mapping[WheelWells, float]] = MappingProxyType(
    {
        WheelWells.OPEN: 0.05328,
        WheelWells.CLOSED: 0.04955,
    }
)


class ExcrescenceCategory(Enum):
    """Rows of the excrescence and leakage list."""

    PROPELLER_AIRCRAFT = "Propeller aircraft"
    JET_TRANSPORT = "Jet transport"
    NON_STEALTH_FIGHTER = "Non-stealth fighter"
    STEALTH_FIGHTER = "Stealth fighter"


# Fraction of the total CD0 [-]
EXCRESCENCE_FRACTION_TABLE: Final[Mapping[ExcrescenceCategory, ValueRange]] = (
    MappingProxyType(
        {
            ExcrescenceCategory.PROPELLER_AIRCRAFT: ValueRange(0.05, 0.10),
            ExcrescenceCategory.JET_TRANSPORT: ValueRange(0.02, 0.05),
            ExcrescenceCategory.NON_STEALTH_FIGHTER: ValueRange(0.10, 0.15),
            ExcrescenceCategory.STEALTH_FIGHTER: ValueRange(0.03, 0.05),
        }
    )
)
