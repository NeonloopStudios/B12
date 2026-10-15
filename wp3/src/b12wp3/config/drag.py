# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Class II drag settings (ADSEE-II, component drag build-up).

The two tables are copied from the ADSEE-II slides "Flat plate skin friction
coefficient": the laminar-flow fraction per aircraft category, and the
equivalent sand roughness k per surface finish. The selected row of each
table is set below.

The interference factors and the miscellaneous drag are not on the slides
used so far. They stay unset (None) until their values are known. The drag
build-up takes them as required arguments, so an unset value cannot enter a
result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final


class AircraftCategory(Enum):
    """Rows of the ADSEE-II laminar-flow table."""

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
    """Rows of the ADSEE-II surface roughness table."""

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


@dataclass(frozen=True)
class InterferenceFactors:
    """Interference factor IF of each component [-].

    Attributes:
        wing: Wing.
        horizontal_tail: Horizontal tail.
        vertical_tail: Vertical tail.
        fuselage: Fuselage.
        nacelle: One nacelle.
        pylon: One pylon.
    """

    wing: float
    horizontal_tail: float
    vertical_tail: float
    fuselage: float
    nacelle: float
    pylon: float

    def __post_init__(self) -> None:
        for name, value in (
            ("wing", self.wing),
            ("horizontal_tail", self.horizontal_tail),
            ("vertical_tail", self.vertical_tail),
            ("fuselage", self.fuselage),
            ("nacelle", self.nacelle),
            ("pylon", self.pylon),
        ):
            if value <= 0.0:
                raise ValueError(f"InterferenceFactors.{name} must be > 0, got {value}")


# --- selected settings ---

# Civil jet, classic production metal: 5 % laminar on the fuselage, 10 % on the
# wing and tails. Nacelles use the fuselage value (body of revolution); pylons
# use the wing-and-tail value (lifting surface).
AIRCRAFT_CATEGORY: Final[AircraftCategory] = AircraftCategory.CIVIL_JET_CLASSIC_METAL
LAMINAR_FRACTION: Final[LaminarFraction] = LAMINAR_FRACTION_TABLE[AIRCRAFT_CATEGORY]

SURFACE_FINISH: Final[SurfaceFinish] = SurfaceFinish.SMOOTH_PAINT
ROUGHNESS_M: Final[float] = ROUGHNESS_TABLE_M[SURFACE_FINISH]

# Not on the slides used so far: set them when the values are known.
INTERFERENCE_FACTORS: Final[InterferenceFactors | None] = None
CD_MISC: Final[float | None] = None
