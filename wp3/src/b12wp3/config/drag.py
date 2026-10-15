# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Class II drag settings: the selected row of each ADSEE-II table and the
method inputs that are not geometry.

The tables themselves are in b12wp3.config.drag_tables. Everything the drag
build-up needs from here is collected in DRAG_SETTINGS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from b12wp3.config.drag_tables import (
    EXCRESCENCE_FRACTION_TABLE,
    FLAP_FACTOR_TABLE,
    LAMINAR_FRACTION_TABLE,
    ROUGHNESS_TABLE_M,
    WHEEL_WELL_COEFF_TABLE,
    AircraftCategory,
    ExcrescenceCategory,
    FlapType,
    InterferenceCase,
    InterferenceChoice,
    LaminarFraction,
    SurfaceFinish,
    WheelWells,
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


@dataclass(frozen=True)
class DragSettings:
    """Everything the drag build-up needs that is not geometry.

    Attributes:
        laminar: Laminar fractions of the selected aircraft category.
        roughness_m: Equivalent sand roughness k [m].
        interference: Interference factor of each component.
        kappa_a: Korn technology factor k_a of the wing airfoil [-].
        wing_mach_critical_normal: Critical Mach number of the wing section
            normal to the wave-drag sweep line [-].
        wave_sweep_chord_fraction: Chord line whose sweep the drag-divergence
            equation uses [-].
        te_flap_factor: F_flap of the trailing-edge flap [-].
        le_flap_factor: F_flap of the leading-edge flap [-].
        wheel_well_coeff: Coefficient of the landing gear drag fit [-].
        excrescence_fraction: Excrescence and leakage drag as a fraction of
            the total CD0 [-].
    """

    laminar: LaminarFraction
    roughness_m: float
    interference: InterferenceFactors
    kappa_a: float
    wing_mach_critical_normal: float
    wave_sweep_chord_fraction: float
    te_flap_factor: float
    le_flap_factor: float
    wheel_well_coeff: float
    excrescence_fraction: float

    def __post_init__(self) -> None:
        if self.roughness_m <= 0.0:
            raise ValueError(f"roughness_m must be > 0, got {self.roughness_m}")
        if not 0.0 < self.wing_mach_critical_normal < 1.0:
            raise ValueError(
                "wing_mach_critical_normal must be in (0, 1), "
                f"got {self.wing_mach_critical_normal}"
            )
        if not 0.0 <= self.wave_sweep_chord_fraction <= 1.0:
            raise ValueError(
                "wave_sweep_chord_fraction must be in [0, 1], "
                f"got {self.wave_sweep_chord_fraction}"
            )
        if not 0.0 <= self.excrescence_fraction < 1.0:
            raise ValueError(
                "excrescence_fraction must be in [0, 1), "
                f"got {self.excrescence_fraction}"
            )


# ============================================================
#  SELECTED SETTINGS
# ============================================================

# --- skin friction ---
# Civil jet, classic production metal: 5 % laminar on the fuselage, 10 % on the
# wing and tails. Nacelles use the fuselage value (body of revolution); pylons
# use the wing-and-tail value (lifting surface).
AIRCRAFT_CATEGORY: Final[AircraftCategory] = AircraftCategory.CIVIL_JET_CLASSIC_METAL
LAMINAR_FRACTION: Final[LaminarFraction] = LAMINAR_FRACTION_TABLE[AIRCRAFT_CATEGORY]

SURFACE_FINISH: Final[SurfaceFinish] = SurfaceFinish.SMOOTH_PAINT
ROUGHNESS_M: Final[float] = ROUGHNESS_TABLE_M[SURFACE_FINISH]

# --- interference factors ---
# Wing: conservative upper end of the un-filleted low wing range.
WING_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(
    InterferenceCase.UNFILLETED_LOW_WING, 1.4
)
# Cruciform tail: the same row for the horizontal and the vertical tail.
HORIZONTAL_TAIL_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(
    InterferenceCase.CRUCIFORM_TAIL
)
VERTICAL_TAIL_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(
    InterferenceCase.CRUCIFORM_TAIL
)
# The fuselage has no row in the table: it is the reference body, IF = 1.0.
FUSELAGE_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(None, 1.0)
# Pylon-mounted nacelle, more than about one diameter away from the surface.
NACELLE_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(
    InterferenceCase.NACELLE_UNDER_FUSELAGE_FAR
)
# The pylon has no row in the table: IF = 1.0.
PYLON_INTERFERENCE: Final[InterferenceChoice] = InterferenceChoice(None, 1.0)

# --- wave drag (Korn / Mason drag-divergence Mach number) ---
# k_a of a supercritical section, the same value as WP2
# (wp2/src/b12wp2/config/scoring.py, KAPPA_A_SUPERCRITICAL).
KAPPA_A: Final[float] = 0.95
# M_crit of NASA SC(2)-0712 normal to the half-chord line, at the cruise Cl_n
# (wp2/results/section/data/mcrit_summary.csv). The drag build-up converts it
# to the streamwise (free-stream) value M_crit,n / cos(sweep).
WING_MACH_CRITICAL_NORMAL: Final[float] = 0.56
# Half chord: the same sweep line as the WP2 section reduction and Korn estimate.
WAVE_SWEEP_CHORD_FRACTION: Final[float] = 0.5

# --- flaps ---
# Trailing edge: Fowler flap, a slotted flap. Leading edge: unslotted nose flap.
TE_FLAP_TYPE: Final[FlapType] = FlapType.SLOTTED
LE_FLAP_TYPE: Final[FlapType] = FlapType.PLAIN

# --- landing gear ---
# The gear doors are open while the gear is down.
WHEEL_WELLS: Final[WheelWells] = WheelWells.OPEN

# --- excrescence and leakage ---
EXCRESCENCE_CATEGORY: Final[ExcrescenceCategory] = ExcrescenceCategory.JET_TRANSPORT
EXCRESCENCE_FRACTION: Final[float] = EXCRESCENCE_FRACTION_TABLE[
    EXCRESCENCE_CATEGORY
].select(0.035)

DRAG_SETTINGS: Final[DragSettings] = DragSettings(
    laminar=LAMINAR_FRACTION,
    roughness_m=ROUGHNESS_M,
    interference=InterferenceFactors(
        wing=WING_INTERFERENCE.factor,
        horizontal_tail=HORIZONTAL_TAIL_INTERFERENCE.factor,
        vertical_tail=VERTICAL_TAIL_INTERFERENCE.factor,
        fuselage=FUSELAGE_INTERFERENCE.factor,
        nacelle=NACELLE_INTERFERENCE.factor,
        pylon=PYLON_INTERFERENCE.factor,
    ),
    kappa_a=KAPPA_A,
    wing_mach_critical_normal=WING_MACH_CRITICAL_NORMAL,
    wave_sweep_chord_fraction=WAVE_SWEEP_CHORD_FRACTION,
    te_flap_factor=FLAP_FACTOR_TABLE[TE_FLAP_TYPE],
    le_flap_factor=FLAP_FACTOR_TABLE[LE_FLAP_TYPE],
    wheel_well_coeff=WHEEL_WELL_COEFF_TABLE[WHEEL_WELLS],
    excrescence_fraction=EXCRESCENCE_FRACTION,
)
