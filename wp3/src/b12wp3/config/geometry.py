# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Every geometric input of the aircraft, in one list.

Each input is one module constant below. Nothing is read from another file
and nothing is derived here. An input that is not designed yet is None.
aircraft_geometry() collects the inputs into frozen dataclasses. If an input
is None, it raises one error that names every unset input.

Conventions:

* Lifting surfaces are trapezoidal (b12wp3.common.planform). AREA is the
  planform area of all panels, AR = b^2 / S. For the vertical tail and a
  pylon (one panel), b is the panel height.
* COVERED_SPAN is the span length inside the fuselage (or nacelle). The
  exposed area S_exp is the planform outside it.
* X_C_MAX_THICKNESS is the chord position of the maximum thickness, (x/c)_m
  in the form-factor equation.
* Bodies (fuselage, nacelle) are a nose of length L1, a cylinder of length
  L2 and a tail cone of length L3, all with the maximum diameter D. This is
  the geometry of the ADSEE-II fuselage wetted-area formula.
* Flap ETA_IN / ETA_OUT are fractions of the wing semi-span from the
  centreline, as in WP2. The flapped area S_flap is the wing planform area
  between them, both halves.
* Landing gear inputs are per gear unit. The drag reference area of one unit
  is tire diameter x tire width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from b12wp3.common.planform import TrapezoidalPlanform

# ============================================================
#  INPUTS
# ============================================================

# --- wing (WP2 final planform, wp2/results/aero_summary_3d.csv) ---
WING_AREA_M2: Final[float | None] = 66.7
WING_ASPECT_RATIO: Final[float | None] = 9.5
WING_TAPER: Final[float | None] = 0.4
WING_SWEEP_C4_DEG: Final[float | None] = 30.0
WING_THICKNESS_TO_CHORD: Final[float | None] = 0.1199  # NASA SC(2)-0712
# Max-thickness location of NASA_SC(2)-0712.dat (wp2/airfoils), streamwise.
WING_X_C_MAX_THICKNESS: Final[float | None] = 0.37
WING_COVERED_SPAN_M: Final[float | None] = None  # fuselage width at the wing
WING_PANELS: Final[int] = 2

# --- horizontal tail ---
HORIZONTAL_TAIL_AREA_M2: Final[float | None] = None
HORIZONTAL_TAIL_ASPECT_RATIO: Final[float | None] = None
HORIZONTAL_TAIL_TAPER: Final[float | None] = None
HORIZONTAL_TAIL_SWEEP_C4_DEG: Final[float | None] = None
HORIZONTAL_TAIL_THICKNESS_TO_CHORD: Final[float | None] = None
HORIZONTAL_TAIL_X_C_MAX_THICKNESS: Final[float | None] = None
HORIZONTAL_TAIL_COVERED_SPAN_M: Final[float | None] = None
HORIZONTAL_TAIL_PANELS: Final[int] = 2

# --- vertical tail ---
VERTICAL_TAIL_AREA_M2: Final[float | None] = None
VERTICAL_TAIL_ASPECT_RATIO: Final[float | None] = None
VERTICAL_TAIL_TAPER: Final[float | None] = None
VERTICAL_TAIL_SWEEP_C4_DEG: Final[float | None] = None
VERTICAL_TAIL_THICKNESS_TO_CHORD: Final[float | None] = None
VERTICAL_TAIL_X_C_MAX_THICKNESS: Final[float | None] = None
VERTICAL_TAIL_COVERED_SPAN_M: Final[float | None] = None
VERTICAL_TAIL_PANELS: Final[int] = 1

# --- fuselage ---
FUSELAGE_NOSE_LENGTH_M: Final[float | None] = None  # L1
FUSELAGE_CYLINDER_LENGTH_M: Final[float | None] = None  # L2
FUSELAGE_TAIL_LENGTH_M: Final[float | None] = None  # L3
FUSELAGE_DIAMETER_M: Final[float | None] = None  # D
FUSELAGE_UPSWEEP_DEG: Final[float | None] = None  # u, aft-body upsweep angle
FUSELAGE_BASE_AREA_M2: Final[float | None] = None  # A_base, 0 for a closed cone

# --- trailing-edge flap (WP2 Fowler flap, wp2/src/b12wp2/config/hld*.py) ---
TE_FLAP_CHORD_RATIO: Final[float | None] = 0.30  # c_f / c
TE_FLAP_ETA_IN: Final[float | None] = 0.12  # fraction of the semi-span
TE_FLAP_ETA_OUT: Final[float | None] = 0.75

# --- leading-edge flap (WP2 nose flap, unslotted) ---
LE_FLAP_CHORD_RATIO: Final[float | None] = 0.15
LE_FLAP_ETA_IN: Final[float | None] = 0.12
LE_FLAP_ETA_OUT: Final[float | None] = 0.95

# --- main landing gear (per unit; d x w is the drag reference area) ---
MAIN_GEAR_COUNT: Final[int | None] = None  # number of main gear units
MAIN_GEAR_FRONTAL_AREA_M2: Final[float | None] = None  # S_A
MAIN_GEAR_TIRE_WIDTH_M: Final[float | None] = None  # w
MAIN_GEAR_TIRE_DIAMETER_M: Final[float | None] = None  # d

# --- nose landing gear (per unit) ---
NOSE_GEAR_COUNT: Final[int | None] = None
NOSE_GEAR_FRONTAL_AREA_M2: Final[float | None] = None
NOSE_GEAR_TIRE_WIDTH_M: Final[float | None] = None
NOSE_GEAR_TIRE_DIAMETER_M: Final[float | None] = None

# --- nacelle (one engine; body of revolution, same formula as the fuselage) ---
NACELLE_NOSE_LENGTH_M: Final[float | None] = None
NACELLE_CYLINDER_LENGTH_M: Final[float | None] = None
NACELLE_TAIL_LENGTH_M: Final[float | None] = None
NACELLE_DIAMETER_M: Final[float | None] = None

# --- pylon (one engine; thin lifting surface, same formula as the tails) ---
PYLON_AREA_M2: Final[float | None] = None
PYLON_ASPECT_RATIO: Final[float | None] = None
PYLON_TAPER: Final[float | None] = None
PYLON_SWEEP_C4_DEG: Final[float | None] = None
PYLON_THICKNESS_TO_CHORD: Final[float | None] = None
PYLON_X_C_MAX_THICKNESS: Final[float | None] = None
PYLON_COVERED_SPAN_M: Final[float | None] = None
PYLON_PANELS: Final[int] = 1

# --- engines ---
ENGINE_COUNT: Final[int | None] = None  # one nacelle and one pylon per engine


# ============================================================
#  INPUT TYPES
# ============================================================


@dataclass(frozen=True)
class LiftingSurfaceInputs:
    """Geometric inputs of one trapezoidal lifting surface.

    Attributes:
        area_m2: Planform area of all panels [m^2].
        aspect_ratio: b^2 / S [-].
        taper: c_tip / c_root [-].
        sweep_c4_deg: Quarter-chord sweep [deg].
        thickness_to_chord: Maximum t/c of the section [-].
        x_c_max_thickness: Chord position of the maximum thickness [-].
        covered_span_m: Span length inside the fuselage or nacelle [m].
        panels: 2 for a symmetric surface, 1 for a single surface.
    """

    area_m2: float
    aspect_ratio: float
    taper: float
    sweep_c4_deg: float
    thickness_to_chord: float
    x_c_max_thickness: float
    covered_span_m: float
    panels: int

    def __post_init__(self) -> None:
        if not 0.0 < self.thickness_to_chord < 1.0:
            raise ValueError(
                f"thickness_to_chord must be in (0, 1), got {self.thickness_to_chord}"
            )
        if not 0.0 < self.x_c_max_thickness < 1.0:
            raise ValueError(
                f"x_c_max_thickness must be in (0, 1), got {self.x_c_max_thickness}"
            )
        self.planform().exposed_area(self.covered_span_m)  # validates the planform

    def planform(self) -> TrapezoidalPlanform:
        """Trapezoidal planform of this surface."""
        return TrapezoidalPlanform(
            area_m2=self.area_m2,
            aspect_ratio=self.aspect_ratio,
            taper=self.taper,
            sweep_c4_rad=math.radians(self.sweep_c4_deg),
            panels=self.panels,
        )


@dataclass(frozen=True)
class BodyInputs:
    """Geometric inputs of one body of revolution (fuselage or nacelle).

    Attributes:
        nose_length_m: Nose length L1 [m], > 0.
        cylinder_length_m: Cylinder length L2 [m], >= 0.
        tail_length_m: Tail-cone length L3 [m], >= 0.
        diameter_m: Maximum diameter D [m], > 0.
    """

    nose_length_m: float
    cylinder_length_m: float
    tail_length_m: float
    diameter_m: float

    def __post_init__(self) -> None:
        if self.nose_length_m <= 0.0:
            raise ValueError(f"nose_length_m must be > 0, got {self.nose_length_m}")
        if self.cylinder_length_m < 0.0:
            raise ValueError(
                f"cylinder_length_m must be >= 0, got {self.cylinder_length_m}"
            )
        if self.tail_length_m < 0.0:
            raise ValueError(f"tail_length_m must be >= 0, got {self.tail_length_m}")
        if self.diameter_m <= 0.0:
            raise ValueError(f"diameter_m must be > 0, got {self.diameter_m}")

    @property
    def length_m(self) -> float:
        """Total length L1 + L2 + L3 [m]."""
        return self.nose_length_m + self.cylinder_length_m + self.tail_length_m

    @property
    def max_frontal_area_m2(self) -> float:
        """Maximum cross-section area A_max = pi D^2 / 4 [m^2]."""
        return math.pi * self.diameter_m**2 / 4.0


@dataclass(frozen=True)
class FuselageAftInputs:
    """Aft-fuselage inputs of the upsweep and base drag.

    Attributes:
        upsweep_deg: Aft-body upsweep angle u [deg], in [0, 90).
        base_area_m2: Base area A_base [m^2], 0 for a closed tail cone.
    """

    upsweep_deg: float
    base_area_m2: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.upsweep_deg < 90.0:
            raise ValueError(f"upsweep_deg must be in [0, 90), got {self.upsweep_deg}")
        if self.base_area_m2 < 0.0:
            raise ValueError(f"base_area_m2 must be >= 0, got {self.base_area_m2}")


@dataclass(frozen=True)
class FlapInputs:
    """Geometric inputs of one flap (both wing halves).

    Attributes:
        chord_ratio: Flap chord over wing chord c_f / c [-], in (0, 1).
        eta_in: Inboard end, fraction of the semi-span [-].
        eta_out: Outboard end, fraction of the semi-span [-].
    """

    chord_ratio: float
    eta_in: float
    eta_out: float

    def __post_init__(self) -> None:
        if not 0.0 < self.chord_ratio < 1.0:
            raise ValueError(f"chord_ratio must be in (0, 1), got {self.chord_ratio}")
        if not 0.0 <= self.eta_in < self.eta_out <= 1.0:
            raise ValueError(
                f"need 0 <= eta_in < eta_out <= 1, got {self.eta_in}, {self.eta_out}"
            )


@dataclass(frozen=True)
class GearInputs:
    """Geometric inputs of one landing gear type (main or nose).

    Attributes:
        count: Number of identical gear units.
        frontal_area_m2: Frontal area S_A of one unit [m^2].
        tire_width_m: Tire width w [m].
        tire_diameter_m: Tire diameter d [m].
    """

    count: int
    frontal_area_m2: float
    tire_width_m: float
    tire_diameter_m: float

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        for name, value in (
            ("frontal_area_m2", self.frontal_area_m2),
            ("tire_width_m", self.tire_width_m),
            ("tire_diameter_m", self.tire_diameter_m),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class AircraftGeometry:
    """All geometric inputs of the aircraft.

    Attributes:
        wing: Wing.
        horizontal_tail: Horizontal tail.
        vertical_tail: Vertical tail.
        fuselage: Fuselage.
        nacelle: One nacelle.
        pylon: One pylon.
        engine_count: Number of engines (nacelles and pylons).
        fuselage_aft: Fuselage upsweep and base area.
        te_flap: Trailing-edge flap.
        le_flap: Leading-edge flap.
        main_gear: Main landing gear.
        nose_gear: Nose landing gear.
    """

    wing: LiftingSurfaceInputs
    horizontal_tail: LiftingSurfaceInputs
    vertical_tail: LiftingSurfaceInputs
    fuselage: BodyInputs
    nacelle: BodyInputs
    pylon: LiftingSurfaceInputs
    engine_count: int
    fuselage_aft: FuselageAftInputs
    te_flap: FlapInputs
    le_flap: FlapInputs
    main_gear: GearInputs
    nose_gear: GearInputs

    def __post_init__(self) -> None:
        if self.engine_count < 1:
            raise ValueError(f"engine_count must be >= 1, got {self.engine_count}")


# ============================================================
#  ASSEMBLY
# ============================================================


def _lifting_surface(
    prefix: str,
    area: float | None,
    aspect_ratio: float | None,
    taper: float | None,
    sweep_c4_deg: float | None,
    thickness_to_chord: float | None,
    x_c_max: float | None,
    covered_span: float | None,
    panels: int,
    missing: list[str],
) -> LiftingSurfaceInputs | None:
    """Build one lifting surface, or add its unset input names to `missing`."""
    if (
        area is None
        or aspect_ratio is None
        or taper is None
        or sweep_c4_deg is None
        or thickness_to_chord is None
        or x_c_max is None
        or covered_span is None
    ):
        named = (
            ("AREA_M2", area),
            ("ASPECT_RATIO", aspect_ratio),
            ("TAPER", taper),
            ("SWEEP_C4_DEG", sweep_c4_deg),
            ("THICKNESS_TO_CHORD", thickness_to_chord),
            ("X_C_MAX_THICKNESS", x_c_max),
            ("COVERED_SPAN_M", covered_span),
        )
        missing.extend(f"{prefix}_{name}" for name, value in named if value is None)
        return None
    return LiftingSurfaceInputs(
        area_m2=area,
        aspect_ratio=aspect_ratio,
        taper=taper,
        sweep_c4_deg=sweep_c4_deg,
        thickness_to_chord=thickness_to_chord,
        x_c_max_thickness=x_c_max,
        covered_span_m=covered_span,
        panels=panels,
    )


def _body(
    prefix: str,
    nose: float | None,
    cylinder: float | None,
    tail: float | None,
    diameter: float | None,
    missing: list[str],
) -> BodyInputs | None:
    """Build one body, or add its unset input names to `missing`."""
    if nose is None or cylinder is None or tail is None or diameter is None:
        named = (
            ("NOSE_LENGTH_M", nose),
            ("CYLINDER_LENGTH_M", cylinder),
            ("TAIL_LENGTH_M", tail),
            ("DIAMETER_M", diameter),
        )
        missing.extend(f"{prefix}_{name}" for name, value in named if value is None)
        return None
    return BodyInputs(
        nose_length_m=nose,
        cylinder_length_m=cylinder,
        tail_length_m=tail,
        diameter_m=diameter,
    )


def _fuselage_aft(missing: list[str]) -> FuselageAftInputs | None:
    """Build the aft-fuselage inputs, or add the unset names to `missing`."""
    if FUSELAGE_UPSWEEP_DEG is None or FUSELAGE_BASE_AREA_M2 is None:
        named = (
            ("FUSELAGE_UPSWEEP_DEG", FUSELAGE_UPSWEEP_DEG),
            ("FUSELAGE_BASE_AREA_M2", FUSELAGE_BASE_AREA_M2),
        )
        missing.extend(name for name, value in named if value is None)
        return None
    return FuselageAftInputs(FUSELAGE_UPSWEEP_DEG, FUSELAGE_BASE_AREA_M2)


def _flap(
    prefix: str,
    chord_ratio: float | None,
    eta_in: float | None,
    eta_out: float | None,
    missing: list[str],
) -> FlapInputs | None:
    """Build one flap, or add its unset input names to `missing`."""
    if chord_ratio is None or eta_in is None or eta_out is None:
        named = (
            ("CHORD_RATIO", chord_ratio),
            ("ETA_IN", eta_in),
            ("ETA_OUT", eta_out),
        )
        missing.extend(f"{prefix}_{name}" for name, value in named if value is None)
        return None
    return FlapInputs(chord_ratio, eta_in, eta_out)


def _gear(
    prefix: str,
    count: int | None,
    frontal_area: float | None,
    tire_width: float | None,
    tire_diameter: float | None,
    missing: list[str],
) -> GearInputs | None:
    """Build one landing gear type, or add its unset input names to `missing`."""
    if (
        count is None
        or frontal_area is None
        or tire_width is None
        or tire_diameter is None
    ):
        named: tuple[tuple[str, float | None], ...] = (
            ("COUNT", count),
            ("FRONTAL_AREA_M2", frontal_area),
            ("TIRE_WIDTH_M", tire_width),
            ("TIRE_DIAMETER_M", tire_diameter),
        )
        missing.extend(f"{prefix}_{name}" for name, value in named if value is None)
        return None
    return GearInputs(count, frontal_area, tire_width, tire_diameter)


_Surfaces = tuple[
    LiftingSurfaceInputs | None,
    LiftingSurfaceInputs | None,
    LiftingSurfaceInputs | None,
    LiftingSurfaceInputs | None,
]


def _surfaces(missing: list[str]) -> _Surfaces:
    """Wing, horizontal tail, vertical tail and pylon from the module inputs."""
    return (
        _lifting_surface(
            "WING",
            WING_AREA_M2,
            WING_ASPECT_RATIO,
            WING_TAPER,
            WING_SWEEP_C4_DEG,
            WING_THICKNESS_TO_CHORD,
            WING_X_C_MAX_THICKNESS,
            WING_COVERED_SPAN_M,
            WING_PANELS,
            missing,
        ),
        _lifting_surface(
            "HORIZONTAL_TAIL",
            HORIZONTAL_TAIL_AREA_M2,
            HORIZONTAL_TAIL_ASPECT_RATIO,
            HORIZONTAL_TAIL_TAPER,
            HORIZONTAL_TAIL_SWEEP_C4_DEG,
            HORIZONTAL_TAIL_THICKNESS_TO_CHORD,
            HORIZONTAL_TAIL_X_C_MAX_THICKNESS,
            HORIZONTAL_TAIL_COVERED_SPAN_M,
            HORIZONTAL_TAIL_PANELS,
            missing,
        ),
        _lifting_surface(
            "VERTICAL_TAIL",
            VERTICAL_TAIL_AREA_M2,
            VERTICAL_TAIL_ASPECT_RATIO,
            VERTICAL_TAIL_TAPER,
            VERTICAL_TAIL_SWEEP_C4_DEG,
            VERTICAL_TAIL_THICKNESS_TO_CHORD,
            VERTICAL_TAIL_X_C_MAX_THICKNESS,
            VERTICAL_TAIL_COVERED_SPAN_M,
            VERTICAL_TAIL_PANELS,
            missing,
        ),
        _lifting_surface(
            "PYLON",
            PYLON_AREA_M2,
            PYLON_ASPECT_RATIO,
            PYLON_TAPER,
            PYLON_SWEEP_C4_DEG,
            PYLON_THICKNESS_TO_CHORD,
            PYLON_X_C_MAX_THICKNESS,
            PYLON_COVERED_SPAN_M,
            PYLON_PANELS,
            missing,
        ),
    )


def _bodies(missing: list[str]) -> tuple[BodyInputs | None, BodyInputs | None]:
    """Fuselage and nacelle from the module inputs."""
    return (
        _body(
            "FUSELAGE",
            FUSELAGE_NOSE_LENGTH_M,
            FUSELAGE_CYLINDER_LENGTH_M,
            FUSELAGE_TAIL_LENGTH_M,
            FUSELAGE_DIAMETER_M,
            missing,
        ),
        _body(
            "NACELLE",
            NACELLE_NOSE_LENGTH_M,
            NACELLE_CYLINDER_LENGTH_M,
            NACELLE_TAIL_LENGTH_M,
            NACELLE_DIAMETER_M,
            missing,
        ),
    )


def _flaps(missing: list[str]) -> tuple[FlapInputs | None, FlapInputs | None]:
    """Trailing-edge and leading-edge flap from the module inputs."""
    return (
        _flap("TE_FLAP", TE_FLAP_CHORD_RATIO, TE_FLAP_ETA_IN, TE_FLAP_ETA_OUT, missing),
        _flap("LE_FLAP", LE_FLAP_CHORD_RATIO, LE_FLAP_ETA_IN, LE_FLAP_ETA_OUT, missing),
    )


def _gears(missing: list[str]) -> tuple[GearInputs | None, GearInputs | None]:
    """Main and nose landing gear from the module inputs."""
    return (
        _gear(
            "MAIN_GEAR",
            MAIN_GEAR_COUNT,
            MAIN_GEAR_FRONTAL_AREA_M2,
            MAIN_GEAR_TIRE_WIDTH_M,
            MAIN_GEAR_TIRE_DIAMETER_M,
            missing,
        ),
        _gear(
            "NOSE_GEAR",
            NOSE_GEAR_COUNT,
            NOSE_GEAR_FRONTAL_AREA_M2,
            NOSE_GEAR_TIRE_WIDTH_M,
            NOSE_GEAR_TIRE_DIAMETER_M,
            missing,
        ),
    )


def aircraft_geometry() -> AircraftGeometry:
    """Collect the module inputs into an AircraftGeometry.

    Returns:
        The validated aircraft geometry.

    Raises:
        ValueError: If one or more inputs are None. The message names all of
            them. Also if a set input is not physical.
    """
    missing: list[str] = []
    wing, horizontal_tail, vertical_tail, pylon = _surfaces(missing)
    fuselage, nacelle = _bodies(missing)
    fuselage_aft = _fuselage_aft(missing)
    te_flap, le_flap = _flaps(missing)
    main_gear, nose_gear = _gears(missing)
    if ENGINE_COUNT is None:
        missing.append("ENGINE_COUNT")
    if (
        missing
        or wing is None
        or horizontal_tail is None
        or vertical_tail is None
        or pylon is None
        or fuselage is None
        or nacelle is None
        or fuselage_aft is None
        or te_flap is None
        or le_flap is None
        or main_gear is None
        or nose_gear is None
        or ENGINE_COUNT is None
    ):
        raise ValueError(
            "geometry inputs not set in b12wp3/config/geometry.py: "
            + ", ".join(missing)
        )
    return AircraftGeometry(
        wing=wing,
        horizontal_tail=horizontal_tail,
        vertical_tail=vertical_tail,
        fuselage=fuselage,
        nacelle=nacelle,
        pylon=pylon,
        engine_count=ENGINE_COUNT,
        fuselage_aft=fuselage_aft,
        te_flap=te_flap,
        le_flap=le_flap,
        main_gear=main_gear,
        nose_gear=nose_gear,
    )
