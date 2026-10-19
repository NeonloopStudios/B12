# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Geometric inputs of the aircraft: the input types and their YAML loader.

The values are in wp3/configs/geometry.yaml, one key per quantity. Each YAML
section maps to one frozen dataclass below; its keys are the dataclass field
names. load_geometry() reads the file, checks the structure, collects every
null (not designed yet) input into one error, and builds the dataclasses,
which run their own physical checks.

Conventions (repeated in the YAML header):

* Lifting surfaces are trapezoidal (b12wp3.common.planform). area_m2 is the
  planform area of all panels, AR = b^2 / S. For the vertical tail (one
  panel), b is the panel height.
* covered_span_m is the span length inside the fuselage (or nacelle). The
  exposed area S_exp is the planform outside it.
* x_c_max_thickness is the chord position of the maximum thickness, (x/c)_m
  in the form-factor equation.
* The fuselage is a nose of length L1, a cylinder of length L2 and a tail
  cone of length L3, all with the maximum diameter D. This is the geometry
  of the ADSEE-II fuselage wetted-area formula.
* A nacelle is the Torenbeek Type B outline: highlight diameter D_h, maximum
  diameter D_n at the fraction beta of the length l_n, exit diameter D_ef
  (scripts/nacelle_sizing.py computes them).
* Flap eta_in / eta_out are fractions of the wing semi-span from the
  centreline, as in WP2. The flapped area S_flap is the wing planform area
  between them, both halves.
* Landing gear inputs are per gear unit. The drag reference area of one unit
  is tire diameter x tire width.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from b12wp3.common.planform import TrapezoidalPlanform
from b12wp3.config.paths import GEOMETRY_YAML
from b12wp3.config.yaml_io import ConfigError, build_section, number, read_mapping

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
class NacelleInputs:
    """Outer geometry of one nacelle (Torenbeek Type B outline).

    Attributes:
        length_m: Nacelle length l_n, without the cone [m].
        highlight_diameter_m: Highlight diameter D_h [m].
        max_diameter_m: Maximum diameter D_n [m].
        max_diameter_position: Position of D_n as a fraction beta of l_n [-].
        exit_diameter_m: Exit diameter D_ef [m].
    """

    length_m: float
    highlight_diameter_m: float
    max_diameter_m: float
    max_diameter_position: float
    exit_diameter_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length_m", self.length_m),
            ("highlight_diameter_m", self.highlight_diameter_m),
            ("exit_diameter_m", self.exit_diameter_m),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if not 0.0 < self.max_diameter_position < 1.0:
            raise ValueError(
                "max_diameter_position must be in (0, 1), "
                f"got {self.max_diameter_position}"
            )
        largest = max(self.highlight_diameter_m, self.exit_diameter_m)
        if self.max_diameter_m < largest:
            raise ValueError(
                f"max_diameter_m {self.max_diameter_m} is smaller than the "
                f"highlight or exit diameter {largest}"
            )

    @property
    def max_frontal_area_m2(self) -> float:
        """Maximum cross-section area pi D_n^2 / 4 [m^2]."""
        return math.pi * self.max_diameter_m**2 / 4.0


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
        engine_count: Number of engines (nacelles).
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
    nacelle: NacelleInputs
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
#  YAML LOADER
# ============================================================

# YAML section -> input type. engine_count is the one top-level scalar.
_SECTIONS: Final[Mapping[str, type]] = {
    "wing": LiftingSurfaceInputs,
    "horizontal_tail": LiftingSurfaceInputs,
    "vertical_tail": LiftingSurfaceInputs,
    "fuselage": BodyInputs,
    "fuselage_aft": FuselageAftInputs,
    "nacelle": NacelleInputs,
    "te_flap": FlapInputs,
    "le_flap": FlapInputs,
    "main_gear": GearInputs,
    "nose_gear": GearInputs,
}
_ENGINE_COUNT_KEY: Final[str] = "engine_count"


class GeometryConfigError(ConfigError):
    """The geometry YAML file has a wrong structure, a wrong type or nulls."""


def _section[T](
    raw: Mapping[str, object], section: str, cls: type[T], missing: list[str]
) -> T | None:
    """build_section with the geometry error type."""
    try:
        return build_section(raw, section, cls, missing)
    except ConfigError as error:
        raise GeometryConfigError(str(error)) from error


def _read_yaml(path: Path) -> Mapping[str, object]:
    """read_mapping with the geometry sections and error type."""
    try:
        return read_mapping(path, set(_SECTIONS) | {_ENGINE_COUNT_KEY})
    except ConfigError as error:
        raise GeometryConfigError(str(error)) from error


def _engine_count(value: object) -> float | int:
    """The engine_count scalar, checked as an integer."""
    try:
        return number(_ENGINE_COUNT_KEY, value, "int")
    except ConfigError as error:
        raise GeometryConfigError(str(error)) from error


def load_geometry(path: Path = GEOMETRY_YAML) -> AircraftGeometry:
    """Load, validate and freeze the aircraft geometry from its YAML file.

    Args:
        path: Geometry YAML file. Default: wp3/configs/geometry.yaml.

    Returns:
        The validated aircraft geometry.

    Raises:
        GeometryConfigError: If the file cannot be read, has a wrong
            structure or type, or has null inputs. One error names every
            null input, as "section.key".
        ValueError: If a set input is not physical (dataclass checks).
    """
    raw = _read_yaml(path)
    missing: list[str] = []
    wing = _section(raw, "wing", LiftingSurfaceInputs, missing)
    horizontal_tail = _section(raw, "horizontal_tail", LiftingSurfaceInputs, missing)
    vertical_tail = _section(raw, "vertical_tail", LiftingSurfaceInputs, missing)
    fuselage = _section(raw, "fuselage", BodyInputs, missing)
    fuselage_aft = _section(raw, "fuselage_aft", FuselageAftInputs, missing)
    nacelle = _section(raw, "nacelle", NacelleInputs, missing)
    te_flap = _section(raw, "te_flap", FlapInputs, missing)
    le_flap = _section(raw, "le_flap", FlapInputs, missing)
    main_gear = _section(raw, "main_gear", GearInputs, missing)
    nose_gear = _section(raw, "nose_gear", GearInputs, missing)
    engine_count_raw = raw[_ENGINE_COUNT_KEY]
    engine_count = (
        None if engine_count_raw is None else int(_engine_count(engine_count_raw))
    )
    if engine_count is None:
        missing.append(_ENGINE_COUNT_KEY)
    if (
        missing
        or wing is None
        or horizontal_tail is None
        or vertical_tail is None
        or fuselage is None
        or fuselage_aft is None
        or nacelle is None
        or te_flap is None
        or le_flap is None
        or main_gear is None
        or nose_gear is None
        or engine_count is None
    ):
        raise GeometryConfigError(
            f"geometry inputs not set (null) in {path}: " + ", ".join(missing)
        )
    return AircraftGeometry(
        wing=wing,
        horizontal_tail=horizontal_tail,
        vertical_tail=vertical_tail,
        fuselage=fuselage,
        nacelle=nacelle,
        engine_count=engine_count,
        fuselage_aft=fuselage_aft,
        te_flap=te_flap,
        le_flap=le_flap,
        main_gear=main_gear,
        nose_gear=nose_gear,
    )
