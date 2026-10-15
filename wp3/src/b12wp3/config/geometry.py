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
  planform area of all panels, AR = b^2 / S. For the vertical tail and a
  pylon (one panel), b is the panel height.
* covered_span_m is the span length inside the fuselage (or nacelle). The
  exposed area S_exp is the planform outside it.
* x_c_max_thickness is the chord position of the maximum thickness, (x/c)_m
  in the form-factor equation.
* Bodies (fuselage, nacelle) are a nose of length L1, a cylinder of length
  L2 and a tail cone of length L3, all with the maximum diameter D. This is
  the geometry of the ADSEE-II fuselage wetted-area formula.
* Flap eta_in / eta_out are fractions of the wing semi-span from the
  centreline, as in WP2. The flapped area S_flap is the wing planform area
  between them, both halves.
* Landing gear inputs are per gear unit. The drag reference area of one unit
  is tire diameter x tire width.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from b12wp3.common.planform import TrapezoidalPlanform
from b12wp3.config.paths import GEOMETRY_YAML

logger = logging.getLogger(__name__)

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
#  YAML LOADER
# ============================================================

# YAML section -> input type. engine_count is the one top-level scalar.
_SECTIONS: Final[Mapping[str, type]] = {
    "wing": LiftingSurfaceInputs,
    "horizontal_tail": LiftingSurfaceInputs,
    "vertical_tail": LiftingSurfaceInputs,
    "fuselage": BodyInputs,
    "fuselage_aft": FuselageAftInputs,
    "nacelle": BodyInputs,
    "pylon": LiftingSurfaceInputs,
    "te_flap": FlapInputs,
    "le_flap": FlapInputs,
    "main_gear": GearInputs,
    "nose_gear": GearInputs,
}
_ENGINE_COUNT_KEY: Final[str] = "engine_count"


class GeometryConfigError(ValueError):
    """The geometry YAML file has a wrong structure, a wrong type or nulls."""


def _number(name: str, value: object, field_type: object) -> float | int:
    """One YAML value, checked against the dataclass field type (int or float)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GeometryConfigError(f"{name} must be a number, got {value!r}")
    if field_type == "int":
        if not isinstance(value, int):
            raise GeometryConfigError(f"{name} must be an integer, got {value!r}")
        return value
    return float(value)


def _section[T](
    raw: Mapping[str, object],
    section: str,
    cls: type[T],
    missing: list[str],
) -> T | None:
    """Build one input dataclass from its YAML section.

    Null keys are added to `missing` as "section.key"; the section is then
    not built and None is returned.
    """
    content = raw[section]
    if not isinstance(content, Mapping):
        raise GeometryConfigError(f"section {section!r} must be a mapping")
    fields = {f.name: f.type for f in dataclasses.fields(cls)}  # type: ignore[arg-type]  # cls is always a dataclass from _SECTIONS
    keys = set(content)
    if keys != set(fields):
        raise GeometryConfigError(
            f"section {section!r}: missing keys {sorted(set(fields) - keys)}, "
            f"unknown keys {sorted(keys - set(fields))}"
        )
    nulls = [f"{section}.{key}" for key in fields if content[key] is None]
    if nulls:
        missing.extend(nulls)
        return None
    values: dict[str, Any] = {
        key: _number(f"{section}.{key}", content[key], field_type)
        for key, field_type in fields.items()
    }
    return cls(**values)


def _read_yaml(path: Path) -> Mapping[str, object]:
    """Read the YAML file and check its top-level sections."""
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        logger.error("cannot read geometry file %s: %s", path, error)
        raise GeometryConfigError(f"cannot read geometry file {path}") from error
    except yaml.YAMLError as error:
        logger.error("invalid YAML in %s: %s", path, error)
        raise GeometryConfigError(f"invalid YAML in {path}") from error
    if not isinstance(raw, Mapping):
        raise GeometryConfigError(f"{path}: the top level must be a mapping")
    expected = set(_SECTIONS) | {_ENGINE_COUNT_KEY}
    keys = set(raw)
    if keys != expected:
        raise GeometryConfigError(
            f"{path}: missing sections {sorted(expected - keys)}, "
            f"unknown sections {sorted(keys - expected)}"
        )
    return raw


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
    nacelle = _section(raw, "nacelle", BodyInputs, missing)
    pylon = _section(raw, "pylon", LiftingSurfaceInputs, missing)
    te_flap = _section(raw, "te_flap", FlapInputs, missing)
    le_flap = _section(raw, "le_flap", FlapInputs, missing)
    main_gear = _section(raw, "main_gear", GearInputs, missing)
    nose_gear = _section(raw, "nose_gear", GearInputs, missing)
    engine_count_raw = raw[_ENGINE_COUNT_KEY]
    engine_count = (
        None
        if engine_count_raw is None
        else int(_number(_ENGINE_COUNT_KEY, engine_count_raw, "int"))
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
        or pylon is None
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
        pylon=pylon,
        engine_count=engine_count,
        fuselage_aft=fuselage_aft,
        te_flap=te_flap,
        le_flap=le_flap,
        main_gear=main_gear,
        nose_gear=nose_gear,
    )
