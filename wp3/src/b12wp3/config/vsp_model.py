# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Inputs of the OpenVSP 3-view model: the input types and their YAML loader.

The values are in wp3/configs/vsp_model.yaml: the positions of the
components and the visual details. The sizes of the components are in
geometry.yaml (b12wp3.config.geometry). Each YAML section maps to one frozen
dataclass below; its keys are the dataclass field names. load_vsp_model()
reads the file, collects every null input into one error, and builds the
dataclasses, which run their own physical checks.

Axes: x aft from the fuselage nose tip, y to the right wing, z up from the
axis of the fuselage cylinder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from b12wp3.config.paths import VSP_MODEL_YAML
from b12wp3.config.yaml_io import ConfigError, build_section, read_mapping

# ============================================================
#  INPUT TYPES
# ============================================================


def _require_positive(**values: float) -> None:
    """Raise ValueError for the first value that is not > 0."""
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")


def _require_count(**values: int) -> None:
    """Raise ValueError for the first count that is not >= 1."""
    for name, value in values.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")


def _require_fraction(**values: float) -> None:
    """Raise ValueError for the first value that is not in (0, 1)."""
    for name, value in values.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")


@dataclass(frozen=True)
class WingPlacement:
    """Position and orientation of the wing.

    Attributes:
        root_le_x_m: x of the root leading edge at the centreline [m].
        root_z_m: z of the root chord line [m].
        dihedral_deg: Dihedral [deg].
        incidence_deg: Incidence, rotation about the root leading edge [deg].
        tip_twist_deg: Tip twist relative to the root, negative = wash-out [deg].
    """

    root_le_x_m: float
    root_z_m: float
    dihedral_deg: float
    incidence_deg: float
    tip_twist_deg: float

    def __post_init__(self) -> None:
        _require_positive(root_le_x_m=self.root_le_x_m)
        for name, value in (
            ("dihedral_deg", self.dihedral_deg),
            ("incidence_deg", self.incidence_deg),
            ("tip_twist_deg", self.tip_twist_deg),
        ):
            if abs(value) >= 15.0:
                raise ValueError(f"|{name}| must be < 15 deg, got {value}")


@dataclass(frozen=True)
class CentreOfGravity:
    """Position of the centre of gravity, the reference of the tail arms.

    Attributes:
        x_m: x of the c.g. from the nose tip [m], > 0.
    """

    x_m: float

    def __post_init__(self) -> None:
        _require_positive(x_m=self.x_m)


@dataclass(frozen=True)
class TailPlacement:
    """Tail arms and the cruciform horizontal-tail height.

    Attributes:
        horizontal_arm_m: Horizontal distance from the c.g. to the
            horizontal-tail MAC quarter chord [m].
        vertical_arm_m: Horizontal distance from the c.g. to the
            vertical-tail MAC quarter chord [m].
        horizontal_height_fraction: Height of the horizontal-tail root above
            the fin root, fraction of the fin span [-], in [0, 1].
        horizontal_incidence_deg: Horizontal-tail incidence [deg].
    """

    horizontal_arm_m: float
    vertical_arm_m: float
    horizontal_height_fraction: float
    horizontal_incidence_deg: float

    def __post_init__(self) -> None:
        _require_positive(
            horizontal_arm_m=self.horizontal_arm_m,
            vertical_arm_m=self.vertical_arm_m,
        )
        if not 0.0 <= self.horizontal_height_fraction <= 1.0:
            raise ValueError(
                "horizontal_height_fraction must be in [0, 1], "
                f"got {self.horizontal_height_fraction}"
            )
        if abs(self.horizontal_incidence_deg) >= 15.0:
            raise ValueError(
                "|horizontal_incidence_deg| must be < 15 deg, "
                f"got {self.horizontal_incidence_deg}"
            )


@dataclass(frozen=True)
class FuselageShape:
    """Visual shape inputs of the fuselage.

    Attributes:
        nose_tip_z_m: z of the nose tip [m]; negative = below the axis.
        nose_sections: Number of cross-sections on the nose, >= 3.
        tail_sections: Number of cross-sections on the tail cone, >= 3.
    """

    nose_tip_z_m: float
    nose_sections: int
    tail_sections: int

    def __post_init__(self) -> None:
        for name, value in (
            ("nose_sections", self.nose_sections),
            ("tail_sections", self.tail_sections),
        ):
            if value < 3:
                raise ValueError(f"{name} must be >= 3, got {value}")


@dataclass(frozen=True)
class NacellePlacement:
    """Position of the right nacelle; the left nacelle is its mirror.

    Attributes:
        axis_y_m: Span station of the nacelle axis [m], > 0.
        inlet_ahead_of_le_m: Inlet plane ahead of the local wing LE [m].
        axis_below_chord_m: Nacelle axis below the local wing chord line [m].
    """

    axis_y_m: float
    inlet_ahead_of_le_m: float
    axis_below_chord_m: float

    def __post_init__(self) -> None:
        _require_positive(
            axis_y_m=self.axis_y_m, axis_below_chord_m=self.axis_below_chord_m
        )


@dataclass(frozen=True)
class NacelleDetail:
    """Visual details of the nacelle inside the Torenbeek outline.

    Attributes:
        lip_thickness_m: Inlet lip wall thickness [m].
        lip_length_fraction: Lip length, fraction of l_n [-].
        exit_wall_thickness_m: Wall thickness at the exit [m].
        fan_face_position: Fan face behind the inlet, fraction of l_n [-].
        spinner_diameter_m: Spinner diameter D_s [m].
        spinner_length_m: Spinner length ahead of the fan face [m].
        plug_diameter_fraction: Exhaust plug diameter at the exit / D_ef [-].
        plug_length_m: Plug length behind the exit [m].
    """

    lip_thickness_m: float
    lip_length_fraction: float
    exit_wall_thickness_m: float
    fan_face_position: float
    spinner_diameter_m: float
    spinner_length_m: float
    plug_diameter_fraction: float
    plug_length_m: float

    def __post_init__(self) -> None:
        _require_positive(
            lip_thickness_m=self.lip_thickness_m,
            exit_wall_thickness_m=self.exit_wall_thickness_m,
            spinner_diameter_m=self.spinner_diameter_m,
            spinner_length_m=self.spinner_length_m,
            plug_length_m=self.plug_length_m,
        )
        _require_fraction(
            lip_length_fraction=self.lip_length_fraction,
            fan_face_position=self.fan_face_position,
            plug_diameter_fraction=self.plug_diameter_fraction,
        )
        if self.fan_face_position <= self.lip_length_fraction:
            raise ValueError(
                f"fan_face_position {self.fan_face_position} must be behind the "
                f"lip, lip_length_fraction {self.lip_length_fraction}"
            )


@dataclass(frozen=True)
class PylonInputs:
    """Visual pylon between the nacelle and the wing.

    Attributes:
        le_position: Pylon root LE behind the inlet, fraction of l_n [-].
        chord_fraction: Pylon chord, fraction of l_n [-].
        sweep_le_deg: Leading-edge sweep from the nacelle to the wing [deg].
        thickness_to_chord: t/c of the NACA 4-digit section [-].
    """

    le_position: float
    chord_fraction: float
    sweep_le_deg: float
    thickness_to_chord: float

    def __post_init__(self) -> None:
        _require_fraction(
            le_position=self.le_position,
            chord_fraction=self.chord_fraction,
            thickness_to_chord=self.thickness_to_chord,
        )
        if self.le_position + self.chord_fraction > 1.0:
            raise ValueError(
                f"pylon root ends behind the nacelle exit: le_position "
                f"{self.le_position} + chord_fraction {self.chord_fraction} > 1"
            )
        if not 0.0 <= self.sweep_le_deg < 80.0:
            raise ValueError(
                f"sweep_le_deg must be in [0, 80), got {self.sweep_le_deg}"
            )


@dataclass(frozen=True)
class NoseGearPlacement:
    """Position of the nose gear unit, on the centreline.

    Attributes:
        contact_x_m: x of the tire contact point [m].
        tires_per_unit: Tires side by side on the axle, >= 1.
        strut_diameter_m: Strut diameter [m].
        tire_gap_m: Gap between neighbouring tires [m].
    """

    contact_x_m: float
    tires_per_unit: int
    strut_diameter_m: float
    tire_gap_m: float

    def __post_init__(self) -> None:
        _require_count(tires_per_unit=self.tires_per_unit)
        _require_positive(
            contact_x_m=self.contact_x_m,
            strut_diameter_m=self.strut_diameter_m,
            tire_gap_m=self.tire_gap_m,
        )


@dataclass(frozen=True)
class MainGearPlacement:
    """Position of the two main gear units, symmetric about the centreline.

    Attributes:
        contact_x_m: x of the tire contact points [m].
        track_m: Distance between the two unit centres [m].
        tires_per_unit: Tires side by side on one axle, >= 1.
        strut_diameter_m: Strut diameter [m].
        tire_gap_m: Gap between neighbouring tires of one unit [m].
    """

    contact_x_m: float
    track_m: float
    tires_per_unit: int
    strut_diameter_m: float
    tire_gap_m: float

    def __post_init__(self) -> None:
        _require_count(tires_per_unit=self.tires_per_unit)
        _require_positive(
            contact_x_m=self.contact_x_m,
            track_m=self.track_m,
            strut_diameter_m=self.strut_diameter_m,
            tire_gap_m=self.tire_gap_m,
        )


@dataclass(frozen=True)
class GroundInputs:
    """Height of the aircraft above the ground, gear down.

    Attributes:
        fuselage_clearance_m: Ground to the fuselage cylinder bottom [m].
    """

    fuselage_clearance_m: float

    def __post_init__(self) -> None:
        _require_positive(fuselage_clearance_m=self.fuselage_clearance_m)


@dataclass(frozen=True)
class VspModelInputs:
    """Everything in vsp_model.yaml.

    Attributes:
        wing: Wing position and orientation.
        centre_of_gravity: c.g. position, the reference of the tail arms.
        tails: Tail arms and horizontal-tail height.
        fuselage_shape: Visual fuselage shape.
        nacelle_position: Right-nacelle position.
        nacelle_detail: Visual nacelle details.
        pylon: Visual pylon.
        nose_gear_position: Nose gear unit position.
        main_gear_position: Main gear units position.
        ground: Height above the ground.
    """

    wing: WingPlacement
    centre_of_gravity: CentreOfGravity
    tails: TailPlacement
    fuselage_shape: FuselageShape
    nacelle_position: NacellePlacement
    nacelle_detail: NacelleDetail
    pylon: PylonInputs
    nose_gear_position: NoseGearPlacement
    main_gear_position: MainGearPlacement
    ground: GroundInputs


# ============================================================
#  YAML LOADER
# ============================================================

# YAML section -> input type, in file order
_SECTIONS: Final[Mapping[str, type]] = {
    "wing": WingPlacement,
    "centre_of_gravity": CentreOfGravity,
    "tails": TailPlacement,
    "fuselage_shape": FuselageShape,
    "nacelle_position": NacellePlacement,
    "nacelle_detail": NacelleDetail,
    "pylon": PylonInputs,
    "nose_gear_position": NoseGearPlacement,
    "main_gear_position": MainGearPlacement,
    "ground": GroundInputs,
}


class VspModelConfigError(ConfigError):
    """The VSP model YAML file has a wrong structure, a wrong type or nulls."""


def load_vsp_model(path: Path = VSP_MODEL_YAML) -> VspModelInputs:
    """Load, validate and freeze the OpenVSP model inputs.

    Args:
        path: VSP model YAML file. Default: wp3/configs/vsp_model.yaml.

    Returns:
        The validated inputs.

    Raises:
        VspModelConfigError: If the file cannot be read, has a wrong
            structure or type, or has null inputs. One error names every
            null input, as "section.key".
        ValueError: If a set input is not physical (dataclass checks).
    """
    missing: list[str] = []
    try:
        raw = read_mapping(path, _SECTIONS)
        sections = {
            name: build_section(raw, name, cls, missing)
            for name, cls in _SECTIONS.items()
        }
    except ConfigError as error:
        raise VspModelConfigError(str(error)) from error
    if missing:
        raise VspModelConfigError(
            f"VSP model inputs not set (null) in {path}: " + ", ".join(missing)
        )
    # every section is built: build_section returns None only with a null key
    return VspModelInputs(**sections)  # type: ignore[arg-type]
