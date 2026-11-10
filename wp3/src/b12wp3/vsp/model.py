# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Resolved OpenVSP model: every component as positions and shapes in metres.

Stage 1 (layout.py, wp3 env) builds an AircraftModel and writes it as JSON.
Stage 2 (builder.py, OpenVSP env) reads the JSON and only calls OpenVSP. This
module is the interface between the two, so it must import in Python 3.11:
no 3.12+ syntax and no import of b12wp3.config.

Axes: x aft from the fuselage nose tip, y to the right wing, z up from the
axis of the fuselage cylinder. A component with symmetric=True is built on
the right side and mirrored about the x-z plane.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

Point2: TypeAlias = tuple[float, float]
Point3: TypeAlias = tuple[float, float, float]

# AirfoilModel.kind values
NACA_FOUR_DIGIT: Final[str] = "naca4"  # symmetric NACA 4-digit, t/c only
POINTS: Final[str] = "points"  # upper and lower surface points

# Material names; the builder holds their colours
MATERIAL_FUSELAGE: Final[str] = "B12 fuselage"
MATERIAL_LIFTING: Final[str] = "B12 lifting surface"
MATERIAL_NACELLE: Final[str] = "B12 nacelle"
MATERIAL_DARK: Final[str] = "B12 dark"  # fan face, spinner
MATERIAL_METAL: Final[str] = "B12 metal"  # struts, exhaust plug


@dataclass(frozen=True)
class BodySection:
    """One circular cross-section of a body.

    OpenVSP fixes the surface tangent at every section (horizontal by
    default, which gives a scalloped outline). The builder sets it from the
    two slopes: outward slope dz_c/dx + dr/dx on top, dr/dx - dz_c/dx at the
    bottom, dr/dx at the sides. A point section (diameter 0) keeps
    OpenVSP's own tip shape.

    Attributes:
        x_m: Position along the body, from its first section [m].
        z_m: Centre height relative to the body origin [m].
        diameter_m: Diameter [m]; 0 is a point (closed end).
        centre_slope: dz_c/dx of the centreline [-].
        radius_slope: dr/dx of the radius [-].
    """

    x_m: float
    z_m: float
    diameter_m: float
    centre_slope: float
    radius_slope: float


@dataclass(frozen=True)
class BodyModel:
    """A body lofted through circular cross-sections (OpenVSP FUSELAGE).

    Attributes:
        name: Geom name.
        material: Material name (a key of the builder's material table).
        origin: Position of the first section [m].
        sections: Cross-sections, x increasing from 0.
        symmetric: Mirror about the x-z plane.
    """

    name: str
    material: str
    origin: Point3
    sections: tuple[BodySection, ...]
    symmetric: bool


@dataclass(frozen=True)
class AirfoilModel:
    """Section of a lifting surface.

    Attributes:
        kind: NACA_FOUR_DIGIT or POINTS.
        thickness_to_chord: t/c [-] (NACA_FOUR_DIGIT).
        upper: Upper surface (x/c, y/c), LE to TE (POINTS).
        lower: Lower surface (x/c, y/c), LE to TE (POINTS).
    """

    kind: str
    thickness_to_chord: float
    upper: tuple[Point2, ...]
    lower: tuple[Point2, ...]


@dataclass(frozen=True)
class SurfaceModel:
    """One straight-tapered lifting surface panel (OpenVSP WING).

    Attributes:
        name: Geom name.
        material: Material name.
        root_le: Root leading edge [m].
        x_rotation_deg: Rotation about x [deg]; 90 makes a vertical panel.
        incidence_deg: Rotation about y, about the root LE [deg].
        span_m: Panel span, root to tip [m].
        root_chord_m: Root chord [m].
        tip_chord_m: Tip chord [m].
        sweep_deg: Sweep of the line at sweep_location [deg].
        sweep_location: Chord fraction of the swept line [-].
        dihedral_deg: Dihedral [deg].
        tip_twist_deg: Tip twist about the quarter chord [deg].
        airfoil: Section, the same at root and tip.
        symmetric: Mirror about the x-z plane.
    """

    name: str
    material: str
    root_le: Point3
    x_rotation_deg: float
    incidence_deg: float
    span_m: float
    root_chord_m: float
    tip_chord_m: float
    sweep_deg: float
    sweep_location: float
    dihedral_deg: float
    tip_twist_deg: float
    airfoil: AirfoilModel
    symmetric: bool


@dataclass(frozen=True)
class RingModel:
    """Annular body: an airfoil section revolved about the x axis
    (OpenVSP BODYOFREVOLUTION, flow-through). Used for the nacelle.

    Attributes:
        name: Geom name.
        material: Material name.
        inlet: Axis point in the inlet plane [m].
        le_diameter_m: Diameter of the section leading edge [m].
        chord_m: Section chord, LE to TE midpoint [m].
        angle_deg: Chord angle; positive = TE closer to the axis [deg].
        upper: Outer surface (x/c, y/c) in the chord frame, LE to TE.
        lower: Inner surface (x/c, y/c) in the chord frame, LE to TE.
        symmetric: Mirror about the x-z plane.
    """

    name: str
    material: str
    inlet: Point3
    le_diameter_m: float
    chord_m: float
    angle_deg: float
    upper: tuple[Point2, ...]
    lower: tuple[Point2, ...]
    symmetric: bool


@dataclass(frozen=True)
class EllipsoidModel:
    """Ellipsoid (OpenVSP ELLIPSOID). Used for the fan face and spinner.

    Attributes:
        name: Geom name.
        material: Material name.
        centre: Centre [m].
        radii: Semi-axes along x, y, z [m].
        symmetric: Mirror about the x-z plane.
    """

    name: str
    material: str
    centre: Point3
    radii: Point3
    symmetric: bool


@dataclass(frozen=True)
class StrutModel:
    """Straight landing-gear strut in an x-z plane (round-ended cylinder).

    Attributes:
        name: Geom name.
        material: Material name.
        bottom: Lower end, at the axle [m].
        top: Upper end, inside the fuselage or the wing [m].
        diameter_m: Strut diameter [m].
        symmetric: Mirror about the x-z plane.
    """

    name: str
    material: str
    bottom: Point3
    top: Point3
    diameter_m: float
    symmetric: bool


@dataclass(frozen=True)
class BogieModel:
    """One gear unit: tires side by side on one axle.

    Attributes:
        name: Unit name.
        contact_x_m: x of the contact point [m].
        contact_y_m: y of the unit centre [m].
        tire_diameter_m: Diameter of one tire [m].
        tire_width_m: Width of one tire [m].
        tires_across: Number of tires side by side.
        tire_gap_m: Gap between neighbouring tires [m].
        symmetric: Mirror about the x-z plane.
    """

    name: str
    contact_x_m: float
    contact_y_m: float
    tire_diameter_m: float
    tire_width_m: float
    tires_across: int
    tire_gap_m: float
    symmetric: bool


@dataclass(frozen=True)
class GearModel:
    """All tires (OpenVSP GEAR), standing on the ground plane.

    Attributes:
        ground_z_m: z of the ground plane [m].
        bogies: Gear units.
    """

    ground_z_m: float
    bogies: tuple[BogieModel, ...]


@dataclass(frozen=True)
class AircraftModel:
    """Every component of the 3-view model.

    Attributes:
        bodies: Fuselage and exhaust plugs.
        surfaces: Wing, tails, pylons.
        rings: Nacelle shells.
        ellipsoids: Fan faces, spinners.
        struts: Landing-gear struts.
        gear: Tires.
    """

    bodies: tuple[BodyModel, ...]
    surfaces: tuple[SurfaceModel, ...]
    rings: tuple[RingModel, ...]
    ellipsoids: tuple[EllipsoidModel, ...]
    struts: tuple[StrutModel, ...]
    gear: GearModel

    def write_json(self, path: Path) -> None:
        """Write the model as JSON.

        Args:
            path: Output file; the parent directory must exist.

        Raises:
            OSError: If the file cannot be written.
        """
        path.write_text(
            json.dumps(dataclasses.asdict(self), indent=1), encoding="utf-8"
        )

    @classmethod
    def read_json(cls, path: Path) -> AircraftModel:
        """Read a model written by write_json.

        Args:
            path: JSON file.

        Returns:
            The model.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the content does not match the model types.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        model = _decode(cls, data, "model")
        if not isinstance(model, AircraftModel):
            raise ValueError(f"{path} does not hold an AircraftModel")
        return model


def _decode(tp: object, value: object, where: str) -> object:
    """Convert JSON data to the annotated type tp (dataclass, tuple, scalar)."""
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected an object, got {value!r}")
        hints = typing.get_type_hints(tp)
        names = {f.name for f in dataclasses.fields(tp)}
        if set(value) != names:
            raise ValueError(
                f"{where}: keys {sorted(value)} differ from {sorted(names)}"
            )
        kwargs = {n: _decode(hints[n], value[n], f"{where}.{n}") for n in names}
        return tp(**kwargs)
    if typing.get_origin(tp) is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)
            )
        if len(value) != len(args):
            raise ValueError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(
            _decode(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value))
        )
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string, got {value!r}")
        return value
    raise ValueError(f"{where}: unsupported type {tp!r}")
