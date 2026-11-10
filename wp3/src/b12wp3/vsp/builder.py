# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Stage 2 of the OpenVSP model: AircraftModel JSON -> OpenVSP geoms -> .vsp3.

Runs in the OpenVSP conda env (Python 3.11), started by scripts/vsp_model.py:

    python builder.py <model_inputs.json> <output.vsp3>

It only calls OpenVSP: every position and shape is already in the JSON
(b12wp3.vsp.layout). It must not import b12wp3.config (Python 3.12+ syntax).

OpenVSP components:

    Fuselage                  FUSELAGE, circular sections
    Wing, tails, pylons       WING, one straight-tapered panel
    Nacelle shells            BODYOFREVOLUTION, file airfoil revolved (open duct)
    Fan faces, spinners       ELLIPSOID
    Gear struts               FUSELAGE, round-ended cylinder, rotated
    Tires                     GEAR, one bogie per gear unit
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

if __package__ in (None, ""):
    # run as a file by the OpenVSP env: make wp3/src/ importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import openvsp as vsp  # noqa: E402

from b12wp3.vsp.model import (  # noqa: E402
    MATERIAL_DARK,
    MATERIAL_FUSELAGE,
    MATERIAL_LIFTING,
    MATERIAL_METAL,
    MATERIAL_NACELLE,
    NACA_FOUR_DIGIT,
    POINTS,
    AircraftModel,
    BodyModel,
    BodySection,
    EllipsoidModel,
    GearModel,
    Point3,
    RingModel,
    StrutModel,
    SurfaceModel,
)

logger = logging.getLogger("b12wp3.vsp.builder")

RGB = tuple[float, float, float]

# Material -> diffuse RGB [0-1]. OpenVSP's built-in materials render close to
# white in the shaded view (wp2 hld_model.py), so all are custom.
MATERIALS: Final[Mapping[str, RGB]] = {
    MATERIAL_FUSELAGE: (0.86, 0.87, 0.89),
    MATERIAL_LIFTING: (0.55, 0.62, 0.72),
    MATERIAL_NACELLE: (0.16, 0.30, 0.52),
    MATERIAL_DARK: (0.10, 0.10, 0.11),
    MATERIAL_METAL: (0.50, 0.50, 0.52),
}

INCH_M: Final[float] = 0.0254  # OpenVSP tire sizes are in inches
SURFACE_CHORD_TESS: Final[int] = 33  # chordwise tessellation, lifting surfaces
SURFACE_SPAN_TESS: Final[int] = 12  # spanwise tessellation, lifting surfaces
BODY_TESS_W: Final[int] = 41  # circumferential tessellation, bodies
SECTION_TESS_U: Final[int] = 10  # tessellation between body sections
TIRE_TESS_U: Final[int] = 41  # tessellation around the tire circumference
TIRE_TESS_W: Final[int] = 17  # tessellation across the tire section
NO_BBOX: Final[float] = 1.0e11  # OpenVSP reports 1e12 for a failed surface


# ============================================================
#  HELPERS
# ============================================================


def _set(geom: str, name: str, group: str, value: float) -> None:
    """Set one geom parameter; raise if it does not exist."""
    parm = vsp.GetParm(geom, name, group)
    if not vsp.ValidParm(parm):
        raise KeyError(f"no OpenVSP parameter {group}:{name} on geom {geom}")
    vsp.SetParmVal(parm, value)


def _set_xsec(xsec: str, name: str, value: float) -> None:
    """Set one cross-section parameter; raise if it does not exist."""
    parm = vsp.GetXSecParm(xsec, name)
    if not vsp.ValidParm(parm):
        raise KeyError(f"no OpenVSP cross-section parameter {name}")
    vsp.SetParmVal(parm, value)


def _place(geom: str, point: Point3, symmetric: bool) -> None:
    """Move a geom to a point and set its x-z symmetry.

    A geom without a parent uses the relative location (the default), which
    is then the global position.
    """
    for axis, value in zip("XYZ", point):
        _set(geom, f"{axis}_Rel_Location", "XForm", value)
    _set(geom, "Sym_Planar_Flag", "Sym", float(vsp.SYM_XZ if symmetric else 0))


def _style(geom: str, name: str, material: str) -> None:
    """Name a geom and give it a material and a matching wireframe colour."""
    if material not in MATERIALS:
        raise KeyError(
            f"unknown material {material!r}, expected one of {sorted(MATERIALS)}"
        )
    vsp.SetGeomName(geom, name)
    vsp.SetGeomMaterialName(geom, material)
    r, g, b = MATERIALS[material]
    vsp.SetGeomWireColor(geom, round(255 * r), round(255 * g), round(255 * b))


def _set_tangents(xsec: str, section: BodySection) -> None:
    """Surface tangent angles of a circular body section (see BodySection).

    OpenVSP angle convention, checked numerically: positive = the surface
    moves away from the axis, equal on both sides of the section.
    """
    outward = {
        "Top": section.centre_slope + section.radius_slope,
        "Bottom": section.radius_slope - section.centre_slope,
        "Right": section.radius_slope,
        "Left": section.radius_slope,
    }
    for flag in ("AllSym", "TBSym", "RLSym"):
        _set_xsec(xsec, flag, 0.0)
    for side, slope in outward.items():
        _set_xsec(xsec, f"{side}LRAngleEq", 1.0)
        _set_xsec(xsec, f"{side}LAngleSet", 1.0)
        _set_xsec(xsec, f"{side}LAngle", math.degrees(math.atan(slope)))


def _check_surface(geom: str, name: str) -> None:
    """Raise RuntimeError if OpenVSP could not build the surface of a geom."""
    vsp.Update()
    for key in ("X_Len", "Y_Len", "Z_Len"):
        value = vsp.GetParmVal(geom, key, "BBox")
        if not math.isfinite(value) or abs(value) > NO_BBOX:
            raise RuntimeError(
                f"OpenVSP failed to build {name!r}: bounding box {key} = {value}"
            )


def _material_terms(diffuse: RGB) -> dict[str, tuple[float, ...]]:
    """Ambient/diffuse/specular/emissive RGBA and shininess of a matte material."""
    return {
        "Ambient": (*(0.3 * c for c in diffuse), 1.0),
        "Diffuse": (*diffuse, 1.0),
        "Specular": (0.25, 0.25, 0.25, 1.0),
        "Emissive": (0.0, 0.0, 0.0, 1.0),
        "Shininess": (30.0,),
    }


def _add_materials() -> None:
    """Register the custom materials in the current model."""
    known = vsp.GetMaterialNames()
    for name, diffuse in MATERIALS.items():
        if name in known:
            continue
        t = _material_terms(diffuse)
        vsp.AddMaterial(
            name,
            *(
                vsp.vec3d(*t[k][:3])
                for k in ("Ambient", "Diffuse", "Specular", "Emissive")
            ),
            1.0,
            t["Shininess"][0],
        )


# ============================================================
#  COMPONENTS
# ============================================================


def add_body(body: BodyModel) -> str:
    """FUSELAGE geom lofted through the circular sections of a body."""
    geom: str = vsp.AddGeom("FUSELAGE")
    length = body.sections[-1].x_m - body.sections[0].x_m
    _set(geom, "Length", "Design", length)
    _set(geom, "Tess_W", "Shape", BODY_TESS_W)
    surf = vsp.GetXSecSurf(geom, 0)
    while vsp.GetNumXSec(surf) < len(body.sections):
        vsp.InsertXSec(geom, vsp.GetNumXSec(surf) - 2, vsp.XS_CIRCLE)
    while vsp.GetNumXSec(surf) > len(body.sections):
        vsp.CutXSec(geom, 1)
    for i, section in enumerate(body.sections):
        shape = vsp.XS_POINT if section.diameter_m == 0.0 else vsp.XS_CIRCLE
        vsp.ChangeXSecShape(surf, i, shape)
        xsec = vsp.GetXSec(surf, i)
        _set_xsec(xsec, "XLocPercent", (section.x_m - body.sections[0].x_m) / length)
        _set_xsec(xsec, "ZLocPercent", section.z_m / length)
        _set_xsec(xsec, "SectTess_U", SECTION_TESS_U)
        if shape == vsp.XS_CIRCLE:
            _set_xsec(xsec, "Circle_Diameter", section.diameter_m)
            _set_tangents(xsec, section)
    _place(geom, body.origin, body.symmetric)
    _style(geom, body.name, body.material)
    _check_surface(geom, body.name)
    return geom


def add_surface(surface: SurfaceModel) -> str:
    """WING geom with one straight-tapered panel."""
    geom: str = vsp.AddGeom("WING")
    _place(geom, surface.root_le, surface.symmetric)
    _set(geom, "X_Rel_Rotation", "XForm", surface.x_rotation_deg)
    _set(geom, "Y_Rel_Rotation", "XForm", surface.incidence_deg)
    _set(geom, "Tess_W", "Shape", SURFACE_CHORD_TESS)
    vsp.SetDriverGroup(
        geom, 1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER
    )
    group = "XSec_1"
    _set(geom, "Span", group, surface.span_m)
    _set(geom, "Root_Chord", group, surface.root_chord_m)
    _set(geom, "Tip_Chord", group, surface.tip_chord_m)
    _set(geom, "Sweep", group, surface.sweep_deg)
    _set(geom, "Sweep_Location", group, surface.sweep_location)
    _set(geom, "Dihedral", group, surface.dihedral_deg)
    _set(geom, "SectTess_U", group, SURFACE_SPAN_TESS)
    for g, twist in (("XSec_0", 0.0), ("XSec_1", surface.tip_twist_deg)):
        _set(geom, "Twist", g, twist)
        _set(geom, "Twist_Location", g, 0.25)
    vsp.Update()
    surf = vsp.GetXSecSurf(geom, 0)
    airfoil = surface.airfoil
    for i in (0, 1):
        if airfoil.kind == NACA_FOUR_DIGIT:
            vsp.ChangeXSecShape(surf, i, vsp.XS_FOUR_SERIES)
            xsec = vsp.GetXSec(surf, i)
            _set_xsec(xsec, "ThickChord", airfoil.thickness_to_chord)
            _set_xsec(xsec, "Camber", 0.0)
        elif airfoil.kind == POINTS:
            vsp.ChangeXSecShape(surf, i, vsp.XS_FILE_AIRFOIL)
            vsp.SetAirfoilPnts(
                vsp.GetXSec(surf, i),
                [vsp.vec3d(x, y, 0.0) for x, y in airfoil.upper],
                [vsp.vec3d(x, y, 0.0) for x, y in airfoil.lower],
            )
        else:
            raise ValueError(f"{surface.name}: unknown airfoil kind {airfoil.kind!r}")
    _style(geom, surface.name, surface.material)
    _check_surface(geom, surface.name)
    return geom


def add_ring(ring: RingModel) -> str:
    """BODYOFREVOLUTION geom: the wall airfoil revolved about the x axis."""
    geom: str = vsp.AddGeom("BODYOFREVOLUTION")
    _set(geom, "Mode", "Design", vsp.BOR_FLOWTHROUGH)
    vsp.ChangeBORXSecShape(geom, vsp.XS_FILE_AIRFOIL)
    vsp.SetBORAirfoilPnts(
        geom,
        [vsp.vec3d(x, y, 0.0) for x, y in ring.upper],
        [vsp.vec3d(x, y, 0.0) for x, y in ring.lower],
    )
    _set(geom, "Diameter", "Design", ring.le_diameter_m)
    _set(geom, "Angle", "Design", ring.angle_deg)
    _set(geom, "Chord", "XSecCurve", ring.chord_m)
    _set(geom, "Tess_W", "Shape", SURFACE_CHORD_TESS)
    _set(geom, "Tess_U", "Shape", BODY_TESS_W)
    _place(geom, ring.inlet, ring.symmetric)
    _style(geom, ring.name, ring.material)
    _check_surface(geom, ring.name)
    return geom


def add_ellipsoid(ellipsoid: EllipsoidModel) -> str:
    """ELLIPSOID geom; OpenVSP places its origin at the centre."""
    geom: str = vsp.AddGeom("ELLIPSOID")
    for name, radius in zip(("A_Radius", "B_Radius", "C_Radius"), ellipsoid.radii):
        _set(geom, name, "Design", radius)
    _set(geom, "Tess_U", "Shape", BODY_TESS_W)
    _set(geom, "Tess_W", "Shape", BODY_TESS_W)
    _place(geom, ellipsoid.centre, ellipsoid.symmetric)
    _style(geom, ellipsoid.name, ellipsoid.material)
    _check_surface(geom, ellipsoid.name)
    return geom


def add_strut(strut: StrutModel) -> str:
    """Round-ended cylinder from the axle to the strut top.

    A FUSELAGE body along +x: a point, the full diameter one radius further,
    the full diameter one radius before the end, a point. A rotation of
    -atan2(dz, dx) about y turns +x to the strut direction (dx, 0, dz).
    """
    dx = strut.top[0] - strut.bottom[0]
    dy = strut.top[1] - strut.bottom[1]
    dz = strut.top[2] - strut.bottom[2]
    if abs(dy) > 1.0e-9:
        raise ValueError(f"{strut.name}: the strut must lie in an x-z plane, dy = {dy}")
    length = math.hypot(dx, dz)
    radius = 0.5 * strut.diameter_m
    if length <= 2.0 * radius:
        raise ValueError(
            f"{strut.name}: length {length:.3f} m is not longer than its diameter "
            f"{strut.diameter_m} m"
        )
    body = BodyModel(
        strut.name,
        strut.material,
        strut.bottom,
        (
            BodySection(0.0, 0.0, 0.0, 0.0, 0.0),
            BodySection(radius, 0.0, strut.diameter_m, 0.0, 0.0),
            BodySection(length - radius, 0.0, strut.diameter_m, 0.0, 0.0),
            BodySection(length, 0.0, 0.0, 0.0, 0.0),
        ),
        strut.symmetric,
    )
    geom = add_body(body)
    _set(geom, "Y_Rel_Rotation", "XForm", -math.degrees(math.atan2(dz, dx)))
    _check_surface(geom, strut.name)
    return geom


def add_gear(gear: GearModel) -> str:
    """GEAR geom: one bogie per unit, contact points on the ground plane."""
    geom: str = vsp.AddGeom("GEAR")
    vsp.SetGeomName(geom, "Tires")
    _set(geom, "m_ModelLenUnits", "Gear", vsp.LEN_M)
    _set(geom, "Z_Rel_Location", "XForm", gear.ground_z_m)
    _set(geom, "ShowNominalGroundPlane", "GroundPlane", 0.0)
    _set(geom, "Tess_U", "Shape", TIRE_TESS_U)
    _set(geom, "Tess_W", "Shape", TIRE_TESS_W)
    for unit in gear.bogies:
        bogie = vsp.CreateAndAddBogie(geom)
        parms = {vsp.GetParmName(p): p for p in vsp.FindContainerParmIDs(bogie)}
        values = {
            "NumAcross": unit.tires_across,
            "NumTandem": 1,
            "SpacingType": vsp.BOGIE_GAP,
            "SpacingGap": unit.tire_gap_m,
            "DiameterIn": unit.tire_diameter_m / INCH_M,
            "WidthIn": unit.tire_width_m / INCH_M,
            "DeflectionPct": 0.0,
            "XContactPt": unit.contact_x_m,
            "YContactPt": unit.contact_y_m,
            "ZAboveGround": 0.0,
            "Symmetrical": float(unit.symmetric),
        }
        for name, value in values.items():
            if name not in parms:
                raise KeyError(f"no OpenVSP bogie parameter {name}")
            vsp.SetParmVal(parms[name], value)
    vsp.SetGeomMaterialName(geom, MATERIAL_DARK)
    _check_surface(geom, "Tires")
    return geom


def build(model: AircraftModel) -> list[str]:
    """Add every component of the model to a cleared OpenVSP model.

    Args:
        model: Resolved model.

    Returns:
        Geom IDs, in build order.
    """
    vsp.ClearVSPModel()
    _add_materials()
    geoms = [add_body(b) for b in model.bodies]
    geoms += [add_surface(s) for s in model.surfaces]
    geoms += [add_ring(r) for r in model.rings]
    geoms += [add_ellipsoid(e) for e in model.ellipsoids]
    geoms += [add_strut(s) for s in model.struts]
    geoms.append(add_gear(model.gear))
    vsp.Update()
    return geoms


def write_vsp3(path: Path) -> None:
    """Write the current model, with the custom materials stored in the file.

    OpenVSP saves only materials flagged as user materials, a flag that
    vsp.AddMaterial does not set, so the file would keep the material names
    but not their colours. They are written into the empty <Materials> node
    in OpenVSP's own format (same fix as wp2 hld_model.write_vsp3).

    Raises:
        RuntimeError: If the file has no empty <Materials/> node.
        OSError: If the file cannot be written or read back.
    """
    vsp.WriteVSPFile(str(path))

    def node(name: str, diffuse: RGB) -> str:
        parts = [f"<Name>{name}</Name>"]
        for key, vals in _material_terms(diffuse).items():
            value = (
                f"{vals[0]:.6f}"
                if key == "Shininess"
                else "".join(f"{v:.6f}, " for v in vals)
            )
            parts.append(f"<{key}>{value}</{key}>")
        return f"<Material>{''.join(parts)}</Material>"

    text = path.read_text(encoding="utf-8")
    if "<Materials/>" not in text:
        raise RuntimeError(f"no empty <Materials/> node in {path}")
    materials = "".join(node(n, d) for n, d in MATERIALS.items())
    path.write_text(
        text.replace("<Materials/>", f"<Materials>{materials}</Materials>", 1),
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    """Build the model from a JSON file and write the .vsp3.

    Args:
        argv: [model JSON path, output .vsp3 path].

    Returns:
        Process exit code: 0 on success, 1 on an error.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if len(argv) != 2:
        logger.error("usage: python builder.py <model_inputs.json> <output.vsp3>")
        return 1
    json_path, vsp3_path = Path(argv[0]), Path(argv[1])
    try:
        model = AircraftModel.read_json(json_path)
        geoms = build(model)
        write_vsp3(vsp3_path)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    logger.info("wrote %s (%d geoms)", vsp3_path, len(geoms))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
