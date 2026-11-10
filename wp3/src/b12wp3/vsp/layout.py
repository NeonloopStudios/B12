# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Stage 1 of the OpenVSP model: inputs -> positions and shapes.

build_model() turns the sizes (geometry.yaml) and the positions
(vsp_model.yaml) into an AircraftModel. No OpenVSP call happens here, so all
of it is testable in the wp3 env.

Shapes:

* Fuselage: circular sections. Nose of length L1 on an ellipse,
  r = R sqrt(1 - (1 - x/L1)^2), its centre falling to nose_tip_z at the tip.
  Cylinder of length L2. Tail cone of length L3, s = (x - L1 - L2) / L3:
  radius r = r_b + (R - r_b)(1 - s^2) down to the base radius
  r_b = sqrt(A_base / pi); centreline z_c = L3 tan(u) s^2. The centreline
  leaves the cylinder axis without a kink, and the line from the cone start
  to the tail end is at the upsweep angle u. A straight centreline at u
  would push the upper line above the cylinder (a hump) for L3 tan u > 0.
* Wing and tails: the trapezoids of geometry.yaml. The tails are placed by
  the tail arms: horizontal distance from the c.g. to the tail MAC c/4.
  The fin root is on the local
  fuselage centreline (geometry.yaml measures the fin span from there).
* Nacelle: the Torenbeek Type B outer line through D_h (inlet), D_n (at
  beta l_n) and D_ef (exit), with a rounded lip and an inner duct line. The
  wall section is revolved about the engine axis (open inlet and exit).
* Landing gear: tires on the ground plane, a straight strut from each axle
  into the fuselage or the wing box.

Axes: x aft from the nose tip, y to the right wing, z up from the axis of
the fuselage cylinder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from b12wp3.common.planform import QUARTER_CHORD, TrapezoidalPlanform
from b12wp3.config.geometry import AircraftGeometry, BodyInputs, NacelleInputs
from b12wp3.config.vsp_model import (
    FuselageShape,
    NacelleDetail,
    VspModelInputs,
    WingPlacement,
)
from b12wp3.vsp.model import (
    MATERIAL_DARK,
    MATERIAL_FUSELAGE,
    MATERIAL_LIFTING,
    MATERIAL_METAL,
    MATERIAL_NACELLE,
    NACA_FOUR_DIGIT,
    POINTS,
    AircraftModel,
    AirfoilModel,
    BodyModel,
    BodySection,
    BogieModel,
    EllipsoidModel,
    GearModel,
    Point2,
    Point3,
    RingModel,
    StrutModel,
    SurfaceModel,
)

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

PROFILE_POINTS: Final[int] = 41  # points per nacelle wall line, LE to TE
ENGINE_COUNT: Final[int] = 2  # one nacelle per wing half
NOSE_GEAR_UNITS: Final[int] = 1
MAIN_GEAR_UNITS: Final[int] = 2
STRUT_RAKE_WARNING_DEG: Final[float] = 20.0  # report a strut raked more than this


# ============================================================
#  RESULT TYPES
# ============================================================


@dataclass(frozen=True)
class LayoutReport:
    """Derived positions and checks, for the run report.

    Attributes:
        wing_mac_c4_x_m: x of the wing MAC quarter chord [m].
        cg_x_m: x of the c.g. [m].
        cg_mac_fraction: c.g. behind the MAC leading edge, fraction of MAC [-].
        vertical_tail_root_le: Fin root leading edge (x, z) [m].
        horizontal_tail_root_le: Horizontal-tail root leading edge (x, z) [m].
        nacelle_inlet: Right nacelle inlet centre (x, y, z) [m].
        nacelle_ground_clearance_m: Ground to the nacelle bottom [m].
        nose_strut_length_m: Axle to strut top [m].
        main_strut_length_m: Axle to strut top [m].
        main_strut_rake_deg: Strut angle from vertical, positive = top aft [deg].
        main_gear_mount: "fuselage" or "wing".
        warnings: Checks that failed but still give a usable picture.
    """

    wing_mac_c4_x_m: float
    cg_x_m: float
    cg_mac_fraction: float
    vertical_tail_root_le: Point2
    horizontal_tail_root_le: Point2
    nacelle_inlet: Point3
    nacelle_ground_clearance_m: float
    nose_strut_length_m: float
    main_strut_length_m: float
    main_strut_rake_deg: float
    main_gear_mount: str
    warnings: tuple[str, ...]


# ============================================================
#  FUSELAGE
# ============================================================


@dataclass(frozen=True)
class FuselageProfile:
    """Radius and centre height of the fuselage along x.

    Attributes:
        body: Nose, cylinder and tail-cone lengths and diameter.
        upsweep_rad: Tail-cone centreline angle u [rad].
        base_radius_m: Radius at the tail end r_b [m].
        nose_tip_z_m: z of the nose tip [m].
    """

    body: BodyInputs
    upsweep_rad: float
    base_radius_m: float
    nose_tip_z_m: float

    def __post_init__(self) -> None:
        if self.body.tail_length_m <= 0.0:
            raise ValueError(
                "the VSP model needs a tail cone, got tail_length_m "
                f"{self.body.tail_length_m}"
            )
        if self.base_radius_m >= 0.5 * self.body.diameter_m:
            raise ValueError(
                f"base radius {self.base_radius_m} m is not smaller than the "
                f"fuselage radius {0.5 * self.body.diameter_m} m"
            )

    @property
    def radius_m(self) -> float:
        """Cylinder radius R [m]."""
        return 0.5 * self.body.diameter_m

    @property
    def tail_start_m(self) -> float:
        """x of the tail-cone start, L1 + L2 [m]."""
        return self.body.nose_length_m + self.body.cylinder_length_m

    def radius(self, x: FloatArray) -> FloatArray:
        """Local radius r(x) [m], x in [0, L].

        Args:
            x: Stations along the fuselage [m].

        Returns:
            Radius at each station [m].

        Raises:
            ValueError: If a station is outside the fuselage.
        """
        self._check(x)
        big_r, l1, l3 = self.radius_m, self.body.nose_length_m, self.body.tail_length_m
        nose = big_r * np.sqrt(np.clip(1.0 - (1.0 - x / l1) ** 2, 0.0, 1.0))
        s = np.clip((x - self.tail_start_m) / l3, 0.0, 1.0)
        tail = self.base_radius_m + (big_r - self.base_radius_m) * (1.0 - s**2)
        result: FloatArray = np.where(
            x < l1, nose, np.where(x > self.tail_start_m, tail, big_r)
        )
        return result

    def centre_z(self, x: FloatArray) -> FloatArray:
        """Height of the section centre z_c(x) [m], x in [0, L].

        Args:
            x: Stations along the fuselage [m].

        Returns:
            Centre height at each station [m].

        Raises:
            ValueError: If a station is outside the fuselage.
        """
        r = self.radius(x)
        nose = self.nose_tip_z_m * (1.0 - r / self.radius_m)
        s = np.clip((x - self.tail_start_m) / self.body.tail_length_m, 0.0, 1.0)
        tail = self.body.tail_length_m * math.tan(self.upsweep_rad) * s**2
        result: FloatArray = np.where(
            x < self.body.nose_length_m,
            nose,
            np.where(x > self.tail_start_m, tail, 0.0),
        )
        return result

    def slopes(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Centreline and radius slopes (dz_c/dx, dr/dx) [-], x in [0, L].

        Nose: dr/dx = R (1 - xi) / (L1 sqrt(1 - (1 - xi)^2)), xi = x/L1,
        infinite at the tip (reported as 0 there: a point has no tangent);
        dz_c/dx = -(z_tip / R) dr/dx. Tail cone: dr/dx = -2 (R - r_b) s / L3,
        dz_c/dx = 2 tan(u) s. Cylinder: both 0.

        Args:
            x: Stations along the fuselage [m].

        Returns:
            (dz_c/dx, dr/dx) at each station.

        Raises:
            ValueError: If a station is outside the fuselage.
        """
        self._check(x)
        big_r, l1, l3 = self.radius_m, self.body.nose_length_m, self.body.tail_length_m
        q = np.clip(1.0 - x / l1, 0.0, 1.0)
        root = np.sqrt(1.0 - q**2)
        nose_dr = np.divide(
            big_r * q,
            l1 * root,
            out=np.zeros_like(x, dtype=np.float64),
            where=root > 0.0,
        )
        s = np.clip((x - self.tail_start_m) / l3, 0.0, 1.0)
        in_nose, in_tail = x < l1, x > self.tail_start_m
        dr: FloatArray = np.where(
            in_nose,
            nose_dr,
            np.where(in_tail, -2.0 * (big_r - self.base_radius_m) * s / l3, 0.0),
        )
        dz: FloatArray = np.where(
            in_nose,
            -self.nose_tip_z_m / big_r * nose_dr,
            np.where(in_tail, 2.0 * math.tan(self.upsweep_rad) * s, 0.0),
        )
        return dz, dr

    def bottom_z(self, x: float, y: float) -> float:
        """z of the lower surface at (x, y), |y| < r(x) [m].

        Args:
            x: Station [m].
            y: Lateral position [m].

        Returns:
            Lower-surface height [m].

        Raises:
            ValueError: If (x, y) is outside the fuselage plan view.
        """
        xs = np.array([x], dtype=np.float64)
        r = float(self.radius(xs)[0])
        if abs(y) >= r:
            raise ValueError(
                f"y = {y} m is outside the fuselage radius {r} m at x = {x} m"
            )
        return float(self.centre_z(xs)[0]) - math.sqrt(r * r - y * y)

    def _check(self, x: FloatArray) -> None:
        """Raise ValueError if a station is outside [0, L]."""
        if np.any(x < 0.0) or np.any(x > self.body.length_m):
            raise ValueError(
                f"stations {x} are outside the fuselage [0, {self.body.length_m}] m"
            )


def fuselage_profile(
    geometry: AircraftGeometry, shape: FuselageShape
) -> FuselageProfile:
    """Fuselage profile from the inputs.

    Args:
        geometry: Sizes (fuselage, upsweep, base area).
        shape: Visual nose droop.

    Returns:
        The profile.
    """
    return FuselageProfile(
        body=geometry.fuselage,
        upsweep_rad=math.radians(geometry.fuselage_aft.upsweep_deg),
        base_radius_m=math.sqrt(geometry.fuselage_aft.base_area_m2 / math.pi),
        nose_tip_z_m=shape.nose_tip_z_m,
    )


def fuselage_sections(
    profile: FuselageProfile, shape: FuselageShape
) -> tuple[BodySection, ...]:
    """Cross-sections of the fuselage: nose, cylinder end, tail cone.

    Nose stations follow the ellipse angle t, x = L1 (1 - cos t), which
    clusters them at the tip. Tail stations are evenly spaced.

    Args:
        profile: Fuselage profile.
        shape: Section counts.

    Returns:
        Sections from the nose tip (a point) to the tail end.
    """
    body = profile.body
    angles = np.linspace(0.0, 0.5 * math.pi, shape.nose_sections + 1, dtype=np.float64)
    x_nose = body.nose_length_m * (1.0 - np.cos(angles))
    x_nose[-1] = body.nose_length_m  # exact end of the nose
    s = np.linspace(0.0, 1.0, shape.tail_sections + 1, dtype=np.float64)[1:]
    x_tail = profile.tail_start_m + s * body.tail_length_m
    x_tail[-1] = body.length_m  # exact end of the fuselage
    parts = [x_nose]
    if body.cylinder_length_m > 0.0:
        parts.append(np.array([profile.tail_start_m], dtype=np.float64))
    parts.append(x_tail)
    x = np.concatenate(parts)
    diameter = 2.0 * profile.radius(x)
    diameter[0] = 0.0  # nose tip point
    z = profile.centre_z(x)
    dz, dr = profile.slopes(x)
    return tuple(
        BodySection(float(xi), float(zi), float(di), float(dzi), float(dri))
        for xi, zi, di, dzi, dri in zip(x, z, diameter, dz, dr, strict=True)
    )


# ============================================================
#  WING AND TAILS
# ============================================================


@dataclass(frozen=True)
class WingFrame:
    """Position of the wing planform in the aircraft axes.

    Attributes:
        planform: Wing trapezoid.
        placement: Root position, dihedral, incidence, twist.
    """

    planform: TrapezoidalPlanform
    placement: WingPlacement

    @property
    def semi_span_y_m(self) -> float:
        """y of the wing tip; OpenVSP measures the panel span along the dihedral."""
        return self.planform.panel_span_m * math.cos(
            math.radians(self.placement.dihedral_deg)
        )

    @property
    def mac_c4_x_m(self) -> float:
        """x of the MAC quarter chord [m]."""
        return self.placement.root_le_x_m + self.planform.mac_quarter_chord_offset_m

    @property
    def mac_le_x_m(self) -> float:
        """x of the MAC leading edge, x_LEMAC [m]."""
        return self.mac_c4_x_m - QUARTER_CHORD * self.planform.mac_m

    def mac_fraction(self, x: float) -> float:
        """Position x behind the MAC leading edge, as a fraction of the MAC [-]."""
        return (x - self.mac_le_x_m) / self.planform.mac_m

    def panel_station(self, y: float) -> float:
        """Distance along the panel span of the lateral position y [m].

        Raises:
            ValueError: If y is outside the wing.
        """
        if not 0.0 <= y <= self.semi_span_y_m:
            raise ValueError(
                f"y = {y} m is outside the wing [0, {self.semi_span_y_m}] m"
            )
        return y / math.cos(math.radians(self.placement.dihedral_deg))

    def le_x(self, y: float) -> float:
        """x of the leading edge at y [m]."""
        tan_le = math.tan(self.planform.sweep_at(0.0))
        return self.placement.root_le_x_m + self.panel_station(y) * tan_le

    def chord(self, y: float) -> float:
        """Local chord at y [m]."""
        return self.planform.chord_at(self.panel_station(y))

    def chord_line_z(self, x: float, y: float) -> float:
        """z of the chord line at (x, y), small-angle incidence and dihedral [m]."""
        s = self.panel_station(y)
        dz_dihedral = s * math.sin(math.radians(self.placement.dihedral_deg))
        dz_incidence = (x - self.placement.root_le_x_m) * math.tan(
            math.radians(self.placement.incidence_deg)
        )
        return self.placement.root_z_m + dz_dihedral - dz_incidence


def _naca(thickness_to_chord: float) -> AirfoilModel:
    """Symmetric NACA 4-digit section."""
    return AirfoilModel(NACA_FOUR_DIGIT, thickness_to_chord, (), ())


def _points(upper: FloatArray, lower: FloatArray) -> tuple[tuple[Point2, ...], ...]:
    """Two (N, 2) arrays as tuples of (x, y) points."""
    return tuple(
        tuple((float(x), float(y)) for x, y in curve) for curve in (upper, lower)
    )


def wing_surface(
    frame: WingFrame, airfoil: tuple[FloatArray, FloatArray]
) -> SurfaceModel:
    """The wing, both halves, from the centreline.

    Args:
        frame: Wing planform and placement.
        airfoil: (upper, lower) section points, LE to TE.

    Returns:
        The wing surface.
    """
    upper, lower = _points(*airfoil)
    p = frame.placement
    return SurfaceModel(
        name="Wing",
        material=MATERIAL_LIFTING,
        root_le=(p.root_le_x_m, 0.0, p.root_z_m),
        x_rotation_deg=0.0,
        incidence_deg=p.incidence_deg,
        span_m=frame.planform.panel_span_m,
        root_chord_m=frame.planform.root_chord_m,
        tip_chord_m=frame.planform.tip_chord_m,
        sweep_deg=math.degrees(frame.planform.sweep_c4_rad),
        sweep_location=QUARTER_CHORD,
        dihedral_deg=p.dihedral_deg,
        tip_twist_deg=p.tip_twist_deg,
        airfoil=AirfoilModel(POINTS, 0.0, upper, lower),
        symmetric=True,
    )


def tail_surfaces(
    geometry: AircraftGeometry,
    inputs: VspModelInputs,
    profile: FuselageProfile,
    warnings: list[str],
) -> tuple[SurfaceModel, SurfaceModel]:
    """Fin and cruciform horizontal tail, placed by the tail arms.

    A tail arm is the horizontal distance from the c.g. to the tail MAC
    quarter chord, so x_root_LE = x_cg + l - (x of the MAC c/4 behind the
    root LE).

    Args:
        geometry: Tail sizes and sections.
        inputs: c.g. position, tail arms and horizontal-tail height.
        profile: Fuselage profile (fin root height).
        warnings: Failed placement checks are appended here.

    Returns:
        (vertical tail, horizontal tail).
    """
    tails = inputs.tails
    vt_in, ht_in = geometry.vertical_tail, geometry.horizontal_tail
    vt, ht = vt_in.planform(), ht_in.planform()
    x_cg = inputs.centre_of_gravity.x_m
    vt_le_x = x_cg + tails.vertical_arm_m - vt.mac_quarter_chord_offset_m
    vt_c4_x = vt_le_x + QUARTER_CHORD * vt.root_chord_m
    length = profile.body.length_m
    if vt_c4_x > length:
        warnings.append(
            f"fin root quarter chord x = {vt_c4_x:.3f} m is behind the fuselage "
            f"end {length:.3f} m"
        )
    vt_root_z = float(profile.centre_z(np.array([min(vt_c4_x, length)]))[0])

    height = tails.horizontal_height_fraction * vt.panel_span_m
    ht_le_x = x_cg + tails.horizontal_arm_m - ht.mac_quarter_chord_offset_m
    ht_z = vt_root_z + height
    fin_le_x = vt_le_x + height * math.tan(vt.sweep_at(0.0))
    fin_te_x = fin_le_x + vt.chord_at(height)
    if not fin_le_x <= ht_le_x <= fin_te_x:
        warnings.append(
            f"horizontal-tail root LE x = {ht_le_x:.3f} m is outside the fin "
            f"chord [{fin_le_x:.3f}, {fin_te_x:.3f}] m at its height"
        )

    def surface(
        name: str,
        planform: TrapezoidalPlanform,
        t_c: float,
        root_le: Point3,
        x_rotation: float,
        incidence: float,
    ) -> SurfaceModel:
        return SurfaceModel(
            name=name,
            material=MATERIAL_LIFTING,
            root_le=root_le,
            x_rotation_deg=x_rotation,
            incidence_deg=incidence,
            span_m=planform.panel_span_m,
            root_chord_m=planform.root_chord_m,
            tip_chord_m=planform.tip_chord_m,
            sweep_deg=math.degrees(planform.sweep_c4_rad),
            sweep_location=QUARTER_CHORD,
            dihedral_deg=0.0,
            tip_twist_deg=0.0,
            airfoil=_naca(t_c),
            symmetric=planform.panels == 2,
        )

    return (
        surface(
            "Vertical tail",
            vt,
            vt_in.thickness_to_chord,
            (vt_le_x, 0.0, vt_root_z),
            90.0,
            0.0,
        ),
        surface(
            "Horizontal tail",
            ht,
            ht_in.thickness_to_chord,
            (ht_le_x, 0.0, ht_z),
            0.0,
            tails.horizontal_incidence_deg,
        ),
    )


# ============================================================
#  NACELLE
# ============================================================


@dataclass(frozen=True)
class NacelleProfile:
    """Wall of the nacelle in (x, r), x from the inlet plane.

    Attributes:
        x: Stations, 0 to l_n, clustered at both ends [m].
        outer: Outer radius at each station [m].
        inner: Inner (duct) radius at each station [m].
    """

    x: FloatArray
    outer: FloatArray
    inner: FloatArray

    def inner_at(self, x: float) -> float:
        """Inner radius at x by linear interpolation [m]."""
        return float(np.interp(x, self.x, self.inner))


def nacelle_profile(nacelle: NacelleInputs, detail: NacelleDetail) -> NacelleProfile:
    """Outer and inner wall lines of the nacelle.

    Outer: R_h + (R_n - R_h) sqrt(1 - (1 - x/(beta l))^2) up to beta l (a
    round lip with a vertical tangent at the inlet), then
    R_n - (R_n - R_ef) s^2 to the exit, s = (x - beta l)/((1 - beta) l).
    Inner: an elliptic lip of thickness t_lip and length lip_length_fraction
    l, then a straight duct to R_ef - t_exit at the exit.

    Args:
        nacelle: Torenbeek outline.
        detail: Lip and exit wall details.

    Returns:
        The wall lines.

    Raises:
        ValueError: If the inner line is not inside the outer line.
    """
    length = nacelle.length_m
    r_h, r_n = 0.5 * nacelle.highlight_diameter_m, 0.5 * nacelle.max_diameter_m
    r_ef = 0.5 * nacelle.exit_diameter_m
    x_max = nacelle.max_diameter_position * length
    x = (
        0.5
        * length
        * (1.0 - np.cos(np.linspace(0.0, math.pi, PROFILE_POINTS, dtype=np.float64)))
    )

    front = r_h + (r_n - r_h) * np.sqrt(np.clip(1.0 - (1.0 - x / x_max) ** 2, 0.0, 1.0))
    s = np.clip((x - x_max) / (length - x_max), 0.0, 1.0)
    outer = np.where(x <= x_max, front, r_n - (r_n - r_ef) * s**2)

    x_lip = detail.lip_length_fraction * length
    r_throat = r_h - detail.lip_thickness_m
    r_exit_in = r_ef - detail.exit_wall_thickness_m
    lip = r_h - detail.lip_thickness_m * np.sqrt(
        np.clip(1.0 - (1.0 - x / x_lip) ** 2, 0.0, 1.0)
    )
    duct = r_throat + (r_exit_in - r_throat) * (x - x_lip) / (length - x_lip)
    inner = np.where(x <= x_lip, lip, duct)
    if r_exit_in <= 0.0 or np.any(inner[1:] >= outer[1:]):
        raise ValueError(
            f"nacelle inner wall is not inside the outer wall: lip "
            f"{detail.lip_thickness_m} m, exit wall {detail.exit_wall_thickness_m} m"
        )
    return NacelleProfile(x=x, outer=outer, inner=inner)


def ring_section(
    profile: NacelleProfile,
) -> tuple[float, float, FloatArray, FloatArray]:
    """The nacelle wall as an airfoil in its chord frame.

    The chord runs from the lip point (0, R_h) to the midpoint of the exit
    wall. Its angle a is positive when the exit is closer to the axis. A
    point (dx, dr) from the lip goes to u = (dx cos a - dr sin a)/c,
    v = (dx sin a + dr cos a)/c.

    Args:
        profile: Wall lines.

    Returns:
        (chord [m], angle [deg], upper (outer) points, lower (inner) points).
    """
    r_le = float(profile.outer[0])
    length = float(profile.x[-1])
    r_te = 0.5 * float(profile.outer[-1] + profile.inner[-1])
    angle = math.atan2(r_le - r_te, length)
    chord = math.hypot(length, r_le - r_te)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def to_chord_frame(r: FloatArray) -> FloatArray:
        dr = r - r_le
        u = (profile.x * cos_a - dr * sin_a) / chord
        v = (profile.x * sin_a + dr * cos_a) / chord
        return np.ascontiguousarray(np.column_stack((u, v)), dtype=np.float64)

    return (
        chord,
        math.degrees(angle),
        to_chord_frame(profile.outer),
        to_chord_frame(profile.inner),
    )


@dataclass(frozen=True)
class NacelleParts:
    """Every component of the right engine installation (mirrored left).

    Attributes:
        shell: Nacelle wall.
        fan: Fan face disc.
        spinner: Spinner.
        pylon: Pylon.
        inlet: Inlet centre (x, y, z) [m].
        bottom_z_m: z of the nacelle bottom [m].
    """

    shell: RingModel
    fan: EllipsoidModel
    spinner: EllipsoidModel
    pylon: SurfaceModel
    inlet: Point3
    bottom_z_m: float


def nacelle_parts(
    geometry: AircraftGeometry,
    inputs: VspModelInputs,
    frame: WingFrame,
    radius_m: float,
) -> NacelleParts:
    """Nacelle, fan face, spinner and pylon of the right engine.

    Args:
        geometry: Nacelle outline, engine count.
        inputs: Nacelle position, details, pylon.
        frame: Wing frame.
        radius_m: Fuselage cylinder radius [m].

    Returns:
        The engine installation.

    Raises:
        ValueError: If the engine count is not 2, or the nacelle is inside
            the fuselage, past the wing tip, or above the wing.
    """
    if geometry.engine_count != ENGINE_COUNT:
        raise ValueError(
            f"the VSP model has one nacelle per wing half ({ENGINE_COUNT} engines), "
            f"got engine_count {geometry.engine_count}"
        )
    nac, pos, det = geometry.nacelle, inputs.nacelle_position, inputs.nacelle_detail
    r_n, length, y = 0.5 * nac.max_diameter_m, nac.length_m, pos.axis_y_m
    if y - r_n <= radius_m or y + r_n >= frame.semi_span_y_m:
        raise ValueError(
            f"nacelle at y = {y} m (radius {r_n} m) must lie between the fuselage "
            f"side {radius_m} m and the wing tip {frame.semi_span_y_m:.3f} m"
        )
    le_x = frame.le_x(y)
    x0 = le_x - pos.inlet_ahead_of_le_m
    z_axis = frame.chord_line_z(le_x, y) - pos.axis_below_chord_m

    profile = nacelle_profile(nac, det)
    chord, angle, upper, lower = ring_section(profile)
    up, lo = _points(upper, lower)
    shell = RingModel(
        "Nacelle",
        MATERIAL_NACELLE,
        (x0, y, z_axis),
        nac.highlight_diameter_m,
        chord,
        angle,
        up,
        lo,
        True,
    )

    x_fan = det.fan_face_position * length
    r_fan = profile.inner_at(x_fan)
    if det.spinner_diameter_m >= 2.0 * r_fan:
        raise ValueError(
            f"spinner diameter {det.spinner_diameter_m} m does not fit the fan "
            f"face diameter {2.0 * r_fan:.3f} m"
        )
    centre = (x0 + x_fan, y, z_axis)
    r_s = 0.5 * det.spinner_diameter_m
    fan = EllipsoidModel(
        "Fan face",
        MATERIAL_DARK,
        centre,
        (0.5 * det.lip_thickness_m, r_fan, r_fan),
        True,
    )
    spinner = EllipsoidModel(
        "Spinner", MATERIAL_DARK, centre, (det.spinner_length_m, r_s, r_s), True
    )
    pylon = _pylon(inputs, frame, profile, (x0, y, z_axis), length)
    return NacelleParts(shell, fan, spinner, pylon, (x0, y, z_axis), z_axis - r_n)


def _pylon(
    inputs: VspModelInputs,
    frame: WingFrame,
    profile: NacelleProfile,
    inlet: Point3,
    length: float,
) -> SurfaceModel:
    """Pylon from inside the nacelle wall up to the wing chord line.

    The root sits mid-wall over the pylon chord. With the LE sweep L and
    incidence i, the tip meets the chord line z_c(x) when
    span (1 + tan L tan i) = z_c(le_x) - (x_root - le_x) tan i - z_root.
    """
    pyl = inputs.pylon
    x0, y, z_axis = inlet
    xs0, xs1 = pyl.le_position * length, (pyl.le_position + pyl.chord_fraction) * length
    in_chord = (profile.x >= xs0) & (profile.x <= xs1)
    inner_max, outer_min = float(profile.inner[in_chord].max()), float(
        profile.outer[in_chord].min()
    )
    z_root = z_axis + 0.5 * (inner_max + outer_min)
    x_root = x0 + xs0
    le_x = frame.le_x(y)
    tan_sweep = math.tan(math.radians(pyl.sweep_le_deg))
    tan_inc = math.tan(math.radians(inputs.wing.incidence_deg))
    span = (frame.chord_line_z(le_x, y) - (x_root - le_x) * tan_inc - z_root) / (
        1.0 + tan_sweep * tan_inc
    )
    if span <= 0.0:
        raise ValueError(
            f"pylon span {span:.3f} m <= 0: the nacelle top is above the wing "
            "chord line"
        )
    chord = pyl.chord_fraction * length
    return SurfaceModel(
        name="Pylon",
        material=MATERIAL_LIFTING,
        root_le=(x_root, y, z_root),
        x_rotation_deg=90.0,
        incidence_deg=0.0,
        span_m=span,
        root_chord_m=chord,
        tip_chord_m=chord,
        sweep_deg=pyl.sweep_le_deg,
        sweep_location=0.0,
        dihedral_deg=0.0,
        tip_twist_deg=0.0,
        airfoil=_naca(pyl.thickness_to_chord),
        symmetric=True,
    )


# ============================================================
#  LANDING GEAR
# ============================================================


@dataclass(frozen=True)
class GearParts:
    """Tires and struts.

    Attributes:
        gear: Tires on the ground plane.
        nose_strut: Nose gear strut.
        main_strut: Right main gear strut (mirrored left).
        main_mount: "fuselage" or "wing".
    """

    gear: GearModel
    nose_strut: StrutModel
    main_strut: StrutModel
    main_mount: str


def gear_parts(
    geometry: AircraftGeometry,
    inputs: VspModelInputs,
    frame: WingFrame,
    profile: FuselageProfile,
) -> GearParts:
    """Nose and main gear, shown down, with the tires on the ground.

    Args:
        geometry: Tire sizes, gear unit counts, flap chords (wing box).
        inputs: Gear positions, ground clearance.
        frame: Wing frame.
        profile: Fuselage profile.

    Returns:
        Tires and struts.

    Raises:
        ValueError: If the unit counts differ from 1 nose and 2 main units,
            or a tire would be inside the fuselage or a strut outside the
            wing.
    """
    nose_in, main_in = geometry.nose_gear, geometry.main_gear
    if nose_in.count != NOSE_GEAR_UNITS or main_in.count != MAIN_GEAR_UNITS:
        raise ValueError(
            f"the VSP model has {NOSE_GEAR_UNITS} nose and {MAIN_GEAR_UNITS} main "
            f"gear units, got {nose_in.count} and {main_in.count}"
        )
    nose_pos, main_pos = inputs.nose_gear_position, inputs.main_gear_position
    ground_z = -profile.radius_m - inputs.ground.fuselage_clearance_m

    nose_bottom = (nose_pos.contact_x_m, 0.0, ground_z + 0.5 * nose_in.tire_diameter_m)
    nose_top_z = profile.bottom_z(nose_pos.contact_x_m, 0.0) + nose_pos.strut_diameter_m
    nose_strut = StrutModel(
        "Nose gear strut",
        MATERIAL_METAL,
        nose_bottom,
        (nose_pos.contact_x_m, 0.0, nose_top_z),
        nose_pos.strut_diameter_m,
        False,
    )

    y = 0.5 * main_pos.track_m
    x_contact = main_pos.contact_x_m
    main_bottom = (x_contact, y, ground_z + 0.5 * main_in.tire_diameter_m)
    r_local = float(profile.radius(np.array([x_contact], dtype=np.float64))[0])
    if y < r_local:
        mount = "fuselage"
        top = (x_contact, y, profile.bottom_z(x_contact, y) + main_pos.strut_diameter_m)
    else:
        mount = "wing"
        le_x, chord = frame.le_x(y), frame.chord(y)
        x_front = le_x + geometry.le_flap.chord_ratio * chord
        x_rear = le_x + (1.0 - geometry.te_flap.chord_ratio) * chord
        x_top = min(max(x_contact, x_front), x_rear)
        top = (x_top, y, frame.chord_line_z(x_top, y))
    main_strut = StrutModel(
        "Main gear strut",
        MATERIAL_METAL,
        main_bottom,
        top,
        main_pos.strut_diameter_m,
        True,
    )
    for strut in (nose_strut, main_strut):
        if strut.top[2] <= strut.bottom[2]:
            raise ValueError(
                f"{strut.name}: the axle z = {strut.bottom[2]:.3f} m is not below "
                f"the strut top z = {strut.top[2]:.3f} m; increase "
                "ground.fuselage_clearance_m"
            )

    def bogie(
        name: str,
        x: float,
        y: float,
        d: float,
        w_pair: float,
        n: int,
        gap: float,
        sym: bool,
    ) -> BogieModel:
        return BogieModel(name, x, y, d, w_pair / n, n, gap, sym)

    gear = GearModel(
        ground_z_m=ground_z,
        bogies=(
            bogie(
                "Nose gear",
                nose_pos.contact_x_m,
                0.0,
                nose_in.tire_diameter_m,
                nose_in.tire_width_m,
                nose_pos.tires_per_unit,
                nose_pos.tire_gap_m,
                False,
            ),
            bogie(
                "Main gear",
                x_contact,
                y,
                main_in.tire_diameter_m,
                main_in.tire_width_m,
                main_pos.tires_per_unit,
                main_pos.tire_gap_m,
                True,
            ),
        ),
    )
    return GearParts(gear, nose_strut, main_strut, mount)


def _strut_length_and_rake(strut: StrutModel) -> tuple[float, float]:
    """Strut length [m] and angle from vertical, positive = top aft [deg]."""
    dx = strut.top[0] - strut.bottom[0]
    dz = strut.top[2] - strut.bottom[2]
    return math.hypot(dx, dz), math.degrees(math.atan2(dx, dz))


# ============================================================
#  WHOLE AIRCRAFT
# ============================================================


def build_model(
    geometry: AircraftGeometry,
    inputs: VspModelInputs,
    wing_airfoil: tuple[FloatArray, FloatArray],
) -> tuple[AircraftModel, LayoutReport]:
    """Resolve every component of the 3-view model.

    Args:
        geometry: Sizes (geometry.yaml).
        inputs: Positions and visual details (vsp_model.yaml).
        wing_airfoil: Wing section (upper, lower), LE to TE.

    Returns:
        (model for the OpenVSP builder, report of positions and checks).

    Raises:
        ValueError: If the inputs give an impossible arrangement (the
            message names the values).
    """
    warnings: list[str] = []
    profile = fuselage_profile(geometry, inputs.fuselage_shape)
    frame = WingFrame(geometry.wing.planform(), inputs.wing)
    fuselage = BodyModel(
        "Fuselage",
        MATERIAL_FUSELAGE,
        (0.0, 0.0, 0.0),
        fuselage_sections(profile, inputs.fuselage_shape),
        False,
    )
    vertical, horizontal = tail_surfaces(geometry, inputs, profile, warnings)
    x_cg = inputs.centre_of_gravity.x_m
    cg_mac = frame.mac_fraction(x_cg)
    if not 0.0 <= cg_mac <= 1.0:
        warnings.append(
            f"c.g. x = {x_cg:.3f} m is at {100.0 * cg_mac:.1f} % MAC, outside the "
            f"wing MAC [{frame.mac_le_x_m:.3f}, "
            f"{frame.mac_le_x_m + frame.planform.mac_m:.3f}] m"
        )
    engine = nacelle_parts(geometry, inputs, frame, profile.radius_m)
    gear = gear_parts(geometry, inputs, frame, profile)

    clearance = engine.bottom_z_m - gear.gear.ground_z_m
    if clearance <= 0.0:
        warnings.append(f"nacelle bottom is {-clearance:.3f} m below the ground")
    nose_length, _ = _strut_length_and_rake(gear.nose_strut)
    main_length, main_rake = _strut_length_and_rake(gear.main_strut)
    if abs(main_rake) > STRUT_RAKE_WARNING_DEG:
        warnings.append(
            f"main strut is raked {main_rake:.1f} deg: the wing box is far from "
            "the contact point"
        )
    for message in warnings:
        logger.warning("%s", message)

    model = AircraftModel(
        bodies=(fuselage,),
        surfaces=(
            wing_surface(frame, wing_airfoil),
            vertical,
            horizontal,
            engine.pylon,
        ),
        rings=(engine.shell,),
        ellipsoids=(engine.fan, engine.spinner),
        struts=(gear.nose_strut, gear.main_strut),
        gear=gear.gear,
    )
    report = LayoutReport(
        wing_mac_c4_x_m=frame.mac_c4_x_m,
        cg_x_m=x_cg,
        cg_mac_fraction=cg_mac,
        vertical_tail_root_le=(vertical.root_le[0], vertical.root_le[2]),
        horizontal_tail_root_le=(horizontal.root_le[0], horizontal.root_le[2]),
        nacelle_inlet=engine.inlet,
        nacelle_ground_clearance_m=clearance,
        nose_strut_length_m=nose_length,
        main_strut_length_m=main_length,
        main_strut_rake_deg=main_rake,
        main_gear_mount=gear.main_mount,
        warnings=tuple(warnings),
    )
    return model, report
