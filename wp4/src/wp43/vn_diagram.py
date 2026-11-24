# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-19
"""(V_EAS-n) manoeuvre diagram: speeds, boundary and corner points.

One diagram per (mass, altitude). All speeds are EAS [m/s]. The speeds come
from wp43.speeds; this module only assembles them.

Construction (AE2111-I project guide):
- Clean envelope, a closed polygon from the origin: n = (V / V_S1)^2 up to
  n_max (point A), flat at n_max to V_D (D), down to n = 0 at V_D (E), linear
  to n_min = -1 at V_C (F), flat at -1 back to V_S1 (H), and n = -(V / V_S1)^2
  back to the origin (CL_min = -CL_max).
- Flaps down: n = (V / V_S0)^2 up to n = 2, then flat at n = 2 until V_F, or
  until it meets n = (V / V_S1)^2 if that comes first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from configs.wp43.settings import N_FLAPS, N_MIN, DesignSpeeds, LiftCoefficients
from wp43 import speeds as sp
from wp43.atmosphere import AtmosphereState

FloatArray: TypeAlias = npt.NDArray[np.float64]

N_CURVE_POINTS: Final[int] = 200  # points per parabola, for a smooth plot


@dataclass(frozen=True)
class VnSpeeds:
    """Speeds and load factor of one V-n diagram, EAS [m/s].

    Attributes:
        v_s0: Stall speed, landing flaps, at the diagram mass [m/s].
        v_s1: Stall speed, flaps retracted, at the diagram mass [m/s].
        v_a: Manoeuvring speed V_A [m/s].
        v_c: Design cruising speed V_C [m/s].
        v_d: Design dive speed V_D at the diagram altitude [m/s].
        v_f: Design wing-flap speed V_F [m/s].
        n_max: Positive limit manoeuvring load factor [-].
    """

    v_s0: float
    v_s1: float
    v_a: float
    v_c: float
    v_d: float
    v_f: float
    n_max: float

    def __post_init__(self) -> None:
        if not 0.0 < self.v_s0 < self.v_s1 < self.v_a < self.v_c < self.v_d:
            raise ValueError(
                "speeds must satisfy 0 < V_S0 < V_S1 < V_A < V_C < V_D, got "
                f"V_S0 = {self.v_s0}, V_S1 = {self.v_s1}, V_A = {self.v_a}, "
                f"V_C = {self.v_c}, V_D = {self.v_d}"
            )
        if not self.v_f > self.v_s0:
            raise ValueError(f"V_F must be > V_S0, got {self.v_f} and {self.v_s0}")
        if not self.n_max > 0.0:
            raise ValueError(f"n_max must be > 0, got {self.n_max}")


def vn_speeds(
    mass_kg: float,
    mtow_kg: float,
    area_m2: float,
    cl: LiftCoefficients,
    speeds: DesignSpeeds,
    atm: AtmosphereState,
) -> VnSpeeds:
    """Speeds of the V-n diagram at one mass and altitude.

    n_max and V_F use the MTOM (V_F also takes MLW = MTOM); the stall speeds
    and V_A use the diagram mass; V_D uses the altitude.

    Args:
        mass_kg: Mass of the diagram [kg].
        mtow_kg: Maximum take-off mass [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.
        speeds: Design speeds (V_C, M_C and the CS 25.335 factors).
        atm: Atmosphere state at the diagram altitude.

    Returns:
        The diagram speeds.

    Raises:
        ValueError: If the speeds are not in the order
            V_S0 < V_S1 < V_A < V_C < V_D.
    """
    return VnSpeeds(
        v_s0=sp.stall_speed_flaps_landing(mass_kg, area_m2, cl),
        v_s1=sp.stall_speed_no_flaps(mass_kg, area_m2, cl),
        v_a=sp.manoeuvering_speed(mass_kg, area_m2, cl, mtow_kg),
        v_c=speeds.v_c_eas_ms,
        v_d=sp.dive_speed(speeds, atm),
        v_f=sp.design_wing_flap_speed(speeds, cl, mtow_kg, area_m2),
        n_max=sp.n_max(mtow_kg),
    )


# --- load factor limits ---


def n_min(v_eas_ms: float, v_c: float, v_d: float) -> float:
    """Negative limit load factor at a speed: -1 up to V_C, linear to 0 at V_D.

    Args:
        v_eas_ms: Speed, EAS [m/s].
        v_c: Design cruising speed V_C, EAS [m/s].
        v_d: Design dive speed V_D, EAS [m/s].

    Returns:
        n_min at that speed [-].

    Raises:
        ValueError: If V_D <= V_C, or the speed is outside [0, V_D] or NaN.
    """
    if not v_d > v_c:
        raise ValueError(f"V_D must be > V_C, got {v_d} and {v_c}")
    if not 0.0 <= v_eas_ms <= v_d:
        raise ValueError(f"v_eas_ms must be in [0, {v_d}], got {v_eas_ms}")
    if v_eas_ms <= v_c:
        return N_MIN
    return N_MIN * (v_d - v_eas_ms) / (v_d - v_c)


# --- envelope ---


@dataclass(frozen=True)
class CornerPoint:
    """A named corner of the envelope; the corners are the critical load cases.

    Attributes:
        name: Point label (A, D, E, F, H).
        v_eas_ms: Speed, EAS [m/s].
        n: Load factor [-].
    """

    name: str
    v_eas_ms: float
    n: float


@dataclass(frozen=True, eq=False)
class VnEnvelope:
    """Boundary of one V-n diagram.

    Attributes:
        speeds: The diagram speeds.
        v_eas: Clean envelope speeds, a closed polygon from the origin [m/s].
        n: Clean envelope load factors, same shape as v_eas [-].
        v_flap: Flaps-down curve speeds [m/s].
        n_flap: Flaps-down curve load factors, same shape as v_flap [-].
        corners: Corner points A, D, E, F, H.
    """

    speeds: VnSpeeds
    v_eas: FloatArray
    n: FloatArray
    v_flap: FloatArray
    n_flap: FloatArray
    corners: tuple[CornerPoint, ...]


def _parabola(
    v_s: float, v_start: float, v_end: float, sign: float, n_points: int
) -> tuple[FloatArray, FloatArray]:
    """Points of the stall curve n = sign (V / V_S)^2 from v_start to v_end."""
    v = np.linspace(v_start, v_end, n_points, dtype=np.float64)
    return v, sign * (v / v_s) ** 2


def _points(*pairs: tuple[float, float]) -> tuple[FloatArray, FloatArray]:
    """Arrays of the (V, n) corner points of straight segments."""
    v, n = zip(*pairs)
    return np.array(v, dtype=np.float64), np.array(n, dtype=np.float64)


def corner_points(vs: VnSpeeds) -> tuple[CornerPoint, ...]:
    """Corners of the clean envelope (A, D, E, F, H).

    Args:
        vs: The diagram speeds.

    Returns:
        A (V_A, n_max), D (V_D, n_max), E (V_D, 0), F (V_C, -1), H (V_S1, -1).
    """
    return (
        CornerPoint("A", vs.v_a, vs.n_max),
        CornerPoint("D", vs.v_d, vs.n_max),
        CornerPoint("E", vs.v_d, 0.0),
        CornerPoint("F", vs.v_c, N_MIN),
        CornerPoint("H", vs.v_s1, N_MIN),
    )


def clean_envelope(
    vs: VnSpeeds, n_points: int = N_CURVE_POINTS
) -> tuple[FloatArray, FloatArray]:
    """Clean (flaps retracted) envelope as a closed polygon from the origin.

    Origin -> n = (V / V_S1)^2 up to n_max -> A -> D -> E -> F -> H ->
    n = -(V / V_S1)^2 back to the origin.

    Args:
        vs: The diagram speeds.
        n_points: Points per parabola.

    Returns:
        (V_EAS [m/s], n [-]), first point equal to the last.

    Raises:
        ValueError: If V_A is below V_S1 sqrt(n_max) (CS 25.335(c)(1)).
    """
    v_stall_n_max = vs.v_s1 * math.sqrt(vs.n_max)
    if not vs.v_a >= v_stall_n_max * (1.0 - 1e-12):
        raise ValueError(
            f"V_A = {vs.v_a} is below V_S1 sqrt(n_max) = {v_stall_n_max}"
        )
    v_up, n_up = _parabola(vs.v_s1, 0.0, v_stall_n_max, 1.0, n_points)
    v_mid, n_mid = _points(
        (vs.v_d, vs.n_max), (vs.v_d, 0.0), (vs.v_c, N_MIN), (vs.v_s1, N_MIN)
    )
    v_down, n_down = _parabola(vs.v_s1, vs.v_s1, 0.0, -1.0, n_points)
    v_all = np.concatenate((v_up, v_mid, v_down[1:]))
    n_all = np.concatenate((n_up, n_mid, n_down[1:]))
    n_all[n_points - 1] = vs.n_max  # exact n_max at the top of the parabola
    return v_all, n_all


def flap_curve(
    vs: VnSpeeds, n_points: int = N_CURVE_POINTS
) -> tuple[FloatArray, FloatArray]:
    """Flaps-down boundary: n = (V / V_S0)^2 up to n = 2, then flat at n = 2.

    The flat part ends at V_F, or where it meets the clean curve
    n = (V / V_S1)^2 (at V_S1 sqrt(2)) if that comes first. If it ends at V_F
    below the clean curve, a vertical line joins it to the clean curve.

    Args:
        vs: The diagram speeds.
        n_points: Points on the parabola.

    Returns:
        (V_EAS [m/s], n [-]) from the origin.

    Raises:
        ValueError: If V_F is below V_S0 sqrt(2), so n = 2 is never reached
            (not possible with the CS 25.335(e) factors, V_F >= 1.8 V_S0).
    """
    v_top = vs.v_s0 * math.sqrt(N_FLAPS)
    if not vs.v_f >= v_top:
        raise ValueError(f"V_F = {vs.v_f} is below V_S0 sqrt(2) = {v_top}")
    v_curve, n_curve = _parabola(vs.v_s0, 0.0, v_top, 1.0, n_points)
    v_meet_clean = vs.v_s1 * math.sqrt(N_FLAPS)
    tail: tuple[tuple[float, float], ...]
    if vs.v_f >= v_meet_clean:
        tail = ((v_meet_clean, N_FLAPS),)
    else:
        tail = ((vs.v_f, N_FLAPS), (vs.v_f, (vs.v_f / vs.v_s1) ** 2))
    v_tail, n_tail = _points(*tail)
    return np.concatenate((v_curve, v_tail)), np.concatenate((n_curve, n_tail))


def manoeuvre_envelope(vs: VnSpeeds, n_points: int = N_CURVE_POINTS) -> VnEnvelope:
    """Clean envelope, flaps-down curve and corner points of one diagram.

    Args:
        vs: The diagram speeds.
        n_points: Points per parabola.

    Returns:
        The envelope.
    """
    v_eas, n = clean_envelope(vs, n_points)
    v_flap, n_flap = flap_curve(vs, n_points)
    return VnEnvelope(
        speeds=vs, v_eas=v_eas, n=n, v_flap=v_flap, n_flap=n_flap,
        corners=corner_points(vs),
    )
