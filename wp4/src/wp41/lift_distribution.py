# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""WP4.1 lift and quarter-chord moment of one load case on the global stations.

Reads the VSPAERO database (out/wp41/lift_distribution.txt and
moment_distribution.txt: cl(y) and cm_c4(y) for each Mach and alpha of the
grid) and gives N'(y) and m'_c4(y) of one load case on wp41.stations.
Design: wp4/README_41.md, section 5.

Method for one load case:

1. Weight: W = W_OEM + W_payload + W_fuel [N] (case_weights for a mass case).
2. Flight state (ISA): q = 0.5 rho0 V_EAS^2, V_TAS = V_EAS sqrt(rho0 / rho),
   M = V_TAS / a.
3. Required lift coefficient: CL_d = n W / (q S).
4. Mach: cubic spline (scipy CubicSpline) over the Mach grid of the wing CL,
   cl(y) and cm_c4(y) at both angles of attack.
5. Alpha: linear between the two database angles a0 and a1, because the
   fixed-wake inviscid solution is linear in alpha:
   s = (CL_d - CL_a0) / (CL_a1 - CL_a0), alpha_d = a0 + s (a1 - a0),
   cl_d(y) = cl_a0(y) + s (cl_a1(y) - cl_a0(y)), and the same for cm_c4.
   s outside [0, 1] is an extrapolation and is flagged.
6. Stations: cl c and cm_c4 c^2 go from the strip centres to the stations
   with a monotone cubic (scipy PchipInterpolator), with 0 at the tip.
7. N'(y) = q cl_d c, m'_c4(y) = q cm_c4_d c^2. Check: 2 integral(N' dy) = n W
   (scipy trapezoid).

Assumptions:

* Inviscid VLM without stall: the V-n diagram must keep CL below CL_max.
* N' = L': cos(alpha) is taken as 1 and the drag term is dropped.
* y is the VSPAERO (projected) spanwise coordinate. The stations are
  y = eta b/2; the 2 deg dihedral (cos 2 deg = 0.9994) is neglected.
* weight_per_span w(y) (structure plus fuel) gives only the inertial load
  n w(y). Its structure is part of W_OEM and its fuel is W_fuel, so w(y) is
  not added to W a second time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from configs.aircraft import AircraftMasses
from configs.wp41 import settings
from configs.wp41.load_cases import LoadCase, MassCase
from wp41.atmosphere import G0, isa
from wp41.contracts import LiftTable
from wp41.stations import station_y

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]
WeightFunction: TypeAlias = Callable[[FloatArray], FloatArray]

RHO0: Final[float] = isa(0.0)[2]  # ISA sea-level density [kg/m^3]

# Round-off allowance on s at the two database angles: s = -1e-16 at alpha_0
# is not an extrapolation.
S_ROUND_OFF: Final[float] = 1.0e-9


# ============================================================
#  DATABASE
# ============================================================


@dataclass(frozen=True)
class AeroDatabase:
    """VSPAERO coefficients of the right half-wing on the (Mach, alpha) grid.

    Attributes:
        mach: Mach grid, ascending, shape (n_mach,).
        alpha_deg: The two angles of attack, ascending, shape (2,) [deg].
        y: Strip centres, ascending, shape (n_strip,) [m].
        chord: Strip chords, shape (n_strip,) [m].
        cl: Strip lift coefficient, shape (n_strip, n_mach, 2).
        cm_c4: Strip moment coefficient about the local c/4, nose-up
            positive, shape (n_strip, n_mach, 2).
        cl_wing: Wing lift coefficient CL, shape (n_mach, 2).
        cm_wing: Wing pitching moment coefficient about the MAC quarter
            chord, nose-up positive, shape (n_mach, 2).
        s_ref_m2: Reference area, both halves [m^2].
        half_span_m: b / 2 [m].
    """

    mach: FloatArray
    alpha_deg: FloatArray
    y: FloatArray
    chord: FloatArray
    cl: FloatArray
    cm_c4: FloatArray
    cl_wing: FloatArray
    cm_wing: FloatArray
    s_ref_m2: float
    half_span_m: float


@dataclass(frozen=True)
class _GridFile:
    """One distribution file sorted into the (Mach, alpha) grid."""

    mach: FloatArray
    alpha: FloatArray
    y: FloatArray
    chord: FloatArray
    values: FloatArray  # (n_strip, n_mach, 2)
    totals: FloatArray  # (n_mach, 2)
    s_ref_m2: float
    half_span_m: float


def _read_header(path: Path, keys: tuple[str, ...]) -> dict[str, FloatArray]:
    """The 'key: values' lines of the file header for the given keys."""
    values: dict[str, FloatArray] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        key, sep, rest = line[1:].partition(":")
        if sep and key.strip() in keys:
            values[key.strip()] = np.array(rest.split(), dtype=np.float64)
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{path}: header keys {missing} not found")
    return values


def _read_grid_file(path: Path, total_key: str) -> _GridFile:
    """Read one distribution file of wp41.aero_loads and sort it into the grid.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header and the columns do not form a full
            (Mach, alpha) grid with two angles of attack, or the strips are
            not ascending inside the half-span.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path} (run wp41_aero_loads.py first)")
    keys = (settings.KEY_S_REF, settings.KEY_HALF_SPAN, settings.KEY_MACH,
            settings.KEY_ALPHA, total_key)
    header = _read_header(path, keys)
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    mach_case = header[settings.KEY_MACH]
    alpha_case = header[settings.KEY_ALPHA]
    total_case = header[total_key]
    n_case = mach_case.size
    if alpha_case.size != n_case or total_case.size != n_case or data.shape[1] != 2 + n_case:
        raise ValueError(
            f"{path}: {n_case} Mach values, {alpha_case.size} alpha values, "
            f"{total_case.size} {total_key} values and {data.shape[1] - 2} columns "
            "must be equal"
        )

    machs = np.unique(mach_case)
    alphas = np.unique(alpha_case)
    if alphas.size != 2 or machs.size * alphas.size != n_case:
        raise ValueError(
            f"{path}: need 2 angles of attack at every Mach, got Mach {machs} "
            f"and alpha {alphas} in {n_case} columns"
        )
    order = np.lexsort((alpha_case, mach_case))  # Mach, then alpha
    grid_mach = mach_case[order].reshape(machs.size, 2)
    grid_alpha = alpha_case[order].reshape(machs.size, 2)
    if not (np.allclose(grid_mach, machs[:, None]) and np.allclose(grid_alpha, alphas)):
        raise ValueError(f"{path}: the (Mach, alpha) columns do not form a full grid")

    y = data[:, 0].copy()
    half_span = float(header[settings.KEY_HALF_SPAN][0])
    if not (np.all(np.diff(y) > 0.0) and y[0] >= 0.0 and y[-1] < half_span):
        raise ValueError(f"{path}: strip y must ascend inside [0, {half_span}), got {y}")

    return _GridFile(
        mach=machs,
        alpha=alphas,
        y=y,
        chord=data[:, 1].copy(),
        values=np.ascontiguousarray(data[:, 2:][:, order].reshape(y.size, machs.size, 2)),
        totals=total_case[order].reshape(machs.size, 2),
        s_ref_m2=float(header[settings.KEY_S_REF][0]),
        half_span_m=half_span,
    )


def read_aero_database(lift_path: Path, moment_path: Path) -> AeroDatabase:
    """Read the lift and the moment distribution files into an AeroDatabase.

    Args:
        lift_path: out/wp41/lift_distribution.txt.
        moment_path: out/wp41/moment_distribution.txt.

    Returns:
        The database, with the cases sorted by Mach, then by alpha.

    Raises:
        FileNotFoundError: If a file does not exist.
        ValueError: If a file is not a full grid, or the two files have
            different strips, grids or reference values.
    """
    lift = _read_grid_file(lift_path, settings.KEY_CL)
    moment = _read_grid_file(moment_path, settings.KEY_CMY)
    same = (
        lift.y.shape == moment.y.shape
        and np.allclose(lift.y, moment.y)
        and np.allclose(lift.chord, moment.chord)
        and np.array_equal(lift.mach, moment.mach)
        and np.array_equal(lift.alpha, moment.alpha)
        and lift.s_ref_m2 == moment.s_ref_m2
        and lift.half_span_m == moment.half_span_m
    )
    if not same:
        raise ValueError(
            f"{lift_path} and {moment_path} do not come from the same VSPAERO run"
        )
    return AeroDatabase(
        mach=lift.mach,
        alpha_deg=lift.alpha,
        y=lift.y,
        chord=lift.chord,
        cl=lift.values,
        cm_c4=moment.values,
        cl_wing=lift.totals,
        cm_wing=moment.totals,
        s_ref_m2=lift.s_ref_m2,
        half_span_m=lift.half_span_m,
    )


# ============================================================
#  FLIGHT STATE AND WEIGHTS
# ============================================================


@dataclass(frozen=True)
class FlightState:
    """Speeds and dynamic pressure of one flight condition.

    Attributes:
        altitude_m: Geopotential altitude [m].
        v_eas: Equivalent airspeed [m/s].
        v_tas: True airspeed [m/s].
        density: Air density [kg/m^3].
        speed_of_sound: Speed of sound [m/s].
        mach: Mach number [-].
        dynamic_pressure: q = 0.5 rho0 V_EAS^2 = 0.5 rho V_TAS^2 [Pa].
    """

    altitude_m: float
    v_eas: float
    v_tas: float
    density: float
    speed_of_sound: float
    mach: float
    dynamic_pressure: float


def _speeds(
    v_eas: FloatArray, altitude_m: float
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """(V_TAS, Mach, q, rho, a) for equivalent airspeeds at one altitude (ISA)."""
    _, a, rho = isa(altitude_m)
    v_tas = v_eas * math.sqrt(RHO0 / rho)
    return v_tas, v_tas / a, 0.5 * RHO0 * v_eas**2, rho, a


def flight_state(v_eas: float, altitude_m: float) -> FlightState:
    """Flight state at an equivalent airspeed and altitude (ISA).

    Args:
        v_eas: Equivalent airspeed [m/s], > 0.
        altitude_m: Geopotential altitude, 0 to 20 km [m].

    Returns:
        The flight state.

    Raises:
        ValueError: If v_eas is not > 0 or the altitude is out of range.
    """
    if not v_eas > 0.0:
        raise ValueError(f"v_eas must be > 0, got {v_eas}")
    v_tas, mach, q, rho, a = _speeds(np.array([v_eas], dtype=np.float64), altitude_m)
    return FlightState(
        altitude_m, v_eas, float(v_tas[0]), rho, a, float(mach[0]), float(q[0])
    )


def case_weights(mass_case: MassCase, masses: AircraftMasses) -> tuple[float, float, float]:
    """(W_OEM, W_payload, W_fuel) [N] of one mass case.

    * oew: OEM only.
    * mtow: OEM + maximum payload + the fuel that gives MTOM.
    * no_fuel: OEM + maximum payload, no fuel.

    Raises:
        ValueError: If the mass case is unknown, or MTOM is below OEM plus
            the maximum payload.
    """
    oem = G0 * masses.oem_kg
    mpw = G0 * masses.max_payload_kg
    if mass_case == "oew":
        return oem, 0.0, 0.0
    if mass_case == "no_fuel":
        return oem, mpw, 0.0
    if mass_case == "mtow":
        fuel = G0 * masses.mtom_kg - oem - mpw
        if not fuel >= 0.0:
            raise ValueError(f"MTOM is below OEM + maximum payload: fuel = {fuel} N")
        return oem, mpw, fuel
    raise ValueError(f"unknown mass case {mass_case!r}")


def _check_mach(database: AeroDatabase, mach: FloatArray) -> None:
    """Raise ValueError if a Mach number is outside the database grid."""
    lo, hi = float(database.mach[0]), float(database.mach[-1])
    outside = mach[~((mach >= lo) & (mach <= hi))]
    if outside.size:
        raise ValueError(
            f"Mach {outside} outside the database grid [{lo}, {hi}]: change the "
            "speed or the altitude, or widen the VSPAERO Mach grid"
        )


def _check_weights(w_oem: float, w_payload: float, w_fuel: float) -> None:
    """Raise ValueError unless W_OEM > 0, W_payload >= 0 and W_fuel >= 0."""
    if not w_oem > 0.0:
        raise ValueError(f"w_oem must be > 0, got {w_oem}")
    if not w_payload >= 0.0:
        raise ValueError(f"w_payload must be >= 0, got {w_payload}")
    if not w_fuel >= 0.0:
        raise ValueError(f"w_fuel must be >= 0, got {w_fuel}")


# ============================================================
#  LIFT DISTRIBUTION
# ============================================================


@dataclass(frozen=True)
class LiftDistribution:
    """Lift and quarter-chord moment of one load case on the global stations.

    Attributes:
        y: Stations y = eta b/2, shape (n_station,) [m].
        chord: Local chord, shape (n_station,) [m].
        cl: Local lift coefficient cl_d, shape (n_station,).
        cm_c4: Local moment coefficient about c/4, nose-up positive.
        lift_per_span: N'(y) = L'(y), up positive [N/m].
        moment_per_span: m'_c4(y), about the local c/4, nose-up positive [N m/m].
        inertial_per_span: n w(y), weight per unit span times the load
            factor, down positive [N/m]. Zero without a weight distribution.
        weight: W = W_OEM + W_payload + W_fuel [N].
        load_factor: n [-].
        cl_wing: Required wing lift coefficient CL_d = n W / (q S) [-].
        alpha_deg: Freestream angle of attack that gives CL_d [deg].
        s: Interpolation weight between the two database angles [-].
        extrapolated: True if s is outside [0, 1].
        state: Flight state of the load case.
    """

    y: FloatArray
    chord: FloatArray
    cl: FloatArray
    cm_c4: FloatArray
    lift_per_span: FloatArray
    moment_per_span: FloatArray
    inertial_per_span: FloatArray
    weight: float
    load_factor: float
    cl_wing: float
    alpha_deg: float
    s: float
    extrapolated: bool
    state: FlightState


def _to_stations(database: AeroDatabase, strip_values: FloatArray, y: FloatArray) -> FloatArray:
    """Strip values to the stations: PCHIP through the strip centres and 0 at the tip."""
    y_points = np.append(database.y, database.half_span_m)
    v_points = np.append(strip_values, 0.0)
    return np.asarray(PchipInterpolator(y_points, v_points)(y), dtype=np.float64)


def _weight_on_stations(weight_per_span: WeightFunction | None, y: FloatArray) -> FloatArray:
    """w(y) on the stations; zero without a weight distribution."""
    if weight_per_span is None:
        return np.zeros(y.shape, dtype=np.float64)
    w = np.asarray(weight_per_span(y), dtype=np.float64)
    if w.shape != y.shape or not np.all(np.isfinite(w)):
        raise ValueError(
            f"weight_per_span must return {y.shape} finite values, got shape {w.shape}"
        )
    return w


def _at_mach(values: FloatArray, database: AeroDatabase, mach: float, axis: int) -> FloatArray:
    """Cubic spline over the database Mach grid, evaluated at one Mach number."""
    spline = CubicSpline(database.mach, values, axis=axis)
    return np.asarray(spline(mach), dtype=np.float64)


def lift_distribution(
    database: AeroDatabase,
    w_oem: float,
    w_payload: float,
    v_eas: float,
    altitude_m: float,
    load_factor: float,
    w_fuel: float = 0.0,
    weight_per_span: WeightFunction | None = None,
) -> LiftDistribution:
    """Lift and quarter-chord moment per unit span of one load case.

    Args:
        database: VSPAERO database (read_aero_database).
        w_oem: Operating empty weight [N], > 0.
        w_payload: Payload weight [N], >= 0.
        v_eas: Equivalent airspeed [m/s], > 0.
        altitude_m: Geopotential altitude [m].
        load_factor: Load factor n [-]; lift = n W.
        w_fuel: Fuel weight [N], >= 0.
        weight_per_span: w(y), structural plus fuel weight per unit span
            [N/m], called with the station y [m]. None means no distribution.

    Returns:
        The lift distribution on wp41.stations.

    Raises:
        ValueError: If an input is out of range, the Mach number is outside
            the database grid, or 2 integral(N' dy) differs from n W by more
            than settings.LIFT_CHECK_TOLERANCE times W.
    """
    _check_weights(w_oem, w_payload, w_fuel)
    if not math.isfinite(load_factor):
        raise ValueError(f"load_factor must be finite, got {load_factor}")
    weight = w_oem + w_payload + w_fuel
    state = flight_state(v_eas, altitude_m)
    _check_mach(database, np.array([state.mach], dtype=np.float64))
    cl_wing = load_factor * weight / (state.dynamic_pressure * database.s_ref_m2)

    # step 4: cubic spline in Mach
    cl_wing_m = _at_mach(database.cl_wing, database, state.mach, axis=0)  # (2,)
    cl_m = _at_mach(database.cl, database, state.mach, axis=1)  # (n_strip, 2)
    cm_m = _at_mach(database.cm_c4, database, state.mach, axis=1)  # (n_strip, 2)

    # step 5: linear in alpha
    s = float((cl_wing - cl_wing_m[0]) / (cl_wing_m[1] - cl_wing_m[0]))
    a0, a1 = database.alpha_deg
    alpha = float(a0 + s * (a1 - a0))
    cl_strips = cl_m[:, 0] + s * (cl_m[:, 1] - cl_m[:, 0])
    cm_strips = cm_m[:, 0] + s * (cm_m[:, 1] - cm_m[:, 0])

    # steps 6 and 7: stations, N'(y), m'_c4(y)
    y = station_y(database.half_span_m)
    chord = np.asarray(  # linear taper
        np.polyval(np.polyfit(database.y, database.chord, 1), y), dtype=np.float64
    )
    lift_coef = _to_stations(database, cl_strips * database.chord, y)  # cl c [m]
    moment_coef = _to_stations(database, cm_strips * database.chord**2, y)  # cm c^2 [m^2]
    lift = state.dynamic_pressure * lift_coef

    total = 2.0 * float(trapezoid(lift, y))
    if not abs(total - load_factor * weight) <= settings.LIFT_CHECK_TOLERANCE * weight:
        logger.error("Lift check failed: 2 int N' dy = %.1f N, n W = %.1f N", total,
                     load_factor * weight)
        raise ValueError(
            f"2 integral(N' dy) = {total:.1f} N differs from n W = "
            f"{load_factor * weight:.1f} N by more than "
            f"{settings.LIFT_CHECK_TOLERANCE:.1%} of W"
        )
    extrapolated = not -S_ROUND_OFF <= s <= 1.0 + S_ROUND_OFF
    logger.info(
        "W = %.0f N, n = %.2f, V_EAS = %.1f m/s, M = %.3f: CL = %.4f, alpha = %.2f deg%s",
        weight, load_factor, v_eas, state.mach, cl_wing, alpha,
        " (extrapolated)" if extrapolated else "",
    )
    return LiftDistribution(
        y=y,
        chord=chord,
        cl=lift_coef / chord,
        cm_c4=moment_coef / chord**2,
        lift_per_span=lift,
        moment_per_span=state.dynamic_pressure * moment_coef,
        inertial_per_span=load_factor * _weight_on_stations(weight_per_span, y),
        weight=weight,
        load_factor=load_factor,
        cl_wing=cl_wing,
        alpha_deg=alpha,
        s=s,
        extrapolated=extrapolated,
        state=state,
    )


def lift_for_case(
    database: AeroDatabase, case: LoadCase, masses: AircraftMasses
) -> LiftDistribution:
    """Lift distribution of one WP4.3 load case (weights from its mass case)."""
    w_oem, w_payload, w_fuel = case_weights(case.mass_case, masses)
    return lift_distribution(
        database, w_oem=w_oem, w_payload=w_payload, v_eas=case.v_eas,
        altitude_m=case.altitude_m, load_factor=case.n, w_fuel=w_fuel,
    )


def to_lift_table(lift: LiftDistribution, case_name: str) -> LiftTable:
    """The lift.csv contract of one load case.

    The database is evaluated at the Mach number of the case, so mach_run
    equals mach_case.
    """
    return LiftTable(
        case=case_name,
        n=lift.load_factor,
        w_n=lift.weight,
        q_pa=lift.state.dynamic_pressure,
        cl_total=lift.cl_wing,
        alpha_d_deg=lift.alpha_deg,
        mach_case=lift.state.mach,
        mach_run=lift.state.mach,
        extrapolated=lift.extrapolated,
        y_m=lift.y,
        chord_m=lift.chord,
        cl_d=lift.cl,
        cm_c4_d=lift.cm_c4,
        n_prime_npm=lift.lift_per_span,
        m_prime_c4_nmpm=lift.moment_per_span,
    )


# ============================================================
#  ANGLE-OF-ATTACK GRID
# ============================================================


@dataclass(frozen=True)
class AoaGrid:
    """Required wing CL and angle of attack over weights and speeds.

    Attributes:
        weights: Aircraft weights, shape (n_w,) [N].
        v_eas: Equivalent airspeeds, shape (n_v,) [m/s].
        altitude_m: Geopotential altitude [m].
        load_factor: n [-].
        v_tas: True airspeeds, shape (n_v,) [m/s].
        mach: Mach numbers, shape (n_v,).
        dynamic_pressure: q, shape (n_v,) [Pa].
        cl_wing: n W / (q S), shape (n_w, n_v).
        alpha_deg: Freestream angle of attack, shape (n_w, n_v) [deg].
    """

    weights: FloatArray
    v_eas: FloatArray
    altitude_m: float
    load_factor: float
    v_tas: FloatArray
    mach: FloatArray
    dynamic_pressure: FloatArray
    cl_wing: FloatArray
    alpha_deg: FloatArray


def aoa_grid(
    database: AeroDatabase,
    weights: FloatArray,
    v_eas: FloatArray,
    altitude_m: float,
    load_factor: float,
) -> AoaGrid:
    """Required CL and angle of attack for every (weight, speed) pair.

    Same method as lift_distribution, steps 2 to 5, for the wing CL only.

    Args:
        database: VSPAERO database.
        weights: Aircraft weights [N], 1-D, > 0.
        v_eas: Equivalent airspeeds [m/s], 1-D, > 0.
        altitude_m: Geopotential altitude [m].
        load_factor: Load factor n [-].

    Returns:
        The grid.

    Raises:
        ValueError: If an input is out of range or a Mach number is outside
            the database grid.
    """
    for name, values in (("weights", weights), ("v_eas", v_eas)):
        if values.ndim != 1 or values.size == 0 or not np.all(values > 0.0):
            raise ValueError(f"{name} must be a non-empty 1-D array of values > 0")
    if not math.isfinite(load_factor):
        raise ValueError(f"load_factor must be finite, got {load_factor}")

    v_tas, mach, q, _, _ = _speeds(v_eas, altitude_m)
    _check_mach(database, mach)
    cl_wing_m = np.asarray(
        CubicSpline(database.mach, database.cl_wing, axis=0)(mach), dtype=np.float64
    )  # (n_v, 2)
    cl_wing = load_factor * weights[:, None] / (q[None, :] * database.s_ref_m2)
    s = (cl_wing - cl_wing_m[None, :, 0]) / (cl_wing_m[None, :, 1] - cl_wing_m[None, :, 0])
    a0, a1 = database.alpha_deg
    return AoaGrid(
        weights=weights,
        v_eas=v_eas,
        altitude_m=altitude_m,
        load_factor=load_factor,
        v_tas=v_tas,
        mach=mach,
        dynamic_pressure=q,
        cl_wing=cl_wing,
        alpha_deg=a0 + s * (a1 - a0),
    )
