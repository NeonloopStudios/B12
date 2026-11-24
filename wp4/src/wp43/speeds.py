# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-19
"""Design speeds and load factors for the (V_EAS-n) diagram (CS 25.335, 25.337).

Follows the AE2111-I project guide. All speeds are EAS [m/s] unless the name
says otherwise.

- V_S1: stall speed with flaps retracted.
- V_S0: stall speed with flaps extended, always for a named flap configuration
  (take-off, approach or landing).
- Stall speed at load factor n: V_S = sqrt(2 n m g0 / (rho0 S CL_max)).
- n_max = 2.1 + 24000 / (W + 10000), W = MTOW in lb (1 lb = 0.454 kg),
  clamped to [2.5, 3.8] (CS 25.337(b)).
- V_A >= V_S1 sqrt(n_max) (CS 25.335(c)(1)).
- V_D: V_C <= 0.8 V_D, or M_D - M_C >= 0.05 when M_C is limited by
  compressibility (CS 25.335(b)); V_D depends on altitude.
"""

from __future__ import annotations

import math
from typing import Final

from configs.wp43.settings import DesignSpeeds, LiftCoefficients
from wp43.atmosphere import ISA_G0, ISA_RHO0_KG_M3, AtmosphereState

KG_PER_LB: Final[float] = 0.454  # project guide: 1 lb = 0.454 kg
N_MAX_LOWER: Final[float] = 2.5
N_MAX_UPPER: Final[float] = 3.8


def kg_to_lb(mass_kg: float) -> float:
    """Convert a mass from kg to lb (1 lb = 0.454 kg).

    Args:
        mass_kg: Mass [kg].

    Returns:
        Mass [lb].
    """
    return mass_kg / KG_PER_LB


def tas_from_eas(v_eas_ms: float, atm: AtmosphereState) -> float:
    """True airspeed from equivalent airspeed, V_TAS = V_EAS / sqrt(sigma).

    Args:
        v_eas_ms: Equivalent airspeed [m/s].
        atm: Atmosphere state at the flight altitude.

    Returns:
        True airspeed [m/s].
    """
    return v_eas_ms / math.sqrt(atm.sigma)


def eas_from_mach(mach: float, atm: AtmosphereState) -> float:
    """Equivalent airspeed from Mach number, V_EAS = M a sqrt(sigma).

    Args:
        mach: Mach number [-].
        atm: Atmosphere state at the flight altitude.

    Returns:
        Equivalent airspeed [m/s].

    Raises:
        ValueError: If mach is negative or NaN.
    """
    if not mach >= 0.0:
        raise ValueError(f"mach must be >= 0, got {mach}")
    return mach * atm.speed_of_sound_ms * math.sqrt(atm.sigma)


def stall_speed_eas(
    mass_kg: float, cl_max: float, area_m2: float, n: float = 1.0
) -> float:
    """Stall speed at load factor n, V_S = sqrt(2 n m g0 / (rho0 S CL_max)), EAS.

    Args:
        mass_kg: Aircraft mass [kg].
        cl_max: Maximum lift coefficient of the configuration [-].
        area_m2: Wing reference area S [m^2].
        n: Load factor [-].

    Returns:
        Stall speed, EAS [m/s].

    Raises:
        ValueError: If any input is not positive or is NaN.
    """
    for name, value in (
        ("mass_kg", mass_kg), ("cl_max", cl_max), ("area_m2", area_m2), ("n", n)
    ):
        if not value > 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    return math.sqrt(2.0 * n * mass_kg * ISA_G0 / (ISA_RHO0_KG_M3 * area_m2 * cl_max))


# --- Load factors ---


def n_max(mtow_kg: float) -> float:
    """Positive limit manoeuvring load factor (CS 25.337(b)).

    n_max = 2.1 + 24000 / (W + 10000), W = maximum take-off weight in lb,
    not less than 2.5 and not higher than 3.8.

    Args:
        mtow_kg: Maximum take-off mass [kg].

    Returns:
        Limit load factor n_max [-].

    Raises:
        ValueError: If mtow_kg is not positive or is NaN.
    """
    if not mtow_kg > 0.0:
        raise ValueError(f"mtow_kg must be > 0, got {mtow_kg}")
    n = 2.1 + 24_000.0 / (kg_to_lb(mtow_kg) + 10_000.0)
    return min(N_MAX_UPPER, max(N_MAX_LOWER, n))


# --- V_S1: flaps retracted ---


def stall_speed_no_flaps(
    mass_kg: float, area_m2: float, cl: LiftCoefficients
) -> float:
    """Stall speed V_S1, flaps retracted, EAS [m/s].

    Args:
        mass_kg: Aircraft mass [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.

    Returns:
        V_S1, EAS [m/s].
    """
    return stall_speed_eas(mass_kg, cl.cl_max_clean, area_m2)


# --- V_S0: flaps extended, per flap configuration ---


def stall_speed_flaps_takeoff(
    mass_kg: float, area_m2: float, cl: LiftCoefficients
) -> float:
    """Stall speed V_S0, flaps in take-off position, EAS [m/s].

    Args:
        mass_kg: Aircraft mass [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.

    Returns:
        V_S0 (take-off flaps), EAS [m/s].
    """
    return stall_speed_eas(mass_kg, cl.cl_max_takeoff, area_m2)


def stall_speed_flaps_approach(
    mass_kg: float, area_m2: float, cl: LiftCoefficients
) -> float:
    """Stall speed V_S0, flaps in approach position, EAS [m/s].

    Args:
        mass_kg: Aircraft mass [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.

    Returns:
        V_S0 (approach flaps), EAS [m/s].
    """
    return stall_speed_eas(mass_kg, cl.cl_max_app, area_m2)


def stall_speed_flaps_landing(
    mass_kg: float, area_m2: float, cl: LiftCoefficients
) -> float:
    """Stall speed V_S0, flaps in landing position, EAS [m/s].

    Args:
        mass_kg: Aircraft mass [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.

    Returns:
        V_S0 (landing flaps), EAS [m/s].
    """
    return stall_speed_eas(mass_kg, cl.cl_max_landing, area_m2)


# --- V_A: flaps retracted ---


def manoeuvering_speed(
    mass_kg: float, area_m2: float, cl: LiftCoefficients, mtow_kg: float
) -> float:
    """Manoeuvring speed V_A = V_S1 sqrt(n_max), EAS [m/s] (CS 25.335(c)(1)).

    Minimum V_A: the point where n = (V / V_S1)^2 reaches n_max (point A).

    Args:
        mass_kg: Aircraft mass of the diagram [kg].
        area_m2: Wing reference area S [m^2].
        cl: Maximum lift coefficients per configuration.
        mtow_kg: Maximum take-off mass, for n_max [kg].

    Returns:
        V_A, EAS [m/s].
    """
    return stall_speed_no_flaps(mass_kg, area_m2, cl) * math.sqrt(n_max(mtow_kg))


# --- V_D: depends on altitude ---


def dive_speed(speeds: DesignSpeeds, atm: AtmosphereState) -> float:
    """Design dive speed V_D at an altitude, EAS [m/s] (CS 25.335(b)).

    V_C <= 0.8 V_D gives V_D = k_vd V_C (k_vd = 1.25). Where M_C is limited by
    compressibility, the margin M_D = M_C + dm_d (dm_d = 0.05) applies instead.
    The lower of the two is taken, so the Mach margin sets V_D at altitude.

    Args:
        speeds: Design speeds (V_C, M_C, k_vd, dm_d).
        atm: Atmosphere state at the altitude.

    Returns:
        V_D, EAS [m/s].
    """
    v_d_from_v_c = speeds.k_vd * speeds.v_c_eas_ms
    v_d_from_m_d = eas_from_mach(speeds.m_c + speeds.dm_d, atm)
    return min(v_d_from_v_c, v_d_from_m_d)


# --- V_F: design wing-flap speed ---


def design_wing_flap_speed(
    speeds: DesignSpeeds, cl: LiftCoefficients, mtow_kg: float, area_m2: float
) -> float:
    """Design wing-flap speed V_F, EAS [m/s] (CS 25.335(e)(3)).

    V_F is the highest of the three minimums:
    - k_vf_takeoff (1.6) x stall speed, take-off flaps, at MTOM;
    - k_vf_approach (1.8) x stall speed, approach flaps, at MLW;
    - k_vf_landing (1.8) x V_S0, landing flaps, at MLW.

    Assumptions: MLW = MTOM (WP3 landing design mass), and the approach
    CL_max equals the landing CL_max (configs/wp43/settings.py).

    Args:
        speeds: Design speeds (the k_vf factors).
        cl: Maximum lift coefficients per configuration.
        mtow_kg: Maximum take-off mass, also used as MLW [kg].
        area_m2: Wing reference area S [m^2].

    Returns:
        V_F, EAS [m/s].
    """
    v_f_takeoff = speeds.k_vf_takeoff * stall_speed_flaps_takeoff(mtow_kg, area_m2, cl)
    v_f_approach = speeds.k_vf_approach * stall_speed_flaps_approach(
        mtow_kg, area_m2, cl
    )
    v_f_landing = speeds.k_vf_landing * stall_speed_flaps_landing(mtow_kg, area_m2, cl)
    return max(v_f_takeoff, v_f_approach, v_f_landing)