# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Miscellaneous drag terms C_D,misc (ADSEE-II, "Miscellaneous drag").

    drag divergence (Mason):  M_DD = k_a/cos L - (t/c)/cos^2 L - C_L/(10 cos^3 L)
    wave drag:                 0                                  M < M_cr
                               0.002 [1 + 2.5 (M_DD - M)/0.05]^-1  M_cr <= M <= M_DD
                               0.002 [1 + (M - M_DD)/0.05]^2.5     M_DD < M
    fuselage upsweep:          D/q = 3.83 u^2.5 A_max              (u in rad)
    fuselage base:             D/q = [0.139 + 0.419 (M - 0.161)^2] A_base
    flap:                      dCD = F_flap (c_f/c)(S_flap/S_ref)(delta - 10)
    landing gear:              dC_Ds = coeff exp(5.615 S_A/(d w)),
                               dC_D,ref = dC_Ds (d w) / S_ref
    excrescence and leakage:   a fraction f of the total CD0

A D/q term is a drag area [m^2]; divide it by S_ref for a drag coefficient.
"""

from __future__ import annotations

import math
from typing import Final

# --- wave drag ---
WAVE_DRAG_AT_MDD: Final[float] = 0.002
WAVE_SUB_MDD_SLOPE: Final[float] = 2.5
WAVE_MACH_STEP: Final[float] = 0.05
WAVE_SUPER_MDD_EXPONENT: Final[float] = 2.5
MASON_CL_DIVISOR: Final[float] = 10.0

# --- fuselage ---
UPSWEEP_COEFF: Final[float] = 3.83
UPSWEEP_EXPONENT: Final[float] = 2.5
BASE_CONSTANT: Final[float] = 0.139
BASE_MACH_COEFF: Final[float] = 0.419
BASE_MACH_OFFSET: Final[float] = 0.161

# --- flap ---
FLAP_THRESHOLD_DEG: Final[float] = 10.0

# --- landing gear ---
GEAR_EXPONENT_COEFF: Final[float] = 5.615
# Frontal area ratio m = S_A / (d w): x-axis range of the ADSEE-II chart.
GEAR_AREA_RATIO_MAX: Final[float] = 1.0


def drag_divergence_mach(
    kappa_a: float,
    thickness_to_chord: float,
    lift_coefficient: float,
    sweep_rad: float,
) -> float:
    """Drag-divergence Mach number of a swept wing (Korn equation, Mason form).

    Args:
        kappa_a: Airfoil technology factor k_a [-].
        thickness_to_chord: Streamwise t/c [-].
        lift_coefficient: Wing C_L [-].
        sweep_rad: Wing sweep angle [rad].

    Returns:
        M_DD [-].

    Raises:
        ValueError: If k_a or t/c is not positive, or |sweep| >= pi/2.
    """
    if kappa_a <= 0.0:
        raise ValueError(f"kappa_a must be > 0, got {kappa_a}")
    if not 0.0 < thickness_to_chord < 1.0:
        raise ValueError(
            f"thickness_to_chord must be in (0, 1), got {thickness_to_chord}"
        )
    if not abs(sweep_rad) < math.pi / 2:
        raise ValueError(f"|sweep_rad| must be < pi/2, got {sweep_rad}")
    cos_sweep = math.cos(sweep_rad)
    return (
        kappa_a / cos_sweep
        - thickness_to_chord / cos_sweep**2
        - lift_coefficient / (MASON_CL_DIVISOR * cos_sweep**3)
    )


def streamwise_critical_mach(mach_critical_normal: float, sweep_rad: float) -> float:
    """Free-stream critical Mach number from the sweep-normal section value.

    Simple sweep theory: the normal Mach number is M cos(sweep), so the
    free-stream M_cr is M_cr,n / cos(sweep).

    Args:
        mach_critical_normal: Critical Mach number normal to the sweep line [-].
        sweep_rad: Sweep angle of that line [rad].

    Returns:
        Free-stream (streamwise) M_cr [-].
    """
    if not 0.0 < mach_critical_normal < 1.0:
        raise ValueError(
            f"mach_critical_normal must be in (0, 1), got {mach_critical_normal}"
        )
    if not abs(sweep_rad) < math.pi / 2:
        raise ValueError(f"|sweep_rad| must be < pi/2, got {sweep_rad}")
    return mach_critical_normal / math.cos(sweep_rad)


def wave_drag(mach: float, mach_critical: float, mach_dd: float) -> float:
    """Wave drag coefficient from the ADSEE-II three-branch relation.

    Args:
        mach: Free-stream Mach number [-].
        mach_critical: Critical Mach number M_cr [-].
        mach_dd: Drag-divergence Mach number M_DD [-].

    Returns:
        Delta C_D of the wave drag [-].

    Raises:
        ValueError: If M < 0 or M_cr > M_DD.
    """
    if mach < 0.0:
        raise ValueError(f"mach must be >= 0, got {mach}")
    if mach_critical > mach_dd:
        raise ValueError(f"mach_critical {mach_critical} > mach_dd {mach_dd}")
    if mach < mach_critical:
        return 0.0
    if mach <= mach_dd:
        return WAVE_DRAG_AT_MDD / (
            1.0 + WAVE_SUB_MDD_SLOPE * (mach_dd - mach) / WAVE_MACH_STEP
        )
    rise: float = (1.0 + (mach - mach_dd) / WAVE_MACH_STEP) ** WAVE_SUPER_MDD_EXPONENT
    return WAVE_DRAG_AT_MDD * rise


def upsweep_drag_area(upsweep_rad: float, max_frontal_area_m2: float) -> float:
    """Drag area D/q of the fuselage upsweep, 3.83 u^2.5 A_max.

    Args:
        upsweep_rad: Aft-body upsweep angle u [rad], >= 0.
        max_frontal_area_m2: Maximum fuselage cross-section area [m^2].

    Returns:
        D/q [m^2].
    """
    if upsweep_rad < 0.0:
        raise ValueError(f"upsweep_rad must be >= 0, got {upsweep_rad}")
    if max_frontal_area_m2 <= 0.0:
        raise ValueError(f"max_frontal_area_m2 must be > 0, got {max_frontal_area_m2}")
    upsweep_term: float = upsweep_rad**UPSWEEP_EXPONENT
    return UPSWEEP_COEFF * upsweep_term * max_frontal_area_m2


def base_drag_area(mach: float, base_area_m2: float) -> float:
    """Drag area D/q of the fuselage base, [0.139 + 0.419 (M - 0.161)^2] A_base.

    Args:
        mach: Free-stream Mach number [-], subsonic only.
        base_area_m2: Base area A_base [m^2], >= 0.

    Returns:
        D/q [m^2].

    Raises:
        ValueError: If M is not in [0, 1) or the base area is negative.
    """
    if not 0.0 <= mach < 1.0:
        raise ValueError(
            f"base drag is subsonic only: mach must be in [0, 1), got {mach}"
        )
    if base_area_m2 < 0.0:
        raise ValueError(f"base_area_m2 must be >= 0, got {base_area_m2}")
    coefficient = BASE_CONSTANT + BASE_MACH_COEFF * (mach - BASE_MACH_OFFSET) ** 2
    return coefficient * base_area_m2


def flap_drag(
    flap_factor: float,
    chord_ratio: float,
    flap_area_m2: float,
    reference_area_m2: float,
    deflection_deg: float,
) -> float:
    """Flap drag coefficient, F_flap (c_f/c)(S_flap/S_ref)(delta - 10).

    Args:
        flap_factor: F_flap [-] (0.0144 plain, 0.0074 slotted).
        chord_ratio: c_f / c [-].
        flap_area_m2: Wing area affected by the flap S_flap [m^2].
        reference_area_m2: Reference area S_ref [m^2].
        deflection_deg: Flap deflection [deg].

    Returns:
        Delta C_D [-]. Zero for a deflection of 10 deg or less.
    """
    if flap_factor <= 0.0:
        raise ValueError(f"flap_factor must be > 0, got {flap_factor}")
    if not 0.0 < chord_ratio < 1.0:
        raise ValueError(f"chord_ratio must be in (0, 1), got {chord_ratio}")
    if flap_area_m2 <= 0.0 or reference_area_m2 <= 0.0:
        raise ValueError(
            f"areas must be > 0, got S_flap={flap_area_m2}, S_ref={reference_area_m2}"
        )
    if deflection_deg < 0.0:
        raise ValueError(f"deflection_deg must be >= 0, got {deflection_deg}")
    if deflection_deg <= FLAP_THRESHOLD_DEG:
        return 0.0
    area_ratio = flap_area_m2 / reference_area_m2
    return (
        flap_factor * chord_ratio * area_ratio * (deflection_deg - FLAP_THRESHOLD_DEG)
    )


def landing_gear_drag(
    wheel_well_coeff: float,
    frontal_area_m2: float,
    tire_width_m: float,
    tire_diameter_m: float,
    reference_area_m2: float,
) -> float:
    """Drag coefficient of one retractable landing gear unit, on S_ref.

    Args:
        wheel_well_coeff: Fit coefficient (0.05328 open, 0.04955 closed) [-].
        frontal_area_m2: Frontal area S_A [m^2].
        tire_width_m: Tire width w [m].
        tire_diameter_m: Tire diameter d [m].
        reference_area_m2: Reference area S_ref [m^2].

    Returns:
        Delta C_D,ref = coeff exp(5.615 S_A/(d w)) (d w) / S_ref [-].

    Raises:
        ValueError: If an input is not positive, or m = S_A/(d w) is outside
            the chart range (0, 1].
    """
    for name, value in (
        ("wheel_well_coeff", wheel_well_coeff),
        ("frontal_area_m2", frontal_area_m2),
        ("tire_width_m", tire_width_m),
        ("tire_diameter_m", tire_diameter_m),
        ("reference_area_m2", reference_area_m2),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    gear_reference_area = tire_diameter_m * tire_width_m
    area_ratio = frontal_area_m2 / gear_reference_area
    if area_ratio > GEAR_AREA_RATIO_MAX:
        raise ValueError(
            f"frontal area ratio m = S_A/(d w) = {area_ratio} is outside the "
            f"chart range (0, {GEAR_AREA_RATIO_MAX}]"
        )
    gear_cd = wheel_well_coeff * math.exp(GEAR_EXPONENT_COEFF * area_ratio)
    return gear_cd * gear_reference_area / reference_area_m2


def excrescence_drag(cd0_without: float, fraction: float) -> float:
    """Excrescence and leakage drag as a fraction of the total CD0.

    With CD0_total = CD0_without + dCD and dCD = f CD0_total:
    dCD = f / (1 - f) * CD0_without.

    Args:
        cd0_without: CD0 of every other term [-].
        fraction: Fraction f of the total CD0 [-], in [0, 1).

    Returns:
        Delta C_D [-].
    """
    if cd0_without < 0.0:
        raise ValueError(f"cd0_without must be >= 0, got {cd0_without}")
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    return fraction / (1.0 - fraction) * cd0_without
