# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Flat-plate skin friction coefficient (ADSEE-II, slide 61).

    laminar:     Cf = 1.328 / sqrt(Re)
    turbulent:   Cf = 0.455 / ( (log10 Re)^2.58 * (1 + 0.144 M^2)^0.65 )
    subsonic:    Re = min( rho V l / mu, 38.21 (l/k)^1.053 )
    transonic:   Re = min( rho V l / mu, 44.62 (l/k)^1.053 M^1.16 )
    total Cf:    Cf = x_lam * Cf_laminar + (1 - x_lam) * Cf_turbulent

The cutoff Reynolds number models the effect of the surface roughness k.
The same Reynolds number is used in the laminar and the turbulent formula.
"""

from __future__ import annotations

import math
from typing import Final

from b12wp3.config.mission import FlowRegime

LAMINAR_COEFF: Final[float] = 1.328
TURBULENT_COEFF: Final[float] = 0.455
TURBULENT_LOG_EXPONENT: Final[float] = 2.58
TURBULENT_MACH_COEFF: Final[float] = 0.144
TURBULENT_MACH_EXPONENT: Final[float] = 0.65

CUTOFF_SUBSONIC_COEFF: Final[float] = 38.21
CUTOFF_TRANSONIC_COEFF: Final[float] = 44.62
CUTOFF_LENGTH_EXPONENT: Final[float] = 1.053
CUTOFF_MACH_EXPONENT: Final[float] = 1.16


def _check_reynolds(reynolds: float) -> None:
    """Raise if the Reynolds number is not positive and finite."""
    if not math.isfinite(reynolds) or reynolds <= 0.0:
        raise ValueError(f"reynolds must be > 0 and finite, got {reynolds}")


def reynolds_actual(
    density_kg_m3: float,
    velocity_m_s: float,
    length_m: float,
    viscosity_pa_s: float,
) -> float:
    """Actual Reynolds number rho V l / mu.

    Args:
        density_kg_m3: Air density [kg/m^3].
        velocity_m_s: True airspeed [m/s].
        length_m: Characteristic length of the component [m].
        viscosity_pa_s: Dynamic viscosity [Pa s].

    Returns:
        Reynolds number [-].

    Raises:
        ValueError: If an input is not positive.
    """
    for name, value in (
        ("density_kg_m3", density_kg_m3),
        ("velocity_m_s", velocity_m_s),
        ("length_m", length_m),
        ("viscosity_pa_s", viscosity_pa_s),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    return density_kg_m3 * velocity_m_s * length_m / viscosity_pa_s


def reynolds_cutoff(
    length_m: float, roughness_m: float, mach: float, regime: FlowRegime
) -> float:
    """Cutoff Reynolds number from the surface roughness.

    Args:
        length_m: Characteristic length of the component [m].
        roughness_m: Equivalent sand roughness k [m].
        mach: Free-stream Mach number [-]. Used in the transonic formula only.
        regime: Selects the subsonic or the transonic formula.

    Returns:
        Cutoff Reynolds number [-].

    Raises:
        ValueError: If length, roughness or (transonic) Mach is not positive.
    """
    if length_m <= 0.0:
        raise ValueError(f"length_m must be > 0, got {length_m}")
    if roughness_m <= 0.0:
        raise ValueError(f"roughness_m must be > 0, got {roughness_m}")
    length_term: float = (length_m / roughness_m) ** CUTOFF_LENGTH_EXPONENT
    if regime is FlowRegime.SUBSONIC:
        return CUTOFF_SUBSONIC_COEFF * length_term
    if mach <= 0.0:
        raise ValueError(f"mach must be > 0 for the transonic cutoff, got {mach}")
    mach_term: float = mach**CUTOFF_MACH_EXPONENT
    return CUTOFF_TRANSONIC_COEFF * length_term * mach_term


def effective_reynolds(actual: float, cutoff: float) -> float:
    """Reynolds number used in Cf: the smaller of the actual and the cutoff value.

    Args:
        actual: Actual Reynolds number [-].
        cutoff: Cutoff Reynolds number [-].

    Returns:
        min(actual, cutoff) [-].
    """
    _check_reynolds(actual)
    _check_reynolds(cutoff)
    return min(actual, cutoff)


def cf_laminar(reynolds: float) -> float:
    """Laminar flat-plate skin friction coefficient, 1.328 / sqrt(Re).

    Args:
        reynolds: Reynolds number [-].

    Returns:
        Cf [-].
    """
    _check_reynolds(reynolds)
    return LAMINAR_COEFF / math.sqrt(reynolds)


def cf_turbulent(reynolds: float, mach: float) -> float:
    """Turbulent flat-plate skin friction coefficient with Mach correction.

    Args:
        reynolds: Reynolds number [-], > 1.
        mach: Free-stream Mach number [-], >= 0.

    Returns:
        Cf [-].

    Raises:
        ValueError: If Re <= 1 (log10 Re <= 0) or the Mach number is negative.
    """
    _check_reynolds(reynolds)
    if reynolds <= 1.0:
        raise ValueError(
            f"reynolds must be > 1 for the turbulent formula, got {reynolds}"
        )
    if mach < 0.0:
        raise ValueError(f"mach must be >= 0, got {mach}")
    log_term: float = math.log10(reynolds) ** TURBULENT_LOG_EXPONENT
    mach_term: float = (1.0 + TURBULENT_MACH_COEFF * mach**2) ** TURBULENT_MACH_EXPONENT
    return TURBULENT_COEFF / (log_term * mach_term)


def cf_mixed(reynolds: float, mach: float, laminar_fraction: float) -> float:
    """Skin friction coefficient as the laminar/turbulent weighted average.

    Args:
        reynolds: Reynolds number [-].
        mach: Free-stream Mach number [-].
        laminar_fraction: Fraction x_lam of laminar flow [-], 0 to 1.

    Returns:
        x_lam * Cf_laminar + (1 - x_lam) * Cf_turbulent [-].

    Raises:
        ValueError: If the laminar fraction is outside [0, 1].
    """
    if not 0.0 <= laminar_fraction <= 1.0:
        raise ValueError(f"laminar_fraction must be in [0, 1], got {laminar_fraction}")
    laminar = cf_laminar(reynolds)
    turbulent = cf_turbulent(reynolds, mach)
    return laminar_fraction * laminar + (1.0 - laminar_fraction) * turbulent
