# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Component form factors (ADSEE-II, slide 62).

    wing, tail, strut, pylon:
        FF = [1 + 0.6/(x/c)_m * (t/c) + 100 (t/c)^4] * [1.34 M^0.18 (cos L_m)^0.28]
    fuselage, smooth canopy:
        FF = 1 + 60/f^3 + f/400
    nacelle, smooth external store:
        FF = 1 + 0.35/f
    fineness ratio:
        f = l / d = l / sqrt((4/pi) A_max)

L_m is the sweep of the maximum-thickness line.
"""

from __future__ import annotations

import math
from typing import Final

THICKNESS_POSITION_COEFF: Final[float] = 0.6
THICKNESS_FOURTH_COEFF: Final[float] = 100.0
MACH_COEFF: Final[float] = 1.34
MACH_EXPONENT: Final[float] = 0.18
SWEEP_EXPONENT: Final[float] = 0.28
FUSELAGE_CUBIC_COEFF: Final[float] = 60.0
FUSELAGE_LINEAR_DIVISOR: Final[float] = 400.0
NACELLE_COEFF: Final[float] = 0.35


def fineness_ratio(length_m: float, max_frontal_area_m2: float) -> float:
    """Fineness ratio f = l / sqrt((4/pi) A_max).

    Args:
        length_m: Body length [m].
        max_frontal_area_m2: Maximum cross-section area A_max [m^2].

    Returns:
        f [-].

    Raises:
        ValueError: If an input is not positive.
    """
    if length_m <= 0.0:
        raise ValueError(f"length_m must be > 0, got {length_m}")
    if max_frontal_area_m2 <= 0.0:
        raise ValueError(f"max_frontal_area_m2 must be > 0, got {max_frontal_area_m2}")
    return length_m / math.sqrt(4.0 / math.pi * max_frontal_area_m2)


def lifting_surface_form_factor(
    thickness_to_chord: float,
    x_c_max_thickness: float,
    mach: float,
    sweep_max_thickness_rad: float,
) -> float:
    """Form factor of a wing, tail, strut or pylon.

    Args:
        thickness_to_chord: Maximum t/c [-], in (0, 1).
        x_c_max_thickness: Chord position of the maximum thickness [-], in (0, 1).
        mach: Free-stream Mach number [-], > 0.
        sweep_max_thickness_rad: Sweep of the max-thickness line [rad].

    Returns:
        FF [-].

    Raises:
        ValueError: If an input is outside its range. M = 0 gives FF = 0, so
            the Mach number must be positive.
    """
    if not 0.0 < thickness_to_chord < 1.0:
        raise ValueError(
            f"thickness_to_chord must be in (0, 1), got {thickness_to_chord}"
        )
    if not 0.0 < x_c_max_thickness < 1.0:
        raise ValueError(
            f"x_c_max_thickness must be in (0, 1), got {x_c_max_thickness}"
        )
    if mach <= 0.0:
        raise ValueError(f"mach must be > 0, got {mach}")
    if not abs(sweep_max_thickness_rad) < math.pi / 2:
        raise ValueError(
            f"|sweep_max_thickness_rad| must be < pi/2, got {sweep_max_thickness_rad}"
        )
    t_c = thickness_to_chord
    thickness_term = (
        1.0
        + THICKNESS_POSITION_COEFF / x_c_max_thickness * t_c
        + THICKNESS_FOURTH_COEFF * t_c**4
    )
    compressibility_term: float = (
        MACH_COEFF
        * mach**MACH_EXPONENT
        * math.cos(sweep_max_thickness_rad) ** SWEEP_EXPONENT
    )
    return thickness_term * compressibility_term


def _check_fineness(fineness: float) -> None:
    """Raise if the fineness ratio is not positive."""
    if fineness <= 0.0:
        raise ValueError(f"fineness must be > 0, got {fineness}")


def fuselage_form_factor(fineness: float) -> float:
    """Form factor of a fuselage or smooth canopy, 1 + 60/f^3 + f/400.

    Args:
        fineness: Fineness ratio f [-].

    Returns:
        FF [-].
    """
    _check_fineness(fineness)
    return 1.0 + FUSELAGE_CUBIC_COEFF / fineness**3 + fineness / FUSELAGE_LINEAR_DIVISOR


def nacelle_form_factor(fineness: float) -> float:
    """Form factor of a nacelle or smooth external store, 1 + 0.35/f.

    Args:
        fineness: Fineness ratio f [-].

    Returns:
        FF [-].
    """
    _check_fineness(fineness)
    return 1.0 + NACELLE_COEFF / fineness
