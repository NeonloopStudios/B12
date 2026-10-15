# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Component wetted areas (ADSEE-II, "Component wetted area").

    wing:          S_wet = 1.07 * 2 * S_exp
    tails, pylon:  S_wet = 1.05 * 2 * S_exp
    body:          S_wet = pi D / 4 * ( 1/(3 L1^2) * [ (4 L1^2 + D^2/4)^1.5 - D^3/8 ]
                                        - D + 4 L2 + 2 sqrt(L3^2 + D^2/4) )

The body formula is the fuselage formula of the slides. It is implemented
exactly as given. Its nose part equals the area of a paraboloid of
revolution minus the disk area pi D^2 / 4, so the nose contribution goes to
zero as L1 goes to zero.

The nacelle wetted area is not on the slides. It is the outer surface of the
Torenbeek Type B outline: two conical frustums, from the highlight to the
maximum diameter and from there to the exit.
"""

from __future__ import annotations

import math
from typing import Final

WING_WETTED_FACTOR: Final[float] = 1.07
TAIL_WETTED_FACTOR: Final[float] = 1.05


def _check_exposed_area(exposed_area_m2: float) -> None:
    """Raise if the exposed area is not positive."""
    if exposed_area_m2 <= 0.0:
        raise ValueError(f"exposed_area_m2 must be > 0, got {exposed_area_m2}")


def wing_wetted_area(exposed_area_m2: float) -> float:
    """Wetted area of the wing.

    Args:
        exposed_area_m2: Exposed planform area S_exp [m^2].

    Returns:
        S_wet = 1.07 * 2 * S_exp [m^2].
    """
    _check_exposed_area(exposed_area_m2)
    return WING_WETTED_FACTOR * 2.0 * exposed_area_m2


def tail_wetted_area(exposed_area_m2: float) -> float:
    """Wetted area of a tail surface or a pylon.

    Args:
        exposed_area_m2: Exposed planform area S_exp [m^2].

    Returns:
        S_wet = 1.05 * 2 * S_exp [m^2].
    """
    _check_exposed_area(exposed_area_m2)
    return TAIL_WETTED_FACTOR * 2.0 * exposed_area_m2


def body_wetted_area(
    diameter_m: float,
    nose_length_m: float,
    cylinder_length_m: float,
    tail_length_m: float,
) -> float:
    """Wetted area of a fuselage or nacelle (nose, cylinder, tail cone).

    Args:
        diameter_m: Maximum diameter D [m], > 0.
        nose_length_m: Nose length L1 [m], > 0.
        cylinder_length_m: Cylinder length L2 [m], >= 0.
        tail_length_m: Tail-cone length L3 [m], >= 0.

    Returns:
        Wetted area [m^2].

    Raises:
        ValueError: If D or L1 is not positive, or L2 or L3 is negative.
    """
    if diameter_m <= 0.0:
        raise ValueError(f"diameter_m must be > 0, got {diameter_m}")
    if nose_length_m <= 0.0:
        raise ValueError(f"nose_length_m must be > 0, got {nose_length_m}")
    if cylinder_length_m < 0.0:
        raise ValueError(f"cylinder_length_m must be >= 0, got {cylinder_length_m}")
    if tail_length_m < 0.0:
        raise ValueError(f"tail_length_m must be >= 0, got {tail_length_m}")
    d = diameter_m
    l1 = nose_length_m
    nose: float = (1.0 / (3.0 * l1**2)) * (
        (4.0 * l1**2 + d**2 / 4.0) ** 1.5 - d**3 / 8.0
    )
    cylinder = 4.0 * cylinder_length_m
    cone = 2.0 * math.sqrt(tail_length_m**2 + d**2 / 4.0)
    return math.pi * d / 4.0 * (nose - d + cylinder + cone)


def _frustum_lateral_area(d_start: float, d_end: float, length: float) -> float:
    """Lateral area of a conical frustum between two diameters [m^2]."""
    slant = math.sqrt(length**2 + ((d_start - d_end) / 2.0) ** 2)
    return math.pi * (d_start + d_end) / 2.0 * slant


def nacelle_wetted_area(
    length_m: float,
    highlight_diameter_m: float,
    max_diameter_m: float,
    max_diameter_position: float,
    exit_diameter_m: float,
) -> float:
    """Outer wetted area of a Type B nacelle as two conical frustums.

    The outline runs from the highlight (D_h) to the maximum diameter (D_n)
    at beta * l_n, then to the exit (D_ef) at l_n. The inlet duct and the
    nozzle inside are not outer wetted area.

    Args:
        length_m: Nacelle length l_n [m].
        highlight_diameter_m: D_h [m].
        max_diameter_m: D_n [m].
        max_diameter_position: beta [-], in (0, 1).
        exit_diameter_m: D_ef [m].

    Returns:
        Wetted area of one nacelle [m^2].
    """
    if length_m <= 0.0:
        raise ValueError(f"length_m must be > 0, got {length_m}")
    if not 0.0 < max_diameter_position < 1.0:
        raise ValueError(
            f"max_diameter_position must be in (0, 1), got {max_diameter_position}"
        )
    for name, value in (
        ("highlight_diameter_m", highlight_diameter_m),
        ("max_diameter_m", max_diameter_m),
        ("exit_diameter_m", exit_diameter_m),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")
    front = _frustum_lateral_area(
        highlight_diameter_m, max_diameter_m, max_diameter_position * length_m
    )
    rear = _frustum_lateral_area(
        max_diameter_m, exit_diameter_m, (1.0 - max_diameter_position) * length_m
    )
    return front + rear
