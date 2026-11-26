# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Inputs of the WP4.1 torsion script that WP4.2 will replace.

The shear centre is taken at the centroid of the wing box (reader, WP4.1;
assumption A-6 of README_41). WP4.2 computes that centroid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TorsionInputs:
    """Wing box axis inputs.

    Attributes:
        f_sc: Chord fraction of the centroid of the wing box [-], in [0, 1].
            None while WP4.2 gives no value.
        h_thrust_m: Height of the thrust line below the centroid [m].
    """

    f_sc: float | None
    h_thrust_m: float

    def __post_init__(self) -> None:
        if self.f_sc is not None and not (0.0 <= self.f_sc <= 1.0):
            raise ValueError(f"f_sc must be in [0, 1], got {self.f_sc}")
        if not math.isfinite(self.h_thrust_m):
            raise ValueError(f"h_thrust_m must be finite, got {self.h_thrust_m}")

    def require_f_sc(self) -> float:
        """Return f_sc.

        Raises:
            ValueError: If f_sc is not set yet.
        """
        if self.f_sc is None:
            raise ValueError(
                "f_sc is not set in configs/wp41/torsion.py: ask WP4.2 for the "
                "chord fraction of the wing box centroid"
            )
        return self.f_sc


# f_sc: x/c of the area centroid of the wing box (spars at 0.2 c and 0.7 c),
# from wp42.centroid.centroid_wingbox with configs/wp42/wingbox.py: 0.43371858.
# The reader allows the shear centre at the centroid (assumption A-6).
# h_thrust_m: the engine axis is 0.8 m below the local chord line at y = 4.4 m
# (wp3/configs/vsp_model.yaml, nacelle_position.axis_below_chord_m). The
# centroid lies 0.0038464 c below the chord line (centroid_z of centroid_wingbox,
# c = 2.8708 m at y = 4.4 m), so h_thrust = 0.8 - 0.0038464 * 2.8708 = 0.7890 m.
# UNVERIFIED: the sign of centroid_z (z positive up) is not documented there.
TORSION_INPUTS: Final[TorsionInputs] = TorsionInputs(f_sc=0.4337186, h_thrust_m=0.7890)
