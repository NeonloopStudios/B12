# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Parabolic drag polar of the aircraft (ADSEE-II slides 49-50).

    CD = CD0 + K CL^2

CD0 is the class II build-up (b12wp3.drag.buildup), K the drag due to lift
(b12wp3.drag.induced). For this polar:

    CL at (L/D)max = sqrt(CD0 / K)
    (L/D)max       = 1 / (2 sqrt(CD0 K))

The uncambered form is used: slide 49 allows it for wings with moderate
camber, for which CL at minimum drag is small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class DragPolar:
    """CD = CD0 + K CL^2.

    Attributes:
        cd0: Zero-lift drag coefficient [-].
        k: Drag-due-to-lift factor [-].
    """

    cd0: float
    k: float

    def __post_init__(self) -> None:
        if self.cd0 <= 0.0:
            raise ValueError(f"cd0 must be > 0, got {self.cd0}")
        if self.k <= 0.0:
            raise ValueError(f"k must be > 0, got {self.k}")

    def cd(self, lift_coefficient: float) -> float:
        """Drag coefficient at a lift coefficient [-]."""
        return self.cd0 + self.k * lift_coefficient**2

    def lift_to_drag(self, lift_coefficient: float) -> float:
        """L/D = CL / CD [-]."""
        return lift_coefficient / self.cd(lift_coefficient)

    @property
    def cl_at_max_lift_to_drag(self) -> float:
        """CL at (L/D)max = sqrt(CD0 / K) [-]."""
        return math.sqrt(self.cd0 / self.k)

    @property
    def max_lift_to_drag(self) -> float:
        """(L/D)max = 1 / (2 sqrt(CD0 K)) [-]."""
        return 1.0 / (2.0 * math.sqrt(self.cd0 * self.k))

    def to_dataframe(self, lift_coefficients: npt.ArrayLike) -> pd.DataFrame:
        """CD0, CDi, CD and L/D at each lift coefficient."""
        cl = np.asarray(lift_coefficients, dtype=np.float64)
        cdi = self.k * cl**2
        cd = self.cd0 + cdi
        return pd.DataFrame(
            {
                "cl": cl,
                "cd0": np.full_like(cl, self.cd0),
                "cdi": cdi,
                "cd": cd,
                "lift_to_drag": cl / cd,
            }
        )
