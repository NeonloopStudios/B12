# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-19
"""Global spanwise stations of WP4.1: every force diagram uses this grid.

lift.csv and weight.csv must have the same stations (wp41.torsion checks
this), so the lift and the weight step both take them from here.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from configs.wp41 import settings

FloatArray: TypeAlias = npt.NDArray[np.float64]

# eta = y / (b/2), from the centreline (0) to the tip (1)
SPAN_STATIONS: Final[FloatArray] = np.linspace(
    0.0, 1.0, settings.N_SPAN_STATIONS, dtype=np.float64
)


def station_y(half_span_m: float) -> FloatArray:
    """Span stations y = eta b/2 [m].

    Args:
        half_span_m: b / 2 [m], > 0.

    Returns:
        The stations, shape (settings.N_SPAN_STATIONS,).

    Raises:
        ValueError: If half_span_m is not > 0.
    """
    if not half_span_m > 0.0:
        raise ValueError(f"half_span_m must be > 0, got {half_span_m}")
    return SPAN_STATIONS * half_span_m
