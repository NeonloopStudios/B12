# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-24
"""Inputs of the WP4.1 weight script that no other config holds.

The wing box is the one of configs/wp42/wingbox.py (spars at 0.2 c and 0.7 c).
wp41 does not import wp42, so the values are copied here with their source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class WeightInputs:
    """Chord fractions of the wing weights and the engine installation.

    Attributes:
        f_str: Chord fraction of the structure centre of gravity [-].
        f_fuel: Chord fraction of the fuel centre of gravity [-].
        engine_inlet_ahead_of_le_m: Inlet plane ahead of the local wing leading
            edge [m].
        nacelle_length_m: Nacelle length [m].
        engine_cg_fraction: Engine centre of gravity behind the inlet, as a
            fraction of the nacelle length [-].
    """

    f_str: float
    f_fuel: float
    engine_inlet_ahead_of_le_m: float
    nacelle_length_m: float
    engine_cg_fraction: float

    def __post_init__(self) -> None:
        for name in ("f_str", "f_fuel", "engine_cg_fraction"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("engine_inlet_ahead_of_le_m", "nacelle_length_m"):
            value = getattr(self, name)
            if not (value > 0.0 and math.isfinite(value)):
                raise ValueError(f"{name} must be > 0 and finite, got {value}")


# f_str, f_fuel: x/c of the area centroid of the wing box, from
# wp42.centroid.centroid_wingbox with configs/wp42/wingbox.py (0.43371858...).
# ASSUMPTION A-11: fuel and structure act at this centroid. The fuel fills the
# box, so this holds for the fuel. For the structure the true centroid of the
# material is not known before WP4.3.
# engine_inlet_ahead_of_le_m: wp3/configs/vsp_model.yaml, nacelle_position.
# nacelle_length_m: wp3/results/propulsion/nacelle.csv, nacelle_length_ln.
# engine_cg_fraction: ASSUMPTION A-12, an educated guess. The fan and the
# compressor are in the front part of a turbofan, so the centre of gravity is
# taken at 40 % of the nacelle length behind the inlet. No source gives it.
WEIGHT_INPUTS: Final[WeightInputs] = WeightInputs(
    f_str=0.4337186,
    f_fuel=0.4337186,
    engine_inlet_ahead_of_le_m=2.86,
    nacelle_length_m=4.071,
    engine_cg_fraction=0.40,
)
