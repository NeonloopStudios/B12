# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""WP4.1 load cases. WP4.3 defines them; WP4.1 reads them as input.

LOAD_CASES holds one TEST case until WP4.3 gives the cases.
Replace it with one LoadCase per case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

MassCase: TypeAlias = Literal["oew", "mtow", "no_fuel"]


@dataclass(frozen=True)
class LoadCase:
    """One symmetric load case.

    Attributes:
        name: Identifier, for example "LC-1". Also the output folder name.
        n: Load factor [-], not zero. Negative for a negative-g case.
        v_eas: Equivalent airspeed [m/s].
        altitude_m: ISA altitude [m].
        mass_case: Weight of the reader: OEW, MTOW, or OEW plus payload without fuel.
        thrust_n: Thrust of one engine [N].
    """

    name: str
    n: float
    v_eas: float
    altitude_m: float
    mass_case: MassCase
    thrust_n: float

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError(f"name must be a plain folder name, got {self.name!r}")
        if not (math.isfinite(self.n) and self.n != 0.0):
            raise ValueError(f"n must be finite and not zero, got {self.n}")
        if not (self.v_eas > 0.0):
            raise ValueError(f"v_eas must be > 0, got {self.v_eas}")
        if not (0.0 <= self.altitude_m <= 20_000.0):
            raise ValueError(f"altitude_m must be in [0, 20000], got {self.altitude_m}")
        if not (self.thrust_n >= 0.0 and math.isfinite(self.thrust_n)):
            raise ValueError(f"thrust_n must be >= 0 and finite, got {self.thrust_n}")


# TEST ONLY. WP4.3 has not given the load cases. LC-TEST exercises the chain.
# The values are not design values: n, V_EAS, altitude and thrust are chosen so
# that the case Mach number (0.54) lies inside the VSPAERO grid and CL_d (0.93)
# lies inside the two VSPAERO angles.
LOAD_CASES: Final[tuple[LoadCase, ...]] = (
    LoadCase(
        name="LC-TEST", n=2.5, v_eas=150.0, altitude_m=3_050.0,
        mass_case="mtow", thrust_n=20_000.0,
    ),
)


def get_case(name: str, cases: tuple[LoadCase, ...] = LOAD_CASES) -> LoadCase:
    """Return the load case with this name.

    Args:
        name: Case identifier.
        cases: Cases to search. Defaults to LOAD_CASES.

    Returns:
        The matching case.

    Raises:
        KeyError: If no case has this name. The message lists the known names.
    """
    for case in cases:
        if case.name == name:
            return case
    known = [case.name for case in cases]
    raise KeyError(
        f"load case {name!r} is not defined in load_cases.py; known: {known}"
    )
