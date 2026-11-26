# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""WP4.1 load cases. WP4.3 defines them; WP4.1 reads them as input.

LOAD_CASES holds the two critical cases of the V-n diagrams (see below).
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


# The two critical cases of the V-n diagrams (wp4/out/wp43, configs/wp43).
# Both use MZFM (OEM + maximum payload, no fuel): the fuel in the wing relieves
# the root loads, so the aircraft without fuel is the worst case.
# LC-1: n_max = 2.5 at V_D at sea level. The dynamic pressure is the highest of
#   the envelope, so the nose-down torque of the pitching moment is the highest.
#   Thrust 0: thrust gives a nose-up torque, so zero thrust is conservative.
# LC-2: n_min = -1 at V_C at FL350 (M = 0.77), the highest speed with n = -1.
#   Thrust = drag of the clean aircraft, D = q S (CD0 + K CL^2), per engine D/2:
#   CD0 = 0.02172 (wp3/results/drag/cd0_summary.csv, cruise), K = 0.05357
#   (wp3/results/iteration/oswald_configurations.csv, clean, e = 0.5593),
#   CL = -0.3987, q = 9903.5 Pa, S = 62.52 m^2: D = 18.72 kN, 9.36 kN per engine.
LOAD_CASES: Final[tuple[LoadCase, ...]] = (
    LoadCase(
        name="LC-1", n=2.5, v_eas=158.946, altitude_m=0.0,
        mass_case="no_fuel", thrust_n=0.0,
    ),
    LoadCase(
        name="LC-2", n=-1.0, v_eas=127.157, altitude_m=10_668.0,
        mass_case="no_fuel", thrust_n=9_360.0,
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
