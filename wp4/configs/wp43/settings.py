# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-19
"""WP4.3 settings: masses, CL_max, design speeds and load factors for the V-n diagram.

Manoeuvre envelope only (CS 25.333-25.337, 25.345); the gust envelope is
neglected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

FT_TO_M: Final[float] = 0.3048


@dataclass(frozen=True)
class Masses:
    """Design masses [kg].

    Attributes:
        mtow_kg: Maximum take-off mass [kg].
        mzfw_kg: Maximum zero-fuel mass [kg].
        oem_kg: Operating empty mass [kg].
    """

    mtow_kg: float
    mzfw_kg: float
    oem_kg: float

    def __post_init__(self) -> None:
        if self.oem_kg <= 0.0:
            raise ValueError(f"oem_kg must be > 0, got {self.oem_kg}")
        if not self.oem_kg < self.mzfw_kg <= self.mtow_kg:
            raise ValueError(
                f"masses must satisfy oem_kg < mzfw_kg <= mtow_kg, got "
                f"{self.oem_kg}, {self.mzfw_kg}, {self.mtow_kg}"
            )


@dataclass(frozen=True)
class LiftCoefficients:
    """Maximum lift coefficients per configuration.

    Attributes:
        cl_max_clean: Maximum CL, clean configuration [-].
        cl_max_takeoff: Maximum CL, take-off flaps [-].
        cl_max_app: Maximum CL, approach flaps [-].
        cl_max_landing: Maximum CL, landing flaps [-].
    """

    cl_max_clean: float
    cl_max_takeoff: float
    cl_max_app: float
    cl_max_landing: float

    def __post_init__(self) -> None:
        if self.cl_max_clean <= 0.0:
            raise ValueError(f"cl_max_clean must be > 0, got {self.cl_max_clean}")
        if not (
            self.cl_max_clean
            < self.cl_max_takeoff
            <= self.cl_max_app
            <= self.cl_max_landing
        ):
            raise ValueError(
                "CL_max must satisfy clean < takeoff <= approach <= landing, got "
                f"{self.cl_max_clean}, {self.cl_max_takeoff}, "
                f"{self.cl_max_app}, {self.cl_max_landing}"
            )


@dataclass(frozen=True)
class DesignSpeeds:
    """Fixed design speeds and the rule factors for the derived speeds (CS 25.335).

    Only V_C and M_C are fixed. V_D, M_D and the flap speeds V_F are computed
    in src/wp43 from these factors:
    V_D = k_vd * V_C, M_D = M_C + dm_d, V_F = k_vf * V_S.

    Attributes:
        v_c_eas_ms: Design cruise speed V_C, EAS [m/s].
        m_c: Design cruise Mach M_C [-].
        k_vd: V_D / V_C [-].
        dm_d: M_D - M_C [-].
        k_vf_takeoff: V_F / V_S1, take-off flaps [-].
        k_vf_approach: V_F / V_S1, approach flaps [-].
        k_vf_landing: V_F / V_S0, landing flaps [-].
    """

    v_c_eas_ms: float
    m_c: float
    k_vd: float
    dm_d: float
    k_vf_takeoff: float
    k_vf_approach: float
    k_vf_landing: float

    def __post_init__(self) -> None:
        if self.v_c_eas_ms <= 0.0:
            raise ValueError(f"v_c_eas_ms must be > 0, got {self.v_c_eas_ms}")
        if not 0.0 < self.m_c < 1.0 - self.dm_d:
            raise ValueError(
                f"m_c must be in (0, 1 - dm_d) so that M_D < 1, got {self.m_c}"
            )
        if self.k_vd <= 1.0:
            raise ValueError(f"k_vd must be > 1, got {self.k_vd}")
        if self.dm_d < 0.05:
            raise ValueError(f"dm_d must be >= 0.05 (CS 25.335(b)), got {self.dm_d}")
        if self.k_vf_takeoff < 1.6:
            raise ValueError(
                f"k_vf_takeoff must be >= 1.6 (CS 25.335(e)), got {self.k_vf_takeoff}"
            )
        if self.k_vf_approach < 1.8 or self.k_vf_landing < 1.8:
            raise ValueError(
                "k_vf_approach and k_vf_landing must be >= 1.8 (CS 25.335(e)), "
                f"got {self.k_vf_approach} and {self.k_vf_landing}"
            )


ALTITUDES_M: Final[tuple[float, ...]] = (0.0, 35_000 * FT_TO_M)

# Regulatory load factors (CS 25.337(c), 25.345(a)). n_max depends on mass and
# is computed in src/wp43, not set here.
N_MIN: Final[float] = -1.0
N_FLAPS: Final[float] = 2.0

MASSES: Final[Masses] = Masses(
    mtow_kg=30_865.0,
    mzfw_kg=25_175.0,
    oem_kg=15_873.0,
)

# Values from wp2/results/hld/ (copied, not imported). Approach CL_max taken
# equal to landing CL_max.
CL: Final[LiftCoefficients] = LiftCoefficients(
    cl_max_clean=1.59,
    cl_max_takeoff=2.354,
    cl_max_app=2.682,
    cl_max_landing=2.682,
)

# V_C and M_C from the cruise condition (M 0.77 at 35 000 ft). The factors are
# CS 25.335 minimums; k_vd = 1.25 is the V_D = V_C / 0.8 rule.
SPEEDS: Final[DesignSpeeds] = DesignSpeeds(
    v_c_eas_ms=127.157,
    m_c=0.77,
    k_vd=1.25,
    dm_d=0.05,
    k_vf_takeoff=1.6,
    k_vf_approach=1.8,
    k_vf_landing=1.8,
)
