# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""The WP2 mission: the flight conditions, the wing geometry they are tied
to, and the sweep-theory reduction from a flight condition to the 2D section
condition XFoil is run at.

Single source of truth for every stage: the reduction is written here once,
as data plus a formula, instead of the numbers being copy-pasted into each
script.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from b12wp2.config.atmosphere import isa_atmosphere, sutherland_viscosity

FT_TO_M = 0.3048


def level_flight_cl(
    weight_n: float, density_kg_m3: float, velocity_m_s: float, wing_area_m2: float
) -> float:
    """Steady level-flight trim lift coefficient: L = W => Cl = 2W / (rho V^2 S).

    First-order by construction: assumes steady 1g flight at a single
    representative weight (not integrated over the mission segment's fuel
    burn-off) and ignores tail-trim download. This is the standard
    "lift equals weight" relation used to size the required wing Cl at a
    given flight condition (e.g. textbook eq. 8.13).
    """
    return 2.0 * weight_n / (density_kg_m3 * velocity_m_s**2 * wing_area_m2)


@dataclass(frozen=True)
class FlightCondition:
    """A single flight condition (e.g. cruise or landing) plus the
    sweep-theory reduction to the 2D section condition XFoil is run at.

    `chord_m` and `cl_wing` are the two inputs sweep theory needs that are
    NOT physics constants -- they come from the aircraft's sizing/
    performance data (WP1). They are left unset (None) here rather than
    guessed; the `reynolds_normal` / `cl_normal` properties raise a clear
    error if used before being filled in, instead of silently computing a
    wrong number from a fabricated chord or Cl.
    """

    name: str
    altitude_m: float
    mach_freestream: float
    sweep_rad: float
    ncrit: float
    chord_m: float | None = None
    cl_wing: float | None = None

    @property
    def cos_sweep(self) -> float:
        return math.cos(self.sweep_rad)

    @property
    def temperature_k(self) -> float:
        return isa_atmosphere(self.altitude_m)[0]

    @property
    def speed_of_sound(self) -> float:
        return isa_atmosphere(self.altitude_m)[1]

    @property
    def density(self) -> float:
        return isa_atmosphere(self.altitude_m)[2]

    @property
    def v_freestream(self) -> float:
        return self.mach_freestream * self.speed_of_sound

    @property
    def mach_normal(self) -> float:
        """M_n = M_inf * cos(sweep) -- the Mach number XFoil is run at."""
        return self.mach_freestream * self.cos_sweep

    @property
    def v_normal(self) -> float:
        return self.v_freestream * self.cos_sweep

    @property
    def cl_normal(self) -> float:
        """Cl_n = Cl_wing / cos^2(sweep) -- required 2D section Cl."""
        if self.cl_wing is None:
            raise ValueError(
                f"FlightCondition({self.name!r}): cl_wing is not set, cannot "
                "derive the required section Cl (Cl_n = Cl_wing / cos^2(sweep)). "
                "Fill it in from the aircraft's sizing/performance data."
            )
        return self.cl_wing / self.cos_sweep**2

    @property
    def reynolds_normal(self) -> float:
        """Re_n using the normal chord and normal velocity."""
        if self.chord_m is None:
            raise ValueError(
                f"FlightCondition({self.name!r}): chord_m is not set, cannot "
                "compute Re_n. Fill it in from the aircraft's sizing data."
            )
        chord_normal = self.chord_m * self.cos_sweep
        mu = sutherland_viscosity(self.temperature_k)
        return self.density * self.v_normal * chord_normal / mu


# --- Mission constants ---

SWEEP_RAD = 0.419271315  # 24.02 deg, quarter-chord sweep

CHORD_M = 2.649904207  # streamwise chord at the analysis station (WP1 sizing);
# fixed wing geometry -- same value for cruise and landing, only V/rho/sweep
# effects differ between flight conditions.

CRUISE = FlightCondition(
    name="cruise",
    altitude_m=35_000 * FT_TO_M,
    mach_freestream=0.77,
    sweep_rad=SWEEP_RAD,
    ncrit=8.0,
    chord_m=CHORD_M,
    # Required 3D wing Cl at cruise, Cl = 2W/(rho V^2 S) (level-flight trim,
    # eq. 8.13), computed externally from WP1 cruise weight and wing area.
    # Locates the cruise operating point on the polar; used by cl_cd_cruise,
    # stall_margin, pitching_moment, and mcrit_sweep.py's Mach-critical sweep.
    cl_wing=0.489433403,
)

# Landing: sea level, 65 m/s approach speed. Mach is derived (V / speed of
# sound at sea level), not a separately-given number, so it stays
# consistent with isa_atmosphere by construction.
LANDING_SPEED_MS = 65.0

LANDING = FlightCondition(
    name="landing",
    altitude_m=0.0,  # sea level
    mach_freestream=LANDING_SPEED_MS / isa_atmosphere(0.0)[1],
    sweep_rad=SWEEP_RAD,
    ncrit=8.0,
    chord_m=CHORD_M,
    # cl_wing intentionally left unset and NOT required: "Cl max landing" in
    # the scorecard is the polar's Cl_max at the landing Re/M, not a trim
    # point -- none of the 6 scoring criteria need a required-Cl at landing.
    cl_wing=None,
)

# --- Mission constants ---

SWEEP_RAD = 0.419271315  # 24.02 deg, quarter-chord sweep

CHORD_M = 2.649904207  # streamwise chord at the analysis station (WP1 sizing);
# fixed wing geometry -- same value for cruise and landing, only V/rho/sweep
# effects differ between flight conditions.

CRUISE = FlightCondition(
    name="cruise",
    altitude_m=35_000 * FT_TO_M,
    mach_freestream=0.77,
    sweep_rad=SWEEP_RAD,
    ncrit=8.0,
    chord_m=CHORD_M,
    # Required 3D wing Cl at cruise, Cl = 2W/(rho V^2 S) (level-flight trim,
    # eq. 8.13), computed externally from WP1 cruise weight and wing area.
    # Locates the cruise operating point on the polar; used by cl_cd_cruise,
    # stall_margin, pitching_moment, and mcrit_sweep.py's Mach-critical sweep.
    cl_wing=0.489433403,
)

# Landing: sea level, 65 m/s approach speed. Mach is derived (V / speed of
# sound at sea level), not a separately-given number, so it stays
# consistent with isa_atmosphere by construction.
LANDING_SPEED_MS = 65.0

LANDING = FlightCondition(
    name="landing",
    altitude_m=0.0,  # sea level
    mach_freestream=LANDING_SPEED_MS / isa_atmosphere(0.0)[1],
    sweep_rad=SWEEP_RAD,
    ncrit=8.0,
    chord_m=CHORD_M,
    # cl_wing intentionally left unset and NOT required: "Cl max landing" in
    # the scorecard is the polar's Cl_max at the landing Re/M, not a trim
    # point -- none of the 6 scoring criteria need a required-Cl at landing.
    cl_wing=None,
)
