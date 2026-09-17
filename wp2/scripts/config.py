"""WP2 mission/aircraft constants and the sweep-theory reduction to 2D
section conditions.

Single source of truth for every other wp2/scripts module. Encodes
wp2/xfoil_plan.md Stage 1 ("Flight conditions -> 2D section conditions") as
data + a formula, not as numbers copy-pasted into each script.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

WP2_DIR = Path(__file__).resolve().parent.parent
AIRFOILS_DIR = WP2_DIR / "airfoils"
RESULTS_DIR = WP2_DIR / "results"

FT_TO_M = 0.3048


def discover_airfoils() -> list[Path]:
    """All candidate airfoil .dat files, sorted for deterministic ordering.

    Deliberately a directory scan, not a hardcoded list -- the candidate set
    has already changed once (EPPLER_395 -> NACA_64212); every later stage
    must pick up whatever is in wp2/airfoils/ without code changes.
    """
    return sorted(AIRFOILS_DIR.glob("*.dat"))


# --- ISA standard atmosphere (0-20 km, SI units) ---
#
# Troposphere (h <= 11 km): linear lapse rate.
# Lower stratosphere (11 km < h <= 20 km): isothermal at 216.65 K.
# Source: ICAO Standard Atmosphere. Constants below are the standard ISA
# values, not fitted to this project.

_T0 = 288.15  # K, sea-level standard temperature
_P0 = 101_325.0  # Pa, sea-level standard pressure
_L = 0.0065  # K/m, tropospheric lapse rate
_R_AIR = 287.05287  # J/(kg K), specific gas constant for dry air
_GAMMA = 1.4  # ratio of specific heats for air
_G0 = 9.80665  # m/s^2, standard gravity
_H_TROPOPAUSE = 11_000.0  # m
_T_TROPOPAUSE = _T0 - _L * _H_TROPOPAUSE  # 216.65 K


def isa_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """Return (temperature K, speed of sound m/s, density kg/m^3) at a given
    geopotential altitude, ISA standard atmosphere, valid 0-20 km.

    Cross-checked against wp2/xfoil_plan.md Sec. 1 (35,000 ft -> T ~ 218.8 K,
    a ~ 296.5 m/s, rho ~ 0.380 kg/m^3) in wp2/tests/test_config.py.
    """
    if altitude_m < 0.0:
        raise ValueError(f"altitude_m must be >= 0, got {altitude_m}")
    if altitude_m <= _H_TROPOPAUSE:
        t = _T0 - _L * altitude_m
        p = _P0 * (t / _T0) ** (_G0 / (_L * _R_AIR))
    elif altitude_m <= 20_000.0:
        t = _T_TROPOPAUSE
        p_tropopause = _P0 * (_T_TROPOPAUSE / _T0) ** (_G0 / (_L * _R_AIR))
        p = p_tropopause * math.exp(-_G0 * (altitude_m - _H_TROPOPAUSE) / (_R_AIR * t))
    else:
        raise ValueError(
            f"altitude_m={altitude_m} is above the validated ISA range (0-20 km) "
            "for this implementation"
        )
    rho = p / (_R_AIR * t)
    a = math.sqrt(_GAMMA * _R_AIR * t)
    return t, a, rho


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


def sutherland_viscosity(temperature_k: float) -> float:
    """Dynamic viscosity of air (Pa s) via Sutherland's law, for Reynolds
    number. Constants are the standard values for air (mu0 at T0=273.15 K).
    """
    mu0, t0, s = 1.716e-5, 273.15, 110.4
    return mu0 * (temperature_k / t0) ** 1.5 * (t0 + s) / (temperature_k + s)


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


# --- Established mission constants (wp2/xfoil_plan.md Sec. 1) ---

SWEEP_RAD = 0.419271315  # 24.02 deg, quarter-chord sweep

CHORD_M = 2.649904207  # streamwise chord at the analysis station (WP1 sizing);
# fixed wing geometry -- same value for cruise and landing, only V/rho/sweep
# effects differ between flight conditions.

CRUISE = FlightCondition(
    name="cruise",
    altitude_m=35_000 * FT_TO_M,
    mach_freestream=0.77,
    sweep_rad=SWEEP_RAD,
    ncrit=9.0,  # TODO: confirm against assumed surface finish / free-stream turbulence
    chord_m=CHORD_M,
    # cl_wing: required 3D wing Cl at cruise, Cl = 2W/(rho V^2 S) (level-flight
    # trim, e.g. textbook eq. 8.13) -- still unset, needs cruise weight W and
    # wing reference area S from WP1 sizing. Locates the cruise operating
    # point on the polar; used by cl_cd_cruise, stall_margin, pitching_moment,
    # and the Stage 5 Mach-critical sweep.
    cl_wing=None,  # TODO(WP1 sizing/performance): W_cruise and S -> Cl = 2W/(rho V^2 S)
)

LANDING = FlightCondition(
    name="landing",
    altitude_m=0.0,  # TODO: confirm landing field altitude (sea level assumed)
    mach_freestream=0.2,  # TODO(WP performance): replace with actual approach Mach
    sweep_rad=SWEEP_RAD,
    ncrit=9.0,
    chord_m=CHORD_M,
    # cl_wing intentionally left unset and NOT required: "Cl max landing" in
    # the scorecard is the polar's Cl_max at the landing Re/M, not a trim
    # point -- none of the 6 scoring criteria need a required-Cl at landing.
    cl_wing=None,
)

# --- WP2 scoring matrix (selection criteria and their weights) ---

SCORING_WEIGHTS: dict[str, float] = {
    "mach_critical": 0.25,
    "cl_cd_cruise": 0.30,
    "cl_max_landing": 0.15,
    "stall_margin": 0.125,  # stall angle minus cruise angle
    "cl_zero_angle": 0.05,
    "pitching_moment": 0.125,
}

if abs(sum(SCORING_WEIGHTS.values()) - 1.0) > 1e-9:
    raise AssertionError(
        f"SCORING_WEIGHTS must sum to 1.0, got {sum(SCORING_WEIGHTS.values())}"
    )
