# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 settings: VSPAERO database, lift distribution, output files.

The VSPAERO database holds cl(y) and cm_c4(y) at each Mach of a grid and at
two angles of attack. The solution (fixed wake, inviscid) is linear in alpha,
so the two angles give the distribution at any C_L by linear interpolation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from configs.paths import OUT_DIR as WP4_OUT_DIR

FT_TO_M: Final[float] = 0.3048


@dataclass(frozen=True)
class VspaeroSettings:
    """VSPAERO sweep, mesh and wake settings.

    Attributes:
        alpha_start_deg: First angle of attack [deg].
        alpha_end_deg: Last angle of attack [deg].
        alpha_npts: Number of angles of attack; must be 2 (linear in alpha).
        mach_start: First Mach number of the grid [-].
        mach_end: Last Mach number of the grid [-].
        mach_npts: Number of Mach numbers [-].
        span_tess: Spanwise tessellation; span_tess - 1 strips per half-wing.
        chord_tess: Points around the airfoil (OpenVSP Tess_W).
        out_cluster: Spanwise clustering, < 1 puts more strips at the tip.
        fixed_wake: True for a flat wake along the freestream.
        wake_nodes: Nodes along each trailing wake line.
        wake_iter: Wake iterations (relaxed wake only).
        n_cpu: CPU count for VSPAERO.
        timeout_s: Time limit of the VSPAERO subprocess [s].
    """

    alpha_start_deg: float
    alpha_end_deg: float
    alpha_npts: int
    mach_start: float
    mach_end: float
    mach_npts: int
    span_tess: int
    chord_tess: int
    out_cluster: float
    fixed_wake: bool
    wake_nodes: int
    wake_iter: int
    n_cpu: int
    timeout_s: float

    def __post_init__(self) -> None:
        if self.alpha_npts != 2:
            raise ValueError(f"alpha_npts must be 2 (linear in alpha), got {self.alpha_npts}")
        if not self.alpha_end_deg > self.alpha_start_deg:
            raise ValueError(
                f"alpha_end_deg ({self.alpha_end_deg}) must be larger than "
                f"alpha_start_deg ({self.alpha_start_deg})"
            )
        if not 0.0 < self.mach_start < self.mach_end < 1.0:
            # Prandtl-Glauert in VSPAERO is valid for subsonic flow only
            raise ValueError(
                f"need 0 < mach_start < mach_end < 1, got {self.mach_start}, {self.mach_end}"
            )
        if not self.mach_npts >= 2:
            raise ValueError(f"mach_npts must be >= 2, got {self.mach_npts}")
        if self.span_tess < 3 or self.chord_tess < 3:
            raise ValueError(
                f"span_tess and chord_tess must be >= 3, got "
                f"{self.span_tess} and {self.chord_tess}"
            )
        if self.wake_nodes < 2:
            raise ValueError(f"wake_nodes must be >= 2, got {self.wake_nodes}")
        if not self.n_cpu >= 1:
            raise ValueError(f"n_cpu must be >= 1, got {self.n_cpu}")


# Mesh study: chord_tess 193 is within 0.6 % of 257 in CL and CMy (33 was 10 %
# low). Fewer than 16 wake nodes give a false jump at the tip strip.
# With span_tess 120 the tip strips are narrower and need a finer near wake:
# tip jump in cl (alpha 10 deg) is 0.11 / 0.04 / 0.03 at 64 / 128 / 256 nodes.
# 256 gives no further gain and takes 3.5 times longer (360 s against 100 s).
# Mach grid: V_stall at sea level up to about V_D at cruise altitude. Above
# M = 0.8 Prandtl-Glauert is outside its valid range on this wing.
VSPAERO: Final[VspaeroSettings] = VspaeroSettings(
    alpha_start_deg=0.0,
    alpha_end_deg=10.0,
    alpha_npts=2,
    mach_start=0.15,
    mach_end=0.85,
    mach_npts=15,
    span_tess=120,
    chord_tess=193,
    out_cluster=0.5,
    fixed_wake=True,
    wake_nodes=128,
    wake_iter=5,
    n_cpu=os.cpu_count() or 1,  # all logical CPUs
    timeout_s=4 * 3600.0,
)

CHECK_TOLERANCE: Final[float] = 1.0e-3  # strip integral vs VSPAERO totals

# --- lift distribution (wp41.stations, wp41.lift_distribution) ---
N_SPAN_STATIONS: Final[int] = 120  # global stations, eta = y / (b/2) from 0 to 1
# Largest allowed difference between 2 * integral(L' dy) and n W, as a
# fraction of W. A larger difference means the interpolation is wrong.
LIFT_CHECK_TOLERANCE: Final[float] = 0.01


@dataclass(frozen=True)
class SpeedRange:
    """Equivalent airspeeds of the angle-of-attack grid at one altitude.

    Attributes:
        altitude_m: Geopotential altitude [m].
        v_eas_start: First equivalent airspeed [m/s].
        v_eas_end: Last equivalent airspeed [m/s].
        v_eas_npts: Number of speeds [-].
    """

    altitude_m: float
    v_eas_start: float
    v_eas_end: float
    v_eas_npts: int

    def __post_init__(self) -> None:
        if not 0.0 < self.v_eas_start < self.v_eas_end:
            raise ValueError(
                f"need 0 < v_eas_start < v_eas_end, got {self.v_eas_start}, {self.v_eas_end}"
            )
        if not self.v_eas_npts >= 2:
            raise ValueError(f"v_eas_npts must be >= 2, got {self.v_eas_npts}")


CRUISE_ALTITUDE_M: Final[float] = 35_000 * FT_TO_M

# Speed ranges keep the Mach number inside the VSPAERO grid (0.15 to 0.85)
GRID_SPEEDS: Final[tuple[SpeedRange, ...]] = (
    SpeedRange(altitude_m=0.0, v_eas_start=60.0, v_eas_end=280.0, v_eas_npts=23),
    SpeedRange(altitude_m=CRUISE_ALTITUDE_M, v_eas_start=30.0, v_eas_end=140.0, v_eas_npts=23),
)
GRID_LOAD_FACTOR: Final[float] = 2.5

# --- OpenVSP interpreter: the environment variable wins over the default ---
OPENVSP_PYTHON_ENV_VAR: Final[str] = "OPENVSP_PYTHON"
OPENVSP_PYTHON_DEFAULT: Final[Path] = (
    Path.home() / "anaconda3" / "envs" / "openvsp" / "python.exe"
)

# --- files ---
OUT_DIR: Final[Path] = WP4_OUT_DIR / "wp41"
RUN_DIR: Final[Path] = OUT_DIR / "vspaero_run"
VSP3_FILE: Final[str] = "wing.vsp3"
STRIPS_FILE: Final[str] = "strips.csv"
TOTALS_FILE: Final[str] = "totals.csv"
LIFT_FILE: Final[str] = "lift_distribution.txt"
MOMENT_FILE: Final[str] = "moment_distribution.txt"
AOA_GRID_FILE: Final[str] = "aoa_grid.txt"
WING_NAME: Final[str] = "Wing"

# --- header keys of the distribution files ("# key: values") ---
KEY_S_REF: Final[str] = "s_ref_m2"
KEY_HALF_SPAN: Final[str] = "half_span_m"
KEY_MAC: Final[str] = "mac_m"
KEY_INCIDENCE: Final[str] = "incidence_deg"
KEY_MACH: Final[str] = "mach"
KEY_ALPHA: Final[str] = "alpha_deg"
KEY_CL: Final[str] = "CL"
KEY_CMY: Final[str] = "CMy_c4_mac"

# --- per-load-case CSV files (wp41.contracts) ---
LIFT_CSV: Final[str] = "lift.csv"
WEIGHT_CSV: Final[str] = "weight.csv"
POINT_LOADS_CSV: Final[str] = "point_loads.csv"
TORQUE_CSV: Final[str] = "torque.csv"


def case_dir(case_name: str) -> Path:
    """Output folder of one load case: wp4/out/wp41/<case_name>."""
    return OUT_DIR / case_name
