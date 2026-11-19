# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 settings: flight condition, VSPAERO run, output files.

One run is one Mach number. The solution is linear in alpha, so two angles
of attack give the distribution at any C_L by linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from configs.paths import OUT_DIR as WP4_OUT_DIR

FT_TO_M: Final[float] = 0.3048


@dataclass(frozen=True)
class FlightCondition:
    """Freestream condition of the VSPAERO run.

    Attributes:
        altitude_m: Geopotential altitude, ISA [m].
        mach: Freestream Mach number [-].
    """

    altitude_m: float
    mach: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.altitude_m <= 20_000.0:
            raise ValueError(f"altitude_m must be in [0, 20000], got {self.altitude_m}")
        if not 0.0 <= self.mach < 1.0:
            raise ValueError(f"mach must be subsonic, in [0, 1), got {self.mach}")


@dataclass(frozen=True)
class VspaeroSettings:
    """VSPAERO sweep, mesh and wake settings.

    Attributes:
        alpha_start_deg: First angle of attack [deg].
        alpha_end_deg: Last angle of attack [deg].
        alpha_npts: Number of angles of attack [-].
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
    span_tess: int
    chord_tess: int
    out_cluster: float
    fixed_wake: bool
    wake_nodes: int
    wake_iter: int
    n_cpu: int
    timeout_s: float

    def __post_init__(self) -> None:
        if self.alpha_npts < 2:
            raise ValueError(f"alpha_npts must be >= 2, got {self.alpha_npts}")
        if self.alpha_end_deg <= self.alpha_start_deg:
            raise ValueError(
                f"alpha_end_deg ({self.alpha_end_deg}) must be larger than "
                f"alpha_start_deg ({self.alpha_start_deg})"
            )
        if self.span_tess < 3 or self.chord_tess < 3:
            raise ValueError(
                f"span_tess and chord_tess must be >= 3, got "
                f"{self.span_tess} and {self.chord_tess}"
            )
        if self.wake_nodes < 2:
            raise ValueError(f"wake_nodes must be >= 2, got {self.wake_nodes}")


FLIGHT: Final[FlightCondition] = FlightCondition(altitude_m=35_000 * FT_TO_M, mach=0.77)

# Mesh study: chord_tess 193 is within 0.6 % of 257 in CL and CMy (33 was 10 %
# low). Fewer than 16 wake nodes give a false jump at the tip strip.
VSPAERO: Final[VspaeroSettings] = VspaeroSettings(
    alpha_start_deg=0.0,
    alpha_end_deg=10.0,
    alpha_npts=2,
    span_tess=25,
    chord_tess=193,
    out_cluster=0.5,
    fixed_wake=True,
    wake_nodes=64,
    wake_iter=5,
    n_cpu=4,
    timeout_s=900.0,
)

CHECK_TOLERANCE: Final[float] = 1.0e-3  # strip integral vs VSPAERO totals

OPENVSP_PYTHON_ENV_VAR: Final[str] = "OPENVSP_PYTHON"
OPENVSP_PYTHON_DEFAULT: Final[Path] = (
    Path.home() / "anaconda3" / "envs" / "openvsp" / "python.exe"
)

OUT_DIR: Final[Path] = WP4_OUT_DIR / "wp41"
RUN_DIR: Final[Path] = OUT_DIR / "vspaero_run"
VSP3_FILE: Final[str] = "wing.vsp3"
STRIPS_FILE: Final[str] = "strips.csv"
TOTALS_FILE: Final[str] = "totals.csv"
LIFT_FILE: Final[str] = "lift_distribution.txt"
MOMENT_FILE: Final[str] = "moment_distribution.txt"
WING_NAME: Final[str] = "Wing"

LIFT_CSV: Final[str] = "lift.csv"
WEIGHT_CSV: Final[str] = "weight.csv"
POINT_LOADS_CSV: Final[str] = "point_loads.csv"
TORQUE_CSV: Final[str] = "torque.csv"


def case_dir(case_name: str) -> Path:
    """Output folder of one load case: wp4/out/wp41/<case_name>."""
    return OUT_DIR / case_name
