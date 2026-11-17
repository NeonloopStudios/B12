# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 aerodynamic-load settings.

One run gives the spanwise distributions at one Mach number. For each
load-case Mach of WP4.3, change FLIGHT and run again.

The two angles of attack follow Appendix B of the WP4-5 reader: the inviscid
solution is linear in alpha, so WP4.1 interpolates between alpha = 0 deg and
alpha = 10 deg to the C_L of each load case.
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
            # Prandtl-Glauert in VSPAERO is valid for subsonic flow only
            raise ValueError(f"mach must be in [0, 1), got {self.mach}")


@dataclass(frozen=True)
class VspaeroSettings:
    """VSPAERO sweep and mesh settings.

    Attributes:
        alpha_start_deg: First angle of attack [deg].
        alpha_end_deg: Last angle of attack [deg].
        alpha_npts: Number of angles of attack [-].
        span_tess: Spanwise tessellation; span_tess - 1 strips per half-wing.
        chord_tess: Chordwise tessellation (OpenVSP Tess_W).
        out_cluster: Spanwise clustering, < 1 puts more strips at the tip.
        wake_iter: Wake iterations.
        n_cpu: CPU count for VSPAERO.
        timeout_s: Time limit of the VSPAERO subprocess [s].
    """

    alpha_start_deg: float
    alpha_end_deg: float
    alpha_npts: int
    span_tess: int
    chord_tess: int
    out_cluster: float
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


# --- flight condition: cruise (wp2 config/mission.py) ---
FLIGHT: Final[FlightCondition] = FlightCondition(altitude_m=35_000 * FT_TO_M, mach=0.77)

# --- VSPAERO: alpha 0 and 10 deg (Appendix B), wp2 mesh ---
VSPAERO: Final[VspaeroSettings] = VspaeroSettings(
    alpha_start_deg=0.0,
    alpha_end_deg=10.0,
    alpha_npts=2,
    span_tess=25,
    chord_tess=33,
    out_cluster=0.5,
    wake_iter=5,
    n_cpu=4,
    timeout_s=900.0,
)

# Largest allowed difference between the strip integral and the VSPAERO
# totals (CL and CMy). A larger difference means the strip data is read wrong.
CHECK_TOLERANCE: Final[float] = 1.0e-3

# --- OpenVSP interpreter: the environment variable wins over the default ---
OPENVSP_PYTHON_ENV_VAR: Final[str] = "OPENVSP_PYTHON"
OPENVSP_PYTHON_DEFAULT: Final[Path] = (
    Path.home() / "anaconda3" / "envs" / "openvsp" / "python.exe"
)

# --- files ---
OUT_DIR: Final[Path] = WP4_OUT_DIR / "wp41"
RUN_DIR: Final[Path] = OUT_DIR / "vspaero_run"  # raw VSPAERO files, gitignored
VSP3_FILE: Final[str] = "wing.vsp3"
STRIPS_FILE: Final[str] = "strips.csv"  # written by vspaero_runner.py
TOTALS_FILE: Final[str] = "totals.csv"  # written by vspaero_runner.py
LIFT_FILE: Final[str] = "lift_distribution.txt"
MOMENT_FILE: Final[str] = "moment_distribution.txt"

WING_NAME: Final[str] = "Wing"  # geom name inside the OpenVSP model
