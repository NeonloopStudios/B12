# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Strip-theory viscous and wave drag corrections for the inviscid VSPAERO
(VLM) solution, using the XFoil section polars from run_cruise_polars.py.

VSPAERO's VLM gives the lift distribution and the induced drag, but no
real profile drag. For every spanwise strip j at every angle of attack:

1. the local streamwise cl_j is reduced to the sweep-normal section
   condition the XFoil polar was run at, cl_n = cl_j / cos^2(sweep)
   (the same simple-sweep-theory reduction as config.FlightCondition);
2. cd_n(cl_n) is read off the converged, pre-stall part of the polar;
3. cd_n is taken back to the streamwise frame (sweep_drag_factor) and
   corrected for the local chord Reynolds number (reynolds_factor);
4. the strips are integrated: CD_profile = sum(cd_j * dA_j) / (S / 2).

A strip whose cl_n lies outside the polar's usable range gets cd = NaN
rather than a clamped or extrapolated value: above the top it is past the
XFoil cruise-condition stall (flagged "stalled"), below the bottom the
polar simply has no data there. Either way the integrated CD_profile of that
angle of attack is NaN, so a guessed number never reaches the drag polar.

Pure numpy/pandas on purpose, so it is unit-testable without OpenVSP.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from scripts import polar_analysis

Array = npt.NDArray[np.float64]

# Flat-plate turbulent skin-friction scaling, cf ~ Re^-0.2.
REYNOLDS_EXPONENT = -0.2

# Strip status codes.
IN_RANGE = 0
STALLED = 1  # cl_n above the polar's cl_max
BELOW_POLAR = -1  # cl_n below the lowest converged cl


def load_section_polar(path: Path) -> pd.DataFrame:
    """Usable part of an XFoil polar CSV for a cd(cl) lookup.

    Keeps converged points up to the stall point found by
    polar_analysis.find_cl_max (so a non-physical post-stall plateau can't
    make cl(alpha) multi-valued), sorted by alpha. Raises if cl is not
    strictly increasing over that range: cd(cl) would then be ambiguous.
    """
    polar = pd.read_csv(path)
    _, alpha_cl_max = polar_analysis.find_cl_max(polar)
    usable = (
        polar[polar["converged"] & (polar["alpha"] <= alpha_cl_max)]
        .sort_values("alpha")
        .reset_index(drop=True)
    )
    if len(usable) < 2:
        raise ValueError(f"{path}: fewer than 2 usable converged points")
    if not bool((np.diff(usable["cl"].to_numpy()) > 0).all()):
        raise ValueError(
            f"{path}: cl is not strictly increasing with alpha up to cl_max, "
            "cd(cl) lookup would be ambiguous"
        )
    return usable[["alpha", "cl", "cd", "cm"]]


def normal_cl(cl: Array | float, sweep_rad: float) -> Array:
    """Streamwise section cl -> sweep-normal cl_n = cl / cos^2(sweep)."""
    return np.asarray(cl, dtype=float) / math.cos(sweep_rad) ** 2


def section_cd(cl_n: Array, polar: pd.DataFrame) -> tuple[Array, npt.NDArray[np.int_]]:
    """cd_n at each cl_n by linear interpolation in the usable polar.

    Returns (cd_n, status), with cd_n = NaN and status STALLED / BELOW_POLAR
    for points outside the polar's cl range.
    """
    cl_n = np.asarray(cl_n, dtype=float)
    cl_tab = polar["cl"].to_numpy(dtype=float)
    cd_tab = polar["cd"].to_numpy(dtype=float)
    cd_n = np.interp(cl_n, cl_tab, cd_tab)
    status = np.full(cl_n.shape, IN_RANGE, dtype=int)
    status[cl_n > cl_tab[-1]] = STALLED
    status[cl_n < cl_tab[0]] = BELOW_POLAR
    cd_n[status != IN_RANGE] = np.nan
    return cd_n, status


def sweep_drag_factor(mode: str, sweep_rad: float) -> float:
    """Factor taking a sweep-normal section cd_n to the streamwise cd.

    "friction": cd = cd_n. Cruise profile drag is dominated by skin friction,
        which acts along the local (swept) streamline and is not reduced by
        sweep. Conservative; used as the baseline.
    "cos3": cd = cd_n cos^3(sweep). Pure simple sweep theory, strictly valid
        only for pressure drag. Optimistic lower bound, reported as a
        sensitivity.
    """
    if mode == "friction":
        return 1.0
    if mode == "cos3":
        return math.cos(sweep_rad) ** 3
    raise ValueError(f"sweep_drag_factor: unknown mode {mode!r}")


def reynolds_factor(chord: Array, chord_ref: float) -> Array:
    """Correction of cd from the polar's reference chord to the local chord.

    Re_n scales linearly with chord at fixed flight condition and sweep, so
    (Re_local / Re_ref)^-0.2 = (c / c_ref)^-0.2.
    """
    return (np.asarray(chord, dtype=float) / chord_ref) ** REYNOLDS_EXPONENT


def strip_profile_drag(
    strips: pd.DataFrame,
    polar: pd.DataFrame,
    *,
    sweep_rad: float,
    chord_ref: float,
    s_ref: float,
    mode: str = "friction",
) -> tuple[pd.DataFrame, float]:
    """Profile drag of one angle of attack from its spanwise strips.

    `strips` needs columns cl, chord, area (dA of one half-wing strip).
    Returns the strips with cl_n, cd_n, cd, status added, and CD_profile
    referenced to the full-wing area s_ref (NaN if any strip is out of range).
    """
    out = strips.copy()
    out["cl_n"] = normal_cl(out["cl"].to_numpy(), sweep_rad)
    cd_n, status = section_cd(out["cl_n"].to_numpy(), polar)
    out["cd_n"] = cd_n
    out["cd"] = (
        cd_n
        * sweep_drag_factor(mode, sweep_rad)
        * reynolds_factor(out["chord"].to_numpy(), chord_ref)
    )
    out["status"] = status
    cd_profile = float((out["cd"] * out["area"]).sum(skipna=False) / (s_ref / 2))
    return out, cd_profile


# ============================================================
#  WAVE DRAG (Korn equation + Lock's approximation)
# ============================================================

# M_crit = M_dd - (0.1 / 80)^(1/3): the Mach at which Lock's
# 20 (M - M_crit)^4 reaches dCD/dM = 0.1, i.e. the drag-divergence definition.
KORN_MCRIT_OFFSET = (0.1 / 80) ** (1 / 3)


def korn_swept(kappa_a: float, t_c: float, cl: float, sweep_rad: float) -> tuple[float, float]:
    """(M_dd, M_crit) of a swept wing from the Korn equation with simple
    sweep theory: M_dd = kappa/cos - (t/c)/cos^2 - CL/(10 cos^3).

    Reduces to compressibility.korn_mdd for zero sweep.
    """
    cos_s = math.cos(sweep_rad)
    m_dd = kappa_a / cos_s - t_c / cos_s**2 - cl / (10 * cos_s**3)
    return m_dd, m_dd - KORN_MCRIT_OFFSET


def lock_wave_drag(mach: float, m_crit: float) -> float:
    """Lock's approximation: CD_wave = 20 (M - M_crit)^4 for M > M_crit."""
    return 20 * (mach - m_crit) ** 4 if mach > m_crit else 0.0
