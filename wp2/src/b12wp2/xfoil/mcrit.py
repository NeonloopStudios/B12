# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Mach-critical determination for one airfoil.

Get the Cp(x) distribution at a safely subsonic baseline Mach and the
cruise-relevant Cl (Cl_n), Karman-Tsien-correct it across a Mach grid, and
find where the corrected Cp_min first crosses the critical (sonic) Cp --
that is M_crit. Cross-checked with the Korn equation for drag-divergence
Mach, M_dd. This is the standard, legitimate use of a panel method at
transonic conditions (it flags an oncoming shock, it does not resolve one).

The sweep settings are config.solvers.MCRIT_*; scripts/mcrit_sweep.py runs
this over every candidate airfoil and writes the CSVs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from b12wp2 import config
from b12wp2.common import compressibility, geometry
from b12wp2.config import solvers
from b12wp2.xfoil import runtime as xfoil_runtime


def run_mcrit_analysis(
    airfoil_path: Path,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Return (M_crit, mach-sweep DataFrame, baseline Cp(x) DataFrame,
    summary dict) for one airfoil.
    """
    airfoil = xfoil_runtime.load_airfoil_dat(airfoil_path)
    cl_n = config.CRUISE.cl_normal
    thickness_to_chord = geometry.max_thickness_to_chord(airfoil)
    kappa_a = config.kappa_a(airfoil_path.stem)

    m_dd = compressibility.korn_mdd(kappa_a, thickness_to_chord, cl_n)
    margin = m_dd - config.CRUISE.mach_normal

    with xfoil_runtime.xfoil_session(
        airfoil,
        mach=solvers.MCRIT_BASELINE_MACH,
        reynolds=config.CRUISE.reynolds_normal,
        ncrit=config.CRUISE.ncrit,
    ) as xf:
        baseline = xfoil_runtime.solve_at_cl(xf, cl_n)
        if not baseline["converged"]:
            summary = {
                "m_crit": float("nan"),
                "m_dd": m_dd,
                "margin_m_dd_minus_m_n": margin,
                "thickness_to_chord": thickness_to_chord,
                "cl_cruise": cl_n,
                "kappa_a": kappa_a,
            }
            empty_sweep = pd.DataFrame(columns=["mach", "cp_min_corrected", "cp_crit"])
            empty_cp = pd.DataFrame(columns=["x", "y", "cp"])
            return None, empty_sweep, empty_cp, summary

        baseline_cp = xfoil_runtime.cp_distribution(xf)

    cp0 = baseline_cp["cp"].to_numpy()
    m_crit, sweep = compressibility.find_mcrit(cp0, solvers.MCRIT_MACH_GRID)
    summary = {
        "m_crit": float("nan") if m_crit is None else m_crit,
        "m_dd": m_dd,
        "margin_m_dd_minus_m_n": margin,
        "thickness_to_chord": thickness_to_chord,
        "cl_cruise": cl_n,
        "kappa_a": kappa_a,
    }
    return m_crit, sweep, baseline_cp, summary
