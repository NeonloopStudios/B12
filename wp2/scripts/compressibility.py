"""Compressibility relations used by Stage 5 (mcrit_sweep.py): the
Karman-Tsien compressibility correction, the critical (sonic) pressure
coefficient, and the Korn equation for drag-divergence Mach.

Pure physics functions, no XFoil dependency -- independently testable
against known structural properties (Cp_crit(1.0) = 0 exactly, Karman-
Tsien reduces to the identity at M=0, etc.) rather than against memorized
textbook table values, which risk being misremembered.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def karman_tsien_cp(cp0: npt.NDArray[np.float64] | float, mach: float) -> npt.NDArray[np.float64]:
    """Karman-Tsien compressibility correction: scale an (approximately)
    incompressible pressure coefficient Cp0 up to a compressible freestream
    Mach.

    Cp = Cp0 / [ sqrt(1-M^2) + (M^2 / (1 + sqrt(1-M^2))) * (Cp0/2) ]

    Reduces to Cp = Cp0 at M=0 (the M^2 term vanishes, denominator is
    sqrt(1)=1 -- verified in test_compressibility.py). Standard
    compressibility correction (see e.g. Anderson, Fundamentals of
    Aerodynamics); more accurate than Prandtl-Glauert closer to M_crit,
    which is exactly the regime this is used in.
    """
    if not (0.0 <= mach < 1.0):
        raise ValueError(f"karman_tsien_cp: mach must be in [0, 1), got {mach}")
    cp0_arr = np.asarray(cp0, dtype=np.float64)
    beta = np.sqrt(1.0 - mach**2)
    return cp0_arr / (beta + (mach**2 / (1.0 + beta)) * (cp0_arr / 2.0))


def cp_crit(mach: float, gamma: float = 1.4) -> float:
    """Critical pressure coefficient: the Cp at which the LOCAL flow first
    reaches sonic (M_local = 1) for a given freestream Mach.

    Cp_crit = (2 / (gamma M^2)) * { [(1 + (gamma-1)/2 M^2) / (1 + (gamma-1)/2)]^(gamma/(gamma-1)) - 1 }

    Verified structurally (test_compressibility.py): Cp_crit(1.0) = 0
    exactly; becomes large and negative as M -> 0 (a small local velocity
    increase over a very slow freestream corresponds to a large relative
    pressure drop to reach sonic); rises monotonically toward 0 as M -> 1.
    This is the standard, well-known Cp_crit(M) shape (see e.g. Anderson) --
    checked structurally rather than against a specific remembered value.
    """
    if not (0.0 < mach <= 1.0):
        raise ValueError(f"cp_crit: mach must be in (0, 1], got {mach}")
    g = gamma
    term = (1.0 + (g - 1.0) / 2.0 * mach**2) / (1.0 + (g - 1.0) / 2.0)
    return (2.0 / (g * mach**2)) * (term ** (g / (g - 1.0)) - 1.0)


def korn_mdd(kappa_a: float, thickness_to_chord: float, cl: float) -> float:
    """Korn equation drag-divergence Mach: M_dd + t/c + Cl/10 = kappa_A."""
    return kappa_a - thickness_to_chord - cl / 10.0


def find_mcrit(
    cp0: npt.NDArray[np.float64], mach_grid: npt.NDArray[np.float64], *, gamma: float = 1.4
) -> tuple[float | None, pd.DataFrame]:
    """Sweep mach_grid, Karman-Tsien-correcting the full Cp0(x) distribution
    at each Mach and comparing its minimum to Cp_crit(M).

    Returns (M_crit, sweep_dataframe); sweep_dataframe has columns
    mach, cp_min_corrected, cp_crit. M_crit is the first (lowest) Mach in
    mach_grid where cp_min_corrected <= cp_crit -- local sonic flow first
    appears there. If no crossing occurs within mach_grid, M_crit is None
    (not guessed or extrapolated) -- the caller must report that
    explicitly, e.g. by widening the grid, not silently accept a missing
    result.

    Correcting the whole Cp0(x) distribution rather than just its most
    negative value: the Karman-Tsien transform is monotonically increasing
    in Cp0 for any fixed Mach in (0,1) (derivative is
    beta / (beta + M^2/(1+beta)*Cp0/2)^2 > 0 for beta=sqrt(1-M^2) > 0), so
    the point with the most negative Cp0 would in principle stay the most
    negative after correction regardless of which point is checked -- but
    correcting the full array and taking min() is the textbook-correct
    approach and doesn't rely on that argument holding at every Mach in the
    grid, so it's used here instead of assuming it.
    """
    rows: list[dict[str, float]] = []
    mcrit: float | None = None
    for mach in mach_grid:
        mach_f = float(mach)
        cp_corrected = karman_tsien_cp(cp0, mach_f)
        cp_min = float(np.min(cp_corrected))
        crit = cp_crit(mach_f, gamma=gamma)
        rows.append({"mach": mach_f, "cp_min_corrected": cp_min, "cp_crit": crit})
        if mcrit is None and cp_min <= crit:
            mcrit = mach_f
    return mcrit, pd.DataFrame(rows)
