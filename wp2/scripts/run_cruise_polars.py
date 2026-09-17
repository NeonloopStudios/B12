"""WP2 Stage 3: cruise polars for every candidate airfoil.

For each airfoil in wp2/airfoils/, runs an angle-of-attack sweep at the
CRUISE flight condition (config.CRUISE: M_n, Re_n, Ncrit) and writes the
full polar to wp2/results/data/<airfoil>_cruise_polar.csv.

This stage only generates and saves the raw polar. Extracting the cruise
operating point (matching Cl(alpha) = Cl_n), stall angle, Cl_max, and
zero-angle Cl from it is Stage 7's job (build_scorecard.py) -- this module
prints a lightweight sanity check (whether Cl_n was actually reached before
the sweep broke down) so a bad run is caught immediately rather than three
stages later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts import config, xfoil_runtime

ALPHA_LOW_DEG = -6.0
ALPHA_HIGH_DEG = 20.0
ALPHA_STEP_DEG = 0.25


def upper_leg_alphas() -> np.ndarray:
    """0 deg up through well past stall. Starts at 0, not the extreme end:
    cold-starting XFoil's BL solve directly at a harsh angle at this Mach
    (M_n~0.70) reliably diverges for several consecutive points before
    reset_bls() recovers it -- verified directly, and at this sweep's 0.25
    deg resolution that eats the whole non-convergence budget before ever
    reaching a point that would actually converge, so the polar comes back
    empty. Warm-starting from a known-good 0 deg point and marching outward
    avoids the cold start entirely; the eventual non-convergence run this
    hits *is* the real stall, not an artifact (verified: NACA 25112 here
    converges cleanly up to ~5 deg before legitimately breaking down).
    """
    return np.arange(0.0, ALPHA_HIGH_DEG + ALPHA_STEP_DEG, ALPHA_STEP_DEG)


def lower_leg_alphas() -> np.ndarray:
    """0 deg down through below zero-lift, same warm-start rationale as
    upper_leg_alphas (mirrored direction).
    """
    return np.arange(0.0, ALPHA_LOW_DEG - ALPHA_STEP_DEG, -ALPHA_STEP_DEG)


def run_cruise_polar(airfoil_path: Path) -> pd.DataFrame:
    """Run the cruise alpha sweep for one airfoil and return its polar.

    Two independent legs, each its own fresh XFoil session starting from
    alpha=0 (see upper_leg_alphas docstring for why): one sweeping up, one
    down. Separate sessions rather than one session run twice, so a bad
    excursion at one extreme (e.g. deep stall on the upper leg) can't leave
    corrupted BL state that taints the other leg's results. The two legs'
    alpha=0 point is identical by construction (same airfoil/condition);
    the lower leg's copy is dropped as a duplicate before merging.
    """
    airfoil = xfoil_runtime.load_airfoil_dat(airfoil_path)
    mach = config.CRUISE.mach_normal
    reynolds = config.CRUISE.reynolds_normal
    ncrit = config.CRUISE.ncrit

    with xfoil_runtime.xfoil_session(airfoil, mach=mach, reynolds=reynolds, ncrit=ncrit) as xf:
        upper = xfoil_runtime.run_alpha_sweep(xf, upper_leg_alphas())

    with xfoil_runtime.xfoil_session(airfoil, mach=mach, reynolds=reynolds, ncrit=ncrit) as xf:
        lower = xfoil_runtime.run_alpha_sweep(xf, lower_leg_alphas())
    lower = lower[lower["alpha"] != 0.0]

    polar = pd.concat([lower, upper], ignore_index=True)
    return polar.sort_values("alpha").reset_index(drop=True)


def _sanity_check(airfoil_stem: str, polar: pd.DataFrame) -> None:
    converged = polar[polar["converged"]]
    n_converged = len(converged)
    n_total = len(polar)
    if n_converged == 0:
        print(f"  [WARN] {airfoil_stem}: no converged points at all")
        return

    cl_n = config.CRUISE.cl_normal
    cl_min, cl_max = converged["cl"].min(), converged["cl"].max()
    reachable = cl_min <= cl_n <= cl_max
    flag = "" if reachable else "  [WARN] Cl_n not bracketed by converged range"
    print(
        f"  {airfoil_stem}: {n_converged}/{n_total} converged, "
        f"alpha in [{converged['alpha'].min():.2f}, {converged['alpha'].max():.2f}] deg, "
        f"cl in [{cl_min:.3f}, {cl_max:.3f}] (Cl_n={cl_n:.3f}){flag}"
    )


def main() -> None:
    out_dir = config.RESULTS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Cruise condition: M_n={config.CRUISE.mach_normal:.4f}, "
        f"Re_n={config.CRUISE.reynolds_normal:,.0f}, Ncrit={config.CRUISE.ncrit}, "
        f"Cl_n={config.CRUISE.cl_normal:.4f}"
    )
    for airfoil_path in config.discover_airfoils():
        polar = run_cruise_polar(airfoil_path)
        out_path = out_dir / f"{airfoil_path.stem}_cruise_polar.csv"
        polar.to_csv(out_path, index=False)
        _sanity_check(airfoil_path.stem, polar)
        print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
