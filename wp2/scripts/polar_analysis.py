# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Physical quantities extracted from a polar DataFrame (the output of
xfoil_runtime.run_alpha_sweep / run_two_leg_polar).

Split out from xfoil_runtime.py because this is polar post-processing, not
XFoil plumbing. run_landing_polars.py's own sanity check needs real stall
detection to report a trustworthy Cl_max_landing, not a naive max() over
every "converged" row -- see find_cl_max's docstring for why.
"""
from __future__ import annotations

import pandas as pd


def find_cl_max(
    polar: pd.DataFrame, *, min_drop: float = 0.02, sustain_points: int = 3
) -> tuple[float, float]:
    """Return (cl_max, alpha_at_cl_max) using the first sustained decrease
    in Cl(alpha) while sweeping upward from alpha=0, not a blind max() over
    every converged row.

    XFoil's boundary-layer solver can keep numerically converging well past
    a real 2D section's actual stall -- verified directly: NACA 25112 at
    this project's landing condition converges cleanly out to alpha=40 deg,
    with Cl dropping from a real peak of 1.80 at 17.25 deg down to a
    plateau around 0.75 by alpha~30 deg and staying converged there. This
    is a known XFoil limitation (the Newton solve can find a spurious
    converged solution representing a flow state that isn't physically
    what a real, massively separated section would do), not a case the
    `converged` flag catches -- those points genuinely report
    converged=True. So `converged` alone cannot be trusted to exclude this
    non-physical region for a Cl_max extraction.

    The first point where Cl has dropped by at least `min_drop` from the
    running peak and stays at least that far down for the next
    `sustain_points` points is taken as real stall onset; Cl_max is the
    running peak up to (and including) that point. `sustain_points` guards
    against a single noisy/borderline point near the true peak being
    mistaken for stall.

    Only alpha >= 0 is considered: stall (the physical event this is
    detecting) is a property of the upper branch, and using the whole
    polar would let a much larger negative-alpha Cl (were one to exist)
    confuse the running-max tracking.
    """
    up = (
        polar[(polar["alpha"] >= 0) & polar["converged"]]
        .sort_values("alpha")
        .reset_index(drop=True)
    )
    if up.empty:
        raise ValueError("find_cl_max: no converged points at alpha >= 0")

    running_max = float(up["cl"].iloc[0])
    running_max_alpha = float(up["alpha"].iloc[0])
    for i in range(1, len(up)):
        cl_i = float(up["cl"].iloc[i])
        if cl_i > running_max:
            running_max = cl_i
            running_max_alpha = float(up["alpha"].iloc[i])
            continue
        if running_max - cl_i >= min_drop:
            window = up["cl"].iloc[i : i + sustain_points]
            if len(window) == 0 or bool((window <= running_max - min_drop).all()):
                return running_max, running_max_alpha

    return running_max, running_max_alpha


def interpolate_at_cl(polar: pd.DataFrame, cl_target: float) -> dict[str, float]:
    """Linearly interpolate alpha, cd, cm at a target Cl, from the full
    converged polar (ascending alpha order, both signs), by finding the
    bracketing pair of points where Cl crosses cl_target.

    Used by build_scorecard.py to locate the cruise operating point
    (Cl = Cl_n) on the cruise polar for the cl_cd_cruise, pitching_moment,
    and stall_margin scorecard criteria. Deliberately NOT restricted to
    alpha >= 0 (unlike find_cl_max, where that restriction is correct --
    stall is specifically an upper-branch phenomenon): the cruise operating
    point can genuinely
    fall at a negative alpha for a heavily-cambered section. Verified
    directly -- NASA_SC(2)-0712's cruise Cl_n=0.5867 sits between
    alpha=-1.00 deg (Cl=0.585) and alpha=0.00 deg (Cl=0.791); restricting
    to alpha >= 0 here (an earlier version of this function did, copying
    find_cl_max's restriction without re-deriving whether it applied)
    raised a false "not bracketed" error even though the converged data
    covers it fine.

    Scans pairs in alpha order (not a blind np.interp against a
    Cl-sorted array) so it doesn't assume Cl(alpha) is globally monotonic,
    only that it crosses cl_target somewhere in the swept range. Raises if
    cl_target isn't bracketed by the converged range -- extrapolation is
    not attempted, a genuine "can't locate this operating point" failure
    should be visible, not guessed past.
    """
    up = polar[polar["converged"]].sort_values("alpha").reset_index(drop=True)
    if len(up) < 2:
        raise ValueError("interpolate_at_cl: fewer than 2 converged points in this polar")

    for i in range(len(up) - 1):
        cl_a, cl_b = float(up["cl"].iloc[i]), float(up["cl"].iloc[i + 1])
        if (cl_a <= cl_target <= cl_b) or (cl_b <= cl_target <= cl_a):
            frac = 0.0 if cl_b == cl_a else (cl_target - cl_a) / (cl_b - cl_a)
            result = {
                col: float(up[col].iloc[i] + frac * (up[col].iloc[i + 1] - up[col].iloc[i]))
                for col in ("alpha", "cd", "cm")
            }
            result["cl"] = cl_target
            return result

    raise ValueError(
        f"interpolate_at_cl: cl_target={cl_target} not bracketed by the converged "
        f"cl range [{up['cl'].min():.4f}, {up['cl'].max():.4f}]"
    )


def cl_at_zero_angle(polar: pd.DataFrame) -> float:
    """Cl at alpha=0 exactly, from the polar's own alpha=0 row.

    Every run_two_leg_polar sweep includes an exact alpha=0.0 row (the
    warm-start seed point both legs start from), which converges in every
    run so far -- see run_cruise_polars.py/run_landing_polars.py output.
    Raises if that row is missing or didn't converge, rather than
    interpolate around it.
    """
    row = polar[(polar["alpha"] == 0.0) & polar["converged"]]
    if row.empty:
        raise ValueError("cl_at_zero_angle: no converged alpha=0 row in this polar")
    return float(row["cl"].iloc[0])
