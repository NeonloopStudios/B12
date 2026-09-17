"""Physical quantities extracted from a polar DataFrame (the output of
xfoil_runtime.run_alpha_sweep / run_two_leg_polar).

Split out from xfoil_runtime.py because this is polar post-processing, not
XFoil plumbing -- and because it's needed earlier than Stage 7
(build_scorecard.py) was originally going to introduce it: Stage 4's own
sanity check needs real stall detection to report a trustworthy
Cl_max_landing, not a naive max() over every "converged" row. See
find_cl_max's docstring for why.
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
