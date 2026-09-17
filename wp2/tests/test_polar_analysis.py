# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Tests for wp2/scripts/polar_analysis.py."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts import polar_analysis


def _polar(alphas: list[float], cls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"alpha": alphas, "cl": cls, "converged": [True] * len(alphas)})


def test_find_cl_max_on_clean_stall_curve() -> None:
    # rises to a clear peak at alpha=10, then drops and stays down
    alphas = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    cls = [0.2, 0.4, 0.6, 0.8, 0.95, 1.05, 0.9, 0.7, 0.6]
    cl_max, alpha_max = polar_analysis.find_cl_max(_polar(alphas, cls))
    assert cl_max == pytest.approx(1.05)
    assert alpha_max == pytest.approx(10.0)


def test_find_cl_max_ignores_spurious_post_stall_convergence() -> None:
    # the exact shape seen for NACA 25112 at the landing condition: a real
    # peak, a real drop, then XFoil keeps "converging" on a non-physical
    # plateau that's well below the real peak but locally flat/noisy
    alphas = [0.0, 5.0, 10.0, 15.0, 17.25, 20.0, 25.0, 30.0, 35.0, 40.0]
    cls = [0.3, 0.9, 1.4, 1.75, 1.80, 1.2, 0.85, 0.75, 0.752, 0.749]
    cl_max, alpha_max = polar_analysis.find_cl_max(_polar(alphas, cls))
    assert cl_max == pytest.approx(1.80)
    assert alpha_max == pytest.approx(17.25)


def test_find_cl_max_ignores_small_noisy_dip_near_peak() -> None:
    # a single small wiggle right at the peak should not be mistaken for
    # stall onset -- the curve keeps climbing after it
    alphas = [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    cls = [0.2, 0.5, 0.8, 1.0, 0.995, 1.05, 1.3, 1.1]
    cl_max, alpha_max = polar_analysis.find_cl_max(_polar(alphas, cls))
    assert cl_max == pytest.approx(1.3)
    assert alpha_max == pytest.approx(10.0)


def test_find_cl_max_ignores_nonconverged_and_negative_alpha_rows() -> None:
    df = pd.DataFrame(
        {
            "alpha": [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            "cl": [-0.3, -0.1, 0.2, 0.5, 0.8, 5.0, 0.6],
            "converged": [True, True, True, True, True, False, True],
        }
    )
    cl_max, alpha_max = polar_analysis.find_cl_max(df)
    # the cl=5.0 outlier at alpha=6 is not converged and must be ignored
    assert cl_max == pytest.approx(0.8)
    assert alpha_max == pytest.approx(4.0)


def test_find_cl_max_raises_on_no_converged_points() -> None:
    df = pd.DataFrame({"alpha": [0.0, 2.0], "cl": [0.2, 0.4], "converged": [False, False]})
    with pytest.raises(ValueError, match="no converged points"):
        polar_analysis.find_cl_max(df)
