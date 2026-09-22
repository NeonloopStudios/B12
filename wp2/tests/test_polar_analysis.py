# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Tests for wp2/src/b12wp2/xfoil/polar_analysis.py."""
from __future__ import annotations

import pandas as pd
import pytest

from b12wp2.xfoil import polar_analysis


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


def _full_polar(alphas: list[float], cls: list[float], cds: list[float], cms: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"alpha": alphas, "cl": cls, "cd": cds, "cm": cms, "converged": [True] * len(alphas)}
    )


def test_interpolate_at_cl_finds_bracketing_pair() -> None:
    polar = _full_polar(
        alphas=[0.0, 2.0, 4.0],
        cls=[0.2, 0.5, 0.8],
        cds=[0.01, 0.015, 0.02],
        cms=[-0.05, -0.06, -0.07],
    )
    # target 0.5 is exactly on the alpha=2.0 point
    result = polar_analysis.interpolate_at_cl(polar, 0.5)
    assert result["alpha"] == pytest.approx(2.0)
    assert result["cd"] == pytest.approx(0.015)
    assert result["cm"] == pytest.approx(-0.06)
    assert result["cl"] == pytest.approx(0.5)


def test_interpolate_at_cl_interpolates_between_points() -> None:
    polar = _full_polar(
        alphas=[0.0, 2.0],
        cls=[0.2, 0.6],
        cds=[0.01, 0.02],
        cms=[-0.05, -0.07],
    )
    # target 0.4 is 50% of the way from 0.2 to 0.6
    result = polar_analysis.interpolate_at_cl(polar, 0.4)
    assert result["alpha"] == pytest.approx(1.0)
    assert result["cd"] == pytest.approx(0.015)
    assert result["cm"] == pytest.approx(-0.06)


def test_interpolate_at_cl_raises_when_not_bracketed() -> None:
    polar = _full_polar(
        alphas=[0.0, 2.0], cls=[0.2, 0.6], cds=[0.01, 0.02], cms=[-0.05, -0.07]
    )
    with pytest.raises(ValueError, match="not bracketed"):
        polar_analysis.interpolate_at_cl(polar, 5.0)


def test_interpolate_at_cl_finds_negative_alpha_operating_point() -> None:
    # the exact shape of NASA_SC(2)-0712's cruise polar: a heavily-cambered
    # section whose Cl already exceeds Cl_n before alpha reaches 0 -- the
    # operating point is at a negative alpha. An earlier version of this
    # function restricted the search to alpha >= 0 (copied from
    # find_cl_max, where that's correct but here isn't) and raised a false
    # "not bracketed" error on exactly this case.
    polar = _full_polar(
        alphas=[-1.0, 0.0, 1.0],
        cls=[0.585, 0.791, 0.977],
        cds=[0.01, 0.011, 0.012],
        cms=[-0.08, -0.09, -0.10],
    )
    result = polar_analysis.interpolate_at_cl(polar, 0.5867)
    assert -1.0 < result["alpha"] < 0.0
    assert result["cl"] == pytest.approx(0.5867)


def test_cl_at_zero_angle_reads_exact_row() -> None:
    polar = _full_polar(
        alphas=[-2.0, 0.0, 2.0], cls=[0.0, 0.3, 0.6], cds=[0.01, 0.01, 0.02], cms=[0.0, 0.0, 0.0]
    )
    assert polar_analysis.cl_at_zero_angle(polar) == pytest.approx(0.3)


def test_cl_at_zero_angle_raises_when_missing() -> None:
    polar = _full_polar(alphas=[-2.0, 2.0], cls=[0.0, 0.6], cds=[0.01, 0.02], cms=[0.0, 0.0])
    with pytest.raises(ValueError, match="no converged alpha=0"):
        polar_analysis.cl_at_zero_angle(polar)
