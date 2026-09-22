# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-09-22
"""Tests for wp2/scripts/viscous_correction.py."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from b12wp2.common import compressibility
from b12wp2.wing import viscous_correction as vc

SWEEP = math.radians(24.0)


def _polar(cls: list[float], cds: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"alpha": np.arange(len(cls), dtype=float), "cl": cls, "cd": cds})


def _strips(cls: list[float], chords: list[float], areas: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"y": np.arange(len(cls), dtype=float), "cl": cls, "chord": chords, "area": areas})


def test_normal_cl_divides_by_cos_squared() -> None:
    assert vc.normal_cl(0.5, SWEEP) == pytest.approx(0.5 / math.cos(SWEEP) ** 2)
    assert vc.normal_cl(0.5, 0.0) == pytest.approx(0.5)


def test_section_cd_interpolates_and_flags_out_of_range() -> None:
    polar = _polar([0.0, 1.0], [0.006, 0.010])
    cd, status = vc.section_cd(np.array([-0.1, 0.5, 1.2]), polar)
    assert np.isnan(cd[0]) and status[0] == vc.BELOW_POLAR
    assert cd[1] == pytest.approx(0.008) and status[1] == vc.IN_RANGE
    assert np.isnan(cd[2]) and status[2] == vc.STALLED


def test_constant_cd_integrates_to_itself() -> None:
    # constant cd_n on a half-wing of area S/2 at the reference chord gives CD_profile = cd_n
    polar = _polar([-1.0, 2.0], [0.007, 0.007])
    strips = _strips([0.2, 0.4, 0.3], [2.0, 2.0, 2.0], [10.0, 20.0, 10.0])
    _, cd_profile = vc.strip_profile_drag(strips, polar, sweep_rad=SWEEP, chord_ref=2.0, s_ref=80.0)
    assert cd_profile == pytest.approx(0.007)


def test_cos3_mode_scales_profile_drag() -> None:
    polar = _polar([-1.0, 2.0], [0.007, 0.007])
    strips = _strips([0.2], [2.0], [40.0])
    _, cd_friction = vc.strip_profile_drag(strips, polar, sweep_rad=SWEEP, chord_ref=2.0, s_ref=80.0)
    _, cd_cos3 = vc.strip_profile_drag(
        strips, polar, sweep_rad=SWEEP, chord_ref=2.0, s_ref=80.0, mode="cos3"
    )
    assert cd_cos3 == pytest.approx(cd_friction * math.cos(SWEEP) ** 3)
    with pytest.raises(ValueError):
        vc.sweep_drag_factor("nonsense", SWEEP)


def test_reynolds_factor_follows_minus_one_fifth_power() -> None:
    assert vc.reynolds_factor(np.array([2.0]), 2.0)[0] == pytest.approx(1.0)
    # larger chord -> higher Re -> lower cd
    assert vc.reynolds_factor(np.array([4.0]), 2.0)[0] == pytest.approx(2.0**-0.2)


def test_one_stalled_strip_makes_profile_drag_nan() -> None:
    polar = _polar([0.0, 1.0], [0.006, 0.010])
    strips = _strips([0.5, 0.95], [2.0, 2.0], [20.0, 20.0])  # 0.95 / cos^2(24) > 1
    out, cd_profile = vc.strip_profile_drag(strips, polar, sweep_rad=SWEEP, chord_ref=2.0, s_ref=80.0)
    assert math.isnan(cd_profile)
    assert list(out["status"]) == [vc.IN_RANGE, vc.STALLED]


def test_korn_swept_reduces_to_unswept_korn() -> None:
    m_dd, m_crit = vc.korn_swept(0.87, 0.12, 0.5, 0.0)
    assert m_dd == pytest.approx(compressibility.korn_mdd(0.87, 0.12, 0.5))
    assert m_dd - m_crit == pytest.approx((0.1 / 80) ** (1 / 3))


def test_lock_wave_drag_is_zero_below_mcrit_and_hits_divergence_slope_at_mdd() -> None:
    assert vc.lock_wave_drag(0.6, 0.65) == 0.0
    m_dd, m_crit = vc.korn_swept(0.87, 0.12, 0.5, SWEEP)
    h = 1e-6
    slope = (vc.lock_wave_drag(m_dd + h, m_crit) - vc.lock_wave_drag(m_dd - h, m_crit)) / (2 * h)
    assert slope == pytest.approx(0.1, rel=1e-4)


def test_load_section_polar_drops_nonconverged_and_post_stall(tmp_path: Path) -> None:
    polar = pd.DataFrame(
        {
            "alpha": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cl": [np.nan, 0.2, 0.4, 0.6, 0.5, 0.45, 0.4, 0.42],
            "cd": [np.nan, 0.006, 0.007, 0.009, 0.02, 0.03, 0.04, 0.05],
            "cm": [np.nan] + [0.0] * 7,
            "converged": [False] + [True] * 7,
        }
    )
    path = tmp_path / "polar.csv"
    polar.to_csv(path, index=False)
    usable = vc.load_section_polar(path)
    assert list(usable["alpha"]) == [0.0, 1.0, 2.0]
