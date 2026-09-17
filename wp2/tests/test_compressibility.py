# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-09-17
"""Tests for wp2/scripts/compressibility.py.

Verified against structural/limiting properties that can be derived with
confidence (Cp_crit(1.0) = 0 exactly, Karman-Tsien = identity at M=0, etc.),
not against specific textbook table values pulled from memory -- see each
function's docstring for the derivation.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts import compressibility


def test_karman_tsien_identity_at_zero_mach() -> None:
    cp0 = np.array([-0.5, -1.2, 0.3, -2.0])
    result = compressibility.karman_tsien_cp(cp0, 0.0)
    assert result == pytest.approx(cp0)


def test_karman_tsien_amplifies_suction_as_mach_increases() -> None:
    cp0 = -1.0
    at_low_m = compressibility.karman_tsien_cp(cp0, 0.3)
    at_high_m = compressibility.karman_tsien_cp(cp0, 0.7)
    # more negative Cp0 (suction) should be amplified (more negative) at
    # higher Mach -- the whole point of the correction
    assert at_high_m < at_low_m < 0


def test_karman_tsien_rejects_out_of_range_mach() -> None:
    with pytest.raises(ValueError):
        compressibility.karman_tsien_cp(-1.0, 1.0)
    with pytest.raises(ValueError):
        compressibility.karman_tsien_cp(-1.0, -0.1)


def test_cp_crit_is_exactly_zero_at_mach_one() -> None:
    assert compressibility.cp_crit(1.0) == pytest.approx(0.0, abs=1e-9)


def test_cp_crit_is_negative_and_monotonically_increasing_below_mach_one() -> None:
    machs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.99]
    values = [compressibility.cp_crit(m) for m in machs]
    assert all(v < 0.0 for v in values)
    assert all(a < b for a, b in zip(values, values[1:]))


def test_cp_crit_rejects_out_of_range_mach() -> None:
    with pytest.raises(ValueError):
        compressibility.cp_crit(0.0)
    with pytest.raises(ValueError):
        compressibility.cp_crit(1.1)


def test_korn_mdd_matches_equation_directly() -> None:
    # M_dd + t/c + Cl/10 = kappa_A => M_dd = kappa_A - t/c - Cl/10
    mdd = compressibility.korn_mdd(kappa_a=0.87, thickness_to_chord=0.12, cl=0.5867)
    assert mdd == pytest.approx(0.87 - 0.12 - 0.05867)


def test_find_mcrit_detects_crossing() -> None:
    # a single, strongly negative Cp0 that will cross cp_crit(M) somewhere
    # in a normal transonic Mach range
    cp0 = np.array([-0.2, -0.5, -1.5, -0.4])
    mach_grid = np.arange(0.3, 0.90, 0.01)
    mcrit, sweep = compressibility.find_mcrit(cp0, mach_grid)
    assert mcrit is not None
    assert 0.3 < mcrit < 0.90
    # at M_crit, the corrected min should have just reached/crossed cp_crit
    # (mcrit came directly from a mach_grid entry via float(mach), so exact
    # equality against the same DataFrame is safe)
    idx = sweep.index[sweep["mach"] == mcrit][0]
    row = sweep.iloc[idx]
    assert row["cp_min_corrected"] <= row["cp_crit"]
    # the row just before should NOT have crossed yet
    if idx > 0:
        prev = sweep.iloc[idx - 1]
        assert prev["cp_min_corrected"] > prev["cp_crit"]


def test_find_mcrit_returns_none_when_no_crossing_in_grid() -> None:
    # a very mild Cp0 that never gets close to critical in a narrow,
    # conservative Mach range
    cp0 = np.array([-0.05, -0.02])
    mach_grid = np.arange(0.3, 0.5, 0.01)
    mcrit, sweep = compressibility.find_mcrit(cp0, mach_grid)
    assert mcrit is None
    assert len(sweep) == len(mach_grid)
