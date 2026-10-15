# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/skin_friction.py.

Spec: ADSEE-II slide 61, "Flat plate skin friction coefficient". The laminar
value is the Blasius flat-plate result, Cf = 1.328 / sqrt(Re). The weighted
average follows the slide example: 10 % laminar gives
Cf = 0.1 Cf_laminar + 0.9 Cf_turbulent.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.config.mission import FlowRegime
from b12wp3.drag.skin_friction import (
    cf_laminar,
    cf_mixed,
    cf_turbulent,
    effective_reynolds,
    reynolds_actual,
    reynolds_cutoff,
)

REYNOLDS = st.floats(min_value=1e4, max_value=1e9)
MACH = st.floats(min_value=0.0, max_value=0.95)


def test_cf_laminar_re_1e6_returns_blasius_value() -> None:
    cf = cf_laminar(1e6)

    assert cf == pytest.approx(1.328e-3)


def test_cf_turbulent_re_1e7_zero_mach_returns_hand_value() -> None:
    # 0.455 / 7^2.58
    cf = cf_turbulent(1e7, 0.0)

    assert cf == pytest.approx(0.003003713133159046)


def test_cf_turbulent_mach_correction_returns_slide_factor() -> None:
    ratio = cf_turbulent(1e7, 0.77) / cf_turbulent(1e7, 0.0)

    assert ratio == pytest.approx((1.0 + 0.144 * 0.77**2) ** -0.65)


@pytest.mark.parametrize(
    ("fraction", "laminar_weight"),
    [(0.0, 0.0), (0.1, 0.1), (0.05, 0.05), (1.0, 1.0)],
)
def test_cf_mixed_weights_laminar_and_turbulent(
    fraction: float, laminar_weight: float
) -> None:
    re, mach = 2e7, 0.77

    cf = cf_mixed(re, mach, fraction)

    expected = laminar_weight * cf_laminar(re) + (1.0 - laminar_weight) * cf_turbulent(
        re, mach
    )
    assert cf == pytest.approx(expected)


def test_reynolds_actual_returns_rho_v_l_over_mu() -> None:
    re = reynolds_actual(1.2, 50.0, 2.0, 1.8e-5)

    assert re == pytest.approx(1.2 * 50.0 * 2.0 / 1.8e-5)


def test_reynolds_cutoff_subsonic_returns_hand_value() -> None:
    # 38.21 * (1 / 1e-5)^1.053
    re = reynolds_cutoff(1.0, 1e-5, 0.3, FlowRegime.SUBSONIC)

    assert re == pytest.approx(7033589.817612875)


def test_reynolds_cutoff_transonic_returns_hand_value() -> None:
    # 44.62 * (1 / 1e-5)^1.053 * 0.77^1.16
    re = reynolds_cutoff(1.0, 1e-5, 0.77, FlowRegime.TRANSONIC)

    assert re == pytest.approx(6065391.064774214)


@pytest.mark.parametrize(("actual", "cutoff"), [(5e7, 6e6), (3e6, 6e6)])
def test_effective_reynolds_returns_smaller_value(actual: float, cutoff: float) -> None:
    re = effective_reynolds(actual, cutoff)

    assert re == min(actual, cutoff)


@pytest.mark.parametrize(
    "call",
    [
        lambda: cf_laminar(0.0),
        lambda: cf_turbulent(1.0, 0.5),
        lambda: cf_turbulent(1e7, -0.1),
        lambda: cf_mixed(1e7, 0.5, 1.1),
        lambda: cf_mixed(1e7, 0.5, -0.1),
        lambda: reynolds_actual(1.2, 50.0, 0.0, 1.8e-5),
        lambda: reynolds_cutoff(1.0, 0.0, 0.5, FlowRegime.SUBSONIC),
        lambda: reynolds_cutoff(1.0, 1e-5, 0.0, FlowRegime.TRANSONIC),
        lambda: effective_reynolds(math.nan, 1e6),
    ],
)
def test_skin_friction_invalid_input_raises(call: object) -> None:
    assert callable(call)

    with pytest.raises(ValueError):
        call()


@given(re_a=REYNOLDS, re_b=REYNOLDS, mach=MACH)
def test_cf_decreases_with_reynolds(re_a: float, re_b: float, mach: float) -> None:
    low, high = sorted((re_a, re_b))

    laminar = (cf_laminar(low), cf_laminar(high))
    turbulent = (cf_turbulent(low, mach), cf_turbulent(high, mach))

    assert laminar[1] <= laminar[0]
    assert turbulent[1] <= turbulent[0]


@given(re=REYNOLDS, mach=MACH, fraction=st.floats(min_value=0.0, max_value=1.0))
def test_cf_mixed_lies_between_laminar_and_turbulent(
    re: float, mach: float, fraction: float
) -> None:
    bounds = sorted((cf_laminar(re), cf_turbulent(re, mach)))

    cf = cf_mixed(re, mach, fraction)

    assert bounds[0] * (1 - 1e-12) <= cf <= bounds[1] * (1 + 1e-12)
