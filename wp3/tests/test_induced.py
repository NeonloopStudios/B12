# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-10-15
"""Tests for wp3/src/b12wp3/drag/induced.py and wp3/src/b12wp3/config/induced.py.

Spec: ADSEE-II slides 72 (Brandt et al. and Raymer Oswald factors, K = 1/(pi AR e)),
75 and 78 (AR_eff = AR + 1.9 (h/b) AR); Vos (2022) eq. 7.62
(Delta e = 0.0026 delta_f, wing-mounted engines). Reference values are
evaluated by hand from these equations.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from b12wp3.common.planform import TrapezoidalPlanform
from b12wp3.config.induced import REFERENCE_WINGLETS, ReferenceWinglet
from b12wp3.drag.induced import (
    InducedDrag,
    flap_oswald_increment,
    induced_drag_factor,
    mean_height_to_span,
    oswald_factor_brandt,
    oswald_factor_raymer_straight,
    oswald_factor_raymer_swept,
    wing_induced_drag,
    winglet_effective_aspect_ratio,
)

# WP3 wing: S = 66.7 m^2, AR = 9.5, taper 0.4, quarter-chord sweep 30 deg
WING = TrapezoidalPlanform(66.7, 9.5, 0.4, math.radians(30.0), 2)


def test_brandt_unswept_ar_zero_limit_is_half() -> None:
    assert oswald_factor_brandt(1e-9, 0.0) == pytest.approx(0.5)


def test_brandt_unswept_high_ar_tends_to_one() -> None:
    # 2 / (2 - A + sqrt(4 + A^2)) -> 1 for A -> inf
    assert oswald_factor_brandt(1e6, 0.0) == pytest.approx(1.0, abs=1e-5)


def test_brandt_wp3_wing_returns_hand_value() -> None:
    # AR = 9.5, Lambda_c/2 = 28.0236 deg: 2 / (2 - 9.5 + sqrt(4 + 90.25 * 1.28243))
    sweep = math.radians(28.023581804818225)
    tan_sq = math.tan(sweep) ** 2
    expected = 2.0 / (2.0 - 9.5 + math.sqrt(4.0 + 90.25 * (1.0 + tan_sq)))

    assert oswald_factor_brandt(9.5, sweep) == pytest.approx(expected)
    assert oswald_factor_brandt(9.5, sweep) == pytest.approx(0.580377, abs=1e-6)


@given(
    ar=st.floats(min_value=1.0, max_value=20.0),
    sweep_deg=st.floats(min_value=0.0, max_value=60.0),
)
def test_brandt_is_between_zero_and_one(ar: float, sweep_deg: float) -> None:
    e = oswald_factor_brandt(ar, math.radians(sweep_deg))

    assert 0.0 < e <= 1.0


@given(sweep_deg=st.floats(min_value=1.0, max_value=60.0))
def test_brandt_decreases_with_sweep(sweep_deg: float) -> None:
    unswept = oswald_factor_brandt(9.5, 0.0)

    assert oswald_factor_brandt(9.5, math.radians(sweep_deg)) < unswept


def test_brandt_rejects_nonpositive_aspect_ratio() -> None:
    with pytest.raises(ValueError):
        oswald_factor_brandt(0.0, 0.0)


def test_raymer_straight_returns_hand_value() -> None:
    # 1.78 (1 - 0.045 * 9.5^0.68) - 0.64
    expected = 1.78 * (1.0 - 0.045 * 9.5**0.68) - 0.64

    assert oswald_factor_raymer_straight(9.5) == pytest.approx(expected)
    assert oswald_factor_raymer_straight(9.5) == pytest.approx(0.769759, abs=1e-6)


def test_raymer_swept_returns_hand_value() -> None:
    # 4.61 (1 - 0.045 * 9.5^0.68) cos(31.9007 deg)^0.15 - 3.1
    sweep = math.radians(31.90073801234116)
    expected = 4.61 * (1.0 - 0.045 * 9.5**0.68) * math.cos(sweep) ** 0.15 - 3.1

    assert oswald_factor_raymer_swept(9.5, sweep) == pytest.approx(expected)
    assert oswald_factor_raymer_swept(9.5, sweep) == pytest.approx(0.462536, abs=1e-6)


def test_raymer_swept_with_effective_ar_raises_cdi() -> None:
    # e falls faster than AR_eff rises: the reason Brandt is used
    sweep = math.radians(31.9)
    ar, ar_eff = 9.5, 10.6

    k_plain = induced_drag_factor(ar, oswald_factor_raymer_swept(ar, sweep))
    k_eff = induced_drag_factor(ar_eff, oswald_factor_raymer_swept(ar_eff, sweep))

    assert k_eff > k_plain


def test_winglet_effective_aspect_ratio_returns_slide_formula() -> None:
    # AR = 9.5, h = 1.5 m, b = 25 m: 9.5 + 1.9 * 0.06 * 9.5 = 10.583
    assert winglet_effective_aspect_ratio(9.5, 1.5, 25.0) == pytest.approx(10.583)


def test_winglet_effective_aspect_ratio_no_winglet_is_geometric() -> None:
    assert winglet_effective_aspect_ratio(9.5, 0.0, 25.0) == 9.5


def test_winglet_effective_aspect_ratio_rejects_negative_height() -> None:
    with pytest.raises(ValueError):
        winglet_effective_aspect_ratio(9.5, -0.1, 25.0)


def test_induced_drag_factor_returns_definition() -> None:
    assert induced_drag_factor(10.0, 0.8) == pytest.approx(1.0 / (math.pi * 8.0))


def test_flap_increment_returns_vos_value() -> None:
    # 0.0026 * 40 deg
    assert flap_oswald_increment(40.0, 0.0026) == pytest.approx(0.104)


def test_flap_increment_rejects_negative_deflection() -> None:
    with pytest.raises(ValueError):
        flap_oswald_increment(-1.0, 0.0026)


def test_mean_height_to_span_is_arithmetic_mean() -> None:
    refs = (
        ReferenceWinglet("a", 2.0, 40.0, "test"),
        ReferenceWinglet("b", 1.0, 25.0, "test"),
    )

    assert mean_height_to_span(refs) == pytest.approx((0.05 + 0.04) / 2.0)


def test_mean_height_to_span_rejects_empty() -> None:
    with pytest.raises(ValueError):
        mean_height_to_span(())


def test_reference_winglets_return_document_values() -> None:
    by_name = {ref.aircraft: ref for ref in REFERENCE_WINGLETS}

    # A320neo: 2.43 m / 35.80 m; CRJ900: (127.9 - 76.0) in / 915.2 in
    assert by_name["A320neo"].height_to_span == pytest.approx(2.43 / 35.80)
    assert by_name["CRJ900"].height_to_span == pytest.approx(51.9 / 915.2)


def test_wing_induced_drag_uses_brandt_at_effective_aspect_ratio() -> None:
    h = 1.5
    case = wing_induced_drag(WING, h)

    ar_eff = 9.5 * (1.0 + 1.9 * h / WING.span_m)
    expected_e = oswald_factor_brandt(ar_eff, WING.sweep_at(0.5))
    assert case.effective_aspect_ratio == pytest.approx(ar_eff)
    assert case.oswald == pytest.approx(expected_e)
    assert case.k == pytest.approx(1.0 / (math.pi * ar_eff * expected_e))


def test_wing_induced_drag_winglet_lowers_cdi() -> None:
    without = wing_induced_drag(WING, 0.0)
    with_winglet = wing_induced_drag(WING, 1.5)

    assert with_winglet.cdi(0.489) < without.cdi(0.489)


def test_wing_induced_drag_flap_adds_increment() -> None:
    clean = wing_induced_drag(WING, 1.5)
    landing = wing_induced_drag(
        WING, 1.5, name="landing", flap_deflection_deg=40.0, oswald_flap_slope_per_deg=0.0026
    )

    assert landing.oswald - clean.oswald == pytest.approx(0.104)
    assert landing.oswald_clean == pytest.approx(clean.oswald)


def test_induced_drag_cdi_is_k_cl_squared() -> None:
    case = InducedDrag("t", 9.5, 10.0, 0.5, 0.0, 0.6, 0.6)

    assert case.cdi(0.5) == pytest.approx(0.25 / (math.pi * 10.0 * 0.6))


def test_induced_drag_rejects_effective_below_geometric() -> None:
    with pytest.raises(ValueError):
        InducedDrag("t", 9.5, 9.0, 0.5, 0.0, 0.6, 0.6)
